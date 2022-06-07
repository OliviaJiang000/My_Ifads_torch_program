import pytorch_lightning as pl
import torch
from torch import nn

from ..metrics import ExpSmoothedMetric, r2_score
from ..utils import transpose_lists
from .modules import augmentations
from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.l2 import compute_l2_penalty


class LFADS(pl.LightningModule):
    def __init__(
        self,
        encod_data_dim: int,
        encod_seq_len: int,
        recon_seq_len: int,
        ext_input_dim: int,
        ic_enc_seq_len: int,
        ic_enc_dim: int,
        ci_enc_dim: int,
        ci_lag: int,
        con_dim: int,
        co_dim: int,
        ic_dim: int,
        gen_dim: int,
        fac_dim: int,
        dropout_rate: float,
        reconstruction: nn.ModuleList,
        co_prior: nn.Module,
        ic_prior: nn.Module,
        ic_post_var_min: float,
        cell_clip: float,
        train_aug_stack: augmentations.AugmentationStack,
        infer_aug_stack: augmentations.AugmentationStack,
        readin: nn.ModuleList,
        readout: nn.ModuleList,
        loss_scale: float,
        recon_reduce_mean: bool,
        lr_scheduler: bool,
        lr_init: float,
        lr_stop: float,
        lr_decay: float,
        lr_patience: int,
        lr_adam_epsilon: float,
        l2_start_epoch: int,
        l2_increase_epoch: int,
        l2_ic_enc_scale: float,
        l2_ci_enc_scale: float,
        l2_gen_scale: float,
        l2_con_scale: float,
        kl_start_epoch: int,
        kl_increase_epoch: int,
        kl_ic_scale: float,
        kl_co_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["ic_prior", "co_prior", "reconstruction", "readin", "readout"],
        )
        # Store `co_prior` on `hparams` so it can be accessed in decoder
        self.hparams.co_prior = co_prior
        # Make sure the nn.ModuleList arguments are all the same length
        assert len(readin) == len(readout) == len(reconstruction)

        # Store the readin network
        self.readin = readin
        # Decide whether to use the controller
        self.use_con = all([ci_enc_dim > 0, con_dim > 0, co_dim > 0])
        # Create the encoder and decoder
        self.encoder = Encoder(hparams=self.hparams)
        self.decoder = Decoder(hparams=self.hparams)
        # Store the readout network
        self.readout = readout
        # Create object to manage reconstruction
        self.recon = reconstruction
        # Store the trainable priors
        self.ic_prior = ic_prior
        if self.use_con:
            self.co_prior = co_prior
        # Create metric for exponentially-smoothed `valid/recon`
        self.valid_recon_smth = ExpSmoothedMetric()
        # Store the data augmentation stacks
        self.train_aug_stack = train_aug_stack
        self.infer_aug_stack = infer_aug_stack

    def forward(self, data, ext_input, sample_posteriors=False, output_means=True):
        # Use list of tensors to allow efficient computation on data of different sizes
        assert type(data) == list and len(data) == len(self.readin)
        # Keep track of sizes so we can split the data
        split_ixs = [len(d) for d in data]
        # Pass the data through the readin networks
        encod_data = torch.cat([readin(d) for readin, d in zip(self.readin, data)])
        # Pass the data through the encoders
        ic_mean, ic_std, ci = self.encoder(encod_data)
        # Create the posterior distribution over initial conditions
        ic_post = self.ic_prior.make_posterior(ic_mean, ic_std)
        # Choose to take a sample or to pass the mean
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean
        # Unroll the decoder to estimate latent states
        (
            gen_init,
            gen_states,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
            factors,
        ) = self.decoder(ic_samp, ci, ext_input, sample_posteriors=sample_posteriors)
        # Convert the factors representation into output distribution parameters
        facs_split = torch.split(factors, split_ixs)
        output_params = [readout(f) for readout, f in zip(self.readout, facs_split)]
        # Separate parameters of the output distribution
        output_params = [
            recon.reshape_output_params(op)
            for recon, op in zip(self.recon, output_params)
        ]
        # Convert the output parameters to means if requested
        if output_means:
            output_params = [
                recon.compute_means(op) for recon, op in zip(self.recon, output_params)
            ]
        # Return the parameter estimates and all intermediate activations
        return (
            output_params,
            factors,
            ic_mean,
            ic_std,
            co_means,
            co_stds,
            gen_states,
            gen_init,
            gen_inputs,
            con_states,
        )

    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=hps.lr_init)
        if hps.lr_scheduler:
            # Create a scheduler to reduce the learning rate over time
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=hps.lr_decay,
                patience=hps.lr_patience,
                threshold=0.0,
                min_lr=hps.lr_stop,
                eps=hps.lr_adam_epsilon,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid/recon_smth",
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        hps = self.hparams
        # Process the batch for each session
        batch = [self.train_aug_stack.process_batch(sess_b) for sess_b in batch]
        # Unpack the batch
        encod_data, recon_data, ext_input, truth, *_ = transpose_lists(batch)
        # Perform the forward pass
        output_params, _, ic_mean, ic_std, co_means, co_stds, *_ = self.forward(
            encod_data,
            torch.cat(ext_input),
            sample_posteriors=True,
            output_means=False,
        )
        # Compute the reconstruction loss
        recon_all = [
            recon.compute_loss(rd, op)
            for recon, rd, op in zip(self.recon, recon_data, output_params)
        ]
        # Apply losses processing
        recon_all = [
            self.train_aug_stack.process_losses(ra, sess_b, self.log, "train")
            for ra, sess_b in zip(recon_all, batch)
        ]
        # Aggregate the heldout cost for logging
        if not hps.recon_reduce_mean:
            recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all]
        # Compute reconstruction loss for each session
        sess_recon = [ra.mean() for ra in recon_all]
        recon = torch.mean(torch.stack(sess_recon))
        # Compute the L2 penalty on recurrent weights
        l2 = compute_l2_penalty(self, self.hparams)
        l2_ramp = (self.current_epoch - hps.l2_start_epoch) / (
            hps.l2_increase_epoch + 1
        )
        # Compute the KL penalty on posteriors
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.hparams.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * self.hparams.kl_co_scale
        kl_ramp = (self.current_epoch - hps.kl_start_epoch) / (
            hps.kl_increase_epoch + 1
        )
        # Clamp the ramps
        l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        kl_ramp = torch.clamp(torch.tensor(kl_ramp), 0, 1)
        # Compute the final loss
        loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))
        # Compute the reconstruction accuracy, if applicable
        output_means = [
            recon.compute_means(op) for recon, op in zip(self.recon, output_params)
        ]
        r2 = torch.mean(
            torch.stack([r2_score(om, t) for om, t in zip(output_means, truth)])
        )
        # Compute batch sizes for logging
        batch_sizes = [len(d) for d in encod_data]
        # Log per-session metrics
        for i, (value, batch_size) in enumerate(zip(sess_recon, batch_sizes)):
            self.log(
                name=f"train/recon/sess{i}",
                value=value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        # Log overall metrics
        self.log_dict(
            {
                "train/loss": loss,
                "train/recon": recon,
                "train/r2": r2,
                "train/wt_l2": l2,
                "train/wt_l2/ramp": l2_ramp,
                "train/wt_kl": ic_kl + co_kl,
                "train/wt_kl/ic": ic_kl,
                "train/wt_kl/co": co_kl,
                "train/wt_kl/ramp": kl_ramp,
            },
            on_step=False,
            on_epoch=True,
            batch_size=sum(batch_sizes),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        hps = self.hparams
        # Process the batch for each session
        batch = [self.infer_aug_stack.process_batch(sess_b) for sess_b in batch]
        # Unpack the batch
        encod_data, recon_data, ext_input, truth, *_ = transpose_lists(batch)
        # Perform the forward pass
        output_params, _, ic_mean, ic_std, co_means, co_stds, *_ = self.forward(
            encod_data,
            torch.cat(ext_input),
            sample_posteriors=True,
            output_means=False,
        )
        # Compute the reconstruction loss
        recon_all = [
            recon.compute_loss(rd, op)
            for recon, rd, op in zip(self.recon, recon_data, output_params)
        ]
        # Apply losses processing
        recon_all = [
            self.infer_aug_stack.process_losses(ra, sess_b, self.log, "valid")
            for ra, sess_b in zip(recon_all, batch)
        ]
        # Aggregate the reconstruction cost
        if not hps.recon_reduce_mean:
            recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all]
        # Compute reconstruction loss for each session
        sess_recon = [ra.mean() for ra in recon_all]
        recon = torch.mean(torch.stack(sess_recon))
        # Update the smoothed reconstruction loss
        self.valid_recon_smth.update(recon)
        # Compute the L2 penalty on recurrent weights
        l2 = compute_l2_penalty(self, self.hparams)
        l2_ramp = (self.current_epoch - hps.l2_start_epoch) / (
            hps.l2_increase_epoch + 1
        )
        # Compute the KL penalty on posteriors
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.hparams.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * self.hparams.kl_co_scale
        kl_ramp = (self.current_epoch - hps.kl_start_epoch) / (
            hps.kl_increase_epoch + 1
        )
        # Clamp the ramps
        l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        kl_ramp = torch.clamp(torch.tensor(kl_ramp), 0, 1)
        # Compute the final loss
        loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))
        # Compute the reconstruction accuracy, if applicable
        output_means = [
            recon.compute_means(op) for recon, op in zip(self.recon, output_params)
        ]
        r2 = torch.mean(
            torch.stack([r2_score(om, t) for om, t in zip(output_means, truth)])
        )
        # Compute batch sizes for logging
        batch_sizes = [len(d) for d in encod_data]
        # Log per-session metrics
        for i, (value, batch_size) in enumerate(zip(sess_recon, batch_sizes)):
            self.log(
                name=f"valid/recon/sess{i}",
                value=value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        # Log overall metrics
        self.log_dict(
            {
                "valid/loss": loss,
                "valid/recon": recon,
                "valid/recon_smth": self.valid_recon_smth,
                "valid/r2": r2,
                "valid/wt_l2": l2,
                "valid/wt_l2/ramp": l2_ramp,
                "valid/wt_kl": ic_kl + co_kl,
                "valid/wt_kl/ic": ic_kl,
                "valid/wt_kl/co": co_kl,
                "valid/wt_kl/ramp": kl_ramp,
                "hp_metric": recon,
                "cur_epoch": float(self.current_epoch),
            },
            on_step=False,
            on_epoch=True,
            batch_size=sum(batch_sizes),
        )

        return loss

    def predict_step(self, batch, batch_ix, sample_posteriors=True):
        # Process the batch for each session
        batch = [self.infer_aug_stack.process_batch(sess_b) for sess_b in batch]
        # Reset to clear any saved masks
        self.infer_aug_stack.reset()
        # Unpack the batch
        encod_data, _, ext_input, *_ = transpose_lists(batch)
        # Perform the forward pass
        return self.forward(
            encod_data,
            torch.cat(ext_input),
            sample_posteriors=sample_posteriors,
            output_means=True,
        )
