import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from nlb_tools.evaluation import (
    bits_per_spike,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)
from scipy.linalg import LinAlgWarning
from sklearn.decomposition import PCA

from .utils import send_batch_to_device

plt.switch_backend("Agg")


def get_tensorboard_summary_writer(loggers):
    """Gets the TensorBoard SummaryWriter from a logger
    or logger collection to allow writing of images.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search for the
        SummaryWriter.

    Returns
    -------
    torch.utils.tensorboard.writer.SummaryWriter
        The SummaryWriter object.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
            return logger.experiment
    else:
        return None


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, n_samples=2, log_every_n_epochs=20):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 2
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 20
        """
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for the TensorBoard SummaryWriter
        writer = get_tensorboard_summary_writer(trainer.loggers)
        if writer is None:
            return
        # Get data samples from the dataloaders
        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Move data to the right device
        batch = send_batch_to_device(batch, pl_module.device)
        # Compute model output
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=False,
        )
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Log a few example outputs for each session
        for s in sessions:
            # Convert everything to numpy
            encod_data = batch[s].encod_data.detach().cpu().numpy()
            recon_data = batch[s].recon_data.detach().cpu().numpy()
            truth = batch[s].truth.detach().cpu().numpy()
            means = output[s].output_params.detach().cpu().numpy()
            inputs = output[s].gen_inputs.detach().cpu().numpy()
            # Compute data sizes
            _, steps_encod, neur_encod = encod_data.shape
            _, steps_recon, neur_recon = recon_data.shape
            # Decide on how to plot panels
            if np.all(np.isnan(truth)):
                plot_arrays = [recon_data, means, inputs]
                height_ratios = [3, 3, 1]
            else:
                plot_arrays = [recon_data, truth, means, inputs]
                height_ratios = [3, 3, 3, 1]
            # Create subplots
            fig, axes = plt.subplots(
                len(plot_arrays),
                self.n_samples,
                sharex=True,
                sharey="row",
                figsize=(3 * self.n_samples, 10),
                gridspec_kw={"height_ratios": height_ratios},
            )
            for i, ax_col in enumerate(axes.T):
                for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                    if j < len(plot_arrays) - 1:
                        ax.imshow(array[i].T, interpolation="none", aspect="auto")
                        ax.vlines(steps_encod, 0, neur_recon, color="orange")
                        ax.hlines(neur_encod, 0, steps_recon, color="orange")
                        ax.set_xlim(0, steps_recon)
                        ax.set_ylim(0, neur_recon)
                    else:
                        ax.plot(array[i])
            plt.tight_layout()
            # Log the plot to tensorboard
            writer.add_figure(f"raster_plot/sess{s}", fig, trainer.global_step)


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for the TensorBoard SummaryWriter
        writer = get_tensorboard_summary_writer(trainer.loggers)
        if writer is None:
            return
        # Get only the validation dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        dataloaders = {s: dls["valid"] for s, dls in pred_dls.items()}
        # Compute outputs and plot for one session at a time
        for s, dataloader in dataloaders.items():
            latents = []
            for batch in dataloader:
                # Move data to the right device
                batch = send_batch_to_device({s: batch}, pl_module.device)
                # Perform the forward pass through the model
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                latents.append(output.factors)
            latents = torch.cat(latents).detach().cpu().numpy()
            # Reduce dimensionality if necessary
            n_samp, n_step, n_lats = latents.shape
            if n_lats > 3:
                latents_flat = latents.reshape(-1, n_lats)
                pca = PCA(n_components=3)
                latents = pca.fit_transform(latents_flat)
                latents = latents.reshape(n_samp, n_step, 3)
                explained_variance = np.sum(pca.explained_variance_ratio_)
            else:
                explained_variance = 1.0
            # Create figure and plot trajectories
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for traj in latents:
                ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
            ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
            ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
            ax.set_title(f"explained variance: {explained_variance:.2f}")
            plt.tight_layout()
            # Log the plot to tensorboard
            writer.add_figure(f"trajectory_plot/sess{s}", fig, trainer.global_step)


class NLBEvaluation(pl.Callback):
    """Computes and logs all evaluation metrics for the Neural Latents
    Benchmark to tensorboard. These include `co_bps`, `fp_bps`,
    `behavior_r2`, `psth_r2`, and `tp_corr`.
    """

    def __init__(self, log_every_n_epochs=20, decoding_cv_sweep=False):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        decoding_cv_sweep : bool, optional
            Whether to run a cross-validated hyperparameter sweep to
            find optimal regularization values, by default False
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.decoding_cv_sweep = decoding_cv_sweep

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get only the validation dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        dataloaders = {s: dls["valid"] for s, dls in pred_dls.items()}
        # Compute outputs and plot for one session at a time
        for s, dataloader in dataloaders.items():
            # Get entire validation dataset from datamodule
            (input_data, recon_data, *_), (behavior,) = trainer.datamodule.valid_data[s]
            recon_data = recon_data.detach().cpu().numpy()
            behavior = behavior.detach().cpu().numpy()
            # Pass the data through the model
            rates = []
            # TODO: Replace this with Trainer.predict? Hesitation is that switching to
            # Trainer.predict for posterior sampling is inefficient because we can't
            # tell it how many forward passes to use.
            for batch in dataloader:
                batch = send_batch_to_device({s: batch}, pl_module.device)
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                rates.append(output.output_params)
            rates = torch.cat(rates).detach().cpu().numpy()
            # Compute co-smoothing bits per spike
            _, n_obs, n_heldin = input_data.shape
            heldout = recon_data[:, :n_obs, n_heldin:]
            rates_heldout = rates[:, :n_obs, n_heldin:]
            co_bps = bits_per_spike(rates_heldout, heldout)
            pl_module.log(f"nlb/co_bps/sess{s}", max(co_bps, -1.0))
            # Compute forward prediction bits per spike
            forward = recon_data[:, n_obs:]
            rates_forward = rates[:, n_obs:]
            fp_bps = bits_per_spike(rates_forward, forward)
            pl_module.log(f"nlb/fp_bps/sess{s}", max(fp_bps, -1.0))
            # Get relevant training dataset from datamodule
            _, (train_behavior,) = trainer.datamodule.train_data[s]
            train_behavior = train_behavior.detach().cpu().numpy()
            # Get model predictions for the training dataset
            train_dataloader = pred_dls[s]["train"]
            train_rates = []
            for batch in train_dataloader:
                batch = send_batch_to_device({s: batch}, pl_module.device)
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                train_rates.append(output.output_params)
            train_rates = torch.cat(train_rates).detach().cpu().numpy()
            # Get firing rates for observed time points
            rates_obs = rates[:, :n_obs]
            train_rates_obs = train_rates[:, :n_obs]
            # Compute behavioral decoding performance
            if "dmfc_rsg" in trainer.datamodule.hparams.dataset_name:
                tp_corr = speed_tp_correlation(heldout, rates_obs, behavior)
                pl_module.log(f"nlb/tp_corr/sess{s}", tp_corr)
            else:
                with warnings.catch_warnings():
                    # Ignore LinAlgWarning from early in training
                    warnings.filterwarnings("ignore", category=LinAlgWarning)
                    behavior_r2 = velocity_decoding(
                        train_rates_obs,
                        train_behavior,
                        trainer.datamodule.train_decode_mask,
                        rates_obs,
                        behavior,
                        trainer.datamodule.eval_decode_mask,
                        self.decoding_cv_sweep,
                    )
                pl_module.log(f"nlb/behavior_r2/sess{s}", max(behavior_r2, -1.0))
            # Compute PSTH reconstruction performance
            if hasattr(trainer.datamodule, "psth"):
                psth = trainer.datamodule.psth
                cond_idxs = trainer.datamodule.val_cond_idxs
                jitter = trainer.datamodule.eval_jitter
                psth_r2 = eval_psth(psth, rates_obs, cond_idxs, jitter)
                pl_module.log(f"nlb/psth_r2/sess{s}", max(psth_r2, -1.0))
