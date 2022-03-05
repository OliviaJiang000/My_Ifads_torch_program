import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

plt.switch_backend("Agg")


def get_tensorboard_summary_writer(writers):
    """Gets the TensorBoard SummaryWriter from a logger
    or logger collection to allow writing of images.

    Parameters
    ----------
    writers : obj or list[obj]
        An object or list of objects to search for the
        SummaryWriter.

    Returns
    -------
    torch.utils.tensorboard.writer.SummaryWriter
        The SummaryWriter object.
    """
    writer_list = writers if isinstance(writers, list) else [writers]
    for writer in writer_list:
        if isinstance(writer, torch.utils.tensorboard.writer.SummaryWriter):
            return writer
    else:
        return None


def fig_to_rgb_array(fig):
    """Converts a matplotlib figure into an array
    that can be logged to tensorboard.
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be converted.
    Returns
    -------
    np.array
        The figure as an HxWxC array of pixel values.
    """
    # Convert the figure to a numpy array
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        fig_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = fig_data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def batch_fwd(model, batch):
    """Performs the forward pass for a given data batch.

    Parameters
    ----------
    model : lfads_torch.models.base_model.LFADS
        The model to pass data through.
    batch : tuple[torch.Tensor]
        A tuple of batched input tensors.

    Returns
    -------
    tuple[torch.Tensor]
        A tuple of batched output tensors.
    """
    input_data, ext = batch[0], batch[3]
    return model(
        input_data.to(model.device),
        ext.to(model.device),
        sample_posteriors=False,
    )


def get_batch_fwd():
    """Utility function for accessing the `batch_fwd` function
    from `hydra` configs.
    """
    return batch_fwd


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
        writer = get_tensorboard_summary_writer(trainer.logger.experiment)
        if writer is None:
            return
        # Get data samples
        dataloader = trainer.datamodule.val_dataloader()
        encod_data, recon_data, _, ext, truth, *_ = next(iter(dataloader))
        # Compute model output
        means, *_, inputs, _ = pl_module(
            encod_data.to(pl_module.device),
            ext.to(pl_module.device),
            sample_posteriors=False,
        )
        # Convert everything to numpy
        recon_data = recon_data.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        means = means.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()
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
            for ax, array in zip(ax_col, plot_arrays):
                ax.imshow(array[i].T, interpolation="none", aspect="auto")
        plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        writer.add_image("raster_plot", im, trainer.global_step, dataformats="HWC")
