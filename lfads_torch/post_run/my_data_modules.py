# 保存在 lfads_torch/datamodules_myh5.py

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import torch
import h5py

class MyH5DataModule(LightningDataModule):
    def __init__(self, h5_path, batch_size=64):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        with h5py.File(self.h5_path, 'r') as f:
            spikes = f['spikes'][:]  # 改成你真正的键
        spikes = torch.tensor(spikes).float()
        n = spikes.shape[0]
        self.train_dataset = TensorDataset(spikes[:int(n*0.8)])
        self.val_dataset = TensorDataset(spikes[int(n*0.8):])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
