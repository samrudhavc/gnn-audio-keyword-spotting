import os
import glob
import h5py
import numpy as np
import torch
import lightning as L
import torch_geometric


from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from data.dataset_tdigits import SpikingDS

device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')

class TDIGITS(L.LightningDataModule):
    def __init__(self,
                 config):
        super().__init__()

        self.data_dir = config.general.data_dir
        self.config = config

    def setup(self, stage=None):
        file = os.path.join(self.data_dir, 'n-tidigits.hdf5')
        self.train_data = SpikingDS(file, self.config, train=True)
        self.test_data = SpikingDS(file, self.config)

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                            batch_size=self.config.train.batch_size,
                            shuffle=True,
                            num_workers=self.config.train.num_workers,
                            # persistent_workers=self.config.train.num_workers > 0,
                            collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.test_data,
                            batch_size=self.config.train.batch_size,
                            shuffle=False,
                            num_workers=self.config.train.num_workers,
                            persistent_workers=self.config.train.num_workers > 0,
                            collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                            batch_size=self.config.train.batch_size,
                            shuffle=False,
                            num_workers=self.config.train.num_workers,
                            persistent_workers=self.config.train.num_workers > 0,
                            collate_fn=self.collate_fn)
    
    @staticmethod
    def collate_fn(data_list):
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch