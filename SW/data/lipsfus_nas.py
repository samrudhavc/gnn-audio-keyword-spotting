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

from data.dataset_lipsfus import SpikingDS

device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')

class LIPSFUS(L.LightningDataModule):
    def __init__(self,
                 config):
        super().__init__()

        self.data_dir = config.general.data_dir
        self.config = config

    def setup(self, stage=None):
        train_data = glob.glob(os.path.join(self.data_dir, 'Train/*/*.aedat'))
        test_data = glob.glob(os.path.join(self.data_dir, 'Test/*/*.aedat'))

        if self.config.preprocess.create_val:
            train_data = np.random.permutation(train_data)
            n_val = int(len(train_data) * self.config.preprocess.val_split)
            val_data = train_data[:n_val]
            train_data = train_data[n_val:]

        try:
            self.val_data = SpikingDS(val_data, self.config)
            print(f'Using {len(val_data)} samples as validation data')
        except:
            self.val_data = SpikingDS(test_data, self.config)
            print('Using test data as validation data')

        self.train_data = SpikingDS(train_data, self.config, train=True)
        self.test_data = SpikingDS(test_data, self.config)

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                            batch_size=self.config.train.batch_size,
                            shuffle=True,
                            num_workers=self.config.train.num_workers,
                            persistent_workers=True,
                            # persistent_workers=True if self.config.train.num_workers > 0 else False,
                            collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                            batch_size=self.config.train.batch_size,
                            shuffle=False,
                            num_workers=self.config.train.num_workers,
                            persistent_workers=True,
                            collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                            batch_size=self.config.train.batch_size,
                            shuffle=False,
                            num_workers=self.config.train.num_workers,
                            persistent_workers=True,
                            collate_fn=self.collate_fn)
    
    @staticmethod
    def collate_fn(data_list):
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch