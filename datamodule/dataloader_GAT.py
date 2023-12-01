import torch,sys
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils import utils
import numpy as np 

# pytorch datalaoder
class MyDataset(Dataset):
    def __init__(self, drug, target, DTI):
        self.drug = drug
        self.target = target
        self.DTI = DTI
        
    def __getitem__(self, index):
        y = self.DTI.iloc[index, 2]
        drug_index = self.DTI.iloc[index, 0]
        target_index = self.DTI.iloc[index, 1]
        x1 = self.drug.iloc[drug_index].values
        x2 = self.target.iloc[target_index].values
        return torch.tensor(x1).float(), torch.tensor(x2).float(), torch.tensor(y).float(), drug_index, target_index

    def __len__(self):
        return len(self.DTI)


class UNIDataModule(pl.LightningDataModule):
    def __init__(self,config,dataset,dm_cfg,splitting,serializer):
        super().__init__()
        self.X_drug = dataset['X_drug']
        self.X_target = dataset['X_target']
        self.train_ind = dataset['train']
        self.val_ind = dataset['val']
        self.test_ind = dataset['test']    
        self.batch_size = dm_cfg['batch_size']
        self.num_workers = dm_cfg['num_workers']
        self.config = config
        
    def setup(self, stage):
        self.train_dataset = MyDataset(self.X_drug, self.X_target, self.train_ind)
        self.val_dataset = MyDataset(self.X_drug, self.X_target, self.val_ind)
        self.test_dataset = MyDataset(self.X_drug, self.X_target, self.test_ind)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.batch_size , shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, drop_last=False)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, drop_last=False)
