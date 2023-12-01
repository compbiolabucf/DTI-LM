import torch,sys
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np 

# pytorch datalaoder
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return torch.tensor(x).float(), torch.tensor(y).float()

    def __len__(self):
        return len(self.data)


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
        '''
        self.X_train = dataset['train']['X']
        self.y_train = dataset['train']['y']
        self.X_val = dataset['val']['X']
        self.y_val = dataset['val']['y']
        self.X_test = dataset['test']['X']
        self.y_test = dataset['test']['y']
        self.batch_size = dm_cfg['batch_size']
        self.num_workers = dm_cfg['num_workers']
        self.config = config
        '''
    def prepare_data(self):
        #pos_weight = np.unique(self.y_train, return_counts=True)[1][0]/np.unique(self.y_train, return_counts=True)[1][1]
        #self.config['module']['criterion']['pos_weight'] = torch.tensor(pos_weight).float()
        self.X_train = np.concatenate((self.X_drug.values[self.train_ind.Drug_ID],self.X_target.values[self.train_ind.Prot_ID]),1)
        self.y_train = np.expand_dims(self.train_ind.label.values,axis=1)
        self.X_val = np.concatenate((self.X_drug.values[self.val_ind.Drug_ID],self.X_target.values[self.val_ind.Prot_ID]),1)
        self.y_val = np.expand_dims(self.val_ind.label.values,axis=1)
        self.X_test = np.concatenate((self.X_drug.values[self.test_ind.Drug_ID],self.X_target.values[self.test_ind.Prot_ID]),1)
        self.y_test = np.expand_dims(self.test_ind.label.values,axis=1)

    def setup(self, stage):
        self.train_dataset = MyDataset(self.X_train, self.y_train)
        self.val_dataset = MyDataset(self.X_val, self.y_val)
        self.test_dataset = MyDataset(self.X_test, self.y_test)

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
