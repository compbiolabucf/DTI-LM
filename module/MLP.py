import torch.nn as nn
import torch,hydra,sys
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchmetrics 
from torch_geometric.nn import MLP
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(pl.LightningModule):
    def __init__(self, cfg,dataset,network,optimizer,criterion):
        super().__init__()
        activation = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()}
        activation_fn = activation[network['activation_fn']]

        '''
        linears = []
        linears.append(nn.Linear(network['drug_dim']+network['prot_dim'],network['layers'][0]))

        for i in range(len(network['layers'])-1):
            linears.append(nn.BatchNorm1d(network['layers'][i]))
            linears.append(activation_fn)
            linears.append(nn.Dropout(network['dropout']))
            linears.append(nn.Linear(network['layers'][i],network['layers'][i+1]))

        linears.append(nn.BatchNorm1d(network['layers'][-1]))
        linears.append(activation_fn)
        linears.append(nn.Dropout(network['dropout']))
        linears.append(nn.Linear(network['layers'][-1],network['output_dim']))
        linears.append(nn.Sigmoid())

        self.model = nn.Sequential(*linears)
        '''

        layers = [network['drug_dim']+network['prot_dim']]+network['layers']+[network['output_dim']]
        layers = list(layers)
        self.model = MLP(layers, dropout=network['dropout'])

        self.optimizer = optimizer
        self.criterion: torch.nn = hydra.utils.instantiate(criterion)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_auc = 0
        self.test_auprc = 0
        self.test_f1 = 0
        self.save_hyperparameters(cfg)
    

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        _, y = batch
        loss, scores = self.common_step(batch, batch_idx)
        self.training_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):    
        train_loss, train_auc = self.common_epoch_end(self.training_step_outputs)
        self.training_step_outputs.clear()
        self.log_dict({"train_loss":train_loss, "train_auc":train_auc})
        
    def validation_step(self, batch, batch_idx):
        _, y = batch
        loss, scores = self.common_step(batch, batch_idx)
        self.validation_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        return loss

    def on_validation_epoch_end(self):
        val_loss, val_auc = self.common_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss":val_loss, "val_auc":val_auc}, prog_bar=True)
  
    def test_step(self, batch, batch_idx):
        _, y = batch
        loss, scores = self.common_step(batch, batch_idx)
        self.test_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        return loss

    def on_test_epoch_end(self):
        test_loss, test_auc, test_auprc, test_bcm, test_f1 = self.for_test_epoch(self.test_step_outputs)
        print(test_bcm)
        self.test_auc, self.test_auprc, self.test_f1 = test_auc, test_auprc, test_f1
        self.test_step_outputs.clear()
        self.log_dict({"test_auc":test_auc, "test_auprc":test_auprc, "test_f1":test_f1})

    def common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.criterion(scores, y)
        return loss, scores

    def common_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        metric1 = torchmetrics.classification.BinaryAUROC(thresholds = None)
        auc = metric1(scores, y)
        return avg_loss, auc
    
    def for_test_epoch(self, outputs):          
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        metric1 = torchmetrics.classification.BinaryAUROC(thresholds = None)
        auc = metric1(scores, y)
        metric2 = torchmetrics.classification.BinaryAveragePrecision(thresholds = None)
        auprc = metric2(scores, y.long())
        metric3 = torchmetrics.classification.BinaryConfusionMatrix(threshold=0.5).to(device)
        bcm = metric3(scores.to(device), y.to(device))
        metric4 = torchmetrics.classification.BinaryF1Score(threshold=0.5).to(device)
        f1 = metric4(scores.to(device), y.to(device))
        return avg_loss, auc, auprc, bcm, f1

    def configure_optimizers(self):
        if self.optimizer['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer['lr'], weight_decay=self.optimizer['weight_decay'])
        elif self.optimizer['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizer['lr'], weight_decay=self.optimizer['weight_decay'],momentum = 0.9)
        elif self.optimizer['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizer['lr'], weight_decay=self.optimizer['weight_decay'], momentum=0.9)
        else:
            print("optimizer not recognized")
            sys.exit()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

         
