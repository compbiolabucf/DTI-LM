import torch.nn as nn
import torch,hydra,sys
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchmetrics 
from torch_geometric.nn import GATConv,MLP,GCNConv
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNMODEL(nn.Module):
    def __init__(self,params):
        super(GCNMODEL, self).__init__()
        self.layers = nn.ModuleList([GCNConv(in_channels=-1, out_channels=params['out_channels']*params['heads'])])
        for i in range(params['num_layers']-1):
            self.layers.append(GCNConv(in_channels=params['out_channels']*params['heads'], out_channels=params['out_channels']*params['heads']))
        self.norm = nn.BatchNorm1d(params['out_channels']*params['heads'])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(params['dropout'])
    def forward(self, x,edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.norm(x)
            x = self.act(x)
            x = self.dropout(x)
        return x,x


class GATMODEL(nn.Module):
    def __init__(self,params,dim,concat=False):
        super(GATMODEL, self).__init__()
        if not concat:
            assert dim%params['heads'] == 0 , "Feature dimension is not divisible by heads"
            params['out_channels'] = dim//params['heads']
        self.layers = nn.ModuleList([GATConv(in_channels=-1, out_channels = params['out_channels'], heads= params['heads'], dropout=params['dropout'], add_self_loops=params['add_self_loops'])])
        for i in range(params['num_layers']-1):
            self.layers.append(GATConv(in_channels=params['out_channels']*params['heads'], out_channels = params['out_channels'], heads= params['heads'], dropout=params['dropout'], add_self_loops=params['add_self_loops']))
        self.norm = nn.LayerNorm(params['out_channels']*params['heads'])
        self.act = nn.ReLU()
    def forward(self, x,edge_index):
        attention_weights = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, edge_index,return_attention_weights=True)
            attention_weights.append(attn)
            #x = self.norm(x)
            #x = self.act(x)
        return x,attention_weights

'''
class GATMODEL(nn.Module):
    def __init__(self,params):
        super(GATMODEL, self).__init__()

        self.layer1 = GATConv(in_channels=-1, out_channels = params['out_channels'], heads= params['heads'], dropout=params['dropout'], add_self_loops=params['add_self_loops'])
        #self.norm1 = nn.BatchNorm1d(params['out_channels']*4)
        #self.act1 = nn.ReLU()
        self.layer2 = GATConv(in_channels=params['out_channels']*4, out_channels = params['out_channels'], heads= params['heads'], dropout=params['dropout'], add_self_loops=params['add_self_loops'])
        #self.norm2 = nn.BatchNorm1d(params['out_channels']*4)
        #self.act2 = nn.ReLU()
    def forward(self, x,edge_index):
        x, attn1 = self.layer1(x, edge_index,return_attention_weights=False)
        #x = self.norm1(x)
        #x = self.act1(x)
        x, attn2 = self.layer2(x, edge_index,return_attention_weights=False)
        #x = self.norm2(x)
        #x = self.act2(x)
        return x, attn1 , attn2
'''

class Net(pl.LightningModule):
    def __init__(self, cfg,dataset,network,optimizer,criterion,GAT_params):
        super().__init__()
        self.drug_layers = GATMODEL(GAT_params['drug_gat'],network['drug_dim'],concat=cfg['module']['GAT_params']['concat']['concat'])
        self.target_layers = GATMODEL(GAT_params['prot_gat'],network['prot_dim'],concat=cfg['module']['GAT_params']['concat']['concat'])
        #self.drug_layers = GCNMODEL(GAT_params['drug_gat'])
        #self.target_layers = GCNMODEL(GAT_params['prot_gat'])
        drug_gat_dim = GAT_params['drug_gat']['out_channels']*GAT_params['drug_gat']['heads']
        prot_gat_dim = GAT_params['prot_gat']['out_channels']*GAT_params['prot_gat']['heads']
        if cfg['module']['GAT_params']['concat']['concat']:
            layers = [network['drug_dim']+network['prot_dim']+drug_gat_dim+prot_gat_dim]+network['layers']+[network['output_dim']]
        else:
            layers = [network['drug_dim']+network['prot_dim']]+network['layers']+[network['output_dim']]
        layers = list(layers)
        self.model = MLP(layers, dropout=network['dropout'])
        self.cfg = cfg
        self.optimizer = optimizer
        if cfg['preprocess']['data_path'].split('/')[1] != 'warm_start_1_1':
            pos_weight = 1
        else:
            pos_weight = 1
        self.criterion: torch.nn = hydra.utils.instantiate(criterion,pos_weight=torch.tensor(pos_weight).float())
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_auc = 0
        self.test_auprc = 0
        self.test_f1 = 0
        #self.ppi = torch.tensor(dataset['ppi'].values).float().to(device)
        #self.ddi = torch.tensor(dataset['ddi']).float().to(device)
        self.save_hyperparameters(cfg)
    
    def forward(self, x1,x2,x1_org,x2_org,x1_network,x2_network,inv_drug,inv_target):
        x1, drug_attn = self.drug_layers(x1, x1_network)
        x2, prot_attn = self.target_layers(x2, x2_network)        
        x1 = x1[inv_drug]
        x2 = x2[inv_target]

        if self.cfg['module']['GAT_params']['concat']['concat']:
            x1 = torch.cat([x1,x1_org],dim=1)
            x2 = torch.cat([x2,x2_org],dim=1)
        else:
            alpha = self.cfg['module']['GAT_params']['concat']['alpha']
            x1 = x1+(x1_org*alpha)
            x2 = x2+(x2_org*alpha)
        
        data = torch.cat([x1,x2],dim=1)
        scores = self.model(data)
        return torch.squeeze(scores, dim=1)
        
    def training_step(self, batch, batch_idx):
        x1,x2,y, drugs, targets = batch
        x1_org , x2_org = x1, x2
        x1,x2,x1_network,x2_network, inv_drug, inv_target = self.common_preprocess(x1,x2,drugs,targets)
        loss, scores = self.common_step(x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx)
        self.training_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):    
        train_loss, train_auc = self.common_epoch_end(self.training_step_outputs)
        self.training_step_outputs.clear()
        self.log_dict({"train_loss":train_loss, "train_auc":train_auc})
        
    def validation_step(self, batch, batch_idx):
        x1,x2,y, drugs, targets = batch
        x1_org , x2_org = x1, x2
        x1,x2,x1_network,x2_network, inv_drug, inv_target = self.common_preprocess(x1,x2,drugs,targets)
        loss, scores = self.common_step(x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx)
        self.validation_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        return loss

    def on_validation_epoch_end(self):
        val_loss, val_auc = self.common_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss":val_loss, "val_auc":val_auc}, prog_bar=True)
  
    def test_step(self, batch, batch_idx):
        x1,x2,y, drugs, targets = batch
        x1_org , x2_org = x1, x2
        x1,x2,x1_network,x2_network, inv_drug, inv_target = self.common_preprocess(x1,x2,drugs,targets)
        loss, scores = self.common_step(x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx)
        self.test_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        return loss

    def on_test_epoch_end(self):
        test_loss, test_auc, test_auprc, test_bcm, test_f1 = self.for_test_epoch(self.test_step_outputs)
        print(test_bcm)
        self.test_auc, self.test_auprc, self.test_f1 = test_auc, test_auprc, test_f1
        self.test_step_outputs.clear()
        self.log_dict({"test_auc":test_auc, "test_auprc":test_auprc, "test_f1":test_f1})

    def common_step(self, x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx):     
        scores = self.forward(x1,x2,x1_org,x2_org,x1_network,x2_network, inv_drug, inv_target)
        loss = self.criterion(scores, y)
        return loss, scores
    
    #def common_ppi_preprocess(self, x1,x2,drugs,targets):




    def common_preprocess(self, x1,x2,drugs,targets):

        #ppi = self.ppi[targets,:][:,targets]
        #drug_ddi = self.ddi[drugs,:][:,drugs]
        #for each row in ddi, find the indices of the top 5 drugs and set the rest to 0
        #values, indices = drug_ddi.topk(5, dim=1)
        #drug_ddi = torch.zeros_like(drug_ddi)
        #drug_ddi.scatter_(1, indices, 1)
        
        mapping = {}
        for index, value in enumerate(drugs):
            if value.item() not in mapping.keys():
                mapping[value.item()] = index
        drugs = torch.tensor([mapping[value.item()] for value in drugs])
        
        mapping = {}
        for index, value in enumerate(targets):
            if value.item() not in mapping.keys():
                mapping[value.item()] = index
        targets = torch.tensor([mapping[value.item()] for value in targets])
        
        drug_index, inv_drug = torch.unique(drugs,return_inverse=True)
        target_index, inv_target = torch.unique(targets,return_inverse=True)

        x1 = x1[drug_index]
        x2 = x2[target_index]
        
        x1_network = torch.cdist(x1,x1,p=2)
        x1_network[x1_network<self.cfg['module']['GAT_params']['drug_gat']['threshold']] = 1
        x1_network[x1_network>=self.cfg['module']['GAT_params']['drug_gat']['threshold']] = 0
        x1_network = torch.triu(x1_network, diagonal=1)
        
        x2_network = torch.cdist(x2,x2,p=2)
        x2_network[x2_network<self.cfg['module']['GAT_params']['prot_gat']['threshold']] = 1
        x2_network[x2_network>=self.cfg['module']['GAT_params']['prot_gat']['threshold']] = 0
        x2_network = torch.triu(x2_network, diagonal=1)
        
        ### doesn't work
        #x1_network = ddi[drug_index,:][:,drug_index]
        #x1_network = torch.triu(x1_network, diagonal=1)

        #x2_network = ppi[target_index,:][:,target_index]
        #x2_network = torch.triu(x2_network, diagonal=1)

        #x1_network = drug_ddi[drug_index,:][:,drug_index]
        
        # find edge index from adj matrix
        x1_network = x1_network.nonzero(as_tuple=False)
        x2_network = x2_network.nonzero(as_tuple=False)
        x1_network = x1_network.t()
        x2_network = x2_network.t()
        return x1,x2,x1_network,x2_network, inv_drug, inv_target

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

         