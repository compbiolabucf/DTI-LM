import re,sys,os
from utils import utils
from utils.preprocess import PREPROCESS 
import numpy as np
import pandas as pd 
import torch
import warnings
warnings.filterwarnings("ignore", category=Warning, module="torchvision")
warnings.filterwarnings('ignore')
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig,OmegaConf 
from typing import Any, List, Optional, Tuple
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import yaml
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import multiprocessing
from multiprocessing import Manager
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#pl.seed_everything(seed=42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
def train(cfg,dataset,shared_metrics,tune=False):
    tb_logger = TensorBoardLogger('../logger_DTI/tb_logs', name=cfg['logger']['name'])
    
    # Init featurizer model
    data_module: LightningModule = hydra.utils.instantiate(
        cfg['datamodule'],cfg, dataset, _recursive_=False
    )
    
    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(
        cfg['module'],cfg,dataset, _recursive_=False
    )
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg['callbacks']
    )

    if tune:
        metrics = {"auc": "val_auc","loss": "val_loss"}
        callbacks.append(TuneReportCallback(metrics, on="validation_end"))
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=cfg['trainer']['max_epochs'], logger=tb_logger,callbacks=callbacks, enable_progress_bar=False)
        trainer.fit(model, data_module)
    else:
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=cfg['trainer']['max_epochs'], logger=tb_logger,callbacks=callbacks,log_every_n_steps=5)
        trainer.fit(model, data_module)
        trainer.validate(model, data_module)
        trainer.test(model, data_module)
        shared_metrics["test_auc"]+= [model.test_auc.item()]
        shared_metrics["test_auprc"]+= [model.test_auprc.item()]
        shared_metrics["test_f1"]+= [model.test_f1.item()]

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",
    "config_name": "config-name",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg) -> Optional[float]:  
    pre_process = PREPROCESS(cfg['preprocess'])
    X_drug, X_target, DTI = pre_process.process_data()
    #ddi,skipped = utils.get_ddi(X_drug)

    logger, logger_dir = utils.get_logger(OmegaConf.to_container(cfg))
    new_dir = logger_dir.split('run')[0]
    
    save_path = cfg['datamodule']['serializer']['save_path']
    drug_name = cfg['datamodule']['serializer']['drug_name']
    target_name = cfg['datamodule']['serializer']['target_name']
    cfg = OmegaConf.to_container(cfg)
    cfg['datamodule']['serializer']['drug_name'] = drug_name
    cfg['datamodule']['serializer']['target_name'] = target_name

    if cfg['datamodule']['serializer']['load_serialized']:
        X_drug = torch.load(save_path+drug_name)
        X_target = torch.load(save_path+target_name)
    else:
        # Init featurizer model
        drug_featurizer: LightningModule = hydra.utils.instantiate(
                cfg['featurizer']['drugfeaturizer'], device, _recursive_=False
        )
        X_drug_features = drug_featurizer.get_representations(X_drug.SMILES.values)
        X_drug = pd.DataFrame(X_drug_features,index=X_drug.index)
        torch.save(X_drug,save_path+drug_name)
        
        # Init featurizer model
        prot_featurizer: LightningModule = hydra.utils.instantiate(
            cfg['featurizer']['protfeaturizer'], device, _recursive_=False
        )
        X_target_features = prot_featurizer.get_representations(X_target.SEQ.values)
        X_target = pd.DataFrame(X_target_features,index=X_target.index)
        torch.save(X_target,save_path+target_name)
        
    manager = Manager()
    shared_metrics = manager.dict()
    shared_metrics["test_auc"],shared_metrics["test_auprc"],shared_metrics["test_f1"] = [], [], []

    if cfg['tuning']['param_search']['tune']:
        optuna_search = OptunaSearch(
            metric="auc",
            mode="max",
        )

        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='auc',
            mode='max',
            max_t=100,
            grace_period=10,
        )

        #check if DTI is a dict, it will be for other presplitted datasets
        if isinstance(DTI,dict):
            DTI = DTI[0]
        dataset = utils.get_dataset(cfg,X_drug, X_target, DTI)  
        cfg = utils.setup_config_tune(cfg['tuning']['param_search']['search_space'],cfg)
        ray.init()
        trainable = tune.with_parameters(train, dataset=dataset, shared_metrics=None, tune=True)
        analysis = tune.run(
            trainable,
            local_dir="/data/tanvir/DTI",
            resources_per_trial={"gpu": 0.5},
            config=cfg,
            num_samples=100,
            search_alg=optuna_search,
            scheduler=asha_scheduler,
        )
        best_trial = analysis.get_best_trial("auc", mode="max")
        print("Best Hyperparameters:")
        print(best_trial.config)
        train(best_trial.config,dataset,shared_metrics)
        ray.shutdown()
    else:
        cfg = utils.update_best_param(cfg)
        if cfg['multiprocessing']['multiprocessing']:
            X_drug_orig, X_target_orig, DTI_orig = X_drug.copy(), X_target.copy(), DTI.copy()
            dataset={}
            for num in range(cfg['multiprocessing']['num_process']):
                if isinstance(DTI_orig,dict):
                    X_drug, X_target, DTI = X_drug_orig.copy(), X_target_orig.copy(), DTI_orig[num].copy()
                else:    
                    X_drug, X_target, DTI = X_drug_orig.copy(), X_target_orig.copy(), DTI_orig.copy()
                
                dataset[num] = utils.get_dataset(cfg,X_drug, X_target, DTI,ddi=None,skipped=None)
            #find tst_ind from all dataset
            test_ind = []
            for num in range(cfg['multiprocessing']['num_process']):
                test_ind += dataset[num]['test'].label.tolist()
            print(np.unique(test_ind,return_counts=True))

            processes = []
            for num in range(cfg['multiprocessing']['num_process']):
                p = multiprocessing.Process(target=train, args=(cfg,dataset[num],shared_metrics))
                processes.append(p)

                if len(processes) >= cfg['multiprocessing']['concurrent_process']:
                    for batch_process in processes:
                        batch_process.start()
                    for batch_process in processes:
                        batch_process.join()
                    processes = []  

            for p in processes:
                p.start()
            for p in processes:
                p.join()
            
            print("All processes have finished.")
            
            logger.info(shared_metrics["test_auc"])
            logger.info(shared_metrics["test_auprc"])
            logger.info(shared_metrics["test_f1"])
            logger.info(f'Mean test AUC: {np.mean(shared_metrics["test_auc"]):.4f}')
            logger.info(f'Mean test AUPRC: {np.mean(shared_metrics["test_auprc"]):.4f}')
            logger.info(f'Mean test F1: {np.mean(shared_metrics["test_f1"]):.4f}')
            new_dir = f'{new_dir}/run_{np.mean(shared_metrics["test_auc"]):.4f}/'
            os.rename(logger_dir, new_dir)
            
        else:
            pl.seed_everything(seed=42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            dataset = utils.get_dataset(cfg,X_drug, X_target, DTI,ddi=None,skipped=None)  
            train(cfg,dataset,shared_metrics)
            
            test_auc = [shared_metrics["test_auc"]]
            test_auprc = [shared_metrics["test_auprc"]]
            test_f1 = [shared_metrics["test_f1"]]
            logger.info(f'Test AUC: {np.mean(test_auc):.4f}')
            logger.info(f'Test AUPRC: {np.mean(test_auprc):.4f}')
            logger.info(f'Test F1: {np.mean(test_f1):.4f}')
            new_dir = f'{new_dir}/run_{np.mean(test_auc):.4f}/'
            os.rename(logger_dir, new_dir)
            
        
if __name__ == "__main__":
    main()

  
