
## DTI-LM: Language Model Powered Drug-target interaction prediction

This repository contains code for DTI-LM, a drug-target interaction prediction framework that leverages pretrained language model and graph attention network to find representations of drugs and proteins.DTI-LM offers following advantages over existing literature.

1. State-of-the-art prediction results.
2. Requires minimum input data in terms of protein amino acid and drug SMILES sequences.
3. Faster runtime than comparable methods. 
4. Investigates the current standing of language model-based DTI prediction capability and limitations.



## Run Locally

Clone the project

```bash
  git clone https://github.com/compbiolabucf/DTI-LM.git
```

Go to the project directory

```bash
  cd DTI
```

Install dependencies

```bash
  conda env create -f environment.yml
```

Run the code

```bash
  python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"
```




## Documentation

Using own data

```bash
1. Modify utils.PREPROCESS to add the data name
2. Add a preprocessing script in utils that returns     
    X_drug: nx1 pd.DataFrame, index=drug names, Column 1=SMILES sequence. 
    X_target: mx1 pd.DataFrame, index=target names, Column 1=protein sequence. 
    DTI: mxn (available) pd.DataFrame, index: 0-mxn, Column 1=Drug names matching X_drug index, Column 2=Target names matching X_target index, Column 3= interaction label (0,1)
3. Add a train.yaml (bindingDB_train_GAT.yaml) in configs defining preprocess and datamodule
```

Using own featurizer 
```bash 
1. add new featurizers in module.featurizer.drug_featurizer and prot_featurizer. Return nxq and mxp embedding (p,q embedding size)
2. Modify configs.featurizer
3. Modify drug_dim and prot_dim in configs.module.GAT
```

Using own classifier
```bash
1. Add a new pipeline in module (Similar to GAT.py and MLP.py).
2. Add necessary file in configs.module following config.module.GAT for GAT.py
```

Hyperparameter tuning
```bash
1. Set tuning.param_search.tune=True
2. Define search space in configs.tuning
3. Modify utils.setup_config_tune to add missing hyperparameter
4. Add best hyperparameter file in config.best_params and link it to train.yaml
```