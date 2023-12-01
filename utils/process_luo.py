import numpy as np 
import pandas as pd
import sys
import torch
np.random.seed(42)


def process_data(config):
    root_path = config['root_path']
    data_path = config['data_path'] 
    seq_path = config['seq_path']
    mapping_path = config['mapping_path']
    #data_name = config['data_name'].split(' ')
    DTI_dict = {}
    for i in range(10):
        data_name = []
        data_name.append('train_fold_'+str(i+1)+'.csv')
        data_name.append('test_fold_'+str(i+1)+'.csv')
        df_drug, df_proseq, DTI = helper_function(root_path,data_path,seq_path,mapping_path,data_name)
        DTI_dict[i] = DTI
 
    return df_drug, df_proseq, DTI_dict

def helper_function(root_path,data_path,seq_path,mapping_path,data_name):
    #root_path = config['root_path']
    #data_path = config['data_path'] 
    #seq_path = config['seq_path']
    #mapping_path = config['mapping_path']
    #data_name = config['data_name'].split(' ')
    
    train_data = pd.read_csv(root_path+data_path+data_name[0])
    test_data = pd.read_csv(root_path+data_path+data_name[1])
    
    df_drug = pd.read_csv(root_path+seq_path+'drug_smiles.csv',index_col=0)
    df_proseq = pd.read_csv(root_path+seq_path+'seq.txt',sep='\t',header=None)

    drug_names = pd.read_csv(root_path+mapping_path+'drug.txt',sep='\t',header=None)
    pro_names = pd.read_csv(root_path+mapping_path+'protein.txt',sep='\t',header=None)

    """
    special processing for luo data as drug protein data names and DTI names don't match
    1. add equivalent drug names as DTI in drug names in new column
    2. replace the names in DTI with corresonponding names in drug names
    3. reverse all changes except for DTI
    4. same for protein names 
    """

    # add equivalent drug names as DTI in drug names in new column
    drug_names['head'] = drug_names.index.values 
    drug_names['head'] = drug_names['head'].apply(lambda x: 'Durg::'+str(x))
    # replace the names in DTI with corresonponding names in drug names
    mapping_dict = dict(zip(drug_names['head'],df_drug['id']))
    train_data['head'] = train_data['head'].map(mapping_dict)
    test_data['head'] = test_data['head'].map(mapping_dict)
    # reverse all changes except for DTI
    drug_names = drug_names.drop(columns=['head'])

    # same for protein names
    pro_names['tail'] = pro_names.index.values
    pro_names['tail'] = pro_names['tail'].apply(lambda x: 'Protein::'+str(x))
    mapping_dict = dict(zip(pro_names['tail'],pro_names[0]))
    train_data['tail'] = train_data['tail'].map(mapping_dict)
    test_data['tail'] = test_data['tail'].map(mapping_dict)
    pro_names = pro_names.drop(columns=['tail'])

    #set drug and protein names as index
    df_drug = df_drug.set_index(drug_names[0].values)
    df_proseq = df_proseq.set_index(pro_names[0].values)

    # rename the columns 
    df_drug = df_drug.rename(columns={'smiles':'SMILES'})
    #set column names for df_proseq
    df_proseq.columns = ['SEQ']
    
    # check how many df_proseq sequences are longer than 700
    length = df_proseq['SEQ'].apply(lambda x: len(x))
    df_proseq = df_proseq[length<700]
    
    df_drug = df_drug.drop(columns=['id'])
    #df_proseq = df_proseq.drop(columns=['0','pro_ids'])

    #remove rows if duplicate index
    df_drug = df_drug[~df_drug.index.duplicated(keep='first')]
    df_proseq = df_proseq[~df_proseq.index.duplicated(keep='first')]

    # add indicator column to show train(1),val(2), test(3)
    train_data['indicator'] = 1
    test_data['indicator'] = 3
    '''
    #doesn't work for luo data
    if data_path.split('/')[1] == 'protein_coldstart':
        unique_prots = train_data['tail'].unique()
        #15% of the unique proteins to be used for validation and flip the indicator to 2
        val_prots = np.random.choice(unique_prots, int(0.15*len(unique_prots)), replace=False)
        train_data['indicator'] = np.where(train_data['tail'].isin(val_prots), 2, train_data['indicator'])
    elif data_path.split('/')[1] == 'drug_coldstart':
        unique_drugs = train_data['head'].unique()
        #15% of the unique drugs to be used for validation and flip the indicator to 2
        val_drugs = np.random.choice(unique_drugs, int(0.15*len(unique_drugs)), replace=False)
        train_data['indicator'] = np.where(train_data['head'].isin(val_drugs), 2, train_data['indicator'])
    else:        
        #flip random 15% of the train data indicator to 0
        train_data['indicator'] = np.where(np.random.rand(len(train_data)) < 0.15, 2, train_data['indicator'])
    '''
    train_data['indicator'] = np.where(np.random.rand(len(train_data)) < 0.01, 2, train_data['indicator'])
    #concatenate the train and test data
    DTI = pd.concat([train_data,test_data],axis=0)

    #keep few columns from DTI
    DTI = DTI[['head', 'tail', 'label', 'indicator']]
    #DTI = DTI.drop(columns=['relation','pred'])
    DTI = DTI.rename(columns={'head':'Drug_ID','tail':'Prot_ID'})

    # remove some rows from DTI as those proteins were remove for being >700
    DTI = DTI[DTI['Prot_ID'].isin(df_proseq.index)]

    return df_drug, df_proseq, DTI
