import pandas as pd 
import numpy as np
import os,sys 
from Bio import SeqIO
from openbabel import pybel
from tqdm import tqdm
np.random.seed(42)
from rdkit import DataStructs,Chem
from rdkit.Chem import rdFingerprintGenerator
from utils.utils import convert_y_unit

def create_fasta(df):
    fasta_file_path = 'output.fasta'

    # Open the FASTA file for writing
    with open(fasta_file_path, 'w') as fasta_file:
        # Iterate through DataFrame rows
        for index, row in tqdm(df.iterrows()):
            protein_name = index
            sequence = row['SEQ']
            
            # Write the FASTA header line (starts with ">")
            fasta_file.write(f">{protein_name}\n")
            
            # Write the protein sequence
            fasta_file.write(f"{sequence}\n")

    print(f"FASTA file '{fasta_file_path}' has been created.")

def drug_similarity(df,save=False):
    ms = [Chem.MolFromSmiles(x) for x in df.SMILES]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    #fps = [fpgen.GetFingerprint(x) for x in ms]
    fps = []
    for x in tqdm(ms):
        try:
            fps.append(fpgen.GetFingerprint(x))
        except:
            print('error, skipped...')
    
    #generate a similarity matrix of fps by fps
    similarity = np.array([[DataStructs.FingerprintSimilarity(i,j) for i in fps] for j in fps])
    #make the diagonal elements 0
    np.fill_diagonal(similarity,0)
    print(f'Average drug similarity: {np.mean(similarity)}')
    if save:
        np.save('drug_similarity.csv',similarity,delimiter=',',fmt='%s')


def get_data(config):
    path = config['data_path']
    '''
    data = pd.read_csv(path+'BindingDB_All_202310.tsv',delimiter='\t',on_bad_lines='skip',dtype=str,usecols=[29,1,42,38,8,9,10,11])
    """
    29:PubChem CID
    1:Ligand SMILES
    42:UniProt (SwissProt) Primary ID of Target Chain
    38:BindingDB Target Chain Sequence
    8,9,10,11: Ki (nM), IC50 (nM), Kd (nM), EC50 (nM)
    """
    #impute missing protein sequence names based on the same protein sequence names from other rows
    data = data.sort_values(by=['BindingDB Target Chain Sequence'])
    data['UniProt (SwissProt) Primary ID of Target Chain'] = data.groupby('BindingDB Target Chain Sequence')['UniProt (SwissProt) Primary ID of Target Chain'].transform(lambda x: x.ffill())
    data = data.sort_values(by=['BindingDB Target Chain Sequence'], ascending=False)
    data['UniProt (SwissProt) Primary ID of Target Chain'] = data.groupby('BindingDB Target Chain Sequence')['UniProt (SwissProt) Primary ID of Target Chain'].transform(lambda x: x.bfill())
    data = data.sort_index()

    #impute missing drug names based on the same drug names from other rows
    data = data.sort_values(by=['Ligand SMILES'])
    data['PubChem CID'] = data.groupby('Ligand SMILES')['PubChem CID'].transform(lambda x: x.ffill())
    data = data.sort_values(by=['Ligand SMILES'], ascending=False)
    data['PubChem CID'] = data.groupby('Ligand SMILES')['PubChem CID'].transform(lambda x: x.bfill())
    data = data.sort_index()

    data.to_csv('bindingDB_partialprocessed.csv',index=False)    
    print(data)
    sys.exit()
    '''
    data = pd.read_csv(path+'bindingDB_partialprocessed.csv',dtype=str)
    data = data.dropna(subset=['UniProt (SwissProt) Primary ID of Target Chain'])
    if config['label'] == 'Kd':
        data = data.drop(columns=['IC50 (nM)','Ki (nM)','EC50 (nM)'])
        data = data.rename(columns={'Kd (nM)':'label'})
        data = data.dropna(subset=['label'])
    elif config['label'] == 'Ki':
        data = data.drop(columns=['IC50 (nM)','Kd (nM)','EC50 (nM)'])
        data = data.rename(columns={'Ki (nM)':'label'})
        data = data.dropna(subset=['label'])
    elif config['label'] == 'IC50':
        data = data.drop(columns=['Ki (nM)','Kd (nM)','EC50 (nM)'])
        data = data.rename(columns={'IC50 (nM)':'label'})
        data = data.dropna(subset=['label'])
    elif config['label'] == 'EC50':
        data = data.drop(columns=['Ki (nM)','Kd (nM)','IC50 (nM)'])
        data = data.rename(columns={'EC50 (nM)':'label'})
        data = data.dropna(subset=['label'])
    else:
        print('select Kd, Ki, IC50 or EC50')
    
    data = data.reset_index(drop=True)
    return data


def BuildDataset(data,config):
    #replace column names
    data = data.rename(columns={'BindingDB Target Chain Sequence':'SEQ','Ligand SMILES':'SMILES','PubChem CID':'Drug_ID','UniProt (SwissProt) Primary ID of Target Chain':'Prot_ID'})
    #rearange columns
    data = data[['Drug_ID','SMILES','Prot_ID','SEQ','label']]
    #DTI = pd.DataFrame({'Drug_ID':data['PubChem CID'],'Prot_ID':data['UniProt (SwissProt) Primary ID of Target Chain'],'label':data.label})
    data['label'] = data['label'].str.replace('>', '')
    data['label'] = data['label'].str.replace('<', '')
    data['label'] = data['label'].astype(float)
    data = data.groupby(['Drug_ID', 'SMILES', 'Prot_ID', 'SEQ']).agg({'label': 'mean'}).reset_index()
    data = data[data.label <= 10000000.0]

    if config['binary']:
        threshold = config['threshold']
        if isinstance(threshold, list):
            data = data[(data.label.values < threshold[0]) | (data.label.values > threshold[1])]
        data['label'] = [1 if i else 0 for i in data.label.values < threshold[0]]
    else:
        if config['convert_to_log']:
            data['label'] = convert_y_unit(data.label.values, 'nM', 'p')
        else:
            data['label'] = data.label.values

    #check protein sequences are valid protein sequences
    data = data[data.SEQ.str.contains('^[ACDEFGHIKLMNPQRSTVWY]+$')]
    #check drug SMILES are valid SMILES
    data = data[data.SMILES.str.contains('^[A-Za-z0-9\(\)\[\]\-\.=#$:/+]*$')]

    X_drug = data[['Drug_ID','SMILES']]
    X_target = data[['Prot_ID','SEQ']]
    y = data[['Drug_ID','Prot_ID','label']]
    return X_target, X_drug, y

def process_data(config):
    data = get_data(config)    
    X_target, X_drug, DTI = BuildDataset(data,config)
    
    #remove X_target that are longer than 700
    ind = np.where([len(i)<=700 for i in X_target.SEQ])[0]
    np.random.shuffle(ind)
    X_target = X_target.iloc[ind]
    X_drug = X_drug.iloc[ind]
    DTI = DTI.iloc[ind]
    
    ind = np.where([len(i)<=510 for i in X_drug.SMILES])[0]
    np.random.shuffle(ind)
    X_target = X_target.iloc[ind]
    X_drug = X_drug.iloc[ind]
    DTI = DTI.iloc[ind]
    
    X_drug = X_drug.set_index(X_drug['Drug_ID'].values)
    X_target = X_target.set_index(X_target['Prot_ID'].values)

    X_drug = X_drug.drop(columns=['Drug_ID'])
    X_target = X_target.drop(columns=['Prot_ID'])

    #drop rows based on duplicate index values
    X_drug = X_drug[~X_drug.index.duplicated(keep='first')]
    X_target = X_target[~X_target.index.duplicated(keep='first')]
  
    return X_drug, X_target, DTI
