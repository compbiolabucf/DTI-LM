import pandas as pd 
import numpy as np
import os,sys 
from Bio import SeqIO
from openbabel import pybel
from tqdm import tqdm
np.random.seed(42)
from rdkit import DataStructs,Chem
from rdkit.Chem import rdFingerprintGenerator

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

    drug_name = []
    drug_seq = []
    # Read in the drugbank drug file
    for mol in pybel.readfile('sdf', path+'structures.sdf'):
        ID, structure = mol.data['DATABASE_ID'], mol.data['SMILES']
        if len(structure)<30:
            continue
        drug_name.append(ID)
        drug_seq.append(structure)
    drug = pd.DataFrame({'DrugBank ID':drug_name,'SMILES':drug_seq})
    drug = drug.set_index('DrugBank ID')

    gene_name = []
    gene_seq = []
    gene = pd.DataFrame(columns=['UniProt ID','SEQ'])
    # Read in the drugbank protein file
    for e, seq_record in enumerate(SeqIO.parse(path+'protein.fasta', "fasta")):
        name = seq_record.id
        name = name.split('|')
        gene.loc[len(gene.index)] = [name[1], "".join(np.array(seq_record.seq))] 
    gene = gene.set_index('UniProt ID')

    # Read in the link file
    link = pd.read_csv(path+'uniprot links.csv',usecols=['DrugBank ID','UniProt ID'])
    link = link.drop_duplicates()
    DTI = link.pivot_table(index='DrugBank ID', columns='UniProt ID', aggfunc=len,fill_value=0)

    xy,x_ind,y_ind = np.intersect1d(drug.index,DTI.index,return_indices=True)
    DTI = DTI.iloc[y_ind,:]
    drug = drug.iloc[x_ind,:]

    xy,x_ind,y_ind = np.intersect1d(gene.index,DTI.columns,return_indices=True)
    DTI = DTI.iloc[:,y_ind]
    gene = gene.iloc[x_ind,:]

    return gene, drug, DTI

def BuildDataset(gene,drug,DTI,ite=1):
    data1=[]
    data2=[]
    label=[]
    
    # find the pisitive and negative instances in MMI
    ind_1 = np.where(DTI.values==1)
    data1_1 = gene.iloc[ind_1[1],:]
    data2_1 = drug.iloc[ind_1[0],:]
    label.extend([1]*len(ind_1[0]))

    pos_gene_names = gene.index[ind_1[1]]
    pos_drug_names = drug.index[ind_1[0]]
    
    ind_0 = np.where(DTI.values==0)
    #sample same number of negative instances as positive instances
    #ind_0_ind = np.random.choice(np.arange(len(ind_0[0])),len(ind_1[0]),replace=False)
    #ind_0 = np.array(ind_0)[:,ind_0_ind]
    data1_2 = gene.iloc[ind_0[1],:]
    data2_2 = drug.iloc[ind_0[0],:]
    label.extend([0]*len(ind_0[0]))
    neg_gene_names = gene.index[ind_0[1]]
    neg_drug_names = drug.index[ind_0[0]]

    data1 = pd.concat([data1_1,data1_2],axis=0,ignore_index=True)
    data2 = pd.concat([data2_1,data2_2],axis=0,ignore_index=True)
    gene_names = np.concatenate([pos_gene_names,neg_gene_names],axis=0)
    drug_names = np.concatenate([pos_drug_names,neg_drug_names],axis=0)
    drug_names = pd.DataFrame(drug_names,columns=['DrugBank_ID'])
    gene_names = pd.DataFrame(gene_names,columns=['UniProt_ID'])
    return data1, data2, np.array(label), gene_names, drug_names

def process_data(config):
    X_target,X_drug,y = get_data(config)    
    X_target, X_drug, y, gene_names, drug_names = BuildDataset(X_target,X_drug,y)

    #remove X_target that are longer than 700
    ind = np.where([len(i)<=700 for i in X_target.SEQ])[0]
    np.random.shuffle(ind)
    
    X_target = X_target.iloc[ind]
    X_drug = X_drug.iloc[ind]
    y = y[ind]
    gene_names = gene_names.iloc[ind]
    drug_names = drug_names.iloc[ind]

    ind = np.where([len(i)<=510 for i in X_drug.SMILES])[0]
    np.random.shuffle(ind)
    X_target = X_target.iloc[ind]
    X_drug = X_drug.iloc[ind]
    y = y[ind]
    gene_names = gene_names.iloc[ind]
    drug_names = drug_names.iloc[ind]

    #create a dataframe with drug_names, gene_names, and y as columns  
    DTI = pd.DataFrame({'Drug_ID':drug_names.DrugBank_ID.values,'Prot_ID':gene_names.UniProt_ID.values,'label':y})
    
    uni_gene_names = gene_names.drop_duplicates(subset=['UniProt_ID'])
    uni_drug_names = drug_names.drop_duplicates(subset=['DrugBank_ID'])
    X_target = X_target.loc[uni_gene_names.index]
    X_drug = X_drug.loc[uni_drug_names.index]

    # add unique gene names as index to X_target
    X_target = X_target.set_index(uni_gene_names.UniProt_ID.values)
    X_drug = X_drug.set_index(uni_drug_names.DrugBank_ID.values)


    """
    measure sequence similarities
    protein: clustal omega
    drug: rdkit
    """
    #create_fasta(X_target)
    #drug_similarity(X_drug,save=False)
    """
    data = pd.read_csv('/home/tahmed/DTI/utils/pim',delimiter='\t',header=None,skiprows=5)
    #split wach row of the dataframe at space 
    new_data =[]
    for i in range(len(data)):
        temp = data.iloc[i].str.split(' ')[0]
        temp_1=[]
        for j in temp:
            if j!='':
                temp_1.append(j)
        new_data.append(temp_1)
    #create a dataframe from the list
    new_data = pd.DataFrame(new_data)
    #remove the first column
    new_data = new_data.drop(columns=[0])
    new_data.set_index(1,inplace=True)
    #remove the diagonal elements
    new_data = new_data.mask(np.eye(len(new_data),dtype=bool)).astype(float)
    print(f'average protein similarity: {np.nanmean(new_data)}')
    """

    return X_drug, X_target, DTI
