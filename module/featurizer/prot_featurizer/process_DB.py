import pandas as pd 
import numpy as np
import os,sys
from Bio import SeqIO
#from openbabel import pybel
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler
from load_data import get_data
from torch.nn.utils.rnn import pad_sequence
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

np.random.seed(42)
# define traning sameple number
train_sample_num = 10
#Full DTI matrix or a subset of it
subset= False
#define size of each sample
sample_size = 1024
# whether to multiply adj matrix with the feature matrix
multiply_adjacency = False
# whether val test data should be balanced
balance_val_test = False



def BuildDataset(gene,drug,DTI,ite=1):
    data1=[]
    data2=[]
    label=[]
    
    if subset:
        for i in tqdm(range(ite)):
            
            #find samples of size sample_size from the DTI matrix
            gene_ind = np.random.choice(DTI.index,sample_size,replace=False)
            DTI_temp = DTI.loc[gene_ind,:]
            
            drug_ind = np.random.choice(DTI_temp.columns,sample_size,replace=False)
            DTI_temp = DTI_temp.loc[:,drug_ind]
            
            # remove rows with sum of 0
            DTI_temp = DTI_temp.loc[(DTI_temp!=0).any(axis=1),:]
            
            # remove columns with sum of 0
            DTI_temp = DTI_temp.loc[:,(DTI_temp!=0).any(axis=0)]  
            
            drug_temp = drug.loc[DTI_temp.columns,:]
            gene_temp = gene.loc[DTI_temp.index,:]
            
 

            # concatenate neighborhood
            '''
            for each row in gene, find the rows in drug 
            that are connected to it and concatenate those drugs
            for each row in drug, find the rows in gene 
            that are connected to it and concatenate those genes
            '''
            
            new_gene=[]
            new_drug=[]
            max_neighbors = 5
            for i in range(len(gene_temp)):
                individual_gene=[]
                ind = np.where(DTI_temp.values[i,:]==1)[0]
                if len(ind)>max_neighbors:
                    ind = np.random.choice(ind,max_neighbors,replace=False)
                for index in ind:
                    #individual_gene.extend(list(np.array(drug_temp.values[index,0].split(' ')).astype(int)))
                    individual_gene.extend(list(np.array(drug_temp.values[index,0].split(' ')).astype(int)))
                new_gene.append(torch.tensor(individual_gene))
            
            max_neighbors = 3
            for i in range(len(drug_temp)):
                individual_drug=[]
                ind = np.where(DTI_temp.values[:,i]==1)[0]
                if len(ind)>max_neighbors:
                    ind = np.random.choice(ind,max_neighbors,replace=False)
                for index in ind:
                    individual_drug.extend(list(np.array(gene_temp.values[index,0].split(' ')).astype(int)))
                new_drug.append(torch.tensor(individual_drug))
            
                
            # find index of drugs that are extra long
            ind = np.where(np.array([len(i) for i in new_drug])<max_length_drug)[0]
            new_drug = [new_drug[i] for i in ind]
            DTI_temp = DTI_temp.iloc[:,ind] 

            ind = np.where(np.array([len(i) for i in new_drug])>min_length_drug)[0]
            new_drug = [new_drug[i] for i in ind]
            DTI_temp = DTI_temp.iloc[:,ind] 

            # find index of genes that are extra long
            ind = np.where(np.array([len(i) for i in new_gene])<max_length_gene)[0]
            new_gene = [new_gene[i] for i in ind]
            DTI_temp = DTI_temp.iloc[ind,:] 


            # find the pisitive and negative instances in MMI
            ind_1 = np.where(DTI_temp.values==1)
            g = [new_gene[g1] for g1 in ind_1[0]]
            data1.extend(g)
            d = [new_drug[d1] for d1 in ind_1[1]]
            data2.extend(d)
            # add labels for positive instances
            label.extend([1]*len(ind_1[0]))


            ind_0 = np.where(DTI_temp.values==0)
            #sample same number of negative instances than positive instances
            ind_0_ind = np.random.choice(np.arange(len(ind_0[0])),len(ind_1[0]),replace=False)
            ind_0 = np.array(ind_0)[:,ind_0_ind]
            g = [new_gene[g1] for g1 in ind_1[0]]
            data1.extend(g)
            d = [new_drug[d1] for d1 in ind_1[1]]
            data2.extend(d)
            # add labels for negative instances
            label.extend([0]*len(ind_0[0]))
    
    else:

        #gene = gene.values
        #drug = drug.values


        # find the pisitive and negative instances in MMI
        ind_1 = np.where(DTI.values==1)

        data1_1 = gene.iloc[ind_1[1],:]
        data2_1 = drug.iloc[ind_1[0],:]
        label.extend([1]*len(ind_1[0]))

   
        ind_0 = np.where(DTI.values==0)
        #sample same number of negative instances as positive instances
        ind_0_ind = np.random.choice(np.arange(len(ind_0[0])),len(ind_1[0]),replace=False)
        ind_0 = np.array(ind_0)[:,ind_0_ind]
        data1_2 = gene.iloc[ind_0[1],:]
        data2_2 = drug.iloc[ind_0[0],:]
        label.extend([0]*len(ind_0[0]))

        data1 = pd.concat([data1_1,data1_2],axis=0,ignore_index=True)
        data2 = pd.concat([data2_1,data2_2],axis=0,ignore_index=True)


    return data1, data2, label



def process_data():
   
    gene,drug,DTI = get_data()    

    X_target, X_drug, y = BuildDataset(gene,drug,DTI)

    #X_target.drop_duplicates(subset=['SEQ'],inplace=True)
    #X_drug.drop_duplicates(subset=['SMILES'],inplace=True)

    #X_target['SEQ'] = X_target['SEQ'].apply(lambda x: " ".join(x))
    #X_drug['SMILES'] = X_drug['SMILES'].apply(lambda x: " ".join(x))

    #np.savetxt("smiles.txt", X_drug.SMILES.values, fmt="%s")
    #np.savetxt("sequence.txt", X_target.SEQ.values, fmt="%s")
    
    '''    
    #sequence = "".join(str(v[0]) for v in gene.values).replace("\n", "")
    with open("sequence.txt", 'w') as f:
        f.write(X_target.SEQ.values)

    #smiles = "".join(str(v[0]) for v in drug.values).replace("\n", "")
    with open("smiles.txt", 'w') as f:
        f.write(X_drug.SMILES.values)
    
    '''
    #sys.exit()


    return X_drug.SMILES.values, X_target.SEQ.values, np.array(y)
