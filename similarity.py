import numpy as np
import pandas as pd
import sys 
from tqdm import tqdm
from rdkit import DataStructs,Chem
from rdkit.Chem import rdFingerprintGenerator
import torch



def create_fasta(df,path):
    fasta_file_path = path+'output.fasta'

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

def drug_similarity(df,path,save=False):
    ms = [Chem.MolFromSmiles(x) for x in df.SMILES]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    #fps = [fpgen.GetFingerprint(x) for x in ms]
    fps = []
    skipped = []
    for e, x in enumerate(ms):
        try:
            fps.append(fpgen.GetFingerprint(x))
        except:
            print('error, skipped...')
            skipped.append(e)
    
    #generate a similarity matrix of fps by fps
    similarity = np.array([[DataStructs.TanimotoSimilarity(i,j) for i in fps] for j in fps])
    #make the diagonal elements 0
    np.fill_diagonal(similarity,0)
    print(f'Average drug similarity: {np.mean(similarity)}')
    if save:
        np.savetxt(path+'drug_similarity.csv',similarity,delimiter=',',fmt='%s')
    return similarity, skipped

def protein_similarity():
    data = pd.read_csv(path+'pim',delimiter='\t',header=None,skiprows=5)
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
    #replace nan with 0
    new_data = new_data.fillna(0).values
    print(f'average protein similarity: {np.nanmean(new_data)/100}')
    return new_data


def check_LLM(drug,target):
    #find how many nan values using pandas
    drug, target = drug.astype(float), target.astype(float)
    drugtensor = torch.tensor(drug.values)
    targettensor = torch.tensor(target.values)
    #define a random tensor
    drugsimilarity = torch.cdist(drugtensor,drugtensor,p=2)
    targetsimilarity = torch.cdist(targettensor,targettensor,p=2)
    print(f'average drug similarity: {1/torch.mean(drugsimilarity)}')
    print(f'average target similarity: {1/torch.mean(targetsimilarity)}')
    #find pearson correlation between drugs
    drugcorr = np.corrcoef(drug.values)
    targetcorr = np.corrcoef(target.values)
    np.fill_diagonal(drugcorr,0)
    np.fill_diagonal(targetcorr,0)
    print(f'average drug correlation: {np.mean(drugcorr)}')
    print(f'average target correlation: {np.mean(targetcorr)}')
    return drugcorr, targetcorr


def check_neighbors(drug,target,drugcorr,targetcorr,num_neigh = 5):
    try:
        y = pd.read_csv(path+'DTI.csv',delimiter=',',index_col=0)
    except:
        y = torch.load(path+'DTI.pt')
        #merge all dataframes in dict y
        y = pd.concat(y.values(),ignore_index=True)
        #drop duplicates based on drug and protein id
        y = y.drop_duplicates(subset=['Drug_ID','Prot_ID'])

    y = y[y['label']==1]
    y['Drug_ID'] = y['Drug_ID'].astype(str)
    y['Prot_ID'] = y['Prot_ID'].astype(str)

    #find top 5 drugs with highest correlation from drugcorr
    total_int, matched_int = 0, 0
    for i in range(len(drug)):
        row = drugcorr[i]
        row = np.argsort(row)[-num_neigh:]
        neigh_names = drug.index.values[row]
        xy = np.intersect1d(neigh_names,y['Drug_ID'].values)
        
        target_int = y[y['Drug_ID']==drug.index.values[i]]
        neigh_int = y[y['Drug_ID'].isin(neigh_names)]

        for prot in target_int['Prot_ID']:
            total_int+=1
            #find how many times prot is in neigh_int
            count = neigh_int[neigh_int['Prot_ID']==prot].shape[0]
            if count>=(num_neigh//2)+1:
                matched_int+=1
    #Percentage of a interaction being covered by majority of top 5 neighbor
    print(f'Total ints:{total_int},matched: {matched_int}, percentage:{matched_int/total_int}')

    empty = 0
    nei_size = 0  
    total_int, matched_int = 0, 0
    for i in range(len(target)):
        row = targetcorr[i]
        row = np.argsort(row)[-num_neigh:]
        neigh_names = target.index.values[row]

        target_int = y[y['Prot_ID']==target.index.values[i]]
        neigh_int = y[y['Prot_ID'].isin(neigh_names)]
        nei_size+=neigh_int.shape[0]

        for drug in target_int['Drug_ID']:
            total_int+=1
            #find how many times prot is in neigh_int
            count = neigh_int[neigh_int['Drug_ID']==drug].shape[0]
            if count>=(num_neigh//2)+1:
                matched_int+=1
    #Percentage of a interaction being covered by majority of top 5 neighbor
    print(f'Total ints:{total_int},matched: {matched_int}, percentage:{matched_int/total_int}')

def main():
    drug = pd.read_csv(path+'X_drug.csv',delimiter=',',index_col=0)
    target = pd.read_csv(path+'X_target.csv',delimiter=',',index_col=0)
    
    #create_fasta(target,path)
    drug_sim,skipped = drug_similarity(drug,path,save=False)
    prot_sim = protein_similarity()
    drug_encoding = torch.load(data_path+'DrugBank_PubChem10M.pt')
    target_encoding = torch.load(data_path+'DrugBank_ESM.pt')
    drugcorr, targetcorr = check_LLM(drug_encoding,target_encoding)

    print(drug_encoding.shape)
    print(target_encoding.shape)
    print(drugcorr.shape)
    print(targetcorr.shape)

    #check_neighbors(drug_encoding,target_encoding,drugcorr,targetcorr,num_neigh = 5)
    if len(skipped)>0:
        drug = drug.drop(index=drug.index[skipped])
    if len(drug) != len(drug_encoding):
        drug_encoding = drug_encoding.loc[drug.index]    

    print(drug_encoding.shape)
    print(target_encoding.shape)
    print(drug_sim.shape)
    print(prot_sim.shape)
    
    #check_neighbors(drug_encoding,target_encoding,drug_sim,prot_sim,num_neigh = 5)

    PPI = pd.read_csv('/data/tanvir/DTI/comppi--interactions--tax_hsapiens_loc_all.txt', sep='\t',usecols=[0,4,8],on_bad_lines='skip')
    DDI = pd.read_csv('/data/tanvir/DTI/ChCh-Miner_durgbank-chem-chem.tsv',sep='\t',on_bad_lines='skip')

    print(PPI)
    print(DDI)
    sys.exit()
    
path = './datasets/similarity/db/'
data_path = './datasets/serialized/'

if __name__ == '__main__':
    main()