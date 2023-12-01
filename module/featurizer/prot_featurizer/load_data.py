import pandas as pd 
import numpy as np
import os,sys
from Bio import SeqIO
from openbabel import pybel


np.random.seed(42)

path = './datasets/drugbank/'

class Data:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "PAD"}
        self.n_words = 2 

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def extra_prep(data):
    new_data = []
    for i in range(len(data)):
        new_data.append([x for x in data.iloc[i,0]])

    return new_data

def get_data():

    drug_name = []
    drug_seq = []
    # Read in the drugbank file
    for mol in pybel.readfile('sdf', path+'structures.sdf'):
        ID, structure = mol.data['DATABASE_ID'], mol.data['SMILES']
        if len(structure)<30:
            continue
        drug_name.append(ID)
        drug_seq.append(structure)

    # create datarame
    drug = pd.DataFrame({'DrugBank ID':drug_name,'SMILES':drug_seq})
    drug = drug.set_index('DrugBank ID')

    gene_name = []
    gene_seq = []

    gene = pd.DataFrame(columns=['UniProt ID','SEQ'])

    # Read in the drugbank file
    for e, seq_record in enumerate(SeqIO.parse(path+'protein.fasta', "fasta")):
        name = seq_record.id
        name = name.split('|')
        #gene_name.append(name[1])
        gene.loc[len(gene.index)] = [name[1], "".join(np.array(seq_record.seq))] 
        #gene_seq.append("".join(np.array(seq_record.seq)))
      
    #gene = pd.DataFrame({'UniProt ID':gene_name,'SEQ':gene_seq})
    gene = gene.set_index('UniProt ID')
    gene.to_csv('gene.csv')


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
    
    '''
    new_gene = extra_prep(gene)
    new_drug = extra_prep(drug)

    gene_data = Data('gene')
    for sentence in new_gene:
        gene_data.addSentence(sentence)

    drug_data = Data('drug')
    for sentence in new_drug:
        drug_data.addSentence(sentence)
    
    print("Counted words:")
    print(gene_data.name, gene_data.n_words)
    print(drug_data.name, drug_data.n_words)

    #print(len(gene_data.word2index.keys()))
    #print(len(drug_data.word2index.keys()))
    #print(gene_data.word2index.keys())
    #print(drug_data.word2index.keys())
    #sys.exit()
    # convert to values
    value_gene = []
    for e, sentence in enumerate(new_gene):
        value_gene.append([gene_data.word2index[word] for word in sentence])

    value_drug = []
    for sentence in new_drug:
        value_drug.append([drug_data.word2index[word] for word in sentence])

    # merge all values in a row to one string
    trans_gene = []
    for sentence in value_gene:
        trans_gene.append(' '.join(str(x) for x in sentence))

    trans_drug = []
    for sentence in value_drug:
        trans_drug.append(' '.join(str(x) for x in sentence))

    # convert to pandas dataframe
    gene = pd.DataFrame(trans_gene,columns=['SEQ'],index=gene.index)
    drug = pd.DataFrame(trans_drug,columns=['SMILES'],index=drug.index)
    '''

    '''
    # save files
    gene.to_csv(path+'gene.csv')
    drug.to_csv(path+'drug.csv')
    DTI.to_csv(path+'DTI.csv')
    '''

    return gene, drug, DTI #, gene_data.n_words+1, drug_data.n_words+1

