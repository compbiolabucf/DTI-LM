from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import sys

class CHEMFEATURE:
    def __init__(self,device):
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        #total parameters in the model
        #print(sum(p.numel() for p in self.model.parameters()))
        #sys.exit()
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.device = device

    def get_representations(self, X_drug):
    
        batch_size = 1
        data = [X_drug[i * batch_size:(i + 1) * batch_size] for i in range((len(X_drug) + batch_size - 1) // batch_size )]

        drug_representations = []
        for temp_data in tqdm(data):
            inputs = self.tokenizer(temp_data.tolist(), padding=True, truncation=True, return_tensors="pt").to(self.device)
            batch_lens = (inputs['attention_mask'] != 0).sum(1)
            
            #return hidden representations
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            token_representations = outputs.hidden_states[-1].to('cpu')

                #token_representations = model(**inputs).logits

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                drug_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            
            del token_representations, inputs
            torch.cuda.empty_cache()

        drug_representations = torch.stack(drug_representations)
        return np.array(drug_representations)

