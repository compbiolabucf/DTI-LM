import esm, torch, sys, os
import numpy as np
from tqdm import tqdm


class ESMFEATURE:
    def __init__(self,device):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        #print(self.model)
        #sys.exit()
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device

    def get_representations(self, X_target):

        data = []
        for i in range(len(X_target)):
            data.append(("protein"+str(i),X_target[i]))
        
        batch_size = 1
        data = [data[i * batch_size:(i + 1) * batch_size] for i in range((len(data) + batch_size - 1) // batch_size )]
        # Process batches (this supports multiple sequence inputs)
        self.model.eval()  # disables dropout for deterministic results

        sequence_representations = []    
        for temp_data in tqdm(data):
            batch_labels, batch_strs, batch_tokens = self.batch_converter(temp_data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            
            batch_tokens = batch_tokens.to(self.device)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33].to('cpu')
            
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                # print sequence representation shape
                
            del results, batch_tokens
            torch.cuda.empty_cache() 

        #use torch stack to convert list of tensors to tensor
        sequence_representations = torch.stack(sequence_representations)

        return np.array(sequence_representations)


