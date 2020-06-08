
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import *
import numpy as np
from .music_bert_base import MusicBert
from tqdm.notebook import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## NOT WORKING AT THE MOMENT
# TODO: adapt this class to do Sequence Transduction (Train to generate)

class MusicBertReviews(nn.Module):
    
    def __init__(self, input_dim, name = "music_bert_generation",
                 hidden_dims = 128):
        super(MusicBertReviews, self).__init__()
        
        self.BERT = MusicBert(dim_sound = input_dim, name = name)

    def generate(self, tokenizer, starting_text="", sound_sample=None, MAX_STEPS = 10):
        self.eval() # Turn on the evaluation mode
        with torch.no_grad():
            mask = tokenizer.convert_tokens_to_ids(["[MASK]"])
            
            tokens = tokenizer.encode(starting_text, add_special_tokens=False)
            tokens = np.append(tokens,mask)
            
            for i in range(MAX_STEPS):
                text = torch.tensor(tokens).unsqueeze(0)
                
                if sound_sample is not None:
                    _, output_text = self.BERT(\
                                      sound_sample.permute(1,0,2).to(device),
                                      text.permute(1,0).to(device))
                else:
                    _, output_text = self.BERT(\
                                      src_tokens=text.permute(1,0).to(device))

                log_probs = output_text.cpu().detach().squeeze()
                next_token = np.argmax(log_probs, axis=-1)

                if next_token.shape != ():
                    next_token = next_token[-1]

                tokens = np.insert(tokens, -1, next_token)
                    
        return tokenizer.decode(tokens)