import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .mb_base import MusicBertBase
from .sound_modules import SoundCNNEncoder, SoundCNNDecoder

class MusicBertConvs(nn.Module):

    def __init__(self, dim_sound, n_convs=4, dim_model=768,
                 hidden_dims = 128, multi_label=False, **kwargs):
        super().__init__()
        
        self.BERT = MusicBertBase(dim_sound = dim_sound, dim_model=dim_model, **kwargs)
        self.convs = n_convs
        
        if n_convs == 0:
            self.sound_compressor = nn.Linear(dim_sound, dim_model)
            self.sound_decompressor = nn.Identity()
        else:
            self.sound_compressor = SoundCNNEncoder([dim_sound] + [dim_model]*(n_convs-1))
            self.sound_decompressor = SoundCNNDecoder([dim_model]*n_convs)
        
        
    def get_device(self):
        return next(self.parameters()).device    
      
        
    def load_pretrained(self):
        self.load_pretrained()
        
    
    def forward(self,
                src_sound=None,
                src_sound2=None,
                src_tokens=None,
                add_special_tokens=True,
                return_encoded_sound = False):
        
        sound_length = 0
        tokens_length = 0
        
        if src_sound is not None:
            sound_length = src_sound.shape[0]
            src_sound = self.sound_compressor(src_sound)
        
        if src_tokens is not None:
            tokens_length = src_tokens.shape[0]
            
        if add_special_tokens:
            tokens_length += 2
            
        output = self.BERT(src_sound, src_tokens, add_special_tokens)
        
        if sound_length>0:
            # if sound was provided, decompresses the sound_sequence
            sound_sequence = output[tokens_length:]
            token_sequence = output[:tokens_length]

            if self.convs > 0:
                sound_sequence = self.sound_decompressor(sound_sequence, target_len=sound_length)
            else:
                sound_sequence = self.sound_decompressor(sound_sequence)
                
            output = torch.cat((token_sequence, sound_sequence), 0)
        
        
        assert output.shape[0] == tokens_length + sound_length,\
           "Expecting that input and output sequences to have the same length, but got {:d} and {:d}"\
           .format(output.shape[0], tokens_length + sound_length)

        return output
        