import numpy as np
import torch
import torch.nn as nn

from .mine_losses import *
from .abstract_mb import AbstractMusicBertPretraining

class MusicBertPretrain(AbstractMusicBertPretraining):

    def __init__(self, dim_model=768, name="music_bert_pretrain", **kwargs):
        super().__init__(name, dim_model=dim_model, **kwargs)    
        
        dim_vggish = 128
        self.loss = CPCLoss(dim_vggish, dim_model)
        
    
    def pretraining_loss(self, batch):
        
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        
        c, extras = self.BERT(src_sound = sample_sound,
                                   mask = True,
                                   return_extras = True) 
        
        z = extras['encoded_sound_1']
        mask = extras['mask_1']
        
        loss = self.loss(z, mask, c)
                
        return loss