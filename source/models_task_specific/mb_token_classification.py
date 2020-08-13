import numpy as np
import torch
import torch.nn as nn

from .losses import ClassificationLoss
from .abstract_mb import AbstractMusicBertTraining

class MusicBertAudioset(AbstractMusicBertTraining):

    def __init__(self, n_class, dim_model=768, multi_label=False, name="mb_audioset_tokens", **kwargs):
        super().__init__(name, dim_model=dim_model, **kwargs)    
        
        dim_vggish = 128
        self.loss = ClassificationLoss(dim_model, 2*dim_model, n_class, multi_label)
        
    
    def training_loss(self, batch):
        
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['encoded_class'].squeeze().long().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        seq_len, batch_size, feat_dim = sample_sound.shape
        
        target = target.unsqueeze(0).repeat(seq_len, 1, 1)
        c = self.BERT(src_sound = sample_sound,
                      add_special_tokens=False)

        
        loss = self.loss(c, target)
                
        return loss
    
    
    def forward(self, sample):
        sample_sound = sample.to(self.get_device())
        
        if len(sample_sound.shape) == 2:
            #Add Batch dimension
            sample_sound = sample_sound.unsqueeze(0)
            
        sample_sound = sample_sound.permute(1,0,2)
        
        c = self.BERT(src_sound = sample_sound,
                      add_special_tokens=False) 
        
        
        output = self.loss.predict(c)
        return output
    
    
    def evaluate_fn(self, batch):
            
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['encoded_class'].squeeze().long().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        seq_len, batch_size, feat_dim = sample_sound.shape
        
        target = target.unsqueeze(0).repeat(seq_len, 1, 1)
        c = self.BERT(src_sound = sample_sound,
                      add_special_tokens=False) 
        
        
        score = self.loss.score(c, target)
        return score
        
        