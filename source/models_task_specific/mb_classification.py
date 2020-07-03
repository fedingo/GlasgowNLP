import numpy as np
import torch
import torch.nn as nn

from .losses import ClassificationLoss
from .abstract_mb import AbstractMusicBertTraining

class MusicBertClassifier(AbstractMusicBertTraining):

    def __init__(self, n_class, dim_model=768, multi_label=False, name="mb_classification", **kwargs):
        super().__init__(name, dim_model=dim_model, **kwargs)    
        
        dim_vggish = 128
        self.loss = ClassificationLoss(dim_model, dim_model, n_class, multi_label)
        
    
    def training_loss(self, batch):
        
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['encoded_class'].squeeze().long().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        
        c, extras = self.BERT(src_sound = sample_sound,
                                   mask = True,
                                   return_extras = True) 
        
        seq_vector = extras['sequence_vector']
        
        loss = self.loss(seq_vector, target)
                
        return loss
    
    
    def forward(self, batch):
        sample_sound = batch['song_features'].to(self.get_device())
        
        if len(sample_sound.shape) == 2:
            #Add Batch dimension
            sample_sound = sample_sound.unsqueeze(0)
            
        sample_sound = sample_sound.permute(1,0,2)
        
        c, extras = self.BERT(src_sound = sample_sound,
                                   mask = True,
                                   return_extras = True) 
        
        seq_vector = extras['sequence_vector']
        
        output = self.loss.predict(seq_vector)
        return nn.functional.softmax(output, dim=-1).detach()
    
    
    def evaluate_fn(self, batch):
            
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['encoded_class'].squeeze().long().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        
        c, extras = self.BERT(src_sound = sample_sound,
                                   mask = True,
                                   return_extras = True) 
        
        seq_vector = extras['sequence_vector']
        
        score = self.loss.accuracy(seq_vector, target)
        return score
        
        