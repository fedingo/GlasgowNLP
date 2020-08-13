import numpy as np
import torch
import torch.nn as nn

from .losses import MSELoss
from .abstract_mb import AbstractMusicBertTraining

class MusicBertRegression(AbstractMusicBertTraining):

    def __init__(self, out_dim, dim_model=768, name="mb_regression", **kwargs):
        super().__init__(name, dim_model=dim_model, **kwargs)    
        
        dim_vggish = 128
        self.loss = MSELoss(dim_model, 2*dim_model, out_dim)
        
    
    def training_loss(self, batch):
        
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['target'].squeeze().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        
        c, extras = self.BERT(src_sound = sample_sound,
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
                              return_extras = True) 
        
        seq_vector = extras['sequence_vector']
        
        output = self.loss.predict(seq_vector)
        return output.detach()
    
    
    def evaluate(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            
            catter_pred = []
            catter_target = []
            for i, batch in enumerate(val_dataloader):
                prediction = self(batch)
                target = batch['target'].squeeze().to(self.get_device())

                catter_pred.append(prediction)
                catter_target.append(target)
            
            predictions = torch.cat(catter_pred, dim=0)
            targets = torch.cat(catter_target, dim=0)
            
        self.train()      
        return self.loss.evaluate(targets, predictions)
        