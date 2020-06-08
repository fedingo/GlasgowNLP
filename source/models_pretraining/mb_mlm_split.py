import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import roll
from .mlm_losses import *
from .abstract_mb import AbstractMusicBertPretraining


class MusicBertMLM_split(AbstractMusicBertPretraining):

    def __init__(self, dim_model=768, CLS_training = "song_matching", MLM_training="mse", name="music_bert_mlm", **kwargs):
        super().__init__(name, dim_model=dim_model, **kwargs)

        
        assert MLM_training in ['MSE', 'CE'], "Unrecognized MLM training: " + MLM_training
        
        if MLM_training == 'MSE':
            self.mask_criterion = MaskedMSELoss(dim_model, 128)
        elif MLM_training == 'CE':
            self.mask_criterion = MaskedCrossEntropyLoss(dim_model, 128)
        
        
        assert CLS_training in ['SM', 'MOP'], "Unrecognized CLS training: " + CLS_training
        
        self.cls_criterion = CLSBinaryLoss(dim_model)
        self.CLS_training = CLS_training

        self.flipper = True
        
        
    def generate_CLS(self, sample_1, sample_2):
        
        self.flipper = not self.flipper
        # Ensures a 50% distribution of positive and negative samples
        if self.flipper:
            
            if self.CLS_training == "SM":
                sample_2 = roll(sample_2, 1, 1)
                #Rolls inside the batch to split song segments

            elif self.CLS_training == "MOP":
                sample_1, sample_2 = sample_2, sample_1
                
            return 0, sample_1, sample_2
        
        else:
            return 1, sample_1, sample_2
    
    
    def pretraining_loss(self, batch):
        
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)

        seq_len, batch_size, feat_dims = sample_sound.shape
        
        ### Preprocessing
        
        cut_index = seq_len // 2
        
        sample_sound_1, sample_sound_2 = sample_sound[:cut_index], sample_sound[cut_index:]
        
        CLS_target, sample_sound_1, sample_sound_2 = self.generate_CLS(sample_sound_1, sample_sound_2)
                
        
        output, extras = self.BERT(src_sound = sample_sound_1,
                                  src_sound2 = sample_sound_2,
                                  mask = True,
                                  return_extras = True) 
        
        ### Output Extraction
        
        encoded_sound_1, encoded_sound_2 =  extras['encoded_sound_1'], extras['encoded_sound_2']
        mask_1, mask_2 = extras['mask_1'], extras['mask_2']
        
        seq_len_1, seq_len_2 = encoded_sound_1.shape[0], encoded_sound_2.shape[0]
        
        out_sound_1 = output[1:seq_len_1+1]
        out_sound_2 = output[seq_len_1+2:seq_len_1+2+seq_len_2]
        
        
        ### CLS LOSS ###
        cls_loss = self.cls_criterion(output[0], CLS_target)

                
        ### MASK LOSS ###
        mask_loss = self.mask_criterion(out_sound_1, mask_1, encoded_sound_1) + \
                    self.mask_criterion(out_sound_2, mask_2, encoded_sound_2)
            
            
        ### Logging
        self.cls_loss_curve = self.append(self.cls_loss_curve, cls_loss.detach())
        self.mask_loss_curve = self.append(self.mask_loss_curve, mask_loss.detach())
        
        return cls_loss + mask_loss