from ..models_base.vggish_torch.vggish import VGGish
from ..models_base.vggish_torch.vggish_utils import preprocess_sound

import torch
from .losses import ClassificationLoss
from .abstract_mb import AbstractMusicBertTraining

## Fine-Tune / Train From scratch the VGGish Model

class VGGishClassifier(AbstractMusicBertTraining):
    
    def __init__(self, n_class, multi_label=False, pretrained=True, **kwargs):
        super().__init__("ClassificationVGGish", initialize_bert=False, **kwargs)
        
        model_urls = {
                    'vggish': 'https://github.com/harritaylor/torchvggish/'
                              'releases/download/v0.1/vggish-10086976.pth',
                    'pca': 'https://github.com/harritaylor/torchvggish/'
                           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
                }
        

        self.vggish = VGGish(urls=model_urls, pretrained=pretrained, preprocess = False, postprocess = True)

        self.loss = ClassificationLoss(128, 128, n_class, multi_label)
        
        
        
    
    def forward(self, sound_sample):
        sound_sample = preprocess_sound(sound_sample)
        encoded_sound = self.vggish(sound_sample)
        
        seq_vector = torch.mean(encoded_sound, dim=0)
        return seq_vector
    
    
    def training_loss(self, batch):
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['encoded_class'].squeeze().long().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        
        seq_vector = self(sample_sound)
        loss = self.loss(seq_vector, target)
                
        return loss
    
    def evaluate_fn(self, batch):
            
        sample_sound = batch['song_features'].to(self.get_device()).permute(1,0,2)
        target = batch['encoded_class'].squeeze().long().to(self.get_device())
        
        # sample_sound dims = seq_len, batch_size, feat_dim
        
        seq_vector = self(sample_sound)
        
        score = self.loss.accuracy(seq_vector, target)
        return score