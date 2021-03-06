import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .mb_base import MusicBertBase
from .vggish_torch.vggish import VGGish
from .vggish_torch.vggish_utils import preprocess_sound
from ..utils.mask_utils import mask_from_map, sample_map


class MusicBertVGGish(nn.Module):

    def __init__(self, dim_model=768, freeze_vggish = True,
                 skip_vggish = False, **kwargs):
        super().__init__()
        
        self.BERT = MusicBertBase(dim_model=dim_model, **kwargs)
                
        self.skip_vggish = skip_vggish
            
        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                      'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                   'releases/download/v0.1/vggish_pca_params-970ea276.pth'
        }

        self.vggish = VGGish(urls=model_urls, preprocess = False, postprocess = True)
        if freeze_vggish:
            self.vggish.set_grad(False) # Freeze the VGGish model
        
        self.vggish_projector = nn.Linear(128, dim_model) # Projection from VGGish features to model size
        self.projector_activation = nn.Tanh()
        self.PATH = "models/" + kwargs['name'] + ".pth"
        
        
    def get_device(self):
        return next(self.parameters()).device
    
    def save_model(self, path=None):
        path = path if path is not None else self.PATH
        torch.save(self.state_dict(), path)
    
    def load_model(self, path=None):
        path = path if path is not None else self.PATH
        self.load_state_dict(torch.load(path))
        
    def load_pretrained(self):
        self.BERT.load_pretrained()
        
        
    def process_sound(self, sound):
        if sound is not None:
            if not self.skip_vggish:
                sound = preprocess_sound(sound)
                encoded_sound = self.vggish(sound)
            else:
                encoded_sound = sound # Signal Provided assumed to be VGGish features
            
            encoded_sound_proj = self.vggish_projector(encoded_sound)
            encoded_sound_proj = self.projector_activation(encoded_sound_proj)
            return encoded_sound, encoded_sound_proj
        else:
            return None, None
        
    
    def mask_sound(self, sound):
        
        # sound dims = seq_len, batch_size, feat_dim
        if sound is not None:

            seq_len, batch_size, feat_dim = sound.shape
            n_masking = seq_len//4 + 1

            mask = sample_map(batch_size, seq_len, n_masking, device=sound.device)
            masked_sound = mask_from_map(sound, mask)

            return masked_sound, mask
        else:
            return None, None
    
    def forward(self,
                src_sound=None,
                src_sound2=None,
                src_tokens=None,
                add_special_tokens=True,
                mask = False,
                return_extras = False):
        
        # Sound Input Shape: (seq_length, Batch_size, N_features)

        encoded_sound, encoded_sound_proj = self.process_sound(src_sound)
        encoded_sound2, encoded_sound2_proj = self.process_sound(src_sound2)

        
        if mask is True:
            encoded_sound_proj,  mask1  = self.mask_sound(encoded_sound_proj)
            encoded_sound2_proj, mask2 = self.mask_sound(encoded_sound2_proj)
        else:
            mask1, mask2 = None, None
                        
        # Sound Output Shape: reduced_seq_length, batch_size, dim_model
            
        output, layers_out = self.BERT(src_sound = encoded_sound_proj, 
                                       src_sound2 = encoded_sound2_proj,
                                       src_tokens = src_tokens, 
                                       add_special_tokens = add_special_tokens,
                                       return_layers_out = True)
           
        sequence_out = output[0] # BERT Pooling
        extras = {
            "encoded_sound_1": encoded_sound,
            "encoded_sound_2": encoded_sound2,
            "mask_1": mask1,
            "mask_2": mask2,
            "layers_out": layers_out,
            "sequence_vector": sequence_out,
        }

        if return_extras:
            return output, extras
        else:
            return output
        