import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .mb_base import MusicBertBase
from .vggish_torch.vggish import VGGish
from .vggish_torch.vggish_utils import preprocess_sound
from ..utils.mask_utils import mask_from_map, sample_map


class MusicBertVGGish(nn.Module):

    def __init__(self, dim_model=768, freeze_vggish = True, **kwargs):
        super().__init__()
        
        self.BERT = MusicBertBase(dim_model=dim_model, **kwargs)
                
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
        
#     def __preprocess_sound(self, log_mel):
        
#         # assumes log_mel is of shape (seq_length, Batch_size, N_features)
        
#         log_mel = log_mel.permute(1,0,2) #makes batch first dimension
        
#         batch_size, seq_length, n_features = log_mel.shape
        
#         if seq_length < 96:
#             # if less than 1 window size pad with zeros
#             padded = torch.zeros(batch_size, 96, n_features).to(self.get_device())
            
#             padded[:, :seq_length, :] = log_mel
#             log_mel = padded
        
#         windows_length = 96 # Parameter from vggish model
#         seq_len = log_mel.shape[1]
#         batch_size = log_mel.shape[0]
#         seq_len -= seq_len%windows_length
        
#         num_frames = seq_len//windows_length
#         shape = (batch_size, num_frames, 1, windows_length, log_mel.shape[-1])
        
#         log_mel = log_mel[:,:seq_len].view(shape).float() # breaks the seq_length in (n_frames, windows)
        
#         # output_shape is (seq_length, batch_size, channels, H, W)
        
#         return log_mel.permute(1,0,2,3,4).contiguous()
        
        
    def process_sound(self, sound):
        if sound is not None:
            sound = preprocess_sound(sound)
            encoded_sound = self.vggish(sound)
            
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
        