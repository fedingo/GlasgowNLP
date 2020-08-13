import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mask_utils import *
from ..utils.generic_utils import roll
from ..utils import string_import

from .mi_estimator import *
from .similarities import *

class AbstractMINE(nn.Module):
    
    def __init__(self, z_dim, c_dim, mi_estimator="infoNCE", similarity="BilinearSimilarity"):
        super().__init__()
        
        mi_class = string_import("source.models_pretraining.mi_estimator." + mi_estimator)
        sim_class = string_import("source.models_pretraining.similarities." + similarity)
        
        self.similarity = sim_class(z_dim, c_dim)
        self.mi_estimator = mi_class(self.similarity)


class CPCLoss(AbstractMINE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.z_normalizer = nn.LayerNorm(args[0]) # z_dim
        self.mask = True

        
    def forward(self, z, mask, c):

        # sample_sound dims = seq_len, batch_size, feat_dim

        sequence_length, batch_size, n_features = z.shape
        mask_len = mask.shape[0]
        neg_map = sample_map(batch_size, sequence_length, mask_len, device=z.device) 

        c_t = gather_from_map(c, mask)
        z_t = gather_from_map(z, mask) # <--- True examples

        masked_z = mask_from_map(z, mask) # Hid true labels
        z_neg = gather_from_map(masked_z, neg_map) # <--- Negative examples
        # uses masked_z to not leak information of the true vector

        z_t = torch.cat([z_t, z_neg], 0)
        z_t = self.z_normalizer(z_t)
        
        c_t = c_t.unsqueeze(0).repeat(2*mask_len, 1, 1, 1)
        z_t = z_t.unsqueeze(1).repeat(1, mask_len, 1, 1)
        
        M = self.similarity(z_t, c_t).squeeze(-1)
        M = torch.softmax(M, 0)
        M = torch.log(torch.diagonal(M) + 1e-3)

        # InfoNCE Loss taken from the CPC paper https://arxiv.org/pdf/1807.03748.pdf
        loss = - torch.mean(M)
        
        return loss
    
    
class DeepInfoMaxLoss(AbstractMINE):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.z_normalizer = nn.LayerNorm(args[0]) # z_dim
        self.mask = False
        
        
    def forward(self, z, mask, c):

        # sample_sound dims = seq_len, batch_size, feat_dim
        sequence_length, batch_size, n_features = z.shape
        
        assert batch_size > 1, "Invalid batch size"

        c = c[0] # Take CLS vector
        
        z_neg_list = []
        for _ in range(batch_size - 1):
            z_neg = roll(z, 1, 1) # batch_dim
            z_neg = self.z_normalizer(z_neg)
            z_neg_list.append(z_neg)
            
        
#         z_neg = roll(z, 1, 1) # batch_dim

        z = self.z_normalizer(z)
        
        c = c.unsqueeze(0).repeat(sequence_length, 1, 1)
        # tiles to the same dimension to z
        
        loss = - self.mi_estimator(c, z, z_neg_list)
        
        return loss
    