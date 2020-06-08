import torch
import torch.nn as nn

from ..utils.mask_utils import *


class CPCLoss(nn.Module):

    def __init__(self, z_dim, c_dim):
        super().__init__()
        
        self.similarity_model = nn.Bilinear(z_dim, c_dim, 1, bias=False)
        self.similarity_activation = nn.Softmax()

    def similarity(self, z, c):
        return self.similarity_model(z,c)
        
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
        # remove last dimension with squeeze
        
        c_t = c_t.unsqueeze(0).repeat(2*mask_len, 1, 1, 1)
        z_t = z_t.unsqueeze(1).repeat(1, mask_len, 1, 1)
        
        M = self.similarity(z_t, c_t).squeeze(-1)
        M = torch.softmax(M, 0)
        M = torch.log(torch.diagonal(M))

        # InfoNCE Loss taken from the CPC paper https://arxiv.org/pdf/1807.03748.pdf
        loss = - torch.mean(M)

        return loss
    
    
#class DeepInfoMaxLoss(nn.Module):