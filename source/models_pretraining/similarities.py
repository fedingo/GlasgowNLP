import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearSimilarity(nn.Module):
    
    def __init__(self, z_dim, c_dim):
        super().__init__()
        self.similarity_model = nn.Bilinear(z_dim, c_dim, 1, bias=False)
                
    def forward(self, z, c):
        return self.similarity_model(z,c)