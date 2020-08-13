import torch
import torch.nn as nn
import torch.nn.functional as F


class JSD(nn.Module):
    
    def __init__(self, similarity_fn):
        super().__init__()
        self.similarity = similarity_fn
        
        
    def forward(self, c, z, z_neg_list):
        
        E_pos = -F.softplus(-self.similarity(z, c)).mean()
        
        E_neg = 0
        for z_neg in z_neg_list:
            E_neg += -F.softplus(-self.similarity(z_neg, c)).mean()
        
        E_neg = E_neg/len(z_neg_list) # Adjust mean
        
        jsd = - (E_pos - E_neg)
        return jsd
    
    
class DV(nn.Module):
    # Donsker-Varadhan (DV)
    
    def __init__(self, similarity_fn):
        super().__init__()
        self.similarity = similarity_fn
        
        
    def forward(self, c, z, z_neg_list):
        
        E_pos = self.similarity(z, c).mean()
        
        u = []
        for z_neg in z_neg_list:
            u.append(self.similarity(z_neg, c))
        u = torch.cat(u, dim=0)
        
        ## Trick for numeric stability
        u_max = torch.max(u)
        E_neg = torch.log(torch.exp(u - u_max).mean() + 1e-6) + u_max
        
        DV = - (E_pos - E_neg)
        
        assert not torch.isinf(DV), "Infinite Loss"
        
        return DV
    
    
class infoNCE(nn.Module):    
    # Noise contrastive estimation-based loss - Oord et al.(2018)
    
    def __init__(self, similarity_fn):
        super().__init__()
        self.similarity = similarity_fn
        
        
    def forward(self, c, z, z_neg_list):
              
        pos = self.similarity(z, c).unsqueeze(0)
        
        negs = []
        for z_neg in z_neg_list:
            neg = self.similarity(z_neg, c).unsqueeze(0)
            negs.append(neg)
        
        u = torch.cat([pos] + negs, dim=0)
        u = F.log_softmax(u, dim=0)
        
        infoNCE = - u[0].mean()
        
        assert not torch.isnan(infoNCE), "NaN Loss"
        
        return infoNCE
        