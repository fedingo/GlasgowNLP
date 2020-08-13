import torch
import torch.nn as nn

from ..utils.mask_utils import gather_from_map


class CLSBinaryLoss(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        
        self.predictor = nn.Linear(input_dim, 1)
        self.cls_criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, cls_vector, target):
        
        CLS_pred = self.predictor(cls_vector) # Shape: Batch_size, 1
        CLS_target = torch.ones_like(CLS_pred)*target #reshape and cast to torch

        cls_loss = self.cls_criterion(CLS_pred, CLS_target)
        return cls_loss


class MaskedMSELoss(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.normalize = nn.LayerNorm(output_dim)
        self.reconstructor = nn.Linear(input_dim, output_dim)
        self.mse_criterion = nn.MSELoss()
        
    
    def forward(self, sequence, mask, target):
        
        reconstruction = self.reconstructor(sequence)
        target_normalized = self.normalize(target)

        assert target.shape == reconstruction.shape, \
                "Mismatch between target and reconstruction segments %s, %s" % \
                    (target.shape, reconstruction.shape)

        assert mask is not None, "Masks should not be None"

        target_gathered = gather_from_map(target_normalized, mask)
        recon_gathered = gather_from_map(reconstruction, mask)

        mask_loss = self.mse_criterion(recon_gathered, target_gathered)
        
        return mask_loss
    
    

class MaskedCrossEntropyLoss(nn.Module):
    
    def __init__(self, input_dim, output_dim, n_binns=256):
        super().__init__()
        
        self.n_binns = n_binns
        
        self.feature_classifier = nn.Linear(input_dim, output_dim*self.n_binns)
        self.ce_criterion  = nn.CrossEntropyLoss()
        
        
    def forward(self, sequence, mask, target):
        
        # sequence shape: seq_len, batch_size, inpud_dims
        
        batch_size = sequence.shape[1]
        
        assert mask is not None, "Masks should not be None"
        
        classified_features = self.feature_classifier(sequence)
        classified_features_gathered = gather_from_map(classified_features, mask)
        
        features_gathered_reshaped = classified_features_gathered.view(-1, batch_size, 128, self.n_binns).contiguous()

        target_gathered = gather_from_map(target, mask)
        
        assert target_gathered.shape == features_gathered_reshaped.shape[:-1], \
                "Mismatch between target and model output segments %s, %s" % \
                    (target_gathered.shape, features_gathered_reshaped.shape)
        
        # Reshape for CrossEntropyLoss
        features_gathered_reshaped = features_gathered_reshaped.view(-1, self.n_binns)
        target_gathered = target_gathered.view(-1)
        
        mask_loss = self.ce_criterion(features_gathered_reshaped, target_gathered.long())
        
        return mask_loss