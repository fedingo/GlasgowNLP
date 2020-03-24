import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.nn.modules import LayerNorm

# TODO: Normalizing function that cuts the input to a multiple of 16,
# so that the Encoder-Decoder architecture have the same input/output

def normalize_seq_len(x, multiplier = None, target_len = None):
    
    # assumed dims (Batch, Channels, Seq_len)
    
    if multiplier is not None:
        pad_size = (- x.shape[-1]) % multiplier
        x = F.pad(x, (0, pad_size), mode='constant', value=0)
        
    if target_len is not None:
        x = x[:,:,:target_len]
        
    return x
        

class SoundCNNEncoder(nn.Module):
    
    def __init__(self, arch, activation=F.relu):
        super(SoundCNNEncoder, self).__init__()
        
        # arch = list of output_channels for each conv_layer
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activation = activation
        
        self.padding_multiplier = 4**(len(arch) - 1)
        
        # Arch --> Conv1D, BatchNorm1D, Relu/Gelu
        current_size = arch[0]
        
        for out_channels_size in arch[1:]:
            self.convs.append(nn.Conv1d(in_channels = current_size, 
                                out_channels = out_channels_size, 
                                kernel_size = 4, # 8
                                stride=4, padding=0))
        
            self.batch_norms.append(nn.BatchNorm1d(out_channels_size))
            
            current_size = out_channels_size
            
        #self.out_layer = nn.Linear(arch[-1], arch[-1])
    
    
    def get_device(self):
        return next(self.parameters()).device
    
    
    def forward(self, x):
        ## Expected input shape: (Seq_len, Batch_size, Channels)
        
        ## Conv takes (Batch, Channels, Seq_len)
        result = x.permute(1,2,0)
        result = normalize_seq_len(result, multiplier=self.padding_multiplier)
                
        for conv, norm in zip(self.convs, self.batch_norms):
            result = conv(result)
            result = norm(result)
            result = self.activation(result)
        
        result = result.permute(2,0,1)

        return result # self.out_layer(result)
    
    
class SoundCNNDecoder(nn.Module):
    
    def __init__(self, arch, activation=F.relu):
        super(SoundCNNDecoder, self).__init__()
        
        # arch = list of output_channels for each conv_layer
        
        self.deconvs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activation = activation
        
        # Architecture --> Conv1D, BatchNorm1D, Relu/Gelu
        
        current_size = arch[0]
        
        for out_channels_size in arch[1:-1]:
            self.deconvs += [nn.ConvTranspose1d(in_channels = current_size, 
                                              out_channels = out_channels_size, 
                                              kernel_size=4, stride=4)]
        
            self.batch_norms += [nn.BatchNorm1d(out_channels_size)]
            
            current_size = out_channels_size
            
        self.out_deconv = nn.ConvTranspose1d(in_channels = current_size, 
                                           out_channels = arch[-1], 
                                           kernel_size=4, stride=4)
        
        #self.out_layer = nn.Linear(arch[-1], arch[-1])
    
    
    def get_device(self):
        return next(self.parameters()).device
    
    
    def forward(self, x, target_len = None):
        ## Expected input shape: (Seq_len, Batch_size, Channels)
        
        # Conv takes (Batch, Channels, Seq_len)
        result = x.permute(1,2,0)
        
        for deconv, norm in zip(self.deconvs, self.batch_norms):
            result = deconv(result)
            result = norm(result)
            result = self.activation(result)
        
        result = self.out_deconv(result)
        
        # Normalizes the length of the output
        if target_len is not None:
            result = normalize_seq_len(result, target_len=target_len)
            
        result = result.permute(2,0,1)
        
        return result # self.out_layer(result)