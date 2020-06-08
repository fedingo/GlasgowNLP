import torch


def sample_map(batch_size, sequence_length, n_sampling, device):
    sample_map = torch.rand((n_sampling, batch_size), device=device)
    sample_map = sample_map*(sequence_length-1)+1
    
    return sample_map.long()


def gather_from_map(x, sample_map):
    #input shape: seq_len, batch_size, feat_dim
    
    emb_dim = x.shape[2]
    process_map = sample_map.unsqueeze(-1).repeat(1,1,emb_dim)
    
    gathered = torch.gather(x, 0, process_map)
    return gathered


def mask_from_map(x, sample_map):
    #input shape: seq_len, batch_size, feat_dim
    
    batch_size = x.shape[1]
    masked = x.clone()
    for i in range(batch_size):
        for j in sample_map[:,i]:
            if torch.rand(1) < 0.9:
                masked[j,i] = 0
        
    return masked


def concat_maps(x, y, offset):
    
    assert x.shape[1] == y.shape[1], \
        "Batch size mismatch. X batch size is %d and Y batch size is % d" % (x.shape[1], y.shape[1])
    
    assert (x < offset).all(), \
        "Offset overlaps the first argument. Max Value is %d and offset is %d" % (torch.max(x), offset)  
    
    return torch.cat([x, y+offset], axis=0)