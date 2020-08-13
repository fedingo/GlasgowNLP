import torch

def preprocess_sound(log_mel):
        
    # assumes log_mel is of shape (seq_length, Batch_size, N_features)
    
    log_mel = log_mel.permute(1,0,2) #makes batch first dimension
    
    batch_size, seq_length, n_features = log_mel.shape
    
    if seq_length < 96:
        # if less than 1 window size pad with zeros
        padded = torch.zeros(batch_size, 96, n_features).to(log_mel.device)
        
        padded[:, :seq_length, :] = log_mel
        log_mel = padded
    
    windows_length = 96 # Parameter from vggish model
    seq_len = log_mel.shape[1]
    batch_size = log_mel.shape[0]
    seq_len -= seq_len%windows_length
    
    num_frames = seq_len//windows_length
    shape = (batch_size, num_frames, 1, windows_length, log_mel.shape[-1])
    
    log_mel = log_mel[:,:seq_len].view(shape).float() # breaks the seq_length in (n_frames, windows)
    
    # output_shape is (seq_length, batch_size, channels, H, W)
    
    return log_mel.permute(1,0,2,3,4).contiguous()