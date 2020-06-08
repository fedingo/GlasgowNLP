import os
import librosa
import numpy as np
from scipy import signal
import torch

from ..models_base.vggish_torch import vggish_params as vggish_params_torch
from ..models_base.vggish_torch import mel_features
from ..models_base.vggish_torch.vggish import VGGish

def log_mel_spectrogram(raw_audio, sr):
    if sr != vggish_params_torch.SAMPLE_RATE:
        raw_audio = librosa.resample(raw_audio, sr, vggish_params_torch.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        raw_audio,
        audio_sample_rate=vggish_params_torch.SAMPLE_RATE,
        log_offset=vggish_params_torch.LOG_OFFSET,
        window_length_secs=vggish_params_torch.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params_torch.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params_torch.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params_torch.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params_torch.MEL_MAX_HZ)
    
    return log_mel
    
    
class BERT_Features(object):
    
    def __init__(self, BERT_instance):
        self.BERT = BERT_instance
        self.BERT.eval()
        
    def __call__(self, sample):
        sample_sound = torch.tensor(sample['song_features']).to(self.BERT.get_device()).unsqueeze(1)
        output = self.BERT(sample_sound).squeeze().detach().cpu()
        
        sample['song_features'] = output.numpy() #Shape: Batch_size, 768
        
        return sample
    

class toVGGishPreprocess(object):
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        
        sample['song_features'] = log_mel_spectrogram(raw_audio, sr)
        return sample
    

class toMFCC(object):

    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        mfcc = librosa.feature.mfcc(raw_audio, sr=sr)
        
        sample['song_features'] = np.transpose(mfcc)
        return sample
    
    
class toMelSpectrogram(object):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        melspectrogram = librosa.feature.melspectrogram(raw_audio, sr, **self.kwargs)
        
        sample['song_features'] = np.transpose(melspectrogram)
        return sample
    

class toLogMelSpectrogram(object):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        melspectrogram = librosa.feature.melspectrogram(raw_audio, sr, **self.kwargs)
        
        log_mel = librosa.power_to_db(melspectrogram, ref=np.max)
        
        sample['song_features'] = np.transpose(log_mel)
        return sample
    
    
class toSpectrogram(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        frequencies, times, spectrogram = signal.spectrogram(raw_audio, sr, **self.kwargs)
        
        sample['song_features'] = np.transpose(spectrogram)
        return sample
    
    
class toChroma(object):
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        chromagram = librosa.feature.chroma_stft(raw_audio, sr=sr)
        
        sample['song_features'] = np.transpose(chromagram)
        return sample


class toCQT(object):
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']

        cqt = np.abs(librosa.cqt(raw_audio, sr=sr)) 
        
        sample['song_features'] = np.transpose(cqt)
        return sample


class toSTFT(object):
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']

        raw_8k = librosa.resample(raw_audio, sr, 8000)
        stft = np.abs(librosa.stft(raw_audio)) 
        
        sample['song_features'] = np.transpose(stft)
        return sample
   
    
class toRaw(object):

    def __init__(self, hops_per_sec):
        self.hops_per_sec = hops_per_sec
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        
        n_feat = sr//self.hops_per_sec
        seq_len = np.min([len(raw_audio)//n_feat + 1, 2048])  
        
        raw_audio = raw_audio.copy()
        raw_audio.resize(seq_len, n_feat)
        
        sample['song_features'] = raw_audio
        return sample
    
    
class toVggishTorch(object):
    
    def __init__(self, postprocess = True, preprocess=True):
        
        

        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                      'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                   'releases/download/v0.1/vggish_pca_params-970ea276.pth'
        }
        
        self.preprocess = preprocess

        self.vggish = VGGish(urls=model_urls, preprocess=preprocess, postprocess = postprocess).cuda()
        
    def __preprocess_sound(self, log_mel):
        
        # assumes log_mel is of shape (seq_length, Batch_size, N_features)
        
        log_mel = log_mel.permute(1,0,2) #makes batch first dimension
        
        batch_size, seq_length, n_features = log_mel.shape
        
        if seq_length < 96:
            # if less than 1 window size pad with zeros
            padded = torch.zeros(batch_size, 96, n_features)
            
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
        
    def __call__(self, sample):
        
        if self.preprocess:
            raw_audio, sr = sample['song_features'], sample['sample_rate']

            features = self.vggish(raw_audio, sr).squeeze()
        else:
            sound = self.__preprocess_sound(torch.tensor(sample['song_features']).unsqueeze(1)).cuda()
            features = self.vggish(sound).squeeze(1)
        
        
        sample['song_features'] = features.cpu().detach().numpy()
        return sample