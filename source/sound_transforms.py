import os
import librosa
import numpy as np
from scipy import signal
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .vggish import vggish_input
from .vggish import vggish_params
from .vggish import vggish_postprocess
from .vggish import vggish_slim

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
    
    
class toVggish(object):
    
    def __init__(self):
        
        self.pproc = vggish_postprocess.Postprocessor('source/vggish/vggish_pca_params.npz')
        
        self.session = tf.Session()
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(self.session, 'source/vggish/vggish_model.ckpt')
        self.features_tensor = self.session.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        self.embedding_tensor = self.session.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            
    
    def __call__(self, sample):
        raw_audio, sr = sample['song_features'], sample['sample_rate']
        example = vggish_input.waveform_to_examples(raw_audio, sr)

        [embedding_batch] = self.session.run([self.embedding_tensor],
                                             feed_dict={self.features_tensor: example})
        postprocessed_batch = self.pproc.postprocess(embedding_batch)
                
        sample['song_features'] = postprocessed_batch #np.transpose(postprocessed_batch)
        return sample