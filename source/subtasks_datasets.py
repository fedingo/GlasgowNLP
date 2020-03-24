import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from .base_dataset import CacheDataset
from .utils import *
import warnings
warnings.simplefilter("ignore")

def create_sample_from_path(path):
    x, sr = librosa.load(path, sr = None)
    sample = {'song_features': x,
              'sample_rate': sr}
    return sample


class ClassificationDatasetNumpy(CacheDataset):
    
    def __init__(self, source_file, length=None, **kwargs):
        super().__init__(**kwargs)

        with open(source_file, "rb")as f:
            self.track_list = np.load(f, allow_pickle=True)
            
        if length is None:
            self.length = len(self.track_list)
        else:
            self.length = length
            
        self.classes = np.array(list(set([track['class'] for track in self.track_list])))
        
        item = self.__getitem__(0)
        
        self.features_size = item['song_features'].shape[-1]
        self.n_classes = len(self.classes)

    def __len__(self):
        return self.length

    def __create_sample__(self, idx):
        item = self.track_list[idx]
        
        sample = create_sample_from_path(item['path'])
        encoded_class, = np.where(self.classes == item['class'])
        sample['encoded_class'] = encoded_class
                    
        return sample

    
class RegressionDatasetNumpy(CacheDataset):
    
    def __init__(self, source_file, length=None, **kwargs):
        super().__init__(**kwargs)

        with open(source_file, "rb")as f:
            self.track_list = np.load(f, allow_pickle=True)
            
        if length is None:
            self.length = len(self.track_list)
        else:
            self.length = length
                    
        item = self.__getitem__(0)
        
        self.features_size = item['song_features'].shape[-1]
        self.output_size = item['target'].shape[-1]

    def __len__(self):
        return self.length

    def __create_sample__(self, idx):
        item = self.track_list[idx]
        
        sample = create_sample_from_path(item['path'])
        sample['target'] = item['target']
                    
        return sample

class GTZANDataset(CacheDataset):
    
    def __init__(self, source_file="data/GTZAN/path.npy", **kwargs):
        super(GTZANDataset, self).__init__(**kwargs)

        with open(source_file, "rb")as f:
            self.track_list = np.load(f, allow_pickle=True)
            
        self.genres = np.array(list(set([track[1] for track in self.track_list])))

    def __len__(self):
        return len(self.track_list)

    def __create_sample__(self, idx):
        item = self.track_list[idx]
        
        sample = create_sample_from_path(item[0])
        encoded_genre, = np.where(self.genres == item[1])
        sample['encoded_class'] = encoded_genre
                    
        return sample

    
class EmoMusicDataset(CacheDataset):

    def __init__(self, source_folder="data/emoMusic/", datatype="train", **kwargs):
        super(EmoMusicDataset, self).__init__(**kwargs)
        
        filename = datatype + "_path.npy"
        with open(source_folder+filename, "rb")as f:
            self.track_list = np.load(f, allow_pickle=True)
            
    def __len__(self):
        return len(self.track_list)

    def __create_sample__(self, idx):
        item = self.track_list[idx]
        
        sample = create_sample_from_path(item['path'])
        target = item['emo_values']
        sample['target'] = [target[0], target[2]] # just take the average values for Arousal and Valence
        
        return sample

    
class MusicSpeechDataset(CacheDataset):

    def __init__(self, source_file="data/music_speech/path.npy", **kwargs):
        super(MusicSpeechDataset, self).__init__(**kwargs)
        
        with open(source_file, "rb") as f:
            self.track_list = np.load(f, allow_pickle=True)
            
        self.classes = np.array(list(set([track['class'] for track in self.track_list])))
            
    def __len__(self):
        return len(self.track_list)

    def __create_sample__(self, idx):
        item = self.track_list[idx]
        
        sample = create_sample_from_path(item['path'])
        encoded_class, = np.where(self.classes == item['class'])
        sample['encoded_class'] = encoded_class
        
        return sample