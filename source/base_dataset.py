
from multiprocessing import Manager
import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    
    
    def __init__(self, transform=None, use_cache=False):
        self.use_cache = use_cache
        self.cache = Manager().dict() #Shared Cache to allow for multiple workers
        self.transform = transform
        
    def __create_sample__(self, idx):
        raise NotImplementedError
        
    def get_features_size(self, log=False):
        item = self.__getitem__(0)
        if log:
            print("== Item shapes ==")
            for key, value in item.items():
                if hasattr(value, 'shape'):
                    print("%s: " % key, value.shape)
                else:
                     print("%s: " % key, value)
        return item['song_features'].shape[-1]
        
    def __getitem__(self, idx):
        
        if self.use_cache:
            if idx in self.cache:
                item = self.cache[idx]
            else:
                item = self.__create_sample__(idx)
                if self.transform:
                    item = self.transform(item)
                    
                self.cache[idx] = item
        else:
            item = self.__create_sample__(idx)
            if self.transform:
                item = self.transform(item)
            
        return item