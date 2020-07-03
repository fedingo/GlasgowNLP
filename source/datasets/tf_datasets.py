from os import listdir
from os.path import isfile, join
import numpy as np

from torch.utils.data import IterableDataset, DataLoader
import torch

import tensorflow as tf
from ..utils.load_utils import collate_function

class AudiosetConfig:
    
    @staticmethod
    def get_embeddings(tf_example):

        res = []
        for bytes_el in tf_example.feature_lists.feature_list['audio_embedding'].feature:
            hexembed = bytes_el.bytes_list.value[0].hex()
            arrayembed = [int(hexembed[i:i+2],16) for i in range(0,len(hexembed),2)]
            res.append(arrayembed)

        return np.array(res)

    @staticmethod
    def get_labels(tf_example):
        labels = tf_example.context.feature['labels'].int64_list.value
        
        res = np.zeros(527, dtype=np.long)
        res[labels] = 1
        return res

    
class HiveTFIterator:
    
    def __init__(self, filenames, config):
    
        self.config = config
        dataset = tf.data.TFRecordDataset(filenames)
        self.iterator = iter(dataset)
            
    def build_item(self, raw_record):
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        
        item = {
            'song_features': self.config.get_embeddings(example),
            'encoded_class': self.config.get_labels(example)
        }
        return item
      
    def __iter__(self):
        return self
    
    
    def __next__(self):
        raw_record = next(self.iterator)
        return self.build_item(raw_record)
        
        
        
class HiveTFDataset(IterableDataset):
    
    def __init__(self, dir_path, num_workers=0, config=AudiosetConfig()):
        super().__init__()
        
        filenames = [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]
              
        self.num_workers = num_workers
        
        self.split = np.array_split(filenames, num_workers or 1) #if num_workers == 0 split with 1        
        self.config = config
    
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading 
            assert self.num_workers == 0, "Mismatch configuration between dataloader and dataset"
            
            return HiveTFIterator(self.split[0], self.config)
        
        else:
            assert worker_info.num_workers == self.num_workers, "Mismatch configuration between dataloader and dataset"
            
            return HiveTFIterator(self.split[worker_info.id], self.config)
    
    
    def get_dataloader(self, batch_size):
        
        return DataLoader(self, batch_size=batch_size, num_workers=self.num_workers, \
                          drop_last=True, collate_fn = collate_function)
