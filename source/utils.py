import numpy as np
import os
import h5py
import string
import torch
import torchtext
import time

from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence
from itertools import zip_longest
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib

#Class that iterates over the iterable x for N times
def generator_repeat(x, N):
    for i in range(N):
        for el in x:
            yield el

def plot_curve(curve, eval_per_epoch, color = "red", metric_label="Accuracy"):
    
    x = np.array(range(len(curve)))//eval_per_epoch
    plt.plot(x, curve, color)
    plt.ylabel(metric_label)
    plt.xlabel('Epochs')
    plt.show()
    
    eval_range = int(3*eval_per_epoch)
    final_score = np.mean(curve[-eval_range:])*100
    print("Model {}: {:.2f}%".format(metric_label, final_score) )


#Timer class to be use to time Scripts
class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __call__(self):
        return time.time() - self.start
    
    def __str__(self):
        return str(self())
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def torch_count_nonzero(x, dim=-1):
    return (x != 0).sum(dim=dim)

def reshape_features(features, multiplier = 16, max_len=512):
    
    length, features_size = features.shape
    
    # makes the vector size a multiple of the multiplier
    length -= length%multiplier
    length = length if length<max_len else max_len
    
    features = features[:length]
    features = features.reshape(length//multiplier, features_size*multiplier)

    return features


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_path_from_id(ID, base):
    #Folder path
    for c in ID[2:5]:
        base += c + '/'
        
    #Filename
    base += ID + ".h5"
    
    #Sanity Check
    if not os.path.isfile(base):
        print('ERROR:',base,'does not exist.')
        return None
    
    return base


# Function that creates a relative path from the MSD_ID
def from_id_to_path(msd_id, extension = "mp3"):
    
    path = ""  
    for c in msd_id[2:5]:
        path += "/" + c    
    path += "/" + msd_id + "." + extension
    
    return path


# Fix Max Size of the Input
def get_features(path, max_length = 512):
    song_file = h5py.File(path, 'r')
    timbre_vectors = np.array(song_file['analysis']['segments_timbre'])[:max_length]
    pitch_vectors = np.array(song_file['analysis']['segments_pitches'])[:max_length]
    
    features_vectors = np.concatenate([timbre_vectors, pitch_vectors], axis = -1)
    song_file.close()
            
    return features_vectors


def id_to_features(ID, base):
    path = get_path_from_id(ID, base)
    if path is None:
        song_features = None
    else:
        song_features = get_features(path).astype(np.float32)

    return song_features


def tokenize(text):
    # Function that tokenizes a string 
    
    #Strip all the punctuation
    text = text.translate(str.maketrans(string.punctuation,\
                                        ' '*len(string.punctuation)))
    
    text = "<sos> " + text + " <eos>"
    
    # Then split using the whitespaces
    tokens = text.lower().split()
    return tokens


def collate_function(batch):
    
    sample = {}
    for key in batch[0].keys():
        sample[key] = pad_sequence([torch.Tensor(item[key]) for item in batch],\
                                   batch_first=True)
    return sample
    
        
def split_and_load(dataset, split_size=0.95, workers = 4, batch_size=16):
    indices = list(range(len(dataset))); np.random.shuffle(indices)
    split = int(split_size*len(dataset))
    
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(dataset, batch_size=batch_size,\
                                  sampler=train_sampler, num_workers=workers,\
                                  drop_last=True, collate_fn = collate_function)
    val_dataloader = DataLoader(dataset, batch_size=batch_size,\
                                sampler=val_sampler, num_workers=workers,\
                                drop_last=True, collate_fn = collate_function)
        
    return train_dataloader, val_dataloader

def just_load(dataset, workers = 4, batch_size=16, shuffle=True):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,\
                            drop_last=True, collate_fn = collate_function, shuffle=shuffle)
    
    return dataloader