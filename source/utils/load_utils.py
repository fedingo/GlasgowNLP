from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import torch


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