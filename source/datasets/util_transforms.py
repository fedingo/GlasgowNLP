import numpy as np


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch
    

class BERT_Pooling(object):
    
    def __call__ (self, sample):
        sample_sound = sample['song_features']
        
        sample['song_features'] = sample_sound[0]
        
        return sample
    

class Random_Pooling(object):
    
    def __call__ (self, sample):
        sample_sound = sample['song_features']
        
        index = np.random.randint(len(sample_sound))
        
        sample['song_features'] = sample_sound[index]
        
        return sample
    

class Index_Pooling(object):
    
    def __init__(self, index):
        self.index = index
    
    def __call__ (self, sample):
        sample_sound = sample['song_features']
        
        if len(sample_sound) >= self.index:
            sample['song_features'] = sample_sound[-1]
        else:
            sample['song_features'] = sample_sound[self.index]
        
        return sample
    
    
class Average_Pooling(object):
    
    def __call__(self, sample):
        sample_sound = sample['song_features']
        
        sample['song_features'] = np.mean(sample_sound, axis = 0)
        return sample
    
    
class Max_Pooling(object):
    
    def __call__(self, sample):
        sample_sound = sample['song_features']
        
        sample['song_features'] = np.max(sample_sound, axis = 0)
        return sample