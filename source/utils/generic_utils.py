import numpy as np
import torch
import time
from itertools import zip_longest
from IPython.display import Audio, display

def allDone():
      display(Audio(url='https://freesound.org/people/InspectorJ/sounds/415510/download/415510__inspectorj__bell-counter-a.wav', autoplay=True))

#Class that iterates over the iterable x for N times
def generator_repeat(x, N):
    for i in range(N):
        for el in x:
            yield el

    
def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


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


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

