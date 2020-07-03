import numpy as np
import lmdb
import csv
from torch.utils.data import Dataset
from multiprocessing import Manager
import torch


# Dataset specific for Log Mel Spectrograms
class MSDDatasetPretrain(Dataset):
    """Million Songs dataset."""

    def __init__(self, length=None):
        
        self.song_env = lmdb.open("data/MSD/MSD_ID_to_log_mel_spectrogram.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)
        
        mumu_dir = '../datasets/mumu/MuMu_dataset/'
        
        self.msd_ids_list = []

        with open(mumu_dir + '/MuMu_dataset_multi-label.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] == "amazon_id":
                    continue
            
                self.msd_ids_list.append(row[2])

        if length is None:
            self.length = len(self.msd_ids_list)
        else:
            self.length = length
            
        self.pseudo_labels_dict = {}
                
    def __len__(self):
        return self.length
    
    def get_features_size(self, log=False):
        return 64
    
    def set_pseudo_label(self, idx, pseudo_label):
        self.pseudo_labels_dict[idx] = pseudo_label

    def __getitem__(self, idx):
        
        if idx >= len(self):
            raise IndexError
        
        msd_id = self.msd_ids_list[idx][0:17] #Cutoff to comply with moodagent modified IDs
        msd_id = msd_id.encode()
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(msd_id)
            
        assert data is not None, "Key Error in database"
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        song_features = torch.Tensor(song_features[:3000,:])
        sample = {'song_features': song_features}
        
        if self.pseudo_labels_dict.get(idx) is not None:
            sample['encoded_class'] = self.pseudo_labels_dict.get(idx)
            sample['target'] = self.pseudo_labels_dict.get(idx)

        return sample
    
    
class MSDDatasetGenre(Dataset):
    """Million Songs dataset for Music Genre Classification."""

    def __init__(self, length=None):
        
        self.song_env = lmdb.open("data/MSD/MSD_ID_to_log_mel_spectrogram.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)

        self.msd_ids_list = []
        
        with open('data/msd_genre/msd_id_to_genre.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] == "msd_id":
                    continue
                    
                item = {
                    'track_id' : row[0].strip(),
                    'genre' : row[1].strip()
                }
                    
                self.msd_ids_list.append(item)
                
        print(len(self.msd_ids_list))

        if length is None:
            self.length = len(self.msd_ids_list)
        else:
            self.length = length
            
        self.genres = np.array(list(set([track["genre"] for track in self.msd_ids_list])))
                
    def __len__(self):
        return self.length
    
    def get_features_size(self, log=False):
        return 64

    def __getitem__(self, idx):
        
        item = self.msd_ids_list[idx]
        
        #Cutoff to comply with moodagent modified IDs
        msd_id = item['track_id'][0:17].encode()
        genre = item['genre']
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(msd_id)
            
        assert data is not None, "Key Error in database"
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        encoded_genre, = np.where(self.genres == item["genre"])
        
        sample = {
            'song_features': song_features[:3000,:],
            'encoded_class': encoded_genre
        }

        return sample