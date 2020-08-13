import numpy as np
import torch
import csv, lmdb

from .base_dataset import CacheDataset


class GTZANFastDataset(CacheDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.db = []
        
        with open("data/GTZAN/filename_to_genre.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            
            for row in reader:
                self.db.append({
                    "filename": row[0],
                    "genre": row[1]
                })
                
        self.song_env = lmdb.open("data/GTZAN/filename_to_log_mel_spectrograms.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)
            
        self.genres = np.array(list(set([track["genre"] for track in self.db])))
        
        self.length = len(self.db)
        self.set_tags("classification")
        
        
    def get_num_classes(self):
        return len(self.genres)

    def __create_sample__(self, idx):
        item = self.db[idx]
        sample = {}
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(item["filename"].encode())
            
        assert data is not None, "Key Error in database"
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        
        encoded_genre, = np.where(self.genres == item["genre"])
        sample['song_features'] = song_features[:3000]
        sample['encoded_class'] = encoded_genre
                    
        return sample
    
    
class EmoMusicFastDataset(CacheDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.db = []
        
        with open("data/emoMusic/id_to_emotion.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            
            for row in reader:
                self.db.append({
                    "song_id": row[0],
                    "emo_values": row[1::2]
                })
                
        self.song_env = lmdb.open("data/emoMusic/filename_to_log_mel_spectrograms.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)
                    
        self.length = len(self.db)
        self.set_tags("regression")


    def __create_sample__(self, idx):
        item = self.db[idx]
        sample = {}
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(item["song_id"].encode())
            
        assert data is not None, "Key Error in database"
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        
        sample['song_features'] = song_features[:3000]
        sample['target'] = np.array(item['emo_values']).astype(np.float)
        
        return sample
    
    
class MusicSpeechFastDataset(CacheDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.db = []
        
        with open("data/music_speech/filename_to_class.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            
            for row in reader:
                self.db.append({
                    "filename": row[0],
                    "class": row[1]
                })
                
        self.song_env = lmdb.open("data/music_speech/filename_to_log_mel_spectrograms.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)
            
        self.classes = np.array(list(set([track["class"] for track in self.db])))
        
        self.length = len(self.db)
        self.set_tags("classification")

    def __create_sample__(self, idx):
        item = self.db[idx]
        sample = {}
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(item["filename"].encode())
            
        assert data is not None, "Key Error in database"
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        
        encoded_genre, = np.where(self.classes == item["class"])
        sample['song_features'] = song_features
        sample['encoded_class'] = encoded_genre
                    
        return sample
    
    
class DeezerFastDataset(CacheDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.db = []
        
        with open("data/deezer/msd_id_to_emotion.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            
            for row in reader:
                self.db.append({
                    "msd_id": row[0],
                    "emo_values": row[1:]
                })
                
        self.song_env = lmdb.open("data/MSD/MSD_ID_to_log_mel_spectrogram.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)
                    
        self.length = len(self.db)
        self.set_tags("regression")


    def __create_sample__(self, idx):
        item = self.db[idx]
        sample = {}
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(item["msd_id"][0:17].encode())
            
        assert data is not None, "Key Error in database"
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        
        sample['song_features'] = song_features[:3000]
        sample['target'] = np.array(item['emo_values']).astype(np.float)
        
        return sample
    

class MTATFastDataset(CacheDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.db = []
        
        invalids = ["35644",
                    "57881",
                    "55753"]
        
        self.top50tags_indeces = [0, 7, 10, 13, 14, 18, 27, 28, 34, 35, 39, 42, 45, 46,
                             47, 55, 59, 63, 67, 68, 70, 71, 78, 80, 81, 82, 84, 90,
                             91, 97, 115, 120, 123, 125, 132, 143, 144, 146, 147, 151,
                             153, 154, 159, 160, 165, 169, 173, 175, 177, 180]
        
        with open("data/MTAT/annotations_final.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader, None)
            
            for row in reader:
                if row[0] not in invalids:
                    
                    classes = []
                    for tag_index in self.top50tags_indeces:
                        classes.append(row[tag_index+1])
                    
                    self.db.append({
                        "clip_id": row[0],
                        "classes": classes
                    })
                
        self.song_env = lmdb.open("data/MTAT/clip_id_to_log_mel_spectrogram.lmdb", 
                                  subdir=False, readonly=True, lock=False,
                                  readahead=False, meminit=False, max_readers=1)
                    
        self.length = len(self.db)
        self.set_tags(["classification", "multi_label"])

        
    def get_num_classes(self):
        return len(self.top50tags_indeces)

    def __create_sample__(self, idx):
        item = self.db[idx]
        sample = {}
        
        with self.song_env.begin(write=False) as txn:
            data = txn.get(item["clip_id"].encode())
            
        assert data is not None, "Key Error in database for key %s" % item['clip_id']
        
        song_features = np.reshape(np.frombuffer(data), (-1, 64))
        
        sample['song_features'] = song_features[:3000]
        sample['encoded_class'] = np.array(item['classes']).astype(np.int)
        
        return sample