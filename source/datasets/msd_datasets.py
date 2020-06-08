import numpy as np
import json
import torch
import lmdb
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from .utils import *
#from .vocab import Vocabulary
from .io.pb_item import pb_Item

TOKEN_SEQ_MAX_LEN = 64
SOUND_SEQ_MAX_LEN = 384

class MSDDatasetPretrain(Dataset):
    """Million Songs dataset."""

    def __init__(self, track_json_path='json/all_tracks.json',\
                 root_dir='../msd/data/'):
        """
        Args:
            track_json (list): List of strings containing the track_ids that we want to process 
            root_dir (string): Directory with all the music files.
                
        Output:
            Dictionary {'song_features': ..., 'encoded_genre': ...}
        """
        
        with open(track_json_path) as json_file:
            track_json = json.load(json_file)
            
        self.track_json = track_json
        self.root_dir = root_dir

    def __len__(self):
        return len(self.track_json)
    
    def __get_internal_item__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.track_json[idx]

    def __getitem__(self, idx):
        item = self.__get_internal_item__(idx)
        
        path = get_path_from_id(item['track_id'], self.root_dir)
        song_features = get_features(path).astype(np.float32)
        #song_features = reshape_features(song_features)
        
        sample = {'song_features': song_features}

        return sample




class MSDDatasetGenre(MSDDatasetPretrain):
    """Million Songs dataset for Music Genre Classification."""

    def __init__(self, track_json_path='json/balanced_track_with_genre.json',\
                 root_dir='../msd/data/'):
        super().__init__(track_json_path)
        
        self.GENRES = np.array(['hip-hop', 'soul and reggae', 'classical', 'metal', 'folk',
                   'jazz and blues', 'punk', 'classic pop and rock',
                   'dance and electronica', 'pop'])

    def __getitem__(self, idx):
        item = self.__get_internal_item__(idx)
        sample = super().__getitem__(idx)
        
        # Adds the Genre field in the dataset items 
        encoded_genre, = np.where(self.GENRES == item['genre'])
        sample['encoded_class'] = encoded_genre

        return sample
    

class MSDDatasetReviewsEchonest(Dataset):
    """Million Songs dataset."""

    def __init__(self, path='data/MSD/', tokenizer=None, length=None):
        """
        Args:
            track_json (list): List of strings containing the track_ids that we want to process 
            root_dir (string): Directory with all the music files.
                
        Output:
            Dictionary {'song_features': ..., 'encoded_genre': ...}
        """
        
        self.review_env = lmdb.open(path+"MSD_ID_to_review.lmdb", readonly=True, lock=False, 
                                    max_spare_txns=2, subdir=False, readahead=False, meminit=False)
        self.song_env = lmdb.open(path+"MSD_ID_to_Echonest.lmdb", readonly=True, lock=False, 
                                    max_spare_txns=2, subdir=False, readahead=False, meminit=False)
        
        if tokenizer is None:
            self.tokenizer = Vocabulary([], load_path="data/vocab.json")
        else:
            self.tokenizer = tokenizer
        
        if length is None:
            with self.review_env.begin() as txn:
                self.length = txn.stat()['entries']
        else:
            self.length = length
                
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        with self.review_env.begin() as txn:
            str_id = '{:08}'.format(idx).encode("ascii")
            raw_datum = txn.get(str_id)
        review_obj = json.loads(raw_datum)
        
        with self.song_env.begin() as txn:
            raw_song = txn.get(review_obj['msd_id'].encode())
        song_features = json.loads(raw_song)
        
        song_features = np.array(song_features).astype(np.float32) 
        #song_features = reshape_features(song_features)
                
        sample = {
            'cls': [1],
            'song_features': song_features,
            'full_text' : self.tokenizer.encode(review_obj['review_text'],
                                                add_special_tokens=False,
                                                max_length=TOKEN_SEQ_MAX_LEN),
            'summary' : self.tokenizer.encode(review_obj['summary'],
                                              add_special_tokens=False,
                                              max_length=TOKEN_SEQ_MAX_LEN)
        }

        return sample

    
class MSDDatasetReviews(Dataset):
    """Million Songs dataset."""

    def __init__(self, path='data/MSD/', features_db="MSD_ID_to_MFCC.lmdb", tokenizer=None, length=None, protobuf=True):
        """
        Args:
            track_json (list): List of strings containing the track_ids that we want to process 
            root_dir (string): Directory with all the music files.
                
        Output:
            Dictionary {'song_features': ..., 'encoded_genre': ...}
        """
        
        self.protobuf = protobuf
        self.review_env = lmdb.open(path+"MSD_ID_to_review_with_CLS.lmdb", readonly=True, lock=False, 
                                    max_spare_txns=2, subdir=False, readahead=False, meminit=False)
        self.song_env = lmdb.open(path+features_db, readonly=True, lock=False, 
                                    max_spare_txns=2, subdir=False, readahead=False, meminit=False)
        
        if tokenizer is None:
            self.tokenizer = Vocabulary([], load_path="data/vocab.json")
        else:
            self.tokenizer = tokenizer
        
        if length is None:
            with self.review_env.begin() as txn:
                self.length = txn.stat()['entries']
        else:
            self.length = length
                
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        with self.review_env.begin() as txn:
            str_id = '{:08}'.format(idx).encode("ascii")
            raw_datum = txn.get(str_id)
        review_obj = json.loads(raw_datum)
        
        msd_id = review_obj['msd_id'][:-1]
        with self.song_env.begin() as txn:
            data = txn.get(msd_id.encode())
            
        if data is None:
            print(review_obj)
        
        if self.protobuf:
            pb_obj = pb_Item()
            pb_obj.load_from_string(data) 

            song_features =  np.transpose(pb_obj.features)
        else:
            song_features = np.reshape(np.frombuffer(data), (-1, 64))
                
        sample = {
            'cls': [review_obj["CLS"]],
            'song_features': song_features[:SOUND_SEQ_MAX_LEN, :],
            'full_text' : self.tokenizer.encode(review_obj['review_text'],
                                                add_special_tokens=False,
                                                max_length=TOKEN_SEQ_MAX_LEN),
            'summary' : self.tokenizer.encode(review_obj['summary'],
                                              add_special_tokens=False,
                                              max_length=TOKEN_SEQ_MAX_LEN)
        }

        return sample