from .item_pb2 import Item, Matrix
import numpy as np

class pb_Item:
    
    def __init__(self,
                 features=None,
                 review=None,
                 seq_class=None,
                 seq_targets=None,
                 token_class=None):
        
        self.features = features
        self.review = review
        self.seq_class = seq_class
        self.seq_targets = seq_targets
        self.token_class = token_class
        
    
    def load_from_string(self, string):
        pb_obj = Item()
        pb_obj.ParseFromString(string)
        self.load_from_pb(pb_obj)
        
    
    def load_from_pb(self, pb_obj):
        assert self.features is None, "Item already initialized"
        
        matrix_obj = pb_obj.features
        features = np.array(matrix_obj.flat_matrix)
        self.features = np.reshape(features, (-1, matrix_obj.feature_size))
        
        self.review = pb_obj.item_review
        self.seq_class = pb_obj.item_class
        self.seq_targets = pb_obj.item_targets
        self.token_class = pb_obj.token_class
        
    
    def get_pb_obj(self):
        #assert self.features is not None, "Item not Initialized yet"
        
        pb_obj = Item()
        if self.review is not None:
            pb_obj.item_review = self.review
        
        if self.seq_class is not None:
            pb_obj.item_class = self.seq_class
            
        if self.seq_targets is not None:
            pb_obj.item_targets.extend(self.seq_targets)
        
        if self.token_class is not None:
            pb_obj.token_class.extend(self.token_class)
        
        pb_obj.features.feature_size = self.features.shape[-1]
        pb_obj.features.flat_matrix.extend(self.features.flatten().tolist())
        
        return pb_obj