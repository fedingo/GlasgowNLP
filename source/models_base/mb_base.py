## Script defining the MusicBert model

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.nn.modules import LayerNorm
from transformers import BertModel, BertTokenizer
from torch.nn import TransformerEncoderLayer
from .transformers.transformers_encoder import TransformerEncoder
from .bert_pretrain_convert.state_dict_translate import translate_from_hugginface_to_torch_BERT


class PositionalEncodings(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        
        self.embedding_layer_norm = LayerNorm(d_model, eps=1e-12)
        
    def get_device(self):
        return next(self.parameters()).device
        
    def forward(self, x, token_type):
        
        input_shape = x.shape[:-1]
        seq_length = input_shape[0]
        if seq_length > 512:
            print(x.shape)
        
        assert seq_length <= 512, "Input shape is longer than the maximum allowed (512)"

        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.get_device())
        position_ids = position_ids.unsqueeze(1).expand(input_shape)

        if token_type == "sound":
            token_type_ids = torch.ones(input_shape, dtype=torch.long, device=self.get_device())
        elif token_type == "text":
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.get_device())
        else:
            raise Exception("Invalid Token Type")
        
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        

        embeddings = x + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        #print(torch.std(embeddings))
        
        return embeddings
    
    
class MusicBertBase(torch.nn.Module):

    def __init__(self, num_tokens=30522, dim_model=768, nhead=12, name="music_bert",\
                 num_encoder_layers=12, d_feed=3072, dropout=0.1):
        super().__init__()

        self.d_model = dim_model
        self.num_tokens = num_tokens
        
        self.PATH = "models/" + name + ".pth"
        
        # Token Embedder
        self.embedder = nn.Embedding(num_tokens, dim_model)
             
        self.position_embeddings = PositionalEncodings(dim_model, dropout)

        encoder_layer = TransformerEncoderLayer(dim_model, nhead, \
                                                dim_feedforward=d_feed, \
                                                dropout=dropout,
                                                activation='gelu')
        
        device = self.get_device()
        
        # Registers IDs in the buffers memory
        self.register_buffer("CLS_ID", torch.tensor([[101]]).to(device).long())
        self.register_buffer("SEP_ID", torch.tensor([[102]]).to(device).long())
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        
        
    def get_device(self):
        return next(self.parameters()).device
        
        
    def load_pretrained(self):
        #load pretrained model from Hugginface
        
        device = self.get_device()
        bertModel = BertModel.from_pretrained("bert-base-uncased").to(device)
        
        bert_state_dict = translate_from_hugginface_to_torch_BERT(bertModel.state_dict())
        self.load_state_dict(bert_state_dict, strict=False)
        
        self.__test_integrity()
        
        
    def __test_integrity(self):
        
        self.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERTmodel = BertModel.from_pretrained('bert-base-uncased').to(self.get_device())
        
        input_ids = torch.tensor(tokenizer.encode("this is a test")).unsqueeze(0).to(self.get_device())
        BERT_outputs = BERTmodel(input_ids)[0]
        my_outputs = self(src_tokens=input_ids.permute(1,0), add_special_tokens=False).permute(1,0,2)
        
        outputs_mean = torch.mean(my_outputs - BERT_outputs)
        outputs_norm = torch.norm(my_outputs - BERT_outputs)
        
        print("== INTEGRITY TEST ==")
        print(f"Mean error: {outputs_mean}")
        print(f"Norm error: {outputs_norm}")
        self.train()
        
    
    def save_model(self, path=None):
        path = path if path is not None else self.PATH
        torch.save(self.state_dict(), path)
        
    
    def load_model(self, path=None):
        path = path if path is not None else self.PATH
        self.load_state_dict(torch.load(path))
        
    
    def forward(self,
                src_sound=None,
                src_tokens=None,
                src_sound2=None,
                add_special_tokens=True,
                return_layers_out=False):
        
        assert src_tokens is not None or src_sound is not None, \
            "Feed at least one sound or token sequence"
        
        # SHAPES: sound  Seq_len, Batch, Feat
        #         tokens Seq_len, Batch
        
        # EXTENSION: allows to take a second sound sequence
        
        if src_sound is not None:
            batch_size = src_sound.shape[1]
        else:
            batch_size = src_tokens.shape[1]
        
        if src_tokens is not None:
            src_tokens = self.embedder(src_tokens.long())
        
        def emb_or_empty(src, token_type):
            if src is not None:
                return self.position_embeddings(src, token_type=token_type)
            else:
                return torch.Tensor([]).to(self.get_device()) #Empty Placeholder
        
        src_sound  = emb_or_empty(src_sound, "sound")
        src_sound2 = emb_or_empty(src_sound2, "sound")
        src_tokens = emb_or_empty(src_tokens, "text")
        
            
        if add_special_tokens:            
            CLS, SEP = self.embedder(torch.cat([self.CLS_ID, self.SEP_ID]))
            CLS = CLS.repeat(1, batch_size, 1)
            SEP = SEP.repeat(1, batch_size, 1)

            sequence = torch.cat([CLS, src_sound, SEP, src_sound2, SEP, src_tokens], 0)
        else:
            sequence = torch.cat([src_sound, src_sound2, src_tokens], 0)
                
        output = self.encoder(sequence, return_layers_out=return_layers_out)
        #output = nn.Sigmoid()(output)
       
        return output