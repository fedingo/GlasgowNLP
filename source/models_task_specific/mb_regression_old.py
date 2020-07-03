import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .mb_convs import MusicBertConvs
from ..utils.generic_utils import generator_repeat
from tqdm.notebook import tqdm

class MusicBertRegressor(nn.Module):

    def __init__(self, out_dim, MB_BASE_CLASS
                 hidden_dims = 128, **kwargs):
        super().__init__()
        
        self.BERT = MB_BASE_CLASS(name = "music_bert_regression", **kwargs)
        
        self.fc1 = nn.Linear(768, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, out_dim)
        
        # Optimizer init
        lr = 1e-3 # learning rate
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        
        # Training curves
        self.loss_curve = []
        self.training_curve = []
        self.validation_curve = []
        
        self.params = self.parameters()
        
        
    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        # Shape of x: (Seq_length, Batch_size, Emb_size)
                
        x = self.BERT(x)
        x = x[0] # Pooling
        
        # Final FF layers to classify the song
        r = F.relu(self.fc1(x))
        r = self.fc2(r)
        
        return r
    
    def train_model(self, train_dataloader, val_dataloader, epochs, eval_per_epoch=4, eval_fn=None):
        
        self.train() # Turn on the train mode
        eval_interval = len(train_dataloader)//eval_per_epoch
        n_epoch= 0
        train_length = len(train_dataloader)
                
        pbar = tqdm(generator_repeat(train_dataloader, epochs),
                    total = train_length*epochs,
                    desc="Train ({:d}/{:d} Epoch) - Loss X.XXX".format(n_epoch+1, epochs))
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            output = self.forward(batch['song_features'].permute(1,0,2).to(self.get_device()))
            target = batch['target'].squeeze().to(self.get_device())

            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 0.5)
            self.optimizer.step()

            self.loss_curve.append(loss.item())
            current_loss = np.mean(self.loss_curve[-100:])
            n_epoch = i//train_length

            pbar.set_description('Train ({:d}/{:d} Epoch) - Loss {:.3f}'\
                                 .format(n_epoch+1, epochs, current_loss))

            if i%eval_interval == -1%eval_interval:
                val_metric = self.evaluate(val_dataloader, eval_fn)
                train_metric = self.evaluate(train_dataloader, eval_fn)
                
                self.validation_curve.append(val_metric)
                self.training_curve.append(train_metric)
                
                tqdm.write('{:d} steps: validation={}, training={}         '\
                           .format(i, str(val_metric), str(train_metric)), end="\r")
                    
        return self.validation_curve


    def evaluate(self, val_dataloader, eval_fn = None):
        self.eval() # Turn on the evaluation mode
        with torch.no_grad():
            output_vector = []
            target_vector = []
            
            score_vector = []
            for batch in val_dataloader:
                
                output = self.forward(batch['song_features'].permute(1,0,2).to(self.get_device()))
                target = batch['target'].squeeze()
                
                output_vector.append(output)
                target_vector.append(target)
                
            outputs = torch.cat(output_vector, axis=0)
            targets = torch.cat(target_vector, axis=0)
        
        self.train()                 
        
        if eval_fn is not None:
            return eval_fn(outputs.cpu(), targets)
        else:
            return self.criterion(output, target.to(self.get_device())).item()