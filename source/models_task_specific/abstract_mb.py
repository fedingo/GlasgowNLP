import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from ..models_base.mb_vggish import MusicBertVGGish
from ..models_base.rnn_vggish import RNNVGGish
from ..utils.generic_utils import generator_repeat, Timer

class AbstractMusicBertTraining(nn.Module):

    def __init__(self, name, dim_model=768, RNN= False, initialize_bert=True, **kwargs):
        super().__init__()
        
        if initialize_bert:
            if RNN:
                self.BERT = RNNVGGish(dim_model=dim_model, name=name,**kwargs)
            else:
                self.BERT = MusicBertVGGish(dim_model=dim_model, name=name,**kwargs)
        else:
            self.BERT = None
        
        self.register_buffer("loss_curve", torch.tensor([]))
        self.register_buffer("validation_curve", torch.tensor([]))
        
        self.optimizer = None
        
    
    def init_optim(self):
        lr = 5e-3 # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.98)
    
        
    def get_device(self):
        return next(self.parameters()).device
    
    
    def load_pretrained(self):
        self.BERT.load_pretrained()
    
    
    def save_model(self, path=None):
        if self.BERT is not None:
            self.BERT.save_model(path)
        
    
    def load_model(self, path=None):
        self.BERT.load_model(path)

        
    def forward(self, src_sound):
        
        output = self.BERT(src_sound)

        return output
    
    @staticmethod
    def append(curve, loss):
        curve = torch.cat([curve, loss.unsqueeze(0)])
        return curve

    
    def training_loss(self, batch):
        
        # abstract function that defines the loss for the training
        
        raise NotImplementedError
        
    def evaluate_fn(self, batch):
        
        # abstract function that defines the Evaluation function for the training
        
        raise NotImplementedError
    
    
    def train_model(self, train_dataloader, val_dataloader, epochs, eval_per_epoch=1):

        if self.optimizer is None:
            self.init_optim()
        
        self.train()
        
        counter_template = "({:d}/{:d} Epoch)"
        try:
            train_length = len(train_dataloader)
        except:
            #Iterable Dataset
            train_length = 0
            counter_template = "({:d} steps of {:d} epochs)"
            
        save_interval = eval_interval = train_length//eval_per_epoch or 1000

        pbar = tqdm(generator_repeat(train_dataloader, epochs),
                    total = train_length*epochs or None, #None in case of Iterable Dataset
                    desc=("Train "+counter_template+" - Loss...").format(0, epochs))
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad() # Reset Grad

            loss = self.training_loss(batch)
                                                         
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            self.loss_curve = self.append(self.loss_curve, loss.detach())
            
            if i%10 == 0:
                n_epoch = i//train_length if train_length!= 0 else i
                display_loss = torch.mean(self.loss_curve[-100:])
                pbar.set_description(("Train "+counter_template+" - Loss {:.6f}")\
                                     .format(n_epoch+1, epochs, display_loss))
                
            if i%eval_interval == -1%eval_interval and val_dataloader is not None:
                tqdm.write("Evaluating...                             ", end="\r")
                eval_loss = self.evaluate(val_dataloader)
                self.validation_curve = self.append(self.validation_curve, eval_loss)
                tqdm.write('Eval Loss ({:d} steps) {:.4f}         '.format(i, eval_loss), end="\r")

            if i%eval_interval == -1%eval_interval:
                self.scheduler.step()
                
            if i%save_interval == -1%save_interval:
                self.save_model()

        
    def evaluate(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            
            scores = torch.tensor([]).to(self.get_device())
            with Timer() as t:
                for i, batch in enumerate(val_dataloader):
                    print("{:d} steps, Elapsed time {:d} s                  ".format(i, int(t())), end="\r")   
                    score = self.evaluate_fn(batch)
                    
                    scores = self.append(scores, score.detach())
        self.train()      
        return torch.mean(scores)