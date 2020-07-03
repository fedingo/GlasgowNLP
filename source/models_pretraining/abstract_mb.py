import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from ..models_base.mb_vggish import MusicBertVGGish
from ..utils.generic_utils import generator_repeat

from .mlm_losses import *

class AbstractMusicBertPretraining(nn.Module):

    def __init__(self, name, dim_model=768, **kwargs):
        super().__init__()
        
        self.BERT = MusicBertVGGish(dim_model=dim_model, name=name,**kwargs)
                
            
        # ACCIPICCHIA QUI NON CI VA LA DEFINIZIONE DELL'OPTIMIZER
        lr = 1e-3 # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        
        device = self.get_device()
        self.register_buffer("loss_curve", torch.tensor([]).to(device))
        self.register_buffer("validation_curve", torch.tensor([]).to(device))
        self.register_buffer("cls_loss_curve", torch.tensor([]).to(device))
        self.register_buffer("mask_loss_curve", torch.tensor([]).to(device))

        
    def get_device(self):
        return next(self.parameters()).device
    
    
    def load_pretrained(self):
        self.BERT.load_pretrained()
    
    
    def save_model(self, path=None):
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

    
    def pretraining_loss(self, batch):
        
        # abstract function that defines the loss for the pretraining
        
        raise NotImplementedError
    
    
    def pretrain_model(self, pretrain_dataloader, val_dataloader, epochs, eval_per_epoch=1):

        self.train()
        pretrain_length = len(pretrain_dataloader)
        save_interval = eval_interval = pretrain_length//eval_per_epoch

        pbar = tqdm(generator_repeat(pretrain_dataloader, epochs),
                    total = pretrain_length*epochs,
                    desc="Train ({:d}/{:d} Epoch) - Loss...".format(1, epochs))
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad() # Reset Grad

            loss = self.pretraining_loss(batch)
                                                         
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            self.loss_curve = self.append(self.loss_curve, loss.detach())
            
            if i%10 == 0:
                n_epoch = i//pretrain_length
                display_loss = torch.mean(self.loss_curve[-100:])
                pbar.set_description('Pre-train ({:d}/{:d} Epoch) - Loss {:.6f}'\
                                     .format(n_epoch+1, epochs, display_loss))
                
            if i%eval_interval == -1%eval_interval and val_dataloader is not None:
                tqdm.write("Evaluating...                             ", end="\r")
                eval_loss = self.evaluate(val_dataloader)
                self.validation_curve = self.append(self.validation_curve, eval_loss)
                tqdm.write('Eval Loss ({:d} steps) {:.4f}'.format(i, eval_loss), end="\r")

            if i%save_interval == -1%save_interval:
                self.save_model()

        
    def evaluate(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            
            losses = torch.tensor([]).to(self.get_device())
            with Timer() as t:
                for i, batch in enumerate(val_dataloader):
                    print("{:d}%, Elapsed time {:d} s                  ".format(i*100//len(val_dataloader), int(t())), end="\r")   
                    loss = self.pretraining_loss(batch)
                    
                    losses = self.append(losses, loss.detach())
        self.train()      
        return torch.mean(losses)