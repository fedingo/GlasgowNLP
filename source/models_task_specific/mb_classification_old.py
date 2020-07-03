import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ..utils.generic_utils import generator_repeat
from tqdm.notebook import tqdm
from ..models_base.mb_vggish import MusicBertVGGish

## OLD ##

class MusicBertClassifier(nn.Module):

    def __init__(self, n_class, hidden_dims = 128, multi_label=False, **kwargs):
        # MB_BASE_CLASS in set (MusicBertConvs, MusicBertVggish)
        
        super().__init__()
        
        self.BERT = MusicBertVGGish(name = "music_bert_classification", **kwargs)
        
        self.fc1 = nn.Linear(768, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, n_class)
        
        # Optimizer init
        lr = 1e-3 # learning rate
        
        # Allows for multi-label classification using a different loss
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss() # Implicitly applies Sigmoid on the output
        else:
            self.criterion = nn.CrossEntropyLoss() # Implicitly applies Softmax on the output
            
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        
        # Training curves
        self.loss_curve = []
        self.validation_curve = []
        self.training_curve = []
        
        self.params = self.parameters()
        
        
    def get_device(self):
        return next(self.parameters()).device
      
    
    def save_model(self, path=None):
        self.BERT.save_model(path)
        
    
    def load_model(self, path=None):
        self.BERT.load_model(path)

        
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
            target = batch['encoded_class'].squeeze().long().to(self.get_device())

            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 0.5)
            self.optimizer.step()

            self.loss_curve.append(loss.item())
            current_loss = np.mean(self.loss_curve[-100:])
            n_epoch = i//train_length

            pbar.set_description('Train ({:d}/{:d} Epoch) - Loss {:.3f}'\
                                 .format(n_epoch+1, epochs, current_loss))

            if i%eval_interval == -1%eval_interval and val_dataloader is not None:
                tqdm.write("Evaluating...              ", end="\r")
                val_accuracy = self.evaluate(val_dataloader, eval_fn=eval_fn)
                train_accuracy = self.evaluate(train_dataloader, eval_fn=eval_fn)
                
                self.validation_curve.append(val_accuracy)
                self.training_curve.append(train_accuracy)
                tqdm.write('Accuracy ({:d} steps) {:.4f}'.format(i, val_accuracy), end="\r")
            
            if i%save_interval == -1%save_interval:
                self.save_model()



    def evaluate(self, val_dataloader, eval_fn=None):
        self.eval() # Turn on the evaluation mode
        with torch.no_grad():
            target_vector = []
            output_vector = []
            

            for batch in val_dataloader:

                output = self.forward(batch['song_features'].permute(1,0,2).to(self.get_device()))
                    
                output_vector.append(output)
                target_vector.append(batch['encoded_class'])
               
            
            outputs = torch.cat(output_vector, axis=0).cpu()
            targets = torch.cat(target_vector, axis=0)    
                  
            self.train()
                
            if eval_fn is not None:
                return eval_fn(outputs, targets)
            else:
                # Defaults on Accuracy
                score_vector = []
                for out, tar in zip(outputs, targets):
                    dist = np.array(out == torch.max(out)).astype(np.int) #Binarization to predicted class
                    score_vector.append(dist[tar.squeeze().long()])
                                                        
                return np.mean(score_vector) 
                    