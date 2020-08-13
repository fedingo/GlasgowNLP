import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score

def torch_accuracy_score(output, target):
    
    return torch.mean((output == target).float())

def torch_recall_score(output, target):
    
    combo = 2*target + output
    tp = torch.sum((combo == 3).float())
    fn = torch.sum((combo == 2).float())
    
    if tp + fn == 0:
        return tp # returns tensor with value 0
    
    return tp/(tp+fn)


class ClassificationLoss(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_class, multi_label=False):
        super().__init__()
        
        self.multi_label = multi_label
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        
        # Allows for multi-label classification using a different loss
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss() # Implicitly applies Sigmoid on the output
            # pos_weight=torch.ones(n_class)*n_class
        else:
            self.criterion = nn.CrossEntropyLoss() # Implicitly applies Softmax on the output
    
    def forward(self, vector, target):
        
        h = F.relu(self.fc1(vector))
        output = self.fc2(h)
        
        if self.multi_label:
            target = target.float()
            
        loss = self.criterion(output, target)
        
        return loss
    
    def predict(self, vector):
        h = F.relu(self.fc1(vector))
        output = self.fc2(h)
        
        if not self.multi_label:
            return F.softmax(output, dim=-1)
        else:
            return torch.sigmoid(output)
    
    def score(self, vector, y_true):
        
        scores = self.predict(vector)
        
        if not self.multi_label:
            y_pred = torch.argmax(scores, dim=-1)
            return torch_accuracy_score(y_true, y_pred)
        else:
            y_pred = (scores > 0.5).int()
            return torch_recall_score(y_true, y_pred)

        
def torch_r2_score(target, prediction):
    
    mean = torch.mean(target, axis=0)
    
    RES = torch.sum((target-prediction)**2, axis=0)
    TOT = torch.sum((target-mean)**2, axis=0)
    
    return torch.mean(1-RES/TOT)

        
class MSELoss(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
        self.criterion = nn.MSELoss()
        
    def forward(self, seq_vector, target):
        
        output = self.predict(seq_vector)
            
        loss = self.criterion(output, target)
        
        return loss
    
    def predict(self, seq_vector):
        h = F.relu(self.fc1(seq_vector))
        output = self.fc2(h)
        return output
    
    def evaluate(self, target, prediction):
        
        return torch_r2_score(target, prediction)