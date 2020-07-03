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
        return 0
    
    return tp/(tp+fn)

class ClassificationLoss(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_class, multi_label=False):
        super().__init__()
        
        self.multi_label = multi_label
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        
        # Allows for multi-label classification using a different loss
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(n_class)*n_class) # Implicitly applies Sigmoid on the output
        else:
            self.criterion = nn.CrossEntropyLoss() # Implicitly applies Softmax on the output
    
    def forward(self, seq_vector, target):
        
        output = self.predict(seq_vector)
        
        if self.multi_label:
            target = target.float()
            
        loss = self.criterion(output, target)
        
        return loss
    
    def predict(self, seq_vector):
        h = F.relu(self.fc1(seq_vector))
        output = self.fc2(h)
        return output
    
    def accuracy(self, seq_vector, y_true):
        
        h = F.relu(self.fc1(seq_vector))
        scores = self.fc2(h)
        
        if not self.multi_label:
            y_pred = torch.argmax(scores, dim=-1)
            return torch_accuracy_score(y_true, y_pred)
        else:
            y_pred = (scores > 0.5).int()
            return torch_recall_score(y_true, y_pred)
        
        
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
    
    def evaluate(self, seq_vector, target):
        
        output = self.predict(seq_vector)
        return torch.Tensor(r2_score(target.cpu(), output.cpu(), multioutput='raw_values')).cuda()