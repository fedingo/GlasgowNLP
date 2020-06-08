import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

class ClassificationLoss(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_class, multi_label=False):
        super().__init__()
        
        self.multi_label = multi_label
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        
        # Allows for multi-label classification using a different loss
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss() # Implicitly applies Sigmoid on the output
        else:
            self.criterion = nn.CrossEntropyLoss() # Implicitly applies Softmax on the output
    
    def forward(self, seq_vector, target):
        
        h = F.relu(self.fc1(seq_vector))
        output = self.fc2(h)
        
        loss = self.criterion(output, target)
        
        return loss
    
    def accuracy(self, seq_vector, y_true):
        
        h = F.relu(self.fc1(seq_vector))
        scores = self.fc2(h)
        
        if not self.multi_label:
            y_pred = torch.argmax(scores, dim=-1)
        else:
            y_pred = (scores > 0.5).int()
            
        return torch.tensor(accuracy_score(y_true.cpu(), y_pred.cpu())).cuda()