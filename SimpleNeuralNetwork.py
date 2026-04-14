import torch
import torch.nn as nn
import torch.nn.functional as F


class SNN(nn.Module):
    def __init__(self,input_dim = 784 ,hidden_dim = 200 ,output_dim = 10):
        super(SNN,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        return

    def forward(self,x):
        x = self.fc1(x)

        x = F.selu(x)

        x = self.fc2(x)

        return x
    
