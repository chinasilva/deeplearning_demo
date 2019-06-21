import torch
import torch.nn as nn
import torch.nn.functional as F

# 
class MyMnistNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model= nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=96,kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=96),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=246,out_channels=15,kernel_size=3, padding=1,groups=3),
            nn.BatchNorm1d(num_features=15),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(1,30)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(30,1)       

    def forward(self, input):
        y = self.model(input)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return y