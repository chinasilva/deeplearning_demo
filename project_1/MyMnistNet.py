import torch
import torch.nn as nn
import torch.nn.functional as F

# 
class MyMnistNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1,30)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(30,20)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(20,5)
        self.relu = nn.ReLU()

        self.linear4 = nn.Linear(5,1)
       

    def forward(self, input):

        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        y = self.relu(y)
        y = self.linear4(y)
        return y