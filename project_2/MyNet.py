import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= nn.Sequential(
            nn.Linear(in_features=100*100*3,out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=4),
        )

    def forward(self,input):
        return self.model(input)