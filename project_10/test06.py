import torch.nn as nn
class Deconder(nn.Module):
    def __init__(self):
        super(Deconder,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128,512*7*7),
            nn.BatchNorm1d(512*7*7),
            nn.ReLU(inplace=True)
        )#N,512,7,7
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(512,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )#N,128,14,14
        self.conv2= nn.Sequential(
            nn.ConvTranspose2d(128,1,3,2,1,1),
            nn.Sigmoid()
        )#N,1,28,28

    def forward(self, x):
        y1 = self.linear(x)
        y1 = y1.reshape(-1,512,7,7)
        y2 = self.conv1(y1)
        y3 = self.conv2(y2)
        return y3