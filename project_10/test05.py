import torch.nn as nn
class Enconder(nn.Module):
    def __init__(self):
        super(Enconder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )#N,128,14,14
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,512,3,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )#N,512,7,7
        self.linear= nn.Sequential(
            nn.Linear(512*7*7,128),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y2 = y2.reshape(-1,512*7*7)
        y3 = self.linear(y2)
        return y3