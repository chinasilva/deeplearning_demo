import torch
import torch.nn as nn

class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pNet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=5,out_channels=16,kernel_size=3),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=1),
        )
        self.classification=nn.Conv2d(in_channels=32,out_channels=2,kernel_size=1)
        self.boundingbox=nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1)
        self.landmark=nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1)

    def forward(self,input):
        y= self.pNet(input)
        classification=self.classification(y)
        boundingbox=self.boundingbox(y)
        landmark=self.landmark(y)
        return classification,boundingbox,landmark

class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rNet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2),
        )
        self.line=nn.Linear(in_features=1000,out_features=128)
        self.classification=nn.Linear(in_features=128,out_features=2)
        self.boundingbox=nn.Linear(in_features=128,out_features=4)
        self.landmark=nn.Linear(in_features=128,out_features=10)

    def forward(self,input):
        y= self.pNet(input)
        classification=self.classification(y)
        boundingbox=self.boundingbox(y)
        landmark=self.landmark(y)
        return classification,boundingbox,landmark

class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rNet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2),
        )
        self.line=nn.Linear(in_features=1000,out_features=256)
        self.classification=nn.Linear(in_features=256,out_features=2)
        self.boundingbox=nn.Linear(in_features=256,out_features=4)
        self.landmark=nn.Linear(in_features=256,out_features=10)

    def forward(self,input):
        y= self.pNet(input)
        classification=self.classification(y)
        boundingbox=self.boundingbox(y)
        landmark=self.landmark(y)
        return classification,boundingbox,landmark
