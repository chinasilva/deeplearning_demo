import torch
import torch.nn as nn

class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.PNet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
            nn.PReLU()
            # nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1),
        )
        self.outputClass=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.sigmod=nn.Sigmoid()
        self.boundingbox=nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1)
        self.landmark=nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1)

    def forward(self,input):
        y= self.PNet(input)
        outputClass=self.outputClass(y)
        outputClass=self.sigmod(outputClass)
        boundingbox=self.boundingbox(y)
        landmark=self.landmark(y)
        return outputClass,boundingbox,landmark

class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2),
        )
        self.line=nn.Linear(in_features=256,out_features=128)
        self.sigmod=nn.Sigmoid()
        self.classification=nn.Linear(in_features=128,out_features=1)
        self.boundingbox=nn.Linear(in_features=128,out_features=4)
        self.landmark=nn.Linear(in_features=128,out_features=10)

    def forward(self,input):
        y= self.RNet(input)
        y=y.view(-1,256)
        y=self.line(y)
        y=self.sigmod(y)
        classification=self.classification(y)
        boundingbox=self.boundingbox(y)
        landmark=self.landmark(y)
        return classification,boundingbox,landmark

class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ONet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2),
            nn.PReLU()
        )
        self.line=nn.Linear(in_features=512,out_features=256)
        self.classification=nn.Linear(in_features=256,out_features=1)
        self.sigmod=nn.Sigmoid()
        self.boundingbox=nn.Linear(in_features=256,out_features=4)
        self.landmark=nn.Linear(in_features=256,out_features=10)

    def forward(self,input):
        y= self.ONet(input)
        y=y.view(-1,512)
        y=self.line(y)
        classification=self.classification(y)
        classification=self.sigmod(classification)
        boundingbox=self.boundingbox(y)
        landmark=self.landmark(y)
        return classification,boundingbox,landmark


if __name__ == "__main__":
    # test=torch.Tensor(2,3,24,24)
    test=torch.Tensor(2,3,100,100)
    # print(a)
    p=PNet()
    # p=RNet()
    a,b,c=p(test)
    print(a)