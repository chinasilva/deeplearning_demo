import torch
import torch.nn as nn
from torch.nn import functional as F

def device_fun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    return device
device=device_fun()

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
        self.myCenterLossLayer=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1)
        self.outputClassMethod=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.sigmod=nn.Sigmoid()
        self.boundingboxMethod=nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1)
        self.landmarkMethod=nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1)
        self.iouClassMethod=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.sigmod2=nn.Sigmoid()


    def forward(self,input):
        y= self.PNet(input)
        feature=self.myCenterLossLayer(y)
        outputClass=self.outputClassMethod(feature)
        outputClass=self.sigmod(outputClass)
        boundingbox=self.boundingboxMethod(feature)
        landmark=self.landmarkMethod(y)
        iou=self.iouClassMethod(feature)

        return outputClass,boundingbox,landmark,iou,feature

class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNet= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1),
            nn.PReLU(),            
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3,stride=1),
            nn.PReLU(),            
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2),
        )
        self.line=nn.Linear(in_features=256,out_features=128)
        self.myCenterLossLayer=nn.Linear(in_features=128,out_features=128)
        self.outputClassMethod=nn.Linear(in_features=128,out_features=1)
        self.sigmod=nn.Sigmoid()
        self.boundingboxMethod=nn.Linear(in_features=128,out_features=4)
        self.landmarkMethod=nn.Linear(in_features=128,out_features=10)
        self.iouClassMethod=nn.Linear(in_features=128,out_features=1)
        self.sigmod2=nn.Sigmoid()

    def forward(self,input):
        y= self.RNet(input)
        y=y.view(-1,256)
        y=self.line(y)
        feature=self.myCenterLossLayer(y)
        classification=self.outputClassMethod(feature)
        classification=self.sigmod(classification)
        boundingbox=self.boundingboxMethod(feature)
        landmark=self.landmarkMethod(feature)
        iou=self.iouClassMethod(feature)
        return classification,boundingbox,landmark,iou,feature

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
        self.myCenterLossLayer=nn.Linear(in_features=256,out_features=256)
        self.classificationMethod=nn.Linear(in_features=256,out_features=1)
        self.sigmod=nn.Sigmoid()
        self.boundingboxMethod=nn.Linear(in_features=256,out_features=4)
        self.landmarkMethod=nn.Linear(in_features=256,out_features=10)
        self.iouClassMethod=nn.Linear(in_features=256,out_features=1)
        self.sigmod2=nn.Sigmoid()

    def forward(self,input):
        y= self.ONet(input)
        y=y.view(-1,512)
        y=self.line(y)
        feature=self.myCenterLossLayer(y)
        classification=self.classificationMethod(feature)
        classification=self.sigmod(classification)
        boundingbox=self.boundingboxMethod(feature)
        landmark=self.landmarkMethod(feature)
        iou=self.iouClassMethod(feature)
        return classification,boundingbox,landmark,iou,feature

class CenterLoss(nn.Module):

    def __init__(self, cls_num, feature_num):
        super(CenterLoss, self).__init__()

        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num, feature_num).to(device))

    def forward(self, xs, ys):
        xs = torch.nn.functional.normalize(xs)
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)
        count_dis = count.index_select(dim=0, index=ys.long())
        return torch.sum(torch.sqrt(torch.sum((xs - center_exp.float()) ** 2, dim=1)) / count_dis.float())



if __name__ == "__main__":
    # test=torch.Tensor(2,3,24,24)
    # test=torch.Tensor(2,3,48,48)
    test=torch.Tensor(2,3,12,12)
    # print(a)
    p=PNet()
    # p=RNet()
    # p=ONet()
    a,b,c,d,e=p(test)
    print(e.size())