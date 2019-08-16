import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from PIL import Image
from PIL import ImageDraw
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches

from MyData import MyData
from MyNet import MyNet
from utils import device_fun

device=device_fun()
print(device)

IMGPATH=r'/home/chinasilva/code/deeplearning_homework/project_9/data'
class MyTrain():
    def __init__(self,path,epoch,batchSize):
        self.path=path
        self.epoch=epoch
        self.batchSize=batchSize
        self.myNet=MyNet().to(device)
        self.myData=MyData(self.path)
        self.optimizer=torch.optim.Adam(self.myNet.parameters())
        # self.lossFun=nn.MSELoss()
        self.lossFun=nn.CrossEntropyLoss()
        self.trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True)


    def train(self):
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            a=datetime.now() 
            for j,(imagePath,x,y) in enumerate(self.trainData):
                with torch.set_grad_enabled(True):
                    output=self.myNet(x.to(device))
                    # output=torch.argmax(output,dim=1).float()
                    loss=self.lossFun(output,y.to(device).long())
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if j%10==0 :
                        print("loss:",loss.data)
                        b=datetime.now()
                        print("第{}轮次,耗时{}秒".format(i,(b-a).seconds))
    
        save_model = torch.jit.trace(self.myNet,  torch.rand(self.batchSize, 3*100*100).to(device))
        save_model.save(r"model/net.pth")



if __name__ == "__main__":
    path=IMGPATH
    epoch=100
    batchSize=10
    myTrain=MyTrain(path,epoch,batchSize)
    myTrain.train()