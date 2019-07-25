import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os
from datetime import datetime
from MyData import MyData
from MyNet import MyNet
from utils import deviceFun,writeTag
import multiprocessing
from cfg import *  

class MyTrain():
    def __init__(self,batchSize,epoch):
        self.device=deviceFun()
        self.batchSize=batchSize
        self.epoch=epoch
        self.myData=MyData()
        self.modelLoction=MODEL_PATH
        self.lossFun=nn.MSELoss()
        self.net=MyNet()
        if os.path.exists(self.modelLoction):
            self.net=torch.load(self.modelLoction)
        self.optimizer=torch.optim.Adam(self.net.parameters())
    
    def myLoss(self,output,target,alpha):
        output=output.permute(0,2,3,1) #(n,h,w,c)
        output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1) #(n,h,w,3,c)
        index=target[...,0]>0
        index2=target[...,0]==0
        loss1=self.lossFun(output[index],target[index]) 
        loss2=self.lossFun(output[index2],target[index2]) 
        loss=alpha*loss1+(1-alpha)*loss2
        return loss

    def train(self):
        trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True,drop_last=True,num_workers=0)
        for i in range(self.epoch):
            print("epoch:",i)
            try:
                for j,(target13,target26,target52,img) in enumerate(trainData):
                    a=datetime.now()
                    o13,o26,o52=self.net(img).to(self.device)
                    loss1=self.myLoss(o13,target13,0.9)                    
                    loss2=self.myLoss(o26,target26,0.9)                    
                    loss3=self.myLoss(o52,target52,0.9)                    
                    loss=loss1+loss2+loss3

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    b=datetime.now()
                    c=(b-a).microseconds//1000
                    print("epoch:{},batch:{}, loss1:{},loss2:{},loss3:{},loss:{},用时{}ms".format(i,j,loss1.data,loss2.data,loss3.data,loss.data,c))
                    torch.save(self.net,MODEL_PATH)
            except Exception as e:
                print("train",str(e))
  
if __name__ == "__main__":
    myTrain=MyTrain(batchSize=10,epoch=1000)
    myTrain.train()

    
