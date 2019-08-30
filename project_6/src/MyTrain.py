import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os
from datetime import datetime
from MyData import MyData
from MyNet import MyNet,DarkNet
from utils import deviceFun,writeTag
import multiprocessing
from cfg import *  

class MyTrain():
    def __init__(self,batchSize,epoch):
        self.device=deviceFun()
        self.batchSize=batchSize
        self.epoch=epoch
        self.myData=MyData()
        self.preModelLoction=PRE_MODEL_PATH
        self.lossFun=nn.MSELoss()
        self.net=MyNet(cls_num=CLASS_NUM).to(self.device)
        self.darkNet=DarkNet().to(self.device)
        self.loadModel(self.preModelLoction)
        self.optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad!=None ,self.net.parameters()))
    
    def loadModel(self,preModelLoction):
        mynetDict=torch.load(MODEL_PATH)
        
        pretrainedDict=torch.load(preModelLoction)#, map_location=self.device
        modelDict = self.darkNet.state_dict()
        # 筛除不加载的层结构,
        # 加载已有预加载模型参数，并将梯度更新状态改为None(默认为False),优化器中只需要过滤掉None的即可
        pretrainedDict2 = {k: v for k, v in pretrainedDict.items() if k in modelDict}
        for k,v in pretrainedDict2.items():
            v.float().requires_grad=None
            pretrainedDict2[k]=v
        
        # 更新当前网络的结构字典
        mynetDict.update(pretrainedDict2)
        # mynetDict.update(pretrainedDict3)
        self.net.load_state_dict(mynetDict)
        # # 梯度不更新
        # for param in self.net.parameters():
        #     param.requeires_grad=False

    def myLoss(self,output,target,alpha):
        output=output.permute(0,2,3,1) #(n,h,w,c)
        output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1) #(n,h,w,3,c)
        index=target[...,0]>0 #置信度大于0和等于0分别求损失
        index2=target[...,0]==0
        print("output:",output.size())
        print("target:",target.size())
        print("output2:",output[index].size())
        print("target2:",target[index].size())
        
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
                    o13,o26,o52=self.net(img.to(self.device))

                    loss1=self.myLoss(o13,target13.float().to(self.device),0.9)
                    loss2=self.myLoss(o26,target26.float().to(self.device),0.9)                    
                    loss3=self.myLoss(o52,target52.float().to(self.device),0.9)                    
                    loss=loss1+loss2+loss3

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    b=datetime.now()
                    c=(b-a).microseconds//1000
                    print("epoch:{},batch:{}, loss1:{},loss2net:{},loss3:{},loss:{},用时{}ms".format(i,j,loss1.data,loss2.data,loss3.data,loss.data,c))
                    torch.save(self.net.state_dict(),MODEL_PATH)
            except Exception as e:
                print("train",str(e))
  
if __name__ == "__main__":
    myTrain=MyTrain(batchSize=5,epoch=5)
    myTrain.train()

    