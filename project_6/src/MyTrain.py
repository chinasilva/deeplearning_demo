﻿import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os
from datetime import datetime
from MyData import MyData
from MyNet import MyNet
from utils import deviceFun,writeTag
import multiprocessing

class MyTrain():
    def __init__(self,batchSize,epoch,tagPath,imgPath,modelLoction):
        self.device=deviceFun()
        self.batchSize=batchSize
        self.epoch=epoch
        self.myData=MyData(tagPath,imgPath)
        self.modelLoction=modelLoction
        # self.testData=MyData(testTagPath,testImgPath)
        self.lossFun=nn.MSELoss()
        self.net=MyNet()
        if os.path.exists(modelLoction):
            self.net=torch.load(modelLoction)
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
        # testData=data.DataLoader(self.testData,batch_size=self.batchSize,shuffle=True)
        # trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=False,drop_last=True)
        # losslst=[]
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
                    torch.save(self.net,self.modelLoction)
                    
                    # if j%10==0:  #每10批次打印loss
                    #     with torch.no_grad():
                    #         self.net.eval()
                    #         correct = 0.
                    #         error = 0.
                    #         total = 0.
                    #         for _,(input, target) in enumerate(testData):
                    #             input, target = input.to(self.device), target.to(self.device)
                    #             output,outputBox,outputLandMark = self.net(input)
                    #             predicted=output[:,0,0,0] #输出的2个值
                    #             target=target[:,0]
                    #             # _, predicted = torch.max(output.data, 1)
                    #             total += target.size(0)
                    #             predicted=torch.where(predicted<0.1,torch.zeros_like(predicted),predicted)
                    #             predicted=torch.where(predicted>0.9,torch.ones_like(predicted),predicted)
                    #             correct += (predicted == target).sum()
                    #             error += (predicted != target).sum()
                    #             accuracy = correct.float() / total
                    #             recall = correct.float() / correct+error

                    #         print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
                    #         tagLst=[self.netName,i + 1,(100 * accuracy),(100 *recall),0]
                    #         writeTag(self.testResult,tagLst)
            except Exception as e:
                print("train",str(e))
  
if __name__ == "__main__":
    
    imgPath=r'/mnt/D/code/deeplearning_homework/project_5/test/48/positive'
    tagPath=r'/mnt/D/code/deeplearning_homework/project_5/test/48'
    testTagPath=r''
    testImgPath=r''
    myTrain=MyTrain(Net='ONet',epoch=1,batchSize=2,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testTagPath)
    myTrain.train()

    
