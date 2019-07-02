import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os
from datetime import datetime
from MyNet import PNet,RNet,ONet
from MyData import MyData
from utils import deviceFun
from MyEnum import MyEnum

class MyTrain():
    def __init__(self,Net,epoch,batchSize,imgPath,tagPath):
        '''
        Net:PNet,RNet,ONet对应需要训练的网络名称
        epoch,batchSize 批次和轮次
        '''
        self.netName=Net
        self.device=deviceFun()
        self.fileLoction= str('C:/Users/liev/Desktop/code/deeplearning_homework/project_5/model/'+self.netName+'.pth')
        if Net=='PNet':
            self.net=PNet().to(self.device)
        elif Net=='RNet':
            self.net=RNet().to(self.device)
        elif Net=='ONet':
            self.net=ONet().to(self.device)
        else:
            raise RuntimeError('训练时,请输入正确的网络名称')
        self.batchSize=batchSize
        self.epoch=epoch
        self.myData=MyData(tagPath,imgPath)
        self.lossFun1=nn.BCELoss()
        self.lossFun2=nn.MSELoss()
        if os.path.exists(self.fileLoction):
            self.net=torch.load(self.fileLoction)
        self.optimizer=torch.optim.Adam(self.net.parameters())
        

    def train(self):
        trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True)
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            for j,(img,offset) in enumerate(trainData):
                #训练分为两种损失
                #1.negative与positive
                #2.positive与part
                # offset=confidence,offsetX1,offsetY1,offsetX2,offsetY2
                a=datetime.now()
                outputClass,outputBox,outputLandMark=self.net(img.to(self.device))
                index=offset[:,0]!=MyEnum.part.value # 过滤部分人脸样本，进行比较
                target1=offset[index] 
                target1=target1[:,:1] #取第0位,置信度
                output1=outputClass[index]
                output1=output1[:,:1] #取第0位,置信度
                loss1=self.lossFun1(output1.to(self.device),target1.to(self.device))

                index2=offset[:,0]!=MyEnum.negative.value # 过滤非人脸样本，进行比较
                target2=offset[index2] 
                target2=target2[:,1:5] #取第1-4位,偏移量
                output2=outputBox[index2]
                output2=output2[:,:5]
                loss2=self.lossFun2(target2.to(self.device),output2.to(self.device))
                
                loss=loss1+loss2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                if j%10==0:  #每10批次打印loss
                    b=datetime.now()
                    c=(b-a).microseconds//1000
                    print("epoch:{}, loss1:{},loss2:{},loss:{},用时{}ms".format(i,loss1.data,loss2.data,loss.data,c))
                    torch.save(self.net,self.fileLoction)

    
if __name__ == "__main__":
    imgPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\pic\48'
    tagPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\txt\48list_bbox_celeba.txt'
    myTrain=MyTrain(Net='ONet',epoch=10,batchSize=512,imgPath=imgPath,tagPath=tagPath)
    myTrain.train()

    
