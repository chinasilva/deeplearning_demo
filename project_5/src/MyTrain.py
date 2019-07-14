import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import os
from datetime import datetime
from MyNet import PNet,RNet,ONet
from MyData import MyData
from utils import deviceFun,writeTag
from MyEnum import MyEnum
import multiprocessing

class MyTrain():
    def __init__(self,Net,epoch,batchSize,imgPath,tagPath,testTagPath,testImgPath,testResult):
        # multiprocessing.set_start_method('spawn', True)
        '''
        Net:PNet,RNet,ONet对应需要训练的网络名称
        epoch,batchSize 批次和轮次
        '''
        self.netName=Net
        self.testResult=testResult
        self.device=deviceFun()
        self.modelPath= str('/home/chinasilva/code/deeplearning_homework/project_5/model/'+self.netName+'.pth')
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
        self.testData=MyData(testTagPath,testImgPath)
        # self.lossFun1=nn.BCEWithLogitsLoss()
        self.lossFun1=nn.BCELoss()
        self.lossFun2=nn.MSELoss()
        if os.path.exists(self.modelPath):
            self.net=torch.load(self.modelPath)

        # self.optimizer=torch.optim.Adam(self.net.parameters())
        self.optimizer=torch.optim.SGD(self.net.parameters(), lr=0.0001)
        
    def train(self):
        trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True,drop_last=True,num_workers=4) #
        testData=data.DataLoader(self.testData,batch_size=self.batchSize,shuffle=True)
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            try:
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
                    output1=outputClass[index].reshape(-1,1)
                    output1=output1[:,:1] #取第0位,置信度
                    loss1=self.lossFun1(output1.to(self.device),target1.to(self.device))

                    index2=offset[:,0]!=MyEnum.negative.value # 过滤非人脸样本，进行比较
                    target2=offset[index2] 
                    target2=target2[:,1:5] #取第1-4位,偏移量
                    output2=outputBox[index2].reshape(-1,4)
                    output2=output2[:,:5]
                    loss2=self.lossFun2(target2.to(self.device),output2.to(self.device))
                    
                    loss=loss1+0.5*loss2

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                    
                    if j%10==0:  #每10批次打印loss
                        b=datetime.now()
                        c=(b-a).microseconds//1000
                        print("epoch:{},batch:{}, loss1:{},loss2:{},loss:{},用时{}ms".format(i,j,loss1.data,loss2.data,loss.data,c))
                        torch.save(self.net,self.modelPath)
                        print("save,success!!!")

                        tagLst=[]
                        with torch.no_grad():
                            accuracyTrain,recallTrain=self.analysize(trainData)
                            accuracy,recall=self.analysize(testData)
                            if accuracy>0.75:
                                print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
                                self.saveModel(accuracy,self.netName)
                                print("save,success!!!")
                            tag1=[self.netName,i + 1,(100 * accuracy),(100 *recall),0,"Test"]
                            tag2=[self.netName,i + 1,(100 * accuracyTrain),(100 *recallTrain),0,"Train"]
                            tagLst.append(tag1)
                            tagLst.append(tag2)
                            writeTag(self.testResult,tagLst)
            except Exception as e:
                print("train",str(e))

    def analysize(self,testData):
        correct =0.
        accuracy=0.
        recall=0.
        total = 0.
        TP=0.
        FN=0.
        tureTotal=0.
        for x,(input, target) in enumerate(testData):
            if x>10:#每次测试选取10*size个样本
                x=x+1
                break
            input, target = input.to(self.device), target.to(self.device)
            output,outputBox,outputLandMark = self.net(input)
            if self.netName=='PNet':
                predicted=output[:,0,0,0] #输出的2个值
            else:
                predicted=output[:,0] #输出的2个值
            target=target[:,0]
            total += target.size(0)
            target=torch.where(target==2,torch.ones_like(target),target) #测试时目标样本为部分样本时当做正样本看待
            
            predicted=torch.where(predicted<0.1,torch.zeros_like(predicted),predicted)
            predicted=torch.where(predicted>0.8,torch.ones_like(predicted),predicted)
            correct += (predicted == target).sum()
            accuracy = correct.float() / (total+0.000001)
            Positive=torch.where(target==1,torch.ones_like(target),torch.zeros_like(target)) 
            tureTotal+=Positive.sum() #所有正样本总数
            indx= Positive!=0
            TP += (predicted[indx] == Positive[indx]).sum() #正样本预测正确个数
            FN += (predicted != target).sum() #所有样本预测错误中，正样本个数
            recall = TP.float() / (tureTotal+0.000001)
            return accuracy,recall

    def saveModel(self,accuracy,netName):
        if((accuracy>0.9 and netName=='ONet') or (accuracy>0.8 and netName=='RNet') or (accuracy>0.75 and netName=='PNet')):
            torch.save(self.net,self.modelPath+str(accuracy.item()))
            

if __name__ == "__main__":
    
    imgPath=r'/mnt/D/code/deeplearning_homework/project_5/test/48/positive'
    tagPath=r'/mnt/D/code/deeplearning_homework/project_5/test/48'
    testTagPath=r''
    testImgPath=r''
    myTrain=MyTrain(Net='ONet',epoch=1,batchSize=2,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testTagPath)
    myTrain.train()

    
