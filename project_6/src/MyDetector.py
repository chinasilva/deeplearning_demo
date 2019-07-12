import torch
import torch.nn
import torchvision.transforms as trans
import numpy as np
import os
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import MyTrain
import MyNet
from utils import nms,deviceFun


class MyDetector:
    def __init__(self,testImagePath,netPath):
        self.device=deviceFun()
        self.net=torch.load(netPath+'/PNet.pth')
        self.testImagePath=testImagePath
        self.net.eval()
        self.imgName=''
    
    def main(self):
        dataset=[]
        dataset.extend(os.listdir(self.testImagePath))
        for i,imgName in enumerate(dataset):
            self.imgName=imgName
            with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
                out=[]
                PLst,PLst2=self.myDetector(img)

    def myDetector(self,img):
        output13,output26,output52=self.net(img).to(self.device)
        self.trans(output13,0.8)
        self.trans(output26,0.8)
        self.trans(output52,0.8)


    def trans(self,output,alpha):
        output=output.permute(0,2,3,1) #(N,H,W,C)
        output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1)#(N,H,W,3,C)

        mask=output[...,0]>alpha#置信度大于给定范围认为圈中,得到对应索引
        indx=mask.nonzero() #得到对应置信度大于给定范围的数据索引，并且维度降至二维
        result=output[indx] #得到对应值
        return indx,result
    
    def convertOldImg(self,indx,result,scole):
        a=indx[:,3] #(N,H,W,3,C),相当于3的那个维度,代表了3个锚框
        # n=indx[:,0] #代表批次N

        '''
        反算原图坐标:
        对应关系，1.索引即标签索引,值即标签索引对应的值
                2.
        '''
        oy=(indx[:,1]+result[:,2])*scole 
        ox=(indx[:,2]+result[:,1])*scole




if __name__ == "__main__":
    testImagePath=r'/mnt/D/code/deeplearning_homework/project_5/test/12/positive'
   
    netPath=r'/mnt/D/code/deeplearning_homework/project_5/model'
    detector=MyDetector(testImagePath,netPath)
    detector.main()

