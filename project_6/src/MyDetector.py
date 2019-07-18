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
import cfg


class MyDetector(nn.Module):
    def __init__(self,testImagePath,netPath):
        self.device=deviceFun()
        self.net=torch.load(netPath+'/PNet.pth')
        self.testImagePath=testImagePath
        self.net.eval()
        self.imgName=''
    
    def forward(self,input, thresh, anchors):
        dataset=[]
        dataset.extend(os.listdir(self.testImagePath))
        # for i,imgName in enumerate(dataset):
        #     self.imgName=imgName
        #     with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
        boxAll=self.myDetector(input, thresh, anchors)
        last_boxes = []
        for n in range(input.size(0)):
            n_boxes = []
            boxes_n = boxAll[boxAll[:, 6] == n]
            for cls in range(cfg.CLASS_NUM):
                boxes_c = boxes_n[boxes_n[:, 5] == cls]
                if boxes_c.size(0) > 0:
                    n_boxes.extend(nms(boxes_c, 0.3))
                else:
                    pass
            last_boxes.append(torch.stack(n_boxes))
        return last_boxes

    def myDetector(self,img,thresh,anchor):
        output13,output26,output52=self.net(img).to(self.device)
        indx,result=self.trans(output13,thresh)
        box13=self.convertOldImg(indx,result,32,anchor[13])

        indx2,result2=self.trans(output26,thresh)
        box26=self.convertOldImg(indx2,result2,16,anchor[26])

        indx3,result3=self.trans(output52,thresh)
        box52=self.convertOldImg(indx3,result3,8,anchor[52])
        boxAll=torch.cat([box13,box26,box52],dim=0)
        return boxAll

    def trans(self,output,thresh):
        output=output.permute(0,2,3,1) #(N,H,W,C)
        output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1)#(N,H,W,3,C)

        torch.sigmod_(output[:,0]) #置信度激活
        torch.sigmod_(output[:,1:3]) #偏移量激活
        mask=output[...,0]>thresh#置信度大于给定范围认为圈中,得到对应索引
        indx=mask.nonzero() #得到对应置信度大于给定范围的数据索引，并且维度降至二维
        result=output[indx] #得到对应值
        return indx,result
    
    def convertOldImg(self,indx,result,t,anchor):
        '''
        反算原图坐标:
        对应关系，1.索引即标签索引,值即标签索引对应的值
                2.index为二维
        '''
        a=indx[:,3] #(N,H,W,3,C),相当于3的那个维度,代表了3个锚框
        n=indx[:,0] #代表批次N
        conf=result[:,0] #置信度
        cls=torch.argmax(result[:,5:],dim=1)

        oy=(indx[:,1]+result[:,2])*t
        ox=(indx[:,2]+result[:,1])*t

        w=torch.exp(result[:3])*anchor[a,0]
        h=torch.exp(result[:4])*anchor[a,1]

        x1=ox-w/2
        y1=oy-h/2
        x2=ox+w/2
        y2=oy+h/2
        res=torch.stack([x1,y1,x2,y2,cls,conf],dim=1)
        return res

if __name__ == "__main__":
    testImagePath=r'/mnt/D/code/deeplearning_homework/project_5/test/12/positive'
   
    netPath=r'/mnt/D/code/deeplearning_homework/project_5/model'
    detector=MyDetector(testImagePath,netPath)
    detector.main()

