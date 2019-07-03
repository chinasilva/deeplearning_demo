import torch
import torch.nn
import numpy as np
import os
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from utils import nms,createImage,pltFun
import MyTrain
import MyNet

class MyDetector():
    def __init__(self,testImagePath,netPath):
        self.pnet=torch.load(netPath+'\PNet.pth')
        self.rnet=torch.load(netPath+'\RNet.pth')
        self.onet=torch.load(netPath+'\ONet.pth')
        self.testImagePath=testImagePath
        self.pnetSize=12
        self.rnetSize=24
        self.OnetSize=48

    def main(self):
        dataset=[]
        dataset.extend(os.listdir(self.testImagePath))
        for i,imgName in enumerate(dataset) :
            with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
                # img=torch.Tensor(2,3,200,100)
                PLst=self.pnetDetector(img)
                RLst=self.rnetDetector(PLst)
                OLst=self.onetDetector(RLst)
                for out in OLst:
                    x1=out[0]
                    y1=out[1]
                    x2=out[2]
                    y2=out[3]
                    image0=((x1,y1),(x2,y2))
                    #进行画图
                    pltFun(image0,img,imgName)
                #显示图片
                plt.show()

    def pnetDetector(self,img):
        PLst=[] #从PNet返回找到人脸的框
        scale=1
        h,w=img.size()
        side=max(h,w)
        while side>self.pnetSize: #缩放至跟PNet建议框同样大小即可
            outputClass,outputBox,outputLandMark=self.pnet(img)
            for index in range(len(outputClass[:,:])):
                if outputClass[index]>0.8: #置信度大于0.8认为有人脸
                    #通过偏移量反找原图位置
                    offset1,offset2,offset3,offset4=outputBox
                    postionX1,postionY1=index*2,index*2 #现图左上角坐标
                    postionX2,postionY2=index*2+self.pnetSize,index*2+self.pnetSize #现图右下角坐标
                    originImgPostionX1=offset1*(self.pnetSize/scale)+postionX1 #映射回原图坐标
                    originImgPostionY1=offset2*(self.pnetSize/scale)+postionY1
                    originImgPostionX2=offset3*(self.pnetSize/scale)+postionX2
                    originImgPostionY2=offset4*(self.pnetSize/scale)+postionY2
                    originImgW=originImgPostionX2-originImgPostionX1
                    originImgH=originImgPostionY2-originImgPostionY1
                    minSide=min(originImgW,originImgH)#获取最短边
                    maxSide=max(originImgW,originImgH)#获取最长边
                    #按照最长边进行抠图，短边进行补全
                    originImgPostionX1=originImgPostionX1-(maxSide-originImgW)/2
                    originImgPostionY1=originImgPostionY1-(originImgH-minSide)/2
                    originImgPostionX2=(maxSide-originImgW)/2+originImgPostionX2
                    originImgPostionY2=(originImgH-minSide)/2+originImgPostionY2
                    originImgPostion=(originImgPostionX1,originImgPostionY1,originImgPostionX2,originImgPostionY2)
                    img2=img.crop(originImgPostionX1,originImgPostionY1,originImgPostionX2,originImgPostionY2)
                    #将Pnet对应的框图形状变换成RNet需要的24*24，
                    img2=img2.resize(self.rnetSize,self.rnetSize)
                    imgInfo=(img,originImgPostion)
                    PLst.append(imgInfo)
            scale=scale * 0.707
            side = side / 0.707 #让面积每次缩放1/2，则边长缩放比例为(2**0.5)/2
            h=h/scale
            w=w/scale
            img=img.resize(h,w)
        PLst=nms(PLst[1:],overlap_threshold=0.7,mode='union')#将PNet出来的图片进行下一步操作
        return PLst

    def rnetDetector(self,PLst):
        RLst=[] #从PNet返回找到人脸的框
        for img,originImgPostion in enumerate(PLst):
            h,w=img.size()
            outputClass,outputBox,outputLandMark=self.rnet(img)
            postionX1,postionY1,postionX2,postionY2=originImgPostion
            for index in range(len(outputClass[:,:])):
                if outputClass[index]>0.99: #置信度大于0.99认为有人脸
                    offset1,offset2,offset3,offset4=outputBox
                    originImgPostionX1=offset1*self.rnetSize+postionX1 #映射回原图坐标
                    originImgPostionY1=offset2*self.rnetSize+postionY1
                    originImgPostionX2=offset3*self.rnetSize+postionX2
                    originImgPostionY2=offset4*self.rnetSize+postionY2
                    originImgW=originImgPostionX2-originImgPostionX1
                    originImgH=originImgPostionY2-originImgPostionY1
                    minSide=min(originImgW,originImgH)#获取最短边
                    maxSide=max(originImgW,originImgH)#获取最长边
                    
                    #按照最长边进行抠图，短边进行补全
                    originImgPostionX1=originImgPostionX1-(maxSide-originImgW)/2
                    originImgPostionY1=originImgPostionY1-(originImgH-minSide)/2
                    originImgPostionX2=(maxSide-originImgW)/2+originImgPostionX2
                    originImgPostionY2=(originImgH-minSide)/2+originImgPostionY2
                    originImgPostion=(originImgPostionX1,originImgPostionY1,originImgPostionX2,originImgPostionY2)
                    img2=img.crop(originImgPostionX1,originImgPostionY1,originImgPostionX2,originImgPostionY2)
                    #将Pnet对应的框图形状变换成ONet需要的48*48，
                    img2=img2.resize(self.OnetSize,self.OnetSize)
                    p=(originImgPostion,img2)
                    RLst.append(p)
        RLst=nms(RLst[1:],overlap_threshold=0.7,mode='union')#将PNet出来的图片进行下一步操作
        return RLst
    
    def onetDetector(self,RLst):
        OLst=[] #从PNet返回找到人脸的框
        for img,originImgPostion in enumerate(RLst):
            h,w=img.size()
            outputClass,outputBox,outputLandMark=self.rnet(img)
            postionX1,postionY1,postionX2,postionY2=originImgPostion
            for index in range(len(outputClass[:,:])):
                if outputClass[index]>0.99: #置信度大于0.99认为有人脸
                    offset1,offset2,offset3,offset4=outputBox
                    originImgPostionX1=offset1*self.rnetSize+postionX1 #映射回原图坐标
                    originImgPostionY1=offset2*self.rnetSize+postionY1
                    originImgPostionX2=offset3*self.rnetSize+postionX2
                    originImgPostionY2=offset4*self.rnetSize+postionY2
                    
                    originImgPostion=(originImgPostionX1,originImgPostionY1,originImgPostionX2,originImgPostionY2)
                    p=(originImgPostion)
                    OLst.append(p)
        #输出OLst坐标
        OLst=nms(OLst[1:],overlap_threshold=0.7,mode='min')
        return OLst

    