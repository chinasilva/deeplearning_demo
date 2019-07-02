import torch
import torch.nn
import MyTrain
import MyNet
from utils import nms

class MyDector():
    def __init__(self):
        self.pnet=MyNet.PNet()
        self.rnet=MyNet.RNet()
        self.onet=MyNet.ONet()

    def main(self):
        img=torch.Tensor(2,3,200,100)
        img2=img
        h,w,c=img.size()
        centPosX,centPosY=w/2,h/2
        side=max(h,w)
        scale=1
        PLst=[] #从PNet返回找到人脸的框
        while  side>12: #缩放至跟PNet建议框同样大小即可
            outputClass,outputBox,outputLandMark=self.pnet(img2)
            if outputClass[:,:1]>0.8: #置信度大于0.8认为有人脸
                #通过偏移量反找原图位置
                offset1,offset2,offset3,offset4=outputBox
                postionX1,postionY1=centPosX-w/2,centPosY-h/2 #现图左上角坐标
                postionX2,postionY2=centPosX+w/2,centPosY+h/2 #现图右下角坐标

                originImgPostionX1=offset1*w+postionX1
                originImgPostionY1=offset2*w+postionY1
                originImgPostionX2=offset3*h+postionX2
                originImgPostionY2=offset4*h+postionY2
                originImgPostion=(originImgPostionX1,originImgPostionY1,originImgPostionX2,originImgPostionY2)
                p=(originImgPostion,outputClass)
                PLst.append(p)
            scale=scale * 0.707
            side = side / 0.707 #让面积每次缩放1/2，则边长缩放比例为(2**0.5)/2
            h=h/scale
            w=w/scale
            img2=img2.resize(h,w)
        PLst=nms(PLst[1:],overlap_threshold=0.7,mode='union')#将PNet出来的图片进行下一步操作
        for pImg in range(PLst): #将PNet输出后的img放入RNet
            ...