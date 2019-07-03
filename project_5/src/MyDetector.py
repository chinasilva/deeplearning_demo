import torch
import torch.nn
import torchvision.transforms as trans
import numpy as np
import os
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import nms,createImage,pltFun,deviceFun
import MyTrain
import MyNet

class MyDetector():
    def __init__(self,testImagePath,netPath):
        self.device=deviceFun()
        self.pnet=torch.load(netPath+'\PNet.pth')
        self.rnet=torch.load(netPath+'\RNet.pth')
        self.onet=torch.load(netPath+'\ONet.pth')
        self.testImagePath=testImagePath
        self.pnetSize=12
        self.rnetSize=24
        self.onetSize=48
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

    def main(self):
        dataset=[]
        dataset.extend(os.listdir(self.testImagePath))
        for i,imgName in enumerate(dataset) :
            with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
                # img=torch.Tensor(2,3,200,100)
                imgLst=[]
                imgLst.append(img)
                PLst,PLst2=self.pnetDetector(imgLst)
                RLst,RLst2=self.rnetDetector(img,PLst,PLst2)
                OLst=self.onetDetector(img,RLst,RLst2)
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

            
    def pnetDetector(self,imgLst):
        '''
        outLst:返回图片
        outLst2:返回图片坐标
        '''
        PLst=[] #从PNet返回找到人脸的框
        # imgDatas=trans.ToTensor()(imgLst)- 0.5
        with torch.no_grad():
            for img in imgLst:
                outLst=[]
                outLst2=[]
                imgData=trans.ToTensor()(img)- 0.5
                imgData=imgData.unsqueeze(0).to(self.device)
                scale=1
                (h,w)=img.size
                side=max(h,w)
                while side>=self.pnetSize: #缩放至跟PNet建议框同样大小即可
                    outputClass,outputBox,outputLandMark=self.pnet(imgData)
                    outputClass=outputClass.permute(0,2,3,1) #置信度后移 (n,h,w,c)
                    outputBox=outputBox.permute(0,2,3,1) #偏移量后移(n,h,w,c)
                    output=outputClass.permute(1,2,3,0) #将hw前移，方便计算总长度 (h,w,c,n)
                    length=(output.size()[0]*output.size()[1])//(self.pnetSize*self.pnetSize) # h*w总长度
                    outputClass=outputClass.view(-1,1)
                    outputBox=outputBox.view(-1,4)
                    for index in range(length):
                        if outputClass[index]>0.5: #置信度大于0.8认为有人脸
                            #通过偏移量反找原图位置
                            offset1,offset2,offset3,offset4=outputBox[index].cpu().numpy()
                            postionX1,postionY1=index*2,index*2 #现图左上角坐标
                            postionX2,postionY2=index*2+self.pnetSize,index*2+self.pnetSize #现图右下角坐标
                            originImgPostionX1=postionX1-offset1*(self.pnetSize/scale) #映射回原图坐标
                            originImgPostionY1=postionY1-offset2*(self.pnetSize/scale)
                            originImgPostionX2=postionX2-offset3*(self.pnetSize/scale)
                            originImgPostionY2=postionY2-offset4*(self.pnetSize/scale)
                            originImgW=originImgPostionX2-originImgPostionX1
                            originImgH=originImgPostionY2-originImgPostionY1

                            # with Image.open(r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12\positive\a000001.jpg') as originImg:
                            #     # 使用matplotlib圈图
                            #     plt.clf()
                            #     fig,ax = plt.subplots(1)
                            #     rect = patches.Rectangle((originImgPostionX1,originImgPostionY1),originImgW,originImgH,linewidth=1,edgecolor='r',facecolor='none')
                            #     ax.add_patch(rect)
                            #     ax.imshow(originImg)
                            #     plt.pause(10)
                            #     plt.show(block=False)
                            minSide=min(originImgW,originImgH)#获取最短边
                            maxSide=max(originImgW,originImgH)#获取最长边
                            #按照最长边进行抠图，短边进行补全
                            originImgPostionX1=originImgPostionX1-(maxSide-originImgW)/2
                            originImgPostionY1=originImgPostionY1-(originImgH-minSide)/2
                            originImgPostionX2=(maxSide-originImgW)/2+originImgPostionX2
                            originImgPostionY2=(originImgH-minSide)/2+originImgPostionY2
                            imgInfo=(int(originImgPostionX1),int(originImgPostionY1),int(originImgPostionX2),int(originImgPostionY2),float(outputClass[index].cpu().numpy()))
                            PLst.append(imgInfo)
                    scale=scale * 0.707 
                    side=side * 0.707 #让面积每次缩放1/2，则边长缩放比例为(2**0.5)/2
                PLst=np.array(PLst)
                PLst=PLst[nms(PLst,overlap_threshold=0.7,mode='union')]#将PNet出来的图片进行下一步操作
                
                for p in PLst:
                    x1,y1,x2,y2=p[0:4]
                    img2=img.crop((x1,y1,x2,y2))
                    img2=img2.resize((self.rnetSize,self.rnetSize))
                    imageData=trans.ToTensor()(img2)- 0.5
                    # 将Pnet对应的框图形状变换成RNet需要的24*24，
                    outLst.append(imageData)
                    outLst2.append(p[0:4])
        return torch.stack(outLst),outLst2

    def rnetDetector(self,img,PLst,PLst2):
        RLst=[] #从PNet返回找到人脸的框
        with torch.no_grad():
            outLst=[]
            outLst2=[]
            # PLst
            # for i,pImg in enumerate(PLst):
            outputClass,outputBox,outputLandMark=self.rnet(PLst.to(self.device))
            length=outputClass.size()[0]
            for index in range(length):
                if outputClass[index]>0.6: #置信度大于0.99认为有人脸
                    postionX1,postionY1,postionX2,postionY2=PLst2[index]
                    offset1,offset2,offset3,offset4=outputBox[index].cpu().numpy()
                    originImgPostionX1=postionX1-offset1*(self.rnetSize)#映射回原图坐标
                    originImgPostionY1=postionY1-offset2*self.rnetSize
                    originImgPostionX2=postionX2-offset3*self.rnetSize
                    originImgPostionY2=postionY2-offset4*self.rnetSize
                    originImgW=originImgPostionX2-originImgPostionX1
                    originImgH=originImgPostionY2-originImgPostionY1
                    minSide=min(originImgW,originImgH)#获取最短边
                    maxSide=max(originImgW,originImgH)#获取最长边
                    #按照最长边进行抠图，短边进行补全
                    originImgPostionX1=originImgPostionX1-(maxSide-originImgW)/2
                    originImgPostionY1=originImgPostionY1-(originImgH-minSide)/2
                    originImgPostionX2=(maxSide-originImgW)/2+originImgPostionX2
                    originImgPostionY2=(originImgH-minSide)/2+originImgPostionY2
                    originImgPostion=(int(originImgPostionX1),int(originImgPostionY1),int(originImgPostionX2),int(originImgPostionY2),float(outputClass[index].cpu().numpy()))
                    RLst.append(originImgPostion)

            RLst=np.array(RLst)
            RLst=RLst[nms(RLst,overlap_threshold=0.7,mode='union')]#将PNet出来的图片进行下一步操作
            
            for r in RLst:
                x1,y1,x2,y2=r[0:4]
                img2=img.crop((x1,y1,x2,y2))
                # 将Pnet对应的框图形状变换成RNet需要的24*24，
                img2=img2.resize((self.rnetSize,self.rnetSize))
                imageData=trans.ToTensor()(img2)- 0.5
                outLst.append(imageData)
                outLst2.append(r)
        return torch.stack(outLst),outLst2
    
    def onetDetector(self,img,RLst,RLst2):
        OLst=[] #从PNet返回找到人脸的框
        with torch.no_grad():
            outputClass,outputBox,outputLandMark=self.rnet(RLst.to(self.device))
            length=outputClass.size()[0]
            for index in range(length):
                if outputClass[index]>0.7: #置信度大于0.99认为有人脸
                    postionX1,postionY1,postionX2,postionY2=outputBox[index]
                    offset1,offset2,offset3,offset4=outputBox[index].cpu().numpy()
                    originImgPostionX1=postionX1-offset1*self.onetSize #映射回原图坐标
                    originImgPostionY1=postionY1-offset2*self.onetSize
                    originImgPostionX2=postionX2-offset3*self.onetSize
                    originImgPostionY2=postionY2-offset4*self.onetSize
                    
                    originImgPostion=(int(originImgPostionX1),int(originImgPostionY1),int(originImgPostionX2),int(originImgPostionY2),float(outputClass[index].cpu().numpy()))
                    OLst.append(originImgPostion)
            #输出OLst坐标
            OLst=np.array(OLst)
            OLst=RLst[nms(OLst[1:],overlap_threshold=0.7,mode='min')]
        return OLst


if __name__ == "__main__":
    testImagePath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12\positive'
    # tagPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12list_bbox_celeba.txt'
    
    netPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\model'
    detector=MyDetector(testImagePath,netPath)
    detector.main()