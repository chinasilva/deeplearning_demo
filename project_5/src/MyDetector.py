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
from utils import nms,nms2,createImage,pltFun,deviceFun,convertToPosition,backoriginImg
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
        self.imgName=''

    def main(self):
        dataset=[]
        dataset.extend(os.listdir(self.testImagePath))
        for i,imgName in enumerate(dataset):
            self.imgName=imgName
            with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
                out=[]
                PLst,PLst2=self.pnetDetector(img)
                if len(PLst)==0:
                    print("PNet Not Found !!!")
                    continue
                RLst,RLst2=self.rnetDetector(img,PLst,PLst2)
                if len(RLst)==0:
                    print("RNet Not Found !!!")
                    continue
                OLst=self.onetDetector(img,RLst,RLst2)
                if len(OLst)==0:
                    print("ONet Not Found !!!")
                    continue
            
    def pnetDetector(self,img):
        '''
        outLst:返回图片
        outLst2:返回图片坐标
        '''
        PLst=[] #从PNet返回找到人脸的框
        scaleRate=0.707
        # imgDatas=trans.ToTensor()(imgLst)- 0.5
        with torch.no_grad():
            outLst=[]
            outLst2=[]
            stride=2
            pos=[]
            scale=1
            (h,w)=img.size
            img2=img
            side=min(h,w)
            while side>=self.pnetSize: #缩放至跟PNet建议框同样大小即可
                box=[]
                imgData=trans.ToTensor()(img2) #- 0.5
                imgData=imgData.unsqueeze(0).to(self.device)
                outputClass,outputBox,outputLandMark=self.pnet(imgData)
                outputClassValue,outputBoxValue=outputClass[0][0].cpu().data, outputBox[0].cpu().data
                #过滤置信度小于0.6的数据,并且返回对应索引
                idxs = torch.nonzero(torch.gt(outputClassValue, 0.9))
                
                # x1=(idxs[:,1] * stride).float() / scale
                # y1=(idxs[:,0] * stride).float() / scale
                # x2=(idxs[:,1] * stride + self.pnetSize).float() / scale
                # y2=(idxs[:,0] * stride + self.pnetSize).float() / scale
                # w=x2-x1
                # w=y2-y1
                # _offset = offset[:, start_index[0], start_index[1]]
                # x1 = _x1 + ow * _offset[0].int()
                # y1 = _y1 + oh * _offset[1].int()
                # x2 = _x2 + ow * _offset[2].int()
                # y2 = _y2 + oh * _offset[3].int()
                for idx in idxs:
                    #通过偏移量反找原图位置
                    box.append(backoriginImg(idx, outputBoxValue, outputClassValue[idx[0], idx[1]], scale))
                scale=scale * scaleRate 
                side=side * scaleRate #让面积每次缩放1/2，则边长缩放比例为(2**0.5)/2
                
                # def backoriginImg(start_index, offset, cls, scale, stride=2, side_len=12):

                #     _x1 = (start_index[1] * stride).float() / scale
                #     _y1 = (start_index[0] * stride).float() / scale
                #     _x2 = (start_index[1] * stride + side_len).float() / scale
                #     _y2 = (start_index[0] * stride + side_len).float() / scale

                #     ow = _x2 - _x1
                #     oh = _y2 - _y1

                #     _offset = offset[:, start_index[0], start_index[1]]
                #     x1 = _x1 + ow * _offset[0].int()
                #     y1 = _y1 + oh * _offset[1].int()
                #     x2 = _x2 + ow * _offset[2].int()
                #     y2 = _y2 + oh * _offset[3].int()

                #     return [x1, y1, x2, y2, cls]

                h2= int(h*scale)
                w2= int(w*scale)

                img2=img2.resize((w2,h2))
            
                keep=nms(np.array(box),overlap_threshold=0.5)#将PNet出来的图片进行下一步操作
                boxex=box[keep]
                PLst.append(boxex)
            #NMS后对图形做变换，方便传入RNet
            PLst=convertToPosition(PLst)
            for p in PLst:
                x1,y1,x2,y2=p[0:4]
                img3=img.crop((x1,y1,x2,y2))
                # img3=img.crop((y1,x1,y2,x2))
                img3=img3.resize((self.rnetSize,self.rnetSize))
                imageData=trans.ToTensor()(img3) #- 0.5
                # 将Pnet对应的框图形状变换成RNet需要的24*24，
                outLst.append(imageData)
                outLst2.append([x1,y1,x2,y2]) 
            outLst2= np.array(outLst2,dtype=int)
            if len(outLst)==0:
                return [],[]
        return torch.stack(outLst),np.stack(outLst2) #将PNet出来的图片进行下一步操作

    def rnetDetector(self,img,PLst,PLst2):
        self.screenImgTest(PLst2,self.imgName,'PNet')
        RLst=[] #从PNet返回找到人脸的框
        with torch.no_grad():
            outLst=[]
            outLst2=[]
            pos=[]
            # PLst
            # for i,pImg in enumerate(PLst):
            outputClass,outputBox,outputLandMark=self.rnet(PLst.to(self.device))
            w=self.rnetSize
            h=self.rnetSize
            outputClass=outputClass.cpu().data.numpy()
            outputBox=outputBox.cpu().data.numpy()
             #置信度大于0.99认为有人脸
            idxs, _ = np.where(outputClass > 0.9)
            for index in idxs:
                postionX1,postionY1,postionX2,postionY2=PLst2[index]
                offset1,offset2,offset3,offset4=outputBox[index]
                originImgPostionX1=offset1*w+postionX1#通过PNet输出的原图坐标和RNet输出的偏移量映射回新图坐标
                originImgPostionY1=offset2*h+postionY1
                originImgPostionX2=offset3*w+postionX2
                originImgPostionY2=offset4*h+postionY2
                originImgPostion=(int(originImgPostionX1),int(originImgPostionY1),int(originImgPostionX2),int(originImgPostionY2),float(outputClass[index]))
                pos.append(originImgPostion)

            RLst=nms2(np.array(pos),thresh=0.5)#将PNet出来的图片进行下一步操作
            for r in RLst:
                x1,y1,x2,y2=r[0:4]
                img2=img.crop((x1,y1,x2,y2))
                # 将RNet对应的框图形状变换成ONet需要的24*24，
                img2=img2.resize((self.onetSize,self.onetSize))
                imageData=trans.ToTensor()(img2)- 0.5
                outLst.append(imageData)
                outLst2.append(r)
            outLst2= np.array(outLst2)
            if len(outLst)==0:
                return [],[]
        return torch.stack(outLst),np.stack(outLst2)
    
    def onetDetector(self,img,RLst,RLst2):
        self.screenImgTest(RLst2,self.imgName,'RNet')
        OLst=[] #从PNet返回找到人脸的框
        with torch.no_grad():
            pos=[]
            w=self.onetSize
            h=self.onetSize
            outputClass,outputBox,outputLandMark=self.onet(RLst.to(self.device))
            outputClass=outputClass.cpu().data.numpy()
            outputBox=outputBox.cpu().data.numpy()
            #置信度大于0.99认为有人脸
            idxs, _ = np.where(outputClass > 0.99)
            for index in idxs:
                postionX1,postionY1,postionX2,postionY2=RLst2[index][0:4]
                offset1,offset2,offset3,offset4=outputBox[index]
                originImgPostionX1=offset1*w+postionX1 #通过RNet输出的原图坐标和ONet输出的偏移量映射回新图坐标
                originImgPostionY1=offset2*h+postionY1
                originImgPostionX2=offset3*w+postionX2
                originImgPostionY2=offset4*h+postionY2
                
                originImgPostion=(int(originImgPostionX1),int(originImgPostionY1),int(originImgPostionX2),int(originImgPostionY2),float(outputClass[index]))
                pos.append(originImgPostion)
            #输出OLst坐标
            OLst=nms2(np.array(pos),thresh=0.3,isMin=True)
            if len(OLst)==0:
                return []
        self.screenImgTest(OLst,self.imgName,'OLst')
        return OLst

    def screenImgTest(self,outLst2,imgName,text):
        with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
            img2=cv2.imread(self.testImagePath+'\\'+imgName)
            for out in outLst2.astype(int):
                    x1=out[0]
                    y1=out[1]
                    x2=out[2]
                    y2=out[3]
                    draw_0 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img2,text,(50,150),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
            cv2.imshow('image',img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
if __name__ == "__main__":
    testImagePath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12\positive'
    # tagPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12list_bbox_celeba.txt'
    
    netPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\model'
    detector=MyDetector(testImagePath,netPath)
    detector.main()

