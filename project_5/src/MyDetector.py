import torch
import torch.nn
import torchvision.transforms as trans
import numpy as np
import os, sys
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
        self.pnet=torch.load(netPath+'/PNet.pth')
        self.rnet=torch.load(netPath+'/RNet.pth')
        self.onet=torch.load(netPath+'/ONet.pth')
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
            scale=1
            (w,h)=img.size
            img2=img
            side=min(h,w)
            while side>=self.pnetSize: #缩放至跟PNet建议框同样大小即可
                # box=[]
                imgData=trans.ToTensor()(img2) #- 0.5
                imgData=imgData.unsqueeze(0).to(self.device)
                outputClass,outputBox,outputLandMark,outIOU=self.pnet(imgData)
                outputClassValue,outputBoxValue=outputClass[0][0].cpu().data, outputBox[0].cpu().data
                outputBoxValue=outputBoxValue.permute(2,1,0)#(w,h,c)
                tmpClass=outputClassValue.permute(1,0)#(w,h)

                #过滤置信度小于0.6的数据,并且返回对应索引
                idxs = torch.nonzero(torch.gt(tmpClass, 0.9))#(w,h)
                x1=((idxs*stride).float() / scale)[:,0] 
                y1=((idxs*stride).float() / scale)[:,1] 
                x2=((idxs*stride + self.pnetSize).float() / scale)[:,0]
                y2=((idxs*stride + self.pnetSize).float() / scale)[:,1]

                ow=x2-x1
                oh=y2-y1

                offset = outputBoxValue[idxs[:,0],idxs[:,1]]
                conf=tmpClass[idxs[:,0],idxs[:,1]]
                x1 = x1 - ow * offset[:,0]
                y1 = y1 - oh * offset[:,1]
                x2 = x2 - ow * offset[:,2]
                y2 = y2 - oh * offset[:,3]
                box=[x1, y1, x2, y2, conf]
                # box=[x1.int(), y1.int(), x2.int(), y2.int(), tmpClass]
                scale=scale * scaleRate 
                side=side * scaleRate #让面积每次缩放1/2，则边长缩放比例为(2**0.5)/2

                h2= int(h*scale)
                w2= int(w*scale)

                img2=img2.resize((h2,w2))

            
                boxes=torch.stack(box,dim=1).view(-1,5)
                # keep=nms(np.array(box),overlap_threshold=0.5)#将PNet出来的图片进行下一步操作
                boxes=nms2(np.array(boxes),thresh=0.5)#将PNet出来的图片进行下一步操作
                if len(boxes)>0:
                    PLst.extend(boxes)
                    # np.stack(boxes)

            PLst=nms2(np.array(PLst),thresh=0.5)
            #NMS后对图形做变换，方便传入RNet
            PLst2=convertToPosition(np.array(PLst))
            for p in PLst2:
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
            outputClass,outputBox,outputLandMark,outIOU=self.rnet(PLst.to(self.device))
            w=self.rnetSize
            h=self.rnetSize
            outputClass=outputClass.cpu().data.numpy()
            outputBox=outputBox.cpu().data.numpy()
             #置信度大于0.99认为有人脸
            idxs, _ = np.where(outputClass > 0.99)

            #过滤置信度小于0.6的数据,并且返回对应索引
    
            postion = PLst2[idxs]
            offset=outputBox[idxs]
            outputClass=outputClass[idxs]
            
            x1 = (postion[:,0] - w * offset[:,0]).reshape(-1,1)
            y1 = (postion[:,1] - h * offset[:,1]).reshape(-1,1)
            x2 = (postion[:,2] - w * offset[:,2]).reshape(-1,1)
            y2 = (postion[:,3] - h * offset[:,3]).reshape(-1,1)
            box=[x1, y1, x2, y2, outputClass]
            box=np.stack(box,axis=1).reshape(-1,5)
            
            RLst=nms2(np.array(box),thresh=0.5)#将PNet出来的图片进行下一步操作
            RLst2=convertToPosition(np.array(RLst))
            for r in RLst2:
                x1,y1,x2,y2=r[0:4]
                img2=img.crop((x1,y1,x2,y2))
                # 将RNet对应的框图形状变换成ONet需要的24*24，
                img2=img2.resize((self.onetSize,self.onetSize))
                imageData=trans.ToTensor()(img2)
                outLst.append(imageData)
                outLst2.append([x1,y1,x2,y2])
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
            outputClass,outputBox,outputLandMark,outIOU=self.onet(RLst.to(self.device))
            outputClass=outputClass.cpu().data.numpy()
            outputBox=outputBox.cpu().data.numpy()

             #置信度大于0.99认为有人脸
            idxs, _ = np.where(outputClass > 0.999)

            #过滤置信度小于0.6的数据,并且返回对应索引
    
            postion = RLst2[idxs]
            offset=outputBox[idxs]
            outputClass=outputClass[idxs]
            
            x1 = (postion[:,0] - w * offset[:,0]).reshape(-1,1)
            y1 = (postion[:,1] - h * offset[:,1]).reshape(-1,1)
            x2 = (postion[:,2] - w * offset[:,2]).reshape(-1,1)
            y2 = (postion[:,3] - h * offset[:,3]).reshape(-1,1)
            box=[x1, y1, x2, y2, outputClass]
            box=np.stack(box,axis=1).reshape(-1,5)
            #输出OLst坐标
            OLst=nms2(np.array(box),thresh=0.5,isMin=True)#将PNet出来的图片进行下一步操作
            if len(OLst)==0:
                return []
        self.screenImgTest(OLst,self.imgName,'OLst')
        return OLst

    def screenImgTest(self,outLst2,imgName,text):
        with Image.open(os.path.join(self.testImagePath,imgName)).convert('RGB') as img:
            img2=cv2.imread(self.testImagePath+'/'+imgName)
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
    testImagePath=r'/home/chinasilva/code/deeplearning_homework/project_5/images_val/mytest'
    # tagPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12list_bbox_celeba.txt'
   
    netPath=r'/home/chinasilva/code/deeplearning_homework/project_5/model'
    detector=MyDetector(testImagePath,netPath)
    detector.main()

