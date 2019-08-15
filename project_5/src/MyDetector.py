import torch
import torch.nn as nn
import torchvision.transforms as trans
import numpy as np
import os, sys
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from utils import nms,nms2,createImage,pltFun,deviceFun,convertToPosition,backoriginImg,to_rgb
import MyTrain
import MyNet
from datetime import datetime
import mynms

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
            try:
                self.imgName=imgName
                img=cv2.imread(self.testImagePath+'/'+imgName)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if gray.ndim == 2:
                    img = to_rgb(gray)
                a=datetime.now()
                PLst,PLst2=self.pnetDetector(img)
                b=datetime.now()
                if len(PLst)==0:
                    print("PNet Not Found !!!")
                    continue
                RLst,RLst2=self.rnetDetector(img,PLst,PLst2)
                c=datetime.now()
                if len(RLst)==0:
                    print("RNet Not Found !!!")
                    continue
                OLst=self.onetDetector(img,RLst,RLst2)
                d=datetime.now()
                if len(OLst)==0:
                    print("ONet Not Found !!!")
                    continue
                pt=(b-a).microseconds//1000
                rt=(c-b).microseconds//1000
                ot=(d-c).microseconds//1000
                print("pnet耗时:{0},rnet耗时:{1},onet耗时:{2},imgName:{3}:".format(pt,rt,ot,self.imgName))
                self.screenImgTest(OLst,self.imgName,'OLst')
            except Exception as e:
                print("************************",str(e))
    def video(self,img,frame):
        try:
            out=[]
            a=datetime.now()
            PLst,PLst2=self.pnetDetector(img)
            b=datetime.now()
            if len(PLst)==0:
                print("PNet Not Found !!!")
                return
            RLst,RLst2=self.rnetDetector(img,PLst,PLst2)
            c=datetime.now()
            if len(RLst)==0:
                print("RNet Not Found !!!")
                return
            OLst=self.onetDetector(img,RLst,RLst2)
            d=datetime.now()
            if len(OLst)==0:
                print("ONet Not Found !!!")
                return
            pt=(b-a).microseconds//1000
            rt=(c-b).microseconds//1000
            ot=(d-c).microseconds//1000
            # print("pnet耗时:{0},rnet耗时:{1},onet耗时:{2},imgName:{3}:".format(pt,rt,ot,self.imgName))
            print("pnet耗时:{0},rnet耗时:{1},onet耗时:{2}".format(pt,rt,ot))
            for out in OLst:
                x1,y1,x2,y2=out[0:4].astype(int)
                if (x2-x1)<15: ##15像素过滤
                    print("********************10像素过滤************")
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame,str(out[4]),(x1, y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        except Exception as e:
            print("-------------------------------",str(e))
        

    def pnetDetector(self,img):
        '''
        outLst:返回图片
        outLst2:返回图片坐标
        '''
        PLst=[] #从PNet返回找到人脸的框
        scaleRate=0.5#0.707
        with torch.no_grad():
            outLst=[]
            outLst2=[]
            stride=2
            scale=1
            #(w,h)=img.size
            h,w=img.shape[0],img.shape[1]
            img2=img
            side=min(h,w)
            a=datetime.now()
            while side>=self.pnetSize: #缩放至跟PNet建议框同样大小即可
                # box=[]
                imgData=trans.ToTensor()(img2)- 0.5
                imgData=imgData.unsqueeze(0).to(self.device)
                outputClass,outputBox,outputLandMark,outIOU,centerLoss=self.pnet(imgData)
                outputClassValue,outputBoxValue=outputClass[0][0].cpu().data, outputBox[0].cpu().data
                outputBoxValue=outputBoxValue.permute(2,1,0)#(w,h,c)
                tmpClass=outputClassValue.permute(1,0)#(w,h)

                #过滤置信度小于0.6的数据,并且返回对应索引
                idxs = torch.nonzero(torch.gt(tmpClass, 0.9))#(w,h)
                tmp=((idxs*stride).float() / scale)
                tmp2=((idxs*stride + self.pnetSize).float() / scale)
                x1=tmp[:,0] 
                y1=tmp[:,1] 
                x2=tmp2[:,0]
                y2=tmp2[:,1]


                offset = outputBoxValue[idxs[:,0],idxs[:,1]]
                conf=tmpClass[idxs[:,0],idxs[:,1]]
                x1 = x1 - self.pnetSize * offset[:,0]
                y1 = y1 - self.pnetSize * offset[:,1]
                x2 = x2 - self.pnetSize * offset[:,2]
                y2 = y2 - self.pnetSize * offset[:,3]
                box=[x1, y1, x2, y2, conf]
                scale=scale * scaleRate 
                side=side * scaleRate #让面积每次缩放1/2，则边长缩放比例为(2**0.5)/2
                h2= int(h*scale)
                w2= int(w*scale)

                img2=cv2.resize(img2, (w2,h2))
                # img2=img2.resize((w2,h2))

                boxes=torch.stack(box,dim=1).view(-1,5)
                boxes=np.array(boxes)
                if len(boxes)>0:
                    PLst.extend(boxes)
            b=datetime.now()
            PLst=np.array(PLst)
            if len(PLst)==0:
                return [],[]
            PLst=PLst[mynms.py_nms(PLst,thresh=0.5)]
            c=datetime.now()
            # PLst=nms2(PLst,thresh=0.5)
            #NMS后对图形做变换，方便传入RNet
            PLst2=convertToPosition(PLst)
            # adaptiveAvgPool = nn.AdaptiveAvgPool2d((self.pnetSize,self.pnetSize))
            for p in PLst2:
                x1,y1,x2,y2=p[0:4].astype(int)
                # img3=img.crop((x1,y1,x2,y2))
                img3=img[y1:y2,x1:x2]#img[y:y+h, x:x+w]
                img3=cv2.resize(img3, (self.rnetSize,self.rnetSize))
                # img3=img3.resize((self.rnetSize,self.rnetSize))
                imageData=trans.ToTensor()(img3) - 0.5
                # 将Pnet对应的框图形状变换成RNet需要的24*24，
                outLst.append(imageData)
                outLst2.append(p[0:5]) 
            # x1=PLst2[:,0]
            # y1=PLst2[:,1]
            # x2=PLst2[:,2]
            # y2=PLst2[:,3]
            # imgs=np.repeat(np.expand_dims(np.asarray(img),axis=0),repeats=x1.size,axis=0)#获取图片数据(N,W,H,C)
            # imgNumpy=np.asarray(imgs)[y1:y2,x1,x2]  # like crop 
            # imgTorch=(torch.from_numpy(imgNumpy).permute(0,3,2,1)).float() #（N,C,H,W）
            # outLst=adaptiveAvgPool(imgTorch)-0.5 # like resize use
            # outLst2=[x1,y1,x2,y2]
            
            outLst2= np.float32(outLst2)
            d=datetime.now()
            pyramidTime=(b-a).microseconds//1000
            nmsTime=(c-b).microseconds//1000
            resizeTime=(d-c).microseconds//1000
            print("pyramidTime耗时:{0},nmsTime耗时:{1},resizeTime耗时:{2},imgName:{3}:".format(pyramidTime,nmsTime,resizeTime,self.imgName))
            if len(outLst)==0:
                return [],[]
        return torch.stack(outLst),np.stack(outLst2) #将PNet出来的图片进行下一步操作

    def rnetDetector(self,img,PLst,PLst2):
        # self.screenImgTest(PLst2,self.imgName,'PNet')
        RLst=[] #从PNet返回找到人脸的框
        with torch.no_grad():
            outLst=[]
            outLst2=[]
            pos=[]
            # PLst
            # for i,pImg in enumerate(PLst):
            outputClass,outputBox,outputLandMark,outIOU,centerLoss=self.rnet(PLst.to(self.device))
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
            box=[ x1,  y1,  x2,  y2,  outputClass]
            box=np.stack(box,axis=1).reshape(-1,5)
            if len(box)==0:
                return [],[]
            RLst=np.array(box)[mynms.py_nms(box,thresh=0.5)]#将PNet出来的图片进行下一步操作
            # RLst=nms2(box,thresh=0.5)#将PNet出来的图片进行下一步操作
            RLst2=convertToPosition(RLst)
            for r in RLst2:
                x1,y1,x2,y2=r[0:4].astype(int)
                # img2=img.crop((x1,y1,x2,y2))
                img2=img[y1:y2,x1:x2]#img[y:y+h, x:x+w]
                # 将RNet对应的框图形状变换成ONet需要的48*48
                # img2=img2.resize((self.onetSize,self.onetSize))
                img2=cv2.resize(img2, (self.onetSize,self.onetSize))
                imageData=trans.ToTensor()(img2)-0.5
                outLst.append(imageData)
                outLst2.append(r[0:5])
            outLst2= np.array(outLst2)
            if len(outLst)==0:
                return [],[]
        return torch.stack(outLst),np.stack(outLst2)
    
    def onetDetector(self,img,RLst,RLst2):
        # self.screenImgTest(RLst2,self.imgName,'RNet')
        OLst=[] #从PNet返回找到人脸的框
        with torch.no_grad():
            pos=[]
            w=self.onetSize
            h=self.onetSize
            outputClass,outputBox,outputLandMark,outIOU,centerLoss=self.onet(RLst.to(self.device))
            outputClass=outputClass.cpu().data.numpy()
            outputBox=outputBox.cpu().data.numpy()

             #置信度大于0.99认为有人脸
            idxs, _ = np.where(outputClass > 0.999)

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
            # OLst=np.array(box)[mynms.py_nms(box,thresh=0.5)]#将PNet出来的图片进行下一步操作
            OLst=nms2(np.array(box),thresh=0.3,isMin=True)#将PNet出来的图片进行下一步操作
            if len(OLst)==0:
                return []
        # self.screenImgTest(OLst,self.imgName,'OLst')
        return OLst

    def screenImgTest(self,outLst2,imgName,text):
        img2=cv2.imread(self.testImagePath+'/'+imgName)
        for out in outLst2:
                x1=out[0].astype(int)
                y1=out[1].astype(int)
                x2=out[2].astype(int)
                y2=out[3].astype(int)
                draw_0 = cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img2,str(out[4]),(x1, y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.imshow('image',img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    testImagePath=r'/home/chinasilva/code/deeplearning_homework/project_5/images_val/mytest'
   
    netPath=r'/home/chinasilva/code/deeplearning_homework/project_5/model'
    detector=MyDetector(testImagePath,netPath)
    detector.main()

