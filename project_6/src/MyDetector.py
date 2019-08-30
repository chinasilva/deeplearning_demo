import torch
import torch.nn as nn
import torchvision.transforms as trans
import numpy as np
import os
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import MyTrain
from MyNet import MyNet
from utils import nms,deviceFun,screenImgTest,readTag
from cfg import * 


class MyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.device=deviceFun()
        self.net=MyNet()
        # self.loadModel()
        self.net=self.net.load_state_dict(torch.load(PRE_MODEL_PATH, map_location=self.device))
# torch.load(PRE_MODEL_PATH,)
        self.trainImagePath=IMG_PATH
        self.net.eval()
        self.imgName=''



        self.net=torch.load(MODEL_PATH)
        self.trainImagePath=IMG_PATH
        self.net.eval()
        self.imgName=''
    
    def forward(self,input, thresh):
        with torch.no_grad():
          anchors=ANCHORS_GROUP
          boxAll=self.myDetector(input, thresh, anchors)
          last_boxes = []
          for n in range(input.size(0)):
              n_boxes = []
              print("boxAll:",boxAll.size())
              if boxAll.size(0)==0:
                  return np.array([])
              boxes_n = boxAll[boxAll[:, 6] == n]
              for cls in range(CLASS_NUM):
                  boxes_c = boxes_n[boxes_n[:, 5] == cls]
                  if boxes_c.size(0) > 0:
                      print("boxes_c",boxes_c.size())
                      n_boxes.extend(nms(boxes_c, 0.3))
                  else:
                      pass
              last_boxes.append(np.stack(n_boxes)) 
          return last_boxes #(N,M,7):N批次,M框个数,7(cf,x,y,h,w,cls,n)

    def myDetector(self,img,thresh,anchor):
        output13,output26,output52=self.net(img.to(self.device))
        indx,result=self.trans(output13,thresh)
        box13=self.convertOldImg(indx,result,32,anchor[13])

        indx2,result2=self.trans(output26,thresh)
        box26=self.convertOldImg(indx2,result2,16,anchor[26])

        indx3,result3=self.trans(output52,thresh)
        box52=self.convertOldImg(indx3,result3,8,anchor[52])
        # print("box13:",box13.size())
        boxAll=torch.cat([box13,box26,box52],dim=0) #(N,7),7:(cf,x,y,h,w,cls,n)
        return boxAll

    def trans(self,output,thresh):
        output=output.permute(0,2,3,1) #(N,H,W,C)
        output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1)#(N,H,W,3,C)

        output = output.cpu()
        # torch.sigmod_(output[...,0]) #置信度激活
        # torch.sigmod_(output[...,1:3]) #偏移量激活
        
        mask=output[...,0]>thresh#置信度大于给定范围认为圈中,得到对应索引
        indx=mask.nonzero() #得到对应置信度大于给定范围的数据索引，并且维度降至二维
        result=output[mask] #得到对应值
        # print("indx:",indx.size())
        # print("result:",result.size())
        return indx,result
    
    def convertOldImg(self,indx,result,t,anchor):
        '''
        反算原图坐标:
        对应关系，1.索引即标签索引,值即标签索引对应的值
                2.index为二维
        '''
        if indx.size(0) == 0:
          return torch.Tensor([])
        anchor=torch.Tensor(anchor)
        a=indx[:,3] #代表了3个建议框中第几个
        n=indx[:,0] #所属图片批次第几批次
        conf=result[:,0] #置信度
        # print("result[:,5:]:",result[:,5:].size())
        print("result:",result.size())#(N,17)
        cls=torch.argmax(result[:,5:],dim=1)#找出哪个位数是分类onehot最大值
        # print("cls:",cls.size())
        # print("cls:",cls)
        oy=(indx[:,1].float()+result[:,2])*t
        ox=(indx[:,2].float()+result[:,1])*t

        w=torch.exp(result[:,3])*anchor[a,0]
        h=torch.exp(result[:,4])*anchor[a,1]

        # x1=ox-w/2
        # y1=oy-h/2
        # x2=ox+w/2
        # y2=oy+h/2
        res=torch.stack([conf.float(),ox,oy,w,h,cls.float(),n.float()],dim=1)
        return res

if __name__ == "__main__":
    detector=MyDetector()
    dataset=[]
    data=readTag(TAG_PATH)# TAG_PATH
    # imgPath=IMG_PATH
    imgPath=IMG_TEST_PATH
    #根据所读的每个标签进行循环
    # for _,line in enumerate(data) :
    #     imageInfo = line.split()#将单个数据分隔开存好
    #     dataset.append(imageInfo)
    dataset.extend(os.listdir(imgPath))
    for i,imgInfo in enumerate(dataset):
            # imgName=imgInfo[0]
            imgName=imgInfo
            imgData= Image.open(os.path.join(imgPath,imgName))
    #根据所读的每个标签进行循环
    for _,line in enumerate(data) :
        imageInfo = line.split()#将单个数据分隔开存好
        dataset.append(imageInfo)
    for i,imgInfo in enumerate(dataset):
            imgName=imgInfo[0]
            imgData= Image.open(os.path.join(IMG_PATH,imgName))
            imgData2=trans.Resize((IMG_WIDTH,IMG_HEIGHT))(imgData)
            imgData2=trans.ToTensor()(imgData2)- 0.5
            imgData2=imgData2.unsqueeze(0)
            #网络
            last_boxes=np.array(detector(imgData2,thresh=0.6))
            
            if last_boxes.size==0:
                print("nothing found!!!")
                continue
            print("imgName:",imgName)
            # 网络
            outLst2=last_boxes[:, :, [1, 2, 3, 4, 5]]
            
            # 标签
            # outLst2=last_boxes[:, :, [1, 2, 3, 4, 0]]
            outLst2=outLst2.reshape(-1,5)
            w_scale, h_scale = imgData.size[0] / IMG_WIDTH, imgData.size[1] / IMG_HEIGHT
            cx=outLst2[:,0]*w_scale
            cy=outLst2[:,1]*h_scale
            w=outLst2[:,2]*w_scale
            h=outLst2[:,3]*h_scale
            cls=outLst2[:,4]
            x=cx-0.5*w
            y=cy-0.5*h
            outLst2[:,0]=x
            outLst2[:,1]=y
            outLst2[:,2]=w
            outLst2[:,3]=h
            outLst2[:,4]=cls
            screenImgTest(testImagePath=imgPath,outLst2=outLst2,imgName=imgName,text="")
