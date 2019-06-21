import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from PIL import Image
from PIL import ImageDraw
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches

from MyData import MyData
from MyNet import MyNet,VGGNet
from MyUtils import MyUtils
from ResNet import resnet34,resnet18
class MyTrain():
    def __init__(self,path,epoch,batchSize):
        self.path=path
        self.epoch=epoch
        self.batchSize=batchSize
        self.myUtils=MyUtils()
        self.device=self.myUtils.deviceFun()
        # self.myNet=MyNet().to(self.device)
        # self.vggNet=VGGNet('MYVGG').to(self.device)
        # self.ResNet=resnet34(4).to(self.device)
        self.ResNet=resnet18(4).to(self.device)

        self.myData=MyData(self.path)
        self.optimizer=torch.optim.Adam(self.ResNet.parameters())
        self.scheduler =torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        # self.lossFun=nn.CrossEntropyLoss()
        self.lossFun=nn.MSELoss()
        self.trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True)
        # 输出图片位置
        self._OUT_DIR="./pic3"


    def train(self):
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            a=datetime.now()
            for j,(imagePath,x,y) in enumerate(self.trainData):

                # x=x.view(-1,100*100*3).to(self.device)
                # output=self.myNet(x).to(self.device)
                # 输入进行图象变换 (N,H,W,C) -> (N,C,H,W)
                x=x.permute(0,3,1,2).to(self.device)
                # output=self.myNet(x).to(self.device)
                output=self.ResNet(x).to(self.device)
                y=y.to(self.device)
                loss=self.lossFun(y,output)
                losslst.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #二十轮次后进行保存图片
                if j%10==0 and i>50:
                    # x与y位置与标签一致
                    result=output[0]
                    x1=result[0].data.item()
                    y1=result[1].data.item()
                    x2=result[2].data.item()
                    y2=result[3].data.item()
                    width=x2-x1
                    height=y2-y1
                    with Image.open(imagePath[j]) as img:
                        originImg =ImageDraw.Draw(img)
                        originImg.polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],outline=(0,255,0))
                        # img.show()
                        # img.save(self._OUT_DIR+'/pic'+ str(i)+'.jpg', format="jpeg")
                        
                        # # 使用matplotlib圈图
                        # plt.clf()
                        # fig,ax = plt.subplots(1)
                        # rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none')
                        # ax.add_patch(rect)
                        # ax.imshow(img)
                        # plt.pause(0.1)
                        # plt.show()
            self.scheduler.step()
            print("loss:",loss.data)
            b=datetime.now()
            print("第{}轮次,耗时{}ms".format(i,(b-a).microseconds/1000))


        
        # save_model = torch.jit.trace(self.ResNet,  torch.rand(self.batchSize, 3*100*100).to(self.device))
        # save_model.save(r"model/net.pth")


        # 保存加载模型所有信息
        # torch.save(self.myNet, r'model/model.pth')  
        # model = torch.load(r'model/model.pth')

        # 保存加载模型参数信息
        torch.save(self.ResNet.state_dict(), r'models/params.pth')  
        # model_object.load_state_dict(torch.load(r'model/params.pth'))
    
    def changeLogoChannel(self):
        self.myUtils.changeChannel()
