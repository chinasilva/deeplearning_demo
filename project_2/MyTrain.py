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
from MyNet import MyNet
from MyUtils import MyUtils

class MyTrain():
    def __init__(self,path,epoch,batchSize):
        self.path=path
        self.epoch=epoch
        self.batchSize=batchSize
        self.myUtils=MyUtils()
        self.device=self.myUtils.deviceFun()
        self.myNet=MyNet().to(self.device)
        self.myData=MyData(self.path)
        self.optimizer=torch.optim.Adam(self.myNet.parameters())
        self.lossFun=nn.MSELoss()
        self.trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True)
        self._OUT_DIR="./pic3"


    def train(self):
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            a=datetime.now()
            for j,(imagePath,x,y) in enumerate(self.trainData):
                x=x.view(-1,100*100*3).to(self.device)
                output=self.myNet(x).to(self.device)
                y=torch.Tensor(y).to(self.device)
                loss=self.lossFun(y,output)
                losslst.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if j%10==0 and i>20:
                    x1=output[0][0].data.item()
                    y1=output[0][1].data.item()
                    x2=output[0][2].data.item()
                    y2=output[0][3].data.item()
                    width=y2-y1
                    with Image.open(imagePath[j]) as img:
                        originImg =ImageDraw.Draw(img)
                        originImg.polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)],outline=(0,255,0))
                        # img.show()
                        img.save(self._OUT_DIR+'/pic'+ str(i)+'.jpg', format="jpeg")
                        
                        # plt.clf()
                        # fig,ax = plt.subplots(1)
                        # rect = patches.Rectangle((x1,y1),width,width,linewidth=1,edgecolor='r',facecolor='none')
                        # ax.add_patch(rect)
                        # ax.imshow(img)
                        # plt.pause(0.1)
                        # # plt.savefig
                        # plt.show(block=False)

            print("loss:",loss.data)
            b=datetime.now()
            print("第{}轮次,耗时{}秒".format(i,(b-a).seconds))


        
        save_model = torch.jit.trace(self.myNet,  torch.rand(self.batchSize, 3*100*100).to(self.device))
        save_model.save(r"model/net.pth")


        # 保存加载模型所有信息
        # torch.save(self.myNet, r'model/model.pth')  
        # model = torch.load(r'model/model.pth')

        # # 保存加载模型参数信息
        # torch.save(self.myNet.state_dict(), r'model/params.pth')  
        # model_object.load_state_dict(torch.load(r'model/params.pth'))
    
    def changeLogoChannel(self):
        self.myUtils.changeChannel()

if __name__ == "__main__":
    path=r"./pic2"
    epoch=100
    batchSize=100
    myTrain=MyTrain(path,epoch,batchSize)
    # myTrain.train()
    myTrain.changeLogoChannel()