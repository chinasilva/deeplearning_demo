import torchvision
import torchvision.transforms as trans
from torch.utils import data
import numpy as np
from PIL import Image
import os
import os.path
import math
import sys
import cfg
from utils import readTag,oneHot


# sys.path[0]="/home/chinasilva/code/deeplearning_homework/project_6/data"

class MyData():
    def __init__(self,tagPath,imgPath):
        super().__init__()
        self.dataset=[]
        self.tagPath=cfg.TAG_PATH
        self.imgPath=cfg.IMG_PATH
        data=readTag(tagPath)
        #根据所读的每个标签进行循环
        for _,line in enumerate(data) :
            imageInfo = line.split()#将单个数据分隔开存好
            self.dataset.append(imageInfo)   
 
    # 获取数据长度
    def __len__(self):
        return len(self.dataset)
    
    # 获取数据中信息
    def __getitem__(self, index):
        try:
            lables={}
            imgInfo=self.dataset[index]
            imgName=imgInfo[0]
            imgData= Image.open(os.path.join(self.imgPath,imgName))
            imgData2=trans.ToTensor()(imgData)- 0.5

            boxes=np.array([float(i) for i in imgInfo[1:]])
            boxes2=np.split(boxes,len(boxes)//5) #标签为多分类，产生多个标记框

            for featureSize,anchors in cfg.ANCHORS_GROUP.items():
                #三种不同的特征图，4个偏移量+1个置信度+N个分类
                lables[featureSize]=np.zeros(shape=(featureSize,featureSize,3, 5+cfg.CLASS_NUM))
                for box in boxes2:
                    cls,x,y,w,h=box[0:5]
                    boxArea=w*h
                    # math.modf
                    xOffset,xIndex=math.modf(x*featureSize/cfg.IMG_WIDTH)
                    yOffset,yIndex=math.modf(y*featureSize/cfg.IMG_HEIGHT)
                    for i,anchor in enumerate(anchors):
                        anchorArea=cfg.ANCHORS_GROUP_AREA[featureSize][i]
                        aw,ah=anchor[0:2]
                        wOffset=w/aw
                        hOffset=h/ah
                        iou=min(anchorArea,boxArea)/max(anchorArea,boxArea)
                        '''
                        加log函数方便求梯度
                        形状为(N,H,W,3,C),相当于增加了一个维度,
                        其中3相当于在做损失时与3个锚框对应,C维度存放偏移量,iou等数据
                        *oneHot:从数组中将值取出来
                        '''
                        lables[featureSize][int(yIndex),int(xIndex),i]=np.array(
                            [iou,xOffset,yOffset,np.log(wOffset),np.log(hOffset),*oneHot(cfg.CLASS_NUM,int(cls))]
                        )

            return lables[13],lables[26],lables[52],imgData2
        except Exception as e:
            print("__getitem__",str(e))



if __name__ == "__main__":
    imagePath=r'/home/chinasilva/code/deeplearning_homework/project_6/data/images'
    tagPath=r'/home/chinasilva/code/deeplearning_homework/project_6/data/person_label.txt'
    
    myData=MyData(tagPath,imagePath)
    trainData=data.DataLoader(myData,batch_size=1,shuffle=False)
    for i,(net13,net26,net52,imgData) in enumerate(trainData):
        print(i)
        # print("net13:",net13.shape())
        # print("net26:",net26.shape())
        # print("net52:",net52.shape())
        # print("imgData:",imgData.shape())

    
    
