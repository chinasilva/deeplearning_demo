import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
from ProcessImage import ProcessImage
from datetime import datetime


class MyData(data.Dataset):
    def __init__(self,path):
        super().__init__()
        self.path=path
        self.dataset=[]
        # 以列表形式链接所有地址
        self.dataset.extend(os.listdir(path))
        self.tagPath=r"C:/Users/liev/Desktop/dataset/celeba/Anno/list_bbox_celeba.txt"
        tagWritePath=r"C:\Users\liev\Desktop\code\deeplearning_homework\project_5\pic\12\negative\list_bbox_celeba.txt"
        saveImgPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\pic\12\negative'
        #处理图片
        self.tagLst=[]
        
        self.process=ProcessImage(self.tagPath,tagWritePath,saveImgPath,self.tagLst)
    
    # 获取数据长度
    def __len__(self):
        return len(self.dataset)
    
    # 获取数据中x,y
    def __getitem__(self, index):
        imgName=self.dataset[index]
        lst=[12,24,48]
        imageInfo,tagLst= self.process.processImage(self.path,imgName,outImgSize=lst)
        return imgName,tagLst


if __name__ == "__main__":
    imagePath=r'C:/Users/liev/Desktop/dataset/celeba/img_celeba/'
    myData=MyData(imagePath)
    trainData=data.DataLoader(myData,batch_size=1,shuffle=False)
    for i,(imgName,tagLst) in enumerate(trainData):
        print(i)
        # a=datetime.now()
        # b=datetime.now()
        # print("第{}轮次,耗时{}ms".format(i,(b-a).microseconds//1000))

        # myData.process.tagWritePath(myData.tagPath,tagLst)

    
    

    
    