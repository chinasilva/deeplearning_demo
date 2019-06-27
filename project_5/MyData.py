import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
from ProcessImage import ProcessImage
class MyData(data.Dataset):
    def __init__(self,path):
        super().__init__()
        self.path=path
        self.dataset=[]
        # 以列表形式链接所有地址
        self.dataset.extend(os.listdir(path))

        tagPath=r"C:/Users/liev/Desktop/dataset/celeba/Anno/list_bbox_celeba.txt"
        self.processImage=ProcessImage(tagPath)
    
    # 获取数据长度
    def __len__(self):
        return len(self.dataset)
    
    # 获取数据中x,y
    def __getitem__(self, index):
        imgName=self.dataset[index]
        
        self.processImage.processImage(self.path,imgName)

        return imgName


if __name__ == "__main__":
    imagePath=r'C:/Users/liev/Desktop/dataset/celeba/img_celeba/'
    myData=MyData(imagePath)
    trainData=data.DataLoader(myData,batch_size=1,shuffle=True)
    for i in enumerate(trainData):
        print(i)
    
    

    
    