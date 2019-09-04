import os
import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as trans
from PIL import Image
from datetime import datetime
from torchvision import transforms


class MyData(data.Dataset):
    def __init__(self,imgPath):
        super().__init__()
        self.dataset=[]
        self.imgPath=imgPath
        imageInfo=os.listdir(imgPath)
        # self.dataset.append(imageInfo)
        self.dataset=imageInfo
 
    # 获取数据长度
    def __len__(self):
        return len(self.dataset)
    
    # 获取数据中x,y
    def __getitem__(self, index):
        try:
            imgInfo=self.dataset[index]
            with Image.open(os.path.join(self.imgPath,imgInfo)).convert('RGB') as img:
                imageData=self.imgTrans(img)
                return imageData,imgInfo
        except Exception as e:
            print("__getitem__",str(e))
            # return torch.Tensor(3,1,1),torch.Tensor(5)
    def imgTrans(self,imgData):
        imgTrans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5]) #使数据归类到0,1之间
        ])
        imgTrans=imgTrans(imgData)#(imgData)
        return imgTrans


if __name__ == "__main__":

    imagePath=r'/home/chinasilva/code/deeplearning_homework/project_11/data/faces'
    myData=MyData(imagePath)
    trainData=data.DataLoader(myData,batch_size=1,shuffle=False)
    for i,(imgData,imgName) in enumerate(trainData):
        print(i)
        # a=datetime.now()
        # b=datetime.now()
        # print("第{}轮次,耗时{}ms".format(i,(b-a).microseconds//1000))

        # myData.process.tagWritePath(myData.tagPath,tagLst)

    
    

    
    