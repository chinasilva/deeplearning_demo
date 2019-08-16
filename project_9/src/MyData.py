import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
class MyData(data.Dataset):
    def __init__(self,path):
        super().__init__()
        # self.dataset=dataset
        self.path=path
        self.dataset=[]
        # 以列表形式链接所有地址
        self.dataset.extend(os.listdir(path))
    
    # 获取数据长度
    def __len__(self):
        # return 20
        return len(self.dataset)
    
    # 获取数据中x,y
    def __getitem__(self, index):
        # data.dataloader()
        imgInfo=self.dataset[index]
        label=str(imgInfo).replace('.png','')
        x1=ord(label[0])
        y1=ord(label[1])
        x2=ord(label[2])
        y2=ord(label[3])
        label=torch.Tensor(np.array([x1,y1,x2,y2],dtype=np.int))
        a=label[label<58]-48 #number to 0-9
        b=label[label>96]-87 #abc to 10-35
        outLabel=torch.cat((a,b),dim=0)
        imagePath=os.path.join(self.path,imgInfo)
        #打开获取图片内容
        img=Image.open(imagePath)
        #imageData 对图片进行归一化，去均值操作
        imageData=torch.Tensor(np.array(img)/255 - 0.5)
        imageData=imageData.permute(2,0,1)
        return imagePath,imageData,outLabel

if __name__ == "__main__":
    myData=MyData(r"/home/chinasilva/code/deeplearning_homework/project_9/data")
    x=myData[900][0]
    y=myData[900][1]
    print("x:",x)
    print("y:",y)