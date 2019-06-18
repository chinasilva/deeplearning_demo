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
        label=str(imgInfo).replace('pic','').replace('.jpg','')
        x1=label.split(',')[0]
        y1=label.split(',')[1]
        x2=label.split(',')[2]
        y2=label.split(',')[3]
        label=torch.Tensor(np.array([x1,y1,x2,y2],dtype=np.int))
        imagePath=os.path.join(self.path,imgInfo)
        #打开获取图片内容
        img=Image.open(imagePath)
        #imageData 对图片进行归一化，去均值操作
        imageData=torch.Tensor(np.array(img)/255 - 0.5)
        # label=np.array(list(label))

        return imagePath,imageData,label

# if __name__ == "__main__":
#     print(np.array([2,3]))
#     myData=MyData("img")
#     x=myData[6000][0]
#     y=myData[6000][1]
#     print("x:",x)
#     print("y:",y)