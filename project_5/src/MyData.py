import os
import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as trans
from PIL import Image
from ProcessImage import ProcessImage
from datetime import datetime
from MyEnum import MyEnum
from utils import readTag


class MyData(data.Dataset):
    def __init__(self,tagPath,imgPath):
        super().__init__()
        self.dataset=[]
        self.tagPath=tagPath
        self.imgPath=imgPath
        data=readTag(tagPath)
        #根据所读的每个标签进行循环
        for i,line in enumerate(data) :
            imageInfo = line.split()#将单个数据分隔开存好
            self.dataset.append(imageInfo)   
 
    # 获取数据长度
    def __len__(self):
        return len(self.dataset)
    
    # 获取数据中x,y
    def __getitem__(self, index):
        try:
            imgInfo=self.dataset[index]
            imgName=imgInfo[0]

            confidence=imgInfo[1]
            offsetX1=imgInfo[2]
            offsetY1=imgInfo[3]
            offsetX2=imgInfo[4]
            offsetY2=imgInfo[5]
            imgPath2=''
            if confidence==str(MyEnum.part.value):
                imgPath2='part'
            elif confidence==str(MyEnum.positive.value):
                imgPath2='positive'
            else :
                imgPath2='negative'
            offset=[confidence,offsetX1,offsetY1,offsetX2,offsetY2]
            offset=torch.Tensor(np.array(offset,dtype=np.float16))
            with Image.open(os.path.join(self.imgPath,imgPath2,imgName)) as img:
                imageData=trans.ToTensor()(img)- 0.5
                return imageData,offset
        except Exception as e:
            print("__getitem__",str(e))
            # return torch.Tensor(3,1,1),torch.Tensor(5)



if __name__ == "__main__":

    imagePath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12'
    tagPath=r'C:\Users\liev\Desktop\code\deeplearning_homework\project_5\test\12list_bbox_celeba.txt'
    
    myData=MyData(tagPath,imagePath)
    trainData=data.DataLoader(myData,batch_size=1,shuffle=False)
    for i,(imgName,tagLst) in enumerate(trainData):
        print(i)
        # a=datetime.now()
        # b=datetime.now()
        # print("第{}轮次,耗时{}ms".format(i,(b-a).microseconds//1000))

        # myData.process.tagWritePath(myData.tagPath,tagLst)

    
    

    
    