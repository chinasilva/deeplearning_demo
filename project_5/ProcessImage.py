import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ProcessImage():
    def __init__(self,tagPath):
        self.tagPath=tagPath
        self.tagLst=self.readTag(tagPath)

    def processImage(self,imagePath,imgName,outImgSize=12):
        '''
        逐张处理图片
        outImgSize:默认处理尺寸为12
        根据标签框出图片，并且进行按照比列切割
        '''
        try:
            # imagePath=r'C:\Users\liev\Desktop\dataset\celeba\img_celeba'
            # imgName=r'000001.jpg' 
            imgDetail=os.path.join(imagePath,imgName)
            with Image.open(imgDetail) as img:
                for line in self.tagLst:
                    if imgName in line:
                        imageLst = line.split()        #将单个数据分隔开存好
                        x1=float(imageLst[1]) 
                        y1=float(imageLst[2])
                        width=float(imageLst[3])
                        height=float(imageLst[4])
                        # 使用matplotlib圈图
                        plt.clf()
                        fig,ax = plt.subplots(1)
                        rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none')
                        ax.add_patch(rect)
                        ax.imshow(img)
                        plt.savefig(imgName)
                        return imageLst
        except:
            print(imgName)

    def readTag(self,path):
        with open(path, 'r') as f:
            data = f.readlines()  #txt中所有字符串读入data
        return data


if __name__ == "__main__":
    tagPath=r"C:/Users/liev/Desktop/dataset/celeba/Anno/list_bbox_celeba.txt"
    process=ProcessImage(tagPath)
    process.processImage(1)


        