import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ProcessImage():
    def __init__(self,tagPath,tagWritePath,saveImgPath):
        self.tagPath=tagPath
        self.tagWritePath=tagWritePath
        self.saveImgPath=saveImgPath

    def processImage(self,imagePath,imgName,outImgSize=300):
        '''
        逐张处理图片
        outImgSize:默认处理尺寸为12
        根据标签框出图片，并且进行按照比列切割
        '''
        try:
            tagLst=self.readTag(self.tagPath)
            imgDetail=os.path.join(imagePath,imgName)
            with Image.open(imgDetail) as img:
                for line in tagLst:
                    if imgName in line:
                        imageInfo = line.split()#将单个数据分隔开存好
                        #resize 图片 
                        targetSize=(outImgSize,outImgSize)
                        newImgName=self.padImage(img,imgName, targetSize,self.saveImgPath)
                        offset=self.changeTag(img,imageInfo,targetSize)
                        # self.writeTag(self.tagWritePath,newImgName,offset=offset)
                        
                        self.myRectangle(imageInfo,img,imgName)

                        return imageInfo
        except:
            print("ERROR:",imgName)

    def readTag(self,path):
        '''
        读取标签文件
        '''
        with open(path, 'r') as f:
            data = f.readlines()  #txt中所有字符串读入data
        return data

    def writeTag(self,path,imgName,offset):
        '''
        写标签文件
        '''
        x1,x2,y1,y2=offset
        with open(path, 'a+') as f:
            f.write(str.format("{0}  {1}  {2}  {3}  {4}",imgName,x1,x2,y1,y2))
            f.write('\n')

    def myRectangle(self,imageInfo,img,imgName):
        '''
        画框
        '''
        x1=float(imageInfo[1]) 
        y1=float(imageInfo[2])
        width=float(imageInfo[3])
        height=float(imageInfo[4])
        # 使用matplotlib画框
        plt.clf()
        fig,ax = plt.subplots(1)
        rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.imshow(img)
        plt.savefig(imgName)
        return None

    def padImage(self, image,imgName, targetSize,savePath):
        '''
        按比例缩放并填充
        '''
        iw, ih = image.size  # 原始图像的尺寸
        w, h = targetSize  # 目标图像的尺寸
        scale = min(w / iw, h / ih)  # 转换的最小比例

        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
        newImage = Image.new('RGB', targetSize, (255,255,255))
        # // 为整数除法，计算图像的位置
        newImage.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
        newImgName=os.path.join(savePath,'12new-'+imgName)
        newImage.save(newImgName)

        return newImgName

    def changeTag(self,image,imageInfo,targetSize):
        '''
        更改标签,将标签改成偏移量形式
        '''
        iw, ih = image.size  # 原始图像的尺寸
        w, h = targetSize  # 目标图像的尺寸
        scale = max(iw / w, ih / h)  # 转换的最大比例倍数
        
        minValue= min(iw, ih)
        padding=(iw-minValue)/2
        
        offsetX1=(targetSize-padding)
        offsetY1=0

        x1=float(imageInfo[1]) 
        y1=float(imageInfo[2])
        width=float(imageInfo[3])
        height=float(imageInfo[4])
        x2=x1+width
        y2=y1+height
        newX1=x1/scale
        newY1=y1/scale
        newX2=x2/scale
        newY2=y2/scale
        offsetX1=x1-newX1
        offsetY1=y1-newY1
        offsetX2=x2-newX2
        offsetY2=y2-newY2
        # offset=(offsetX1,offsetY1,offsetY2,offsetY2)
        offset=(newX1,newY1,width,height)

        return offset

# if __name__ == "__main__":
#     tagPath=r"C:/Users/liev/Desktop/dataset/celeba/Anno/list_bbox_celeba.txt"
#     tagWritePath=r"C:\Users\liev\Desktop\code\deeplearning_homework\project_5\pic\12\negative\12_list_bbox_celeba.txt"
#     process=ProcessImage(tagPath,tagWritePath)
#     # process.processImage(1)


