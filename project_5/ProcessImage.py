import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class ProcessImage():
    def __init__(self,tagPath,tagWritePath,saveImgPath,newTagLst):
        self.tagPath=tagPath
        self.tagWritePath=tagWritePath
        self.saveImgPath=saveImgPath
        self.newTagLst=newTagLst

    def processImage(self,imagePath,imgName,outImgSize=12):
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
                        offset=self.padImage2(img,imgName,imageInfo, targetSize,self.saveImgPath)
                        self.newTagLst.append(offset)
                        if len(self.newTagLst)%100==0: # 每1000次写一次
                            self.writeTag(self.tagWritePath,self.newTagLst)
                            self.newTagLst=[]
                        return imageInfo,self.newTagLst
        except:
            print("ERROR:",imgName)

    def readTag(self,path):
        '''
        读取标签文件
        '''
        with open(path, 'r') as f:
            data = f.readlines()  #txt中所有字符串读入data
        return data

    def writeTag(self,path,tagLst):
        '''
        写标签文件
        '''
        with open(path, 'a+') as f:
            for i,tag in enumerate(tagLst):
                newImgName,x1,x2,y1,y2=tag
                f.write(str.format("{0}  {1}  {2}  {3}  {4}",newImgName,x1,x2,y1,y2))
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
        按比例缩放并填充,先填充后缩放
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

    def padImage2(self, image,imgName,imageInfo, targetSize,savePath):
        '''
        1.按比例缩放并填充，先缩放后填充
        2.进行记录标签要更改的偏移量
        '''
        x1=float(imageInfo[1]) 
        y1=float(imageInfo[2])
        width=float(imageInfo[3])
        height=float(imageInfo[4])

        iw, ih = image.size  # 原始图像的尺寸
        w, h = targetSize  # 目标图像的尺寸
        maxValue= max(iw, ih)
        paddingW=(maxValue-iw)//2
        paddingH=(maxValue-ih)//2
        # 先填充,后resize
        newImage = Image.new('RGB', (maxValue,maxValue), (255,255,255))
        newImage.paste(image, (paddingW,paddingH))  # 将图像填充为中间图像，两侧为灰色的样式
        
        #求出偏移量，并且对偏移量进行放缩，默认目标图w=h
        offsetX1= round( (x1+paddingW)/w, 2)
        offsetY1= round((y1+paddingH)/w,2)
        offsetX2= round(((x1+width)-iw+paddingW)/w,2)
        offsetY2= round(((y1+height)-ih+paddingH)/w,2)

        #进行缩放
        newImage = newImage.resize((w, h), Image.BICUBIC)  # 缩小图像
        newImgName=os.path.join(savePath,'12new-'+imgName)
        newImage.save(newImgName)
        offset=(imgName,offsetX1,offsetY1,offsetX2,offsetY2)
        return offset


# if __name__ == "__main__":
#     tagPath=r"C:/Users/liev/Desktop/dataset/celeba/Anno/list_bbox_celeba.txt"
#     tagWritePath=r"C:\Users\liev\Desktop\code\deeplearning_homework\project_5\pic\12\negative\12_list_bbox_celeba.txt"
#     process=ProcessImage(tagPath,tagWritePath)
#     # process.processImage(1)


