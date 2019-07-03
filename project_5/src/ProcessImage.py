import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from utils import iouFun,nms,readTag,writeTag,processImage
from MyEnum import MyEnum

class ProcessImage():
    def __init__(self,imagePath,tagPath,saveImgPath,saveTagPath):
        # self.sizeLst=[48,12,24]
        self.sizeLst=[24]
        self.tagPath=tagPath
        self.saveTagPath=saveTagPath
        self.saveImgPath=saveImgPath
        self.imagePath=imagePath

    def main(self):
        try:
            # newImgNameLst=['a','b','c','d','e']
            newImgNameLst=['2a','2b','2c','2d','2e']
            # 根据生成图片及标签尺寸进行循环
            for size in self.sizeLst:
                tagLst=readTag(self.tagPath)
                #根据所读的每个标签进行循环
                for i,line in enumerate(tagLst) :
                    imageInfo = line.split()#将单个数据分隔开存好
                    imgName=imageInfo[0]
                    x=int(imageInfo[1])
                    y=int(imageInfo[2])
                    # 更改默认标签框，由于原图框略大
                    w=int(imageInfo[3])*0.9
                    h=int(imageInfo[4])*0.85
                    if w==0 or h==0:
                        continue
                    x2=x+w
                    y2=y+h
                    #中心点坐标
                    centX=x+w/2
                    centY=y+h/2
                    #对下面操作执行多次，产生多张图片
                    j=0
                    while j<5:
                        #1.移动最小w，h的正负0.2倍，定义最大边长为最小w,h 0.8~最大w,h 1.2 倍
                        #2.按最小边倍数随机移动，得到最新框图，并且计算出坐标
                        movePosition1=random.uniform(-0.6, 0.6) * min(w,h)
                        movePosition2=random.uniform(-0.6, 0.6) * min(w,h)
                        rand=random.uniform(0.8, 1.2)
                        side=random.uniform(rand * min(w,h), rand* max(w,h)) 
                        #新图形中心点坐标
                        newCentX=centX+movePosition1
                        newCentY=centY+movePosition2
                        newImgLeftTopX=newCentX-side/2
                        newImgLeftTopY=newCentY-side/2
                        newImgRightBottomX=newCentX+side/2
                        newImgRightBottomY=newCentY+side/2
                        offsetX1=(newImgLeftTopX-x)/side
                        offsetY1=(newImgLeftTopY-y)/side
                        offsetX2=(newImgRightBottomX-x2)/side
                        offsetY2=(newImgRightBottomY-y2)/side
                        #对原图和新图求IOU
                        p1=(x,y)
                        p2=(x2,y2)
                        newP1=(newImgLeftTopX,newImgLeftTopY)
                        newP2=(newImgRightBottomX,newImgRightBottomY)
                        newImgPosition=(newImgLeftTopX,newImgLeftTopY,newImgRightBottomX,newImgRightBottomY)
                        iouValue= iouFun((p1,p2,0),(newP1,newP2,0))
                        #使用三个不同值进行范围缩放
                        #分别执行，1.从原图抠图 2.保存不同尺寸图片 3.保存坐标文件
                        imgPath2=''
                        confidence=0
                        # if iouValue>0.7:
                        #     imgPath2='positive'
                        #     confidence=MyEnum.positive.value
                        # elif iouValue<0.1:
                        #     imgPath2='negative'
                        #     confidence=MyEnum.negative.value
                        # elif iouValue>0.1 and iouValue<0.35:
                        if iouValue>=0 and iouValue<0.35:
                            imgPath2='part'
                            confidence=MyEnum.part.value
                        if imgPath2:
                            newImgName=newImgNameLst[j]+imgName
                            offset=(newImgName,confidence,offsetX1,offsetY1,offsetX2,offsetY2)
                            j=j+1
                            print("第{}轮，第{}次".format(i,j))
                            processImage(newImgName,imgName,self.imagePath,self.saveImgPath,imgPath2,self.saveTagPath,offset,newImgPosition,outImgSize=size)
            print("Done...............",size)
        except Exception as e:
            print("ERROR:","main"+str(e))

    def negetiveMain(self):
        try:
            newImgNameLst=['0a','0b','0c','0d','0e']
            # 根据生成图片及标签尺寸进行循环
            for size in self.sizeLst:
                dataset=[]
                dataset.extend(os.listdir(self.imagePath))
                #根据所读的每个图片名称进行遍历
                for i,imgName in enumerate(dataset) :
                    j=0
                    if i<1423:
                        continue
                    with Image.open(os.path.join(imagePath,imgName)).convert('RGB') as img:
                        w,h=img.size
                        centX=w/2
                        centY=h/2
                        newTagLst=[]
                        while j<30:
                            newImgName=newImgNameLst[j//6]+str(j)+imgName
                            img1=img.resize((size,size))
                            savePath=saveImgPath+"/"+str(size)+"/"+'negative'
                            if  not os.path.exists(savePath):
                                os.makedirs(savePath)
                            savePath=savePath+"/"+newImgName
                            img1.save(savePath)
                            offset=(newImgName,MyEnum.negative.value,0,0,0,0)
                            newTagLst.append(offset)
                            saveTagPath2=saveTagPath+"/"+str(size)+'list_bbox_celeba.txt'
                            j=j+1
                            print("第{}轮，第{}次".format(i,j))
                        writeTag(saveTagPath2,newTagLst)
                            
            print("Done...............",size)
        except Exception as e:
            print("ERROR:","main"+str(e))

if __name__ == "__main__":
    # imagePath='F:/deeplearning/datasets/celeba/img_celeba/'
    imagePath='D:/pic/'
    tagPath="F:/deeplearning/datasets/celeba/Anno/list_bbox_celeba.txt"

    saveImgPath='D:/my_celebea/negative'
    saveTagPath="D:/my_celebea/negative"
    process=ProcessImage(imagePath,tagPath,saveImgPath,saveTagPath)
    # process.main()
    process.negetiveMain()


