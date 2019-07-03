import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
from PIL import Image,ImageDraw
import os
import torch
# %matplotlib inline

def iouFun (image0,image1):
    (r01,r02,confidence)=image0
    (r11,r12,confidence)=image1
    (x01,y01)=r01
    (x02,y02)=r02
    (x11,y11)=r11
    (x12,y12)=r12
    leftXPoint=np.where(x01>x11,x01,x11)
    leftYPoint=np.where(y01>y11,y01,y11)   
    rightXPoint=np.where(x02>x12,x12,x02)
    rightYPoint=np.where(y02>y12,y12,y02)
    size=(x02-x01)*(y02-y01)
    size2=(x12-x11)*(y12-y11)
    size3=(leftXPoint-rightXPoint)*(leftYPoint-rightYPoint)
    iou=size3/(size+size2-size3)
    if iou<0:
        return 0
    return iou
# 默认iou=0.3为同一类
def nmsFun(defaultIou=0.3,*arg):
    #存储移除的图象索引
    index=[]
    for i,(image0,image1,confidence) in enumerate(arg):
        j=i+1
        # 存在需要移除图象则即跳出循环
        if i in enumerate(index):
            break
        #每次大循环前将maxConfidence至为0,lst存储Iou以及对应的图象,
        maxConfidence=0
        for _,(image0,image1,confidence) in enumerate(arg[j:]):
             # 存在需要移除图象则即跳出循环
            if j in enumerate(index):
                break
            newIou= iouFun(arg[i],arg[j])
            if newIou < defaultIou:
                maxConfidence=max(arg[i][2],arg[j][2])
                #如果是最大的置信度则保留,相同分类中置信度小的添加到移除索引
                if arg[i][2]==maxConfidence:
                    index.append(j)
                else:
                    index.append(i)
    index=list(set(index))
    lst=list(arg)
    #根据需要移除的坐标删除数据
    for index in sorted(index, reverse=True):
        del lst[index]
    return lst

def readTag(path):
    '''
    读取标签文件
    '''
    with open(path, 'r') as f:
        data = f.readlines()  #txt中所有字符串读入data
    return data

def writeTag(path,tagLst):
    '''
    写标签文件
    tagLst:[newImgName,x1,x2,y1,y2,
        newImgName2,x1,x2,y1,y2
    ]
    '''
    with open(path, 'a+') as f:
        for i,tag in enumerate(tagLst):
            newImgName,flag,x1,x2,y1,y2=tag
            f.write(str.format("{0}  {1}  {2}  {3}  {4} {5}",newImgName,flag,x1,x2,y1,y2))
            f.write('\n')
def processImage(newImgName,imgName,imagePath,saveImgPath,imgPath2,saveTagPath,offset,newImgPosition, outImgSize=12):
    '''
    逐张处理图片
    outImgSize:默认处理尺寸为12
    根据标签从原图抠出图片，并且进行按照比列切割
    offset=(imgName,flag,x1,x2,y1,y2)
    '''
    try:
        newTagLst=[]
        with Image.open(os.path.join(imagePath,imgName)) as img:
            img1=img.crop(newImgPosition)
            img1=img1.resize((outImgSize,outImgSize))
            savePath=saveImgPath+"/"+str(outImgSize)+"/"+imgPath2
            if  not os.path.exists(savePath):
                os.makedirs(savePath)
            savePath=savePath+"/"+newImgName
            img1.save(savePath)
            newTagLst.append(offset)
            # if len(self.newTagLst)%100==0: # 每100次写一次
            saveTagPath2=saveTagPath+str(outImgSize)+'list_bbox_celeba.txt'
            writeTag(saveTagPath2,newTagLst)
    except Exception as e:
        print("ERROR:",imgName+str(e))

def pltFun(image0,img,imgName):
    (r01,r02,confidence)=image0
    (x01,y01)=r01
    (x02,y02)=r02
    
    dr = ImageDraw.Draw(img)
    (x01,y01)=r01  
    (x02,y02)=r02
    dr.rectangle(((x01,y01),(x02,y02)), fill=None, outline = 0)
    img.save(imgName)
    
def createImage(imgName):
    img = Image.new('RGB', (100, 100), color = (255,255,255))   
    img.save(imgName)
    return img

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.
    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def deviceFun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


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
    newImgName=os.path.join(savePath,imgName)
    newImage.save(newImgName)
    offset=(imgName,offsetX1,offsetY1,offsetX2,offsetY2)
    return offset

def offsetImage(self, image,imgName,imageInfo, targetSize,savePath):
    '''
    1.按比例缩放并填充，先缩放后填充
    2.进行记录标签要更改的偏移量
    '''
    x1=float(imageInfo[1])
    y1=float(imageInfo[2])
    width=float(imageInfo[3])
    height=float(imageInfo[4])

    box=(x1,y1,width,height)
    image=image.crop(box)

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
    newImgName=os.path.join(savePath,imgName)
    newImage.save(newImgName)
    offset=(imgName,offsetX1,offsetY1,offsetX2,offsetY2)
    return offset

if __name__ == "__main__":
    box=[]
    box1=(1,2,3,4,0)
    box2=(2,3,4,5,1)
    box.append(box1)
    box.append(box2)
    nms(box)
    # # 创建图象坐标点
    # image0_x=np.array((0,0))
    # image0_y=np.array((5,5))
    # image1_x=np.array((3,3))
    # image1_y=np.array((10,10))
    # image2_x=np.array((8,8))
    # image2_y=np.array((13,13))
    # image3_x=np.array((15,15))
    # image3_y=np.array((30,30))
    # image4_x=np.array((45,45))
    # image4_y=np.array((66,66))
    # image5_x=np.array((33,33))
    # image5_y=np.array((77,77))
    # image0=(image0_x,image0_y,0.7)
    # image1=(image1_x,image1_y,0.8)
    # image2=(image2_x,image2_y,0.65)
    # image3=(image3_x,image3_y,0.8)
    # image4=(image4_x,image4_y,0.6)
    # image5=(image5_x,image5_y,0.7)
    # # 创建图片
    # imgName="rectangle.png"
    # img=createImage(imgName)
    # #进行画图
    # pltFun(image0,img,imgName)
    # pltFun(image1,img,imgName)
    # pltFun(image2,img,imgName)
    # pltFun(image3,img,imgName)
    # pltFun(image4,img,imgName)
    # pltFun(image5,img,imgName)

    # #显示图片
    # pil_im = Image.open(imgName, 'r')
    # imshow(np.asarray(pil_im))
    # plt.show()

    # #计算IOU
    # print("IOU:",iouFun(image0,image1))
    # print("IOU3:",iouFun(image0,image5))

    # # 创建图片
    # imgName2="rectangle2.png"
    # img2=createImage(imgName2)
    # res=nmsFun(0.8,image0,image1,image2,image3,image4,image5)
    # #进行画图
    # for i in range(len(res)):
    #     image0=(res[i][0],res[i][1],res[i][2])
    #     pltFun(image0,img2,imgName2)
    # #显示图片
    # pil_im2 = Image.open(imgName2, 'r')
    # imshow(np.asarray(pil_im2))
    # plt.show()
