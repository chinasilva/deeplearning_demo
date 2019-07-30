# 踩坑经历：
# 1.在求损失之前，输出不能够进行类型变换，如:output=output[:].data,这样会导致反向求导中没有梯度
# 2.在求交叉熵损失的时候，输出与目标参数位置不能变。
# 3.在进行GPU训练时需要对输入，网络(初始化)，输出等进行数据传入GPU

import torch
import numpy as np
import os
import cv2
from PIL import Image,ImageDraw
import torch.nn.functional as F
import torch.nn as nn
import math
from MyNet import PNet,RNet,ONet
from utils import nms,createImage,pltFun,deviceFun

# test= torch.Tensor(3,3,100,100)
# test2= torch.Tensor(3,3,100,100)
# lst=[test,test2,test2]
# print(torch.stack(lst,dim=4).size())

# a= torch.range(11,21)
# output=a.reshape(11,1)
# outLen=int((output.size()[0]*0.7))
# output2=output[:][torch.argsort(output[:,0],dim=0,descending=True)]
# output2=torch.mean(output2[0:outLen+1])

# print(output2)

# print(torch.where(output==1,torch.ones_like(output),torch.zeros_like(output)).sum())
# # print(a[1]-a[0])
# # a1,b1=torch.max(output, 1)

# output=[x for x in output if x > 5]
# print("output",output)
# print("a,b",a1,b1)
# b=a[:,:1]
# print("---",b) 
# # net=PNet()
# # a,b,c=net(test)
# # a=a.reshape(3,1,1,1)
# # # a=a[:,:,:,0].reshape(-1,1)
# # a=a.reshape(-1,1) #.squeeze(2).squeeze(2)
# print(a)
# print(a.size())


# PLst=np.array([
# (188,188,201,198,0.9496198892593384),
# (189,189,201,198,0.9496198892593384)
# ])

# a=PLst[nms(PLst,overlap_threshold=0.7,mode='union')]
# print(a[0][0])

# print(np.arange(0,10))

# nx,ny = (2,2)
# #从0开始到1结束，返回一个numpy数组,nx代表数组中元素的个数
# x = np.linspace(0,1,nx)
# #[ 0.   0.5  1. ]
# y = np.linspace(0,1,ny)
# # [0.  1.]
# xv,yv = np.meshgrid(x,y)
# print("xv",xv)
# print("yv",yv)
# test=torch.rand(3,3)
# print("test:",test)
# print("test2:",torch.gt(test,0.6))


if __name__ == "__main__":
    root='/home/chinasilva/code/deeplearning_homework/project_5/images_val/mytest'
    testImagePath=root+'/img'
    dataset=[]
    dataset.extend(os.listdir(testImagePath))
    m = nn.AdaptiveAvgPool2d((48,48))

    for i,imgName in enumerate(dataset):
        with Image.open(os.path.join(testImagePath,imgName)).convert('RGB') as img:
            # img=cv2.imread(testImagePath+'/'+imgName)
            # originPosition=np.arrage
            # w=img2.shape[0]
            # h=img2.shape[1]
            # convertToPosition()
            # img.size()
            # spp=SpatialPyramidPool2D(out_side=(3,100,100))
            # img3=img.crop((440,357,582,468))
            # img3=Image.fromarray(imgNumpy)
            # img4=img3.resize((48,48))
            # img3.save(root+'/result/a.jpg')
            # img4.save(root+'/result/b.jpg')
            a=[[357,468,440,551],
            [357,468,440,551],
            [357,468,440,551],
            [357,468,440,551],
            [357,468,440,551]]
            # b=[359:470,442:553]
            

            # imgNumpy=np.asarray(img)[357:468,440:551]
            # imgTorch=(torch.from_numpy(imgNumpy).permute(2,0,1).unsqueeze(dim=0)).float() #（N,C,H,W）
            # imgTorch2=m(imgTorch).squeeze(dim=0).permute(1,2,0)
            # imgNumpy2=np.uint8(imgTorch2)
            # img5=Image.fromarray(imgNumpy2)
            # img5.save(root+'/result/c.jpg')

            imgs=np.repeat(np.expand_dims(np.asarray(img),axis=0),repeats=5,axis=0)#获取图片数据(N,W,H,C)
            # [PLst2.astype(int)[:,2]:PLst2.astype(int)[:,3],PLst2.astype(int)[:,0]:PLst2.astype(int)[:,1]]
            imgNumpy=np.asarray(imgs)[a]  # like crop 
            imgTorch=(torch.from_numpy(imgNumpy).permute(0,3,2,1)).float() #（N,C,H,W）
            # outLst=adaptiveAvgPool(imgTorch)-0.5 # like resize use
            # outLst2=[x1,y1,x2,y2]

            # img2=img.resize((100,100))
            # img.show()
            # img2.show()
            # m = nn.AdaptiveAvgPool2d(100)
            # img3=torch.from_numpy(np.expand_dims(np.asarray(img).transpose(2,0,1), axis=0)) # (n,c,h,w)
            # input = torch.randn(1, 64, 10, 9)
            # img3=m(img3)
            # img3=img3.numpy()
            # img3=Image.fromarray(np.uint8(img3))
            # img3.show()
            # a=input("test!!!!!!!!!!1")
            # cv2.imshow('image',img)
            # cv2.imshow('image2',img2)
