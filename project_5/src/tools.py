import os
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer. 
    
    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """
    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side
    
    def forward(self, x):
        out = None
        for n in self.out_side:
            w_r, h_r = map(lambda s: math.ceil(s/n), x.size()[2:])    # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s/n), x.size()[2:])   # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out

def convertToPosition(originPosition):
    '''
    根据原图坐标进行最短边补齐
    '''
    newImg=originPosition.copy()
    if len(originPosition) == 0:
        return []
    originImgW=originPosition[:,2]-originPosition[:,0]
    originImgH=originPosition[:,3]-originPosition[:,1]
    maxSide=np.maximum(originImgW,originImgH) #获取最长边max(originImgW,originImgH)
    #按照最长边进行抠图，短边进行补全
    newImg[:,0]=originPosition[:,0]+originImgW*0.5-maxSide*0.5
    newImg[:,1]=originPosition[:,1]+originImgH*0.5-maxSide*0.5
    newImg[:,2]=newImg[:,0]+maxSide 
    newImg[:,3]=newImg[:,1]+maxSide 
    newImg[:,4]=originPosition[:,4]
    return newImg

if __name__ == "__main__":
    testImagePath=r'/home/chinasilva/code/deeplearning_homework/project_5/images_val/mytest'
    dataset=[]
    dataset.extend(os.listdir(testImagePath))
    for i,imgName in enumerate(dataset):
        with Image.open(os.path.join(testImagePath,imgName)).convert('RGB') as img:
            # img=cv2.imread(testImagePath+'/'+imgName)
            # originPosition=np.arrage
            # w=img2.shape[0]
            # h=img2.shape[1]
            # convertToPosition()
            # img.size()
            # spp=SpatialPyramidPool2D(out_side=(3,100,100))
            img2=img.resize((100,100))
            img.show()
            img2.show()
            m = nn.AdaptiveAvgPool2d(100)
            img3=torch.from_numpy(np.expand_dims(np.asarray(img).transpose(2,0,1), axis=0)) # (n,c,h,w)
            input = torch.randn(1, 64, 10, 9)
            img3=m(img3)
            img3=img3.numpy()
            img3=Image.fromarray(np.uint8(img3))
            img3.show()
            a=input("test!!!!!!!!!!1")
            # cv2.imshow('image',img)
            # cv2.imshow('image2',img2)
