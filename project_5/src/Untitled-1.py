# 踩坑经历：
# 1.在求损失之前，输出不能够进行类型变换，如:output=output[:].data,这样会导致反向求导中没有梯度
# 2.在求交叉熵损失的时候，输出与目标参数位置不能变。
# 3.在进行GPU训练时需要对输入，网络(初始化)，输出等进行数据传入GPU

import torch
import numpy as np

from MyNet import PNet,RNet,ONet
from utils import nms,createImage,pltFun,deviceFun

# test= torch.Tensor(3,3,100,100)
# test2= torch.Tensor(3,3,100,100)
# lst=[test,test2,test2]
# print(torch.stack(lst,dim=4).size())

a= torch.range(1,10)
a=a.reshape(2,5)

print(a[1]-a[0])
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