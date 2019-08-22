'''
LayerNorm
InstanceNorm2d
BatchNorm2d
运算方式
'''
import torch
import torch.nn as nn
'''
方差计算
http://www.ab126.com/shuxue/1837.html
'''
# myrange=torch.range(1,50)
# print("myrange:",myrange)
# newrange= myrange.reshape((1,2,5,5)) 
# print("newrange:",newrange)
# # ## 所有层进行平均和标准差
# myLayNorm= nn.LayerNorm((1,2,5,5))
# print("myLayNorm:",myLayNorm(newrange))
# ## 在当前层进行求平均和标准差
# myIntanceNorm=nn.InstanceNorm2d(2)
# print("myIntanceNorm:",myIntanceNorm(newrange))

# # ##
# batchrange=torch.range(1,12)
# # batchrange=torch.randn([1,50])
# print("batchrange:",batchrange)
# newBatchrange= batchrange.reshape((1,3,2,2)) 
# print("newBatchrange:",newBatchrange)

# myBatchNorm=nn.BatchNorm2d(3)
# print("myBatchNorm:",myBatchNorm(newBatchrange))
