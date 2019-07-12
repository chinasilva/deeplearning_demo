
import numpy as np
import torch
# def oneHot(clsNum,v):
#     a=np.zeros(clsNum)
#     a[v]=1
#     return a

# def oneHot2(clsNum,v):
#     dic={}
#     a=np.zeros(clsNum)
#     b=np.zeros(clsNum)
#     a[v]=1
#     b[v+1]=1
#     dic[0]=a
#     dic[1]=b
#     return dic

# print(*oneHot(10,2))


# a=torch.arange(0,27)
# a=torch.arange(1,28)
# a=a.reshape(3,3,3)
a=torch.arange(0,16)
a=a.reshape(2,2,2,2)
print("0",a)
print(a.nonzero()) 

# a2=a[...,0]
# print("0-1",a2)
# b=a[...,0]>5
# print("1:",b)
# c=a[b]
# print("2",c)