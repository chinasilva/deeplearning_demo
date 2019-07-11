
import numpy as np

def oneHot(clsNum,v):
    a=np.zeros(clsNum)
    a[v]=1
    return a

def oneHot2(clsNum,v):
    dic={}
    a=np.zeros(clsNum)
    b=np.zeros(clsNum)
    a[v]=1
    b[v+1]=1
    dic[0]=a
    dic[1]=b
    return dic

print(*oneHot(10,2))