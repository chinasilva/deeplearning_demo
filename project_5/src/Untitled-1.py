# 踩坑经历：
# 1.在求损失之前，输出不能够进行类型变换，如:output=output[:].data,这样会导致反向求导中没有梯度
# 2.在求交叉熵损失的时候，输出与目标参数位置不能变。
# 3.在进行GPU训练时需要对输入，网络(初始化)，输出等进行数据传入GPU

import torch
from MyNet import PNet,RNet,ONet

test= torch.Tensor(3,3,100,100)
# a= torch.range(1,3)
net=PNet()
a,b,c=net(test)
a=a.reshape(3,1,1,1)
# a=a[:,:,:,0].reshape(-1,1)
a=a.reshape(-1,1) #.squeeze(2).squeeze(2)
print(a)
print(a.size())
