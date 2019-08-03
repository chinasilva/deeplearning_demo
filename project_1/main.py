import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from MyMnistNet import MyMnistNet

def device_fun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    print(device)
    return device

def main():
    epochs = 100
    batch_size = 6400
    # in_features=10
    # nb_classes=10

    net = MyMnistNet()
    criterion=nn.MSELoss()  


    optimizer = optim.Adam(net.parameters(), weight_decay=0,
                           amsgrad=False, lr=0.01, betas=(0.9, 0.999), eps=1e-08)

    # 数据构造
    # unsqueeze函数可以将一维数据变成二维数据，在torch中只能处理二维数据
    # 数据点按顺序排序，否则会出现数据点散乱
    # x_init=torch.Tensor(np.random.random(30))
    # x_init=torch.Tensor(np.sort(np.random.random(1000)))


    x_init=torch.Tensor(np.sort(np.random.random(1000)*1000)) 
    x_init=torch.Tensor(np.array(x_init)/1000 - 0.5)


    x= torch.unsqueeze(x_init,dim=1)
    # 拟合任意点
    # y= torch.unsqueeze(torch.Tensor(np.random.random(10)),dim=1)
    # 拟合任意方程组
    y= torch.unsqueeze(torch.Tensor(0.5*x_init**5+3*x_init**4+0.13*x_init**2+0.05*x_init+3),dim=1)
    # print("x___",type(x))
    losses=[]
    x_list=[]
    y_list=[]

    for p in range(1000):
        print(str.format("批次:{}",p))
        # for input, target  in zip(list(x),list(y)):
            # input=input.to(device)
        input=x
        net = net #.float()
        output = net(input)  
        output=output 
        # target=target.to(device)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)
        losses=list(filter(lambda x: x<2,losses)) #过滤部分损失，使图象更直观
        
        x_list.append(input)
        y_list.append(output)
        #画点
        plt.clf()
        plt.subplot(2, 1, 1)
        
        plt.scatter(x,y,c=u'g',marker=u'*')
        #参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        # ax.plot(x.numpy(), output.data.numpy(), color='b', linewidth=1, alpha=0.6)
        # 没有误差点
        plt.plot(x.numpy(), y.numpy(), color='b', linewidth=1, alpha=0.6)
        # 实时训练点
        plt.plot(x.numpy(), output.data.numpy(), color='r', linewidth=1, alpha=0.6)
        # plt.pause(0.1)
        plt.ylabel('Test plot')

        plt.subplot(2, 1, 2)
        plt.plot(losses)
        plt.ylabel('Test losses')
        plt.pause(0.1)

    plt.show()

if __name__ == "__main__":
    main()