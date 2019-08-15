import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.MyNet import MyNet

def device_fun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    print(device)
    return device

def main():
    epochs = 100
    batch_size = 6400
    # in_features=10
    # nb_classes=10

    net = MyNet()
    criterion=nn.CrossEntropyLoss()

    # 定义使用GPU
    device=device_fun()
    print(device)

    #调用cuda
    net.to(device)
    # criterion.to(device)

    # criterion = nn.MSELoss(reduce=None, size_average=None, reduction='mean')
    optimizer = optim.Adam(net.parameters(), weight_decay=0,
                           amsgrad=False, lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST("datasets/", train=True,
                             download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    testdataset = datasets.MNIST(
        "datasets/", train=False, download=True, transform=transform)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=False)
 
    losses = []
    plt.ion() # 画动态图
    for i in range(epochs):
        print("epochs: {}".format(i))
        scheduler.step()
        for j, (input, target) in enumerate(dataloader):
            # if iscuda

            input=input.to(device)


            output = net(input)
            # output=F.softmax(output, dim=1)
            # output=F.log_softmax(output, dim=1) # log_softmax 输出激活
            output = output.to(device)

            # # MSE 需要进行转化成one-hot编码形式，交叉熵则不需要进行转换，内置函数进行转换
            # target = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 1)

            target=target.to(device)


            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                print("[epochs - {0} - {1}/{2}]loss: {3}".format(i,
                                                                 j, len(dataloader), loss.float()))
            if j % 100 == 0:
                losses.append(loss.float())
        # print("--------------------------------0",losses)
        accuracyLst=[]
        with torch.no_grad():
            correct = 0
            total = 0
            for k,(input, target) in enumerate(testdataloader):
                input=input.to(device)    # GPU 
                target=target.to(device) # GPU

                output = net(input)
                

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                accuracy = correct.float() / total
                if k % 10 == 0:
                    accuracyLst.append(accuracy)
            print(
                "[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
        
        # plt.clf()#清空内容
        losses=list(filter(lambda x: x<1.7,losses)) #过滤部分损失，使图象更直观
        
        x=range(len(losses)*(i),len(losses)*(i+1))
        plt.subplot(2, 1, 1)
        plt.plot(x,losses)
        plt.pause(0.5)
        plt.ylabel('Test losses')

        x2=range(len(accuracyLst)*(i),len(accuracyLst)*(i+1))
        # x2=range(0,len(accuracyLst))
        plt.subplot(2, 1, 2)
        plt.plot(x2,accuracyLst)
        plt.pause(0.5)
        plt.ylabel('Test accuracy')

        accuracyLst=[]
        losses=[]


    save_model = torch.jit.trace(net,  torch.rand(1, 1, 28, 28).to(device))
    save_model.save("models/net.pth")

    
    plt.savefig("accuracy_loss.jpg")
    plt.ioff() # 画动态图
    plt.show() # 保留最后一张，程序结束后不关闭