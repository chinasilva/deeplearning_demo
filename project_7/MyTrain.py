import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from MyNet import VGGNet

modelPath='/home/chinasilva/code/deeplearning_homework/project_7/models/net.pth'
datasetsPath='/home/chinasilva/code/deeplearning_homework/project_7/'
def device_fun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    print(device)
    return device

def main():
    epochs = 1000
    batch_size = 6400

    if os.path.exists(modelPath):
        net=torch.load(modelPath)
    else:
        net = VGGNet('VGG11')

    # 定义使用GPU
    device=device_fun()

    #调用cuda
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(datasetsPath, train=True,
                             download=False, transform=transform)
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
        # scheduler.step()
        for j, (input, target) in enumerate(dataloader):
            input=input.to(device)

            features,output = net(input)
            output = output.to(device)
            target=target.to(device)

            loss = net.getloss(outputs=output,features=features,labels=target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                print("[epochs - {0} - {1}/{2}]loss: {3}".format(i,
                                                                 j, len(dataloader), loss.float()))
            if j % 100 == 0:
                losses.append(loss.float())
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
        # losses=list(filter(lambda x: x<1.7,losses)) #过滤部分损失，使图象更直观
        
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

    torch.save(modelPath)

    
    plt.savefig("accuracy_loss.jpg")
    plt.ioff() # 画动态图
    plt.show() # 保留最后一张，程序结束后不关闭


if __name__ == "__main__":
    main()    