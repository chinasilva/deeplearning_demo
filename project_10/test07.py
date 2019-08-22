import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from test05 import Enconder
from test06 import Deconder
import matplotlib.pyplot as plt
trans = transforms.Compose([
    transforms.ToTensor()
])

mnist_data = datasets.MNIST("data",train=True,transform=trans,download=True)
train_data = DataLoader(mnist_data,batch_size=100,shuffle=True)

en_net = Enconder()
de_net = Deconder()

loss_fn = nn.MSELoss()
en_optimizer = torch.optim.Adam(en_net.parameters())
de_optimizer = torch.optim.Adam(de_net.parameters())

for epoch in range(10):
    for i ,(img,label) in enumerate(train_data):
        feature=en_net(img)
        out_img = de_net(feature)
        loss = loss_fn(out_img,img)
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        loss.backward()
        en_optimizer.step()
        de_optimizer.step()

        if i%10 == 0:
            print("Epoch:[{}/{}],loss:{:.3f}".format(epoch,10,loss.item()))
            img = out_img.permute([0,2,3,1])
            plt.imshow(img.data[0].reshape(28,28))
            plt.pause(0.1)