import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net,self).__init__()
        self.dnet = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.dnet(x)
        return y

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net,self).__init__()
        self.gnet = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784),
        )

    def forward(self, x):
        y = self.gnet(x)
        return y

if __name__ == '__main__':
    batch_size = 100
    num_epoch = 10

    if not os.path.exists("./gan_img"):
        os.mkdir("./gan_img")
    if not os.path.exists("./gan_params"):
        os.mkdir("./gan_params")

    mnist = datasets.MNIST("./data",
    train=True,transform=transforms.ToTensor(),download=True)

    dataloader = DataLoader(mnist,batch_size=batch_size,shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)

    d_net.state_dict(torch.load("./gan_params/d_params.pth"))
    g_net.state_dict(torch.load("./gan_params/g_params.pth"))

    loss_fn = nn.BCELoss()

    d_opt = torch.optim.Adam(d_net.parameters())
    g_opt = torch.optim.Adam(g_net.parameters())

    for epoch in range(num_epoch):
        for i,(img,label) in enumerate(dataloader):
            real_img = img.reshape(img.size(0),-1).to(device)
            real_label = torch.ones(img.size(0),1).to(device)
            fake_label = torch.zeros(img.size(0),1).to(device)

            '''训练判别器'''
            real_out = d_net(real_img)
            d_loss_real = loss_fn(real_out,real_label)#把真实图片判别为真、1
            real_scores = real_out

            z = torch.randn(img.size(0),128).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_loss_fake = loss_fn(fake_out,fake_label)#把假图片判别为假、0
            fake_scores = fake_out

            d_loss = d_loss_fake+d_loss_real
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            '''训练生成器'''
            z = torch.randn(img.size(0), 128).to(device)
            fake_img = g_net(z)
            output = d_net(fake_img)
            g_loss = loss_fn(output,real_label)#把假图片的分数学习为真图片的分数

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i%10 == 0:
                print("Epoch:[{}/{}],d_loss:{:.3f},"
                "g_loss:{:.3f},d_real:{:.3f},d_fake:{:.3f}"
                .format(epoch,num_epoch,d_loss,g_loss,real_scores.data.mean(),fake_scores.data.mean()))

                fake_img = fake_img.cpu().data.reshape([-1,1,28,28])
                real_img = real_img.cpu().data.reshape([-1,1,28,28])
                save_image(fake_img, "./gan_img/{}-fkae_img.png"
                           .format(epoch + 1), nrow=10, normalize=True, scale_each=True)
                save_image(real_img, "./gan_img/{}-real_img.png"
                           .format(epoch + 1), nrow=10, normalize=True, scale_each=True)

                torch.save(d_net.state_dict(), "./gan_params/d_params.pth")
                torch.save(g_net.state_dict(), "./gan_params/g_params.pth")