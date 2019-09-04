import torch
import torch.nn as nn
import os
from MyNet import D_Net,G_Net
from torchvision.utils import save_image
from MyData import MyData
from cfg import *
from torch.utils import data

class MyTrain():
    def __init__(self):
        self.myData=MyData(IMG_PATH)
        self.device=deviceFun()
        
    def main(self):
        if not os.path.exists(DCGAN_IMG):
            os.mkdir(DCGAN_IMG)
        if not os.path.exists(DCGAN_MODEL):
            os.mkdir(DCGAN_MODEL)

        dataloader = data.DataLoader(self.myData,batch_size=BATCH_SIZE,shuffle=True)

        d_net = D_Net(12,3).to(self.device)
        g_net = G_Net(1,12,3).to(self.device)

        if os.path.exists(f"{DCGAN_MODEL}/d_params.pth"):
            d_net.load_state_dict(torch.load(f"{DCGAN_MODEL}/d_params.pth"),strict=False)
            g_net.load_state_dict(torch.load(f"{DCGAN_MODEL}/g_params.pth"),strict=False)

        loss_fn = nn.BCELoss()

        d_opt = torch.optim.Adam(d_net.parameters(),betas=(0.5,0.999))
        # d_opt = torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))
        g_opt = torch.optim.Adam(g_net.parameters(),betas=(0.5,0.999))
        # g_opt = torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))

        for epoch in range(NUM_EPOCH):
            for i,(img,label) in enumerate(dataloader):
                real_img = img.to(self.device)
                real_label = torch.ones(img.size(0),1,3,3).to(self.device)
                fake_label = torch.zeros(img.size(0),1,3,3).to(self.device)

                '''训练判别器'''
                real_out = d_net(real_img)
                d_loss_real = loss_fn(real_out,real_label)#把真实图片判别为真、1
                real_scores = real_out

                z = torch.randn(img.size(0),1,3,3).to(self.device)
                fake_img = g_net(z)
                fake_out = d_net(fake_img)
                d_loss_fake = loss_fn(fake_out,fake_label)#把假图片判别为假、0
                fake_scores = fake_out

                d_loss = d_loss_fake+d_loss_real
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                '''训练生成器'''
                z = torch.randn(img.size(0), 1,3,3).to(self.device)
                fake_img = g_net(z)
                output = d_net(fake_img)
                g_loss = loss_fn(output,real_label)#把假图片的分数学习为真图片的分数

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                if i%5 == 0:
                    print("Epoch:[{}/{}],d_loss:{:.3f},"
                    "g_loss:{:.3f},d_real:{:.3f},d_fake:{:.3f}"
                    .format(epoch,NUM_EPOCH,d_loss,g_loss,real_scores.data.mean(),fake_scores.data.mean()))

                    fake_img = fake_img.cpu().data
                    real_img = real_img.cpu().data 
                    save_image(fake_img, "{}/{}-fkae_img.png"
                            .format(DCGAN_IMG,epoch + 1), nrow=10, normalize=True, scale_each=True)
                    save_image(real_img, "{}/{}-real_img.png"
                            .format(DCGAN_IMG,epoch + 1), nrow=10, normalize=True, scale_each=True)

                    torch.save(d_net.state_dict(), f"{DCGAN_MODEL}/d_params.pth")
                    torch.save(g_net.state_dict(), f"{DCGAN_MODEL}/g_params.pth")

def deviceFun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device

if __name__ == "__main__":
    myTrain=MyTrain()    
    myTrain.main()