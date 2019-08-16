import torch
import torch.nn as nn
from utils import device_fun

device=device_fun()
print(device)

w=200
h=80
hidden_size=128
num_layers=4
fout_features=256
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1=nn.Sequential(
            nn.LSTM(input_size=3*w,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        )
        self.fc1=nn.Linear(in_features=hidden_size,out_features=36)
        # self.rnn2=nn.Sequential(
        #     nn.LSTM(input_size=fout_features,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        # )
        # self.fc2=nn.Linear(in_features=hidden_size,out_features=fout_features)
    def forward(self,x):
        # x=x.permute(0,1,3,2)#(n,c,w,h)
        x=x.reshape(-1,h,3*w)#(n*3,s,v)
        batch=x.size(0)
        h0=torch.zeros(num_layers,batch,hidden_size)
        c0=torch.zeros(num_layers,batch,hidden_size)
        xout,(hn,cn)=self.rnn1(x)#,(h0,c0)
        xout=xout[:,-1:,:]
        xout=xout.expand(-1,4,-1)#(N,4,V)
        xout=xout.reshape(-1,hidden_size)
        xout=self.fc1(xout)
        yout=xout.reshape(batch,-1,4)
        # yout,(hn2,cn2)=self.rnn2(xout)
        
        return yout

# def encode(self,x):
    #     pass
    # def decode(self,x):
    #     return x
    # def mainNet(self,x):
    #     xout=self.encode(x)
    #     out=self.decode(xout)
    #     return out
if __name__ == "__main__":
    x=torch.randn(10,3,80,200)
    myNet=MyNet()
    out=myNet(x)
    print(out.shape)