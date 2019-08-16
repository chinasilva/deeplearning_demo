import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import device_fun

device=device_fun()
print(device)
class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1=nn.Sequential(
            nn.Linear(in_features=28*28,out_features=32*32)
        )
        self.rnn=nn.LSTM(input_size=32,hidden_size=64,num_layers=2,batch_first=True)#,bidirectional=True
        self.fc2=nn.Sequential(
            nn.Linear(in_features=64,out_features=10)
        )

    def forward(self, x):
        x=self.fc1(x.reshape(-1,28*28))
        x=x.reshape(-1,32,32)#(n*c*h,w)->(n,s=32(c*h),v=32)
        # print("x:",x.shape)
        batch=x.size(0)
        h0=torch.zeros(2,batch,64).to(device)#(num_layers * num_directions, batch, hidden_size)
        c0=torch.zeros(2,batch,64).to(device)
        xout,(hn,cn)=self.rnn(x,(h0,c0))#(n,s,v)
        # print("xout:",xout.shape)
        output=xout[:,-1,:]#(N,S,V)get last map info,S=(c*h),cut h
        # print("output:",output.shape)
        output=self.fc2(output)
        # print("output:",output.shape)
        return output

