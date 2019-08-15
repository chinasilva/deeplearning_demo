import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1=nn.Sequential(
            nn.Linear(in_features=28*28,out_features=32*32)
        )
        self.rnn=nn.LSTM(input_size=32,hidden_size=32,num_layers=1,batch_first=True)
        

    def forward(self, x):
        x=self.fc1(x)
        x=x.permute(0,1,3,2)#(n,c,w,h) cut h
        x=x.reshape(-1,32)#(n*c*w,h)
        x=x.reshape(-1,1,32)#(n*c*w,h)
        batch=x.size(0)
        h0=torch.zeros(1,batch,32)#(num_layers * num_directions, batch, hidden_size)
        c0=torch.zeros(1,batch,32)
        xout,(hn,cn)=self.rnn(x,(h0,c0))#(n,s,v)
        # output=xout[]
        print(xout.size())
        return xout
