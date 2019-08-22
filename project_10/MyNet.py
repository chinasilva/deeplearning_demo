import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,128,3,2,1),
            nn.ReLU()
        )#1*128*5
        self.conv2 = nn.Sequential(
            nn.Conv1d(128,512,3,2,1),
            nn.ReLU()
        )#1*512*3
        self.conv3 = nn.Sequential(
            nn.Conv1d(512,1024,3,2,1),
            nn.ReLU()
        )#1*1024*2
        self.conv4 = nn.Sequential(
            nn.Conv1d(1024,512,3,2,1),
            nn.ReLU()
        )#1*512*1
        self.conv5 = nn.Sequential(
            nn.Conv1d(512,128,1,1,0),
            nn.ReLU()
        )#1*128*1
        self.linear = nn.Sequential(
            nn.Linear(1*128*1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y5 = y5.reshape(y5.size(0),-1)
        y6 = self.linear(y5)
        return y6
