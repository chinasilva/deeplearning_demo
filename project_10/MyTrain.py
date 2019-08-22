import torch
import torch.nn as nn
import numpy as np
from MyData import get_data
from MyNet import Net
import os
from torch.utils.tensorboard import SummaryWriter


path = r"/home/chinasilva/code/signProcess/data1.xlsx"
modelPath="/home/chinasilva/code/signProcess/model/params.pth"
write="/home/chinasilva/code/signProcess/runs/MyNet"
net = Net()
if os.path.exists(modelPath):
    net.load_state_dict(torch.load(modelPath))

writer = SummaryWriter(log_dir=write)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters())
datas = get_data.red_excel(path)
max_data = np.max(datas)
train_data = np.array(datas)/max_data
print(train_data)

for epoch in range(10000):
    for i in range(len(train_data)-9):
        x = train_data[i:i+9]
        y = train_data[i+9:i+10]
        xs = torch.reshape(torch.tensor(x,dtype=torch.float32),[-1,1,9])
        ys = torch.tensor(y,dtype=torch.float32)
        _y = net(xs)
        loss = loss_fn(_y,ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        out = int(_y*max_data)
        label = int(ys*max_data)
        writer.add_scalar("test Loss",loss,global_step=epoch)
        if i%10==0:
            print(f"loss:{loss.item()},\n out{out},\n label{label},\n {i} /{epoch}")

        # print(i,"/",epoch)
    torch.save(net.state_dict(),modelPath)