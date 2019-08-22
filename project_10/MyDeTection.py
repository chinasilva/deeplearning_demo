import torch
import torch.nn as nn
import numpy as np
from MyData import get_data
from MyNet import Net
import matplotlib.pyplot as plt
path = r"/home/chinasilva/code/signProcess/data1.xlsx"
modelPath="/home/chinasilva/code/signProcess/model/params.pth"
net = Net()
net.load_state_dict(torch.load(modelPath))
loss_fn = nn.BCELoss()
# optimizer = torch.optim.Adam(net.parameters())
datas = get_data.red_excel(path)
max_data = np.max(datas)
train_data = np.array(datas)/max_data
print(train_data)
plt.ion()
a = []
b = []
c = []
for i in range(len(train_data)-9):
    x = train_data[i:i+9]
    y = train_data[i+9:i+10]
    xs = torch.reshape(torch.tensor(x,dtype=torch.float32),[-1,1,9])
    ys = torch.tensor(y,dtype=torch.float32)
    _y = net(xs)
    loss = loss_fn(_y,ys)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    print(loss.item())
    out = int(_y*max_data)
    label = int(ys*max_data)
    print(out)
    print(label)
    print(i)
    a.append(i)
    b.append(label)
    c.append(out)
    plt.plot(a,b,linewidth = 1,color = "red")
    plt.plot(a,c,linewidth = 1,color = "blue")
    plt.pause(0.1)
