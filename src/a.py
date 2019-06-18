import torch
import torch.nn as nn
# print(torch.__version__)

# device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
# print(device)

m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
print(output)