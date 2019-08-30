import torch
import torch.nn as nn


class depthwise_separable_conv(nn.Module):
'''
img input:DF*DF*M,output:DF*DF*N,kernel:DK*DK
origin conv2d:DK*DK*M*N*DF*DF
depthwise separable conv2d :DK*DK*M*DF*DF+M*N*DF*DF
compare:(DK*DK*M*DF*DF+M*N*DF*DF)/(DK*DK*M*N*DF*DF)==(1/N)+(1/DK**2)
'''
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out