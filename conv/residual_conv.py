import torch
import torch.nn as nn

class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)
