
'''
https://github.com/oeway/pytorch-deform-conv/blob/master/torch_deform_conv/cnn.py
'''

import torch
import torch.nn as nn
class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

class DeformConvNet(nn.Module):
    def __init__(self):
        super(DeformConvNet, self).__init__()
        
        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.offset12 = ConvOffset2D(32)
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.offset21 = ConvOffset2D(64)
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.offset22 = ConvOffset2D(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)
        
        x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)
        
        x = self.offset21(x)
        x = F.relu(self.conv21(x))
        x = self.bn21(x)
        
        x = self.offset22(x)
        x = F.relu(self.conv22(x))
        x = self.bn22(x)
        
        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))
        x = F.softmax(x)
        return x