import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGNet(nn.Module):
    
    def __init__(self,vgg_name):
        super().__init__()
        self.layers = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(10, 2)
        self.classifier2 = nn.Linear(2, 10)
        self.center_loss_layer = CenterLoss(10, 2)
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        features = self.classifier1(out)
        outputs = self.classifier2(features) 
        return features,outputs
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def getloss(self, outputs, features, labels):
        loss_cls = self.crossEntropy(outputs, labels)
        loss_center = self.center_loss_layer(features, labels)
        loss = loss_cls + loss_center
        return loss

class CenterLoss(nn.Module):

    def __init__(self, cls_num, feature_num):
        super(CenterLoss, self).__init__()

        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num, feature_num))

    def forward(self, xs, ys):
        xs = torch.nn.functional.normalize(xs)
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)
        count_dis = count.index_select(dim=0, index=ys.long())
        return torch.sum(torch.sqrt(torch.sum((xs - center_exp) ** 2, dim=1)) / count_dis)



