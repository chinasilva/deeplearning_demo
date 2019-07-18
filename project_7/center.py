import torch
import torch.nn as nn


class CenterLoss(torch.nn.Module):

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


class MainNet(torch.nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(784, 120),
            nn.PRule(),
            nn.Linear(120, 2),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(2, 10),
        )

        self.center_loss_layer = CenterLoss(10, 2)
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, xs):
        features = self.hidden_layer(xs)
        outputs = self.output_layer(features)
        return features, outputs

    def getloss(self, outputs, features, labels):
        loss_cls = self.crossEntropy(outputs, labels)
        loss_center = self.center_loss_layer(features, labels)
        loss = loss_cls + loss_center
        return loss
