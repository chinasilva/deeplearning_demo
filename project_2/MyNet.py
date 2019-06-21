import torch
import torch.nn as nn
cfg = {
    'MYVGG': [64, 'M', 128, 'M', 256, 'M', 128, 64],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3, padding=1,groups=3),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=96,out_channels=246,kernel_size=3, padding=1,groups=3),
            nn.BatchNorm2d(num_features=246),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=246,out_channels=15,kernel_size=3, padding=1,groups=3),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            #通道融合
            # nn.Conv2d(in_channels=96,out_channels=15,kernel_size=3, padding=1),

        )
        self.line=nn.Sequential( 
            nn.Linear(in_features=2160,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=4),
        )
            # nn.Linear(in_features=100*100*3,out_features=2048),
            # nn.Linear(in_features=2048,out_features=512),

    def forward(self,input):
        output=self.model(input)
        output = output.view(output.size(0), -1)
        return self.line(output)


class VGGNet(nn.Module):

	def __init__(self,vgg_name):
		super().__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.classifier = nn.Linear(64, 4)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

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


