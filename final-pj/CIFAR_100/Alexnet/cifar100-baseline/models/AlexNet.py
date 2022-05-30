import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            ## c1: input:3*32*32
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=2), ## 96*16*16
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2), ## 96*16*16
            nn.MaxPool2d(kernel_size=2), ## 96*8*8
            ## c2: input:96*8*8
            nn.Conv2d(in_channels=96, out_channels=228, kernel_size=5, stride=1, padding=2), ## 228*16*16
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2), ## ## 228*8*8
            nn.MaxPool2d(kernel_size=2), ## 228*4*4
            ## 228*4*4
            nn.Conv2d(in_channels=228, out_channels=784, kernel_size=3, padding=1), ## 784*4*4
            nn.ReLU(inplace=True),
            ## 784*4*4
            nn.Conv2d(in_channels=784, out_channels=256, kernel_size=3, padding=1), ## 256*4*4
            nn.ReLU(inplace=True),
            ## 256*4*4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride =1, padding=1), ## 256*4*4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            ## 256*2*2
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=(256 * 2 * 2), out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x