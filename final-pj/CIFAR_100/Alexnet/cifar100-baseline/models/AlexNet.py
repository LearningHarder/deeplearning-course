import torch.nn as nn
class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            ## c1: input:3*32*32
            nn.Conv2d(in_channels=3, out_channels=132, kernel_size=5, stride=2, padding=2), ## 64*16*16
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2), ## 32*16*16
            nn.MaxPool2d(kernel_size=2), ## 64*8*8
            ## c2: input:32*8*8
            nn.Conv2d(in_channels=132, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2), ## 3*32*32
            nn.MaxPool2d(kernel_size=2),
            ## 3*32*32
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ## 3*32*32
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ## 3*32*32
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride =1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            ## 3*32*32
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=(192 * 2 * 2), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 192 * 2 * 2)
        x = self.classifier(x)
        return x