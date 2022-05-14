import torch
import torch.nn as nn


class AlexNet(nn.Module):


    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            #input:3*32*32
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            #input:96*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            #input:96*8*8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1,padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            #input:256*8*8
            nn.MaxPool2d(kernel_size=2, stride=2),
            #input:256*4*4
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #input:384*4*4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #input:384*4*4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #input:384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2)
            #output:384*2*2
        )
 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(384 * 2 * 2), out_features=1536),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1536, out_features=1536),
            nn.ReLU(),
            nn.Linear(in_features=1536, out_features=100),
        )


    def forward(self, inputs):
        x = self.features(inputs)
        x = x.view(-1, 384 * 2 * 2)
        x = self.classifier(x)
        return x