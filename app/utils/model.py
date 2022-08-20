import torch 
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(3))
    return nn.Sequential(*layers)


# Model Architecture

class Mymodel(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64, pool=True)
        self.conv3 = ConvBlock(64, 64) 
        self.conv4 = ConvBlock(64, 128, pool=True) 
        self.conv5 = ConvBlock(128, 128) 
        self.conv6 = ConvBlock(128, 256, pool=True) 
        self.conv7 = ConvBlock(256, 256) 
        self.conv8 = ConvBlock(256, 512, pool=True) 
        self.conv9 = ConvBlock(512, 512) 
        self.conv10 = ConvBlock(512, 1568) 
        self.classifier = nn.Sequential(nn.MaxPool2d(3),
                                       nn.Flatten(),
                                       nn.Linear(1568,1568),nn.Dropout(p=0.5),nn.Linear(1568,num_diseases))
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.classifier(out)
        return out