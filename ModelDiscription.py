import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from consts import *

def printshape(title, x):
  print(title, x.shape)

class MuxNetwork(nn.Module):
    def __init__(self):
        super(MuxNetwork, self).__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 3),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels = 12, out_channels = 28, kernel_size = 5),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels = 28, out_channels = 32, kernel_size = 4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fc_layer_1 = nn.Sequential(
            nn.Linear(in_features=3200, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.fc_layer_2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc_layer_3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=2, bias=True),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x1, x2):
        # x1 shape = batch(32) * 3 * PATCH_SIZE * PATCH_SIZE
        # x2 shape = batch(32) * 3 * PATCH_SIZE * PATCH_SIZE
        
        x1 = x1.cuda()
        x1 = self.conv_layer_1(x1)
        x1 = self.conv_layer_2(x1)
        x1 = self.conv_layer_3(x1)
        x1 = self.conv_layer_4(x1)

        x2 = x2.cuda()
        x2 = self.conv_layer_1(x2)
        x2 = self.conv_layer_2(x2)
        x2 = self.conv_layer_3(x2)
        x2 = self.conv_layer_4(x2)

        x1 = x1.view(x1.size(0), -1) 
        x2 = x2.view(x2.size(0), -1) 
        
        xc = torch.cat([x1, x2], dim=1)
        xc = xc.cuda()

        xc = self.fc_layer_1(xc)
        xc = self.fc_layer_2(xc)
        xc = self.fc_layer_3(xc)
        
        return xc