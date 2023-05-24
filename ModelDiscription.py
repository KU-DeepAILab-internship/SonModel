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
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 0)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)
        
    def forward(self, x1, x2):
        # x1 shape = batch(32) * 3 * 35 * 35
        # x2 shape = batch(32) * 3 * 35 * 35
        
        # printshape("x1", x1)

        x1 = self.conv1(x1)
        # printshape("x1_conv1", x1)

        x1 = F.relu(x1)
        x1 = self.pool(x1)
        # printshape("x1_pool", x1)
        
        x1 = self.conv2(x1)
        # printshape("x1_conv2", x1)
        
        x1 = F.relu(x1)
        x1 = self.pool(x1)
        # printshape("x1_pool", x1)

        x1 = self.conv3(x1)
        # printshape("x1_conv3", x1)
        
        x1 = F.relu(x1)
        x1 = self.pool(x1)
        # printshape("x1_pool", x1)

        # printshape("x2", x2)
        x2 = self.conv1(x2)
        # printshape("x2_conv1", x2)

        x2 = F.relu(x2)
        x2 = self.pool(x2)
        # printshape("x2_pool", x2)

        x2 = self.conv2(x2)
        # printshape("x2_conv2", x2)

        x2 = F.relu(x2)
        x2 = self.pool(x2)
        # printshape("x2_pool", x2)

        x2 = self.conv3(x2)
        # printshape("x2_conv3", x2)

        x2 = F.relu(x2)
        x2 = self.pool(x2)
        # printshape("x2_pool", x2)

        x1 = x1.view(x1.size(0), -1) 
        x2 = x2.view(x2.size(0), -1) 
        
        xc = torch.cat([x1, x2], dim=1)
        # printshape("xc", xc)

        xc = self.fc1(xc)
        # printshape("xc_fc1", xc)

        xc = F.relu(xc)
        xc = self.fc2(xc)
        # printshape("xc_fc2", xc)

        xc = F.relu(xc)
        xc = self.fc3(xc)
        # printshape("xc_fc3", xc)

        xc = F.relu(xc)
        xc = self.fc4(xc)
        # printshape("xc_fc4", xc)

        xc = F.log_softmax(xc, dim=0)
        return xc