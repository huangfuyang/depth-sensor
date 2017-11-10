# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from params import *


class HandNet(nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 5)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(32, 64, 3)
        self.conv3 = nn.Conv3d(64, 128, 3)
        self.fc1 = nn.Linear(128*4*4*4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, JOINT_POS_LEN)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1,128*4*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x