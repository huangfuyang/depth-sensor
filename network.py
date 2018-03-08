# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from params import *
import torch


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
        x = self.fc3(x)
        return x


class HandSensorNet(nn.Module):
    def __init__(self):
        super(HandSensorNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 5)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(32, 64, 3)
        self.conv3 = nn.Conv3d(64, 128, 3)
        self.fc1 = nn.Linear(128*4*4*4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512+15, JOINT_POS_LEN)

    def forward(self, x,y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1,128*4*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x,y),dim=1)
        x = self.fc3(x)
        return x


class SensorNet(nn.Module):
    def __init__(self):
        super(SensorNet, self).__init__()
        self.fc1 = nn.Linear(15, 30)
        self.fc2 = nn.Linear(30, 45)
        self.fc3 = nn.Linear(45, JOINT_POS_LEN)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HeatmapNet(nn.Module):
    def __init__(self):
        super(HeatmapNet, self).__init__()
        self.conv11 = nn.Conv2d(1,16,5)
        self.conv12 = nn.Conv2d(16,32,6)
        self.conv21 = nn.Conv2d(1,16,5)
        self.conv22 = nn.Conv2d(16,32,5)
        self.conv31 = nn.Conv2d(1,16,4)
        self.conv32 = nn.Conv2d(16,32,4)
        self.pool2 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(7776,6804)
        self.fc2 = nn.Linear(6804,6804)

    def forward(self, x):
        x[0] = self.pool4(F.relu(self.conv11(x[0])))
        x[0] = self.pool2(F.relu(self.conv12(x[0])))
        x[1] = self.pool2(F.relu(self.conv21(x[1])))
        x[1] = self.pool2(F.relu(self.conv22(x[1])))
        x[2] = F.relu(self.conv31(x[2]))
        x[2] = self.pool2(F.relu(self.conv32(x[2])))
        x = torch.cat((x[0].view(-1, 9*9*32),x[1].view(-1, 9*9*32),x[2].view(-1, 9*9*32)),dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x