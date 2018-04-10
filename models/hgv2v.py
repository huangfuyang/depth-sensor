import torch.nn as nn
import math
from params import *
from res3d import Res3D


# def make_res_layer(self, block, planes, blocks, shortcut_type, stride=1):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion:
#         if shortcut_type == 'A':
#             downsample = partial(
#                 self.downsample_basic_block,
#                 planes=planes * block.expansion,
#                 stride=stride)
#         else:
#             downsample = nn.Sequential(
#                 nn.Conv3d(
#                     self.inplanes,
#                     planes * block.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False), nn.BatchNorm3d(planes * block.expansion))
#
#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#         layers.append(block(self.inplanes, planes))
#
#     return nn.Sequential(*layers)
#



class Hourglass3D(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass3D, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Res3D(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool3d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(Res3D(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass3D(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Res3D(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Res3D(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2


class Hourglass3DNet(nn.Module):
    def __init__(self, nStack, nModules, nFeats):
        super(Hourglass3DNet, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.conv1_ = nn.Conv3d(1, 64, bias=True, kernel_size=3, stride=2, padding=1)
        # self.conv1_ = nn.Conv3d(1, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Res3D(64, 128)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.r4 = Res3D(128, 128)
        self.r5 = Res3D(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass3D(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Res3D(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv3d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm3d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv3d(self.nFeats, JOINT_LEN, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv3d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv3d(JOINT_LEN, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        # x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_

        return out

        # print HourglassNet(1,1,256)