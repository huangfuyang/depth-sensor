from models.layers.Residual import Residual
import torch.nn as nn
from params import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.res3d import BasicBlock


class EncoDecoder(nn.Module):
    def __init__(self, inchannel):
        super(EncoDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.inplanes = inchannel
        self.down1 = nn.MaxPool3d(2,2)
        self.res1 = self._make_layer(BasicBlock,inchannel*2,1,"B")
        self.down2 = nn.MaxPool3d(2,2)
        self.res2 = self._make_layer(BasicBlock,inchannel*4,1,"B")

        # decoder
        self.res3 = self._make_layer(BasicBlock,inchannel*4,1,"B")
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(inchannel*4, inchannel*2, 2, stride=2, padding=0),
            nn.BatchNorm3d(inchannel*2),
            self.relu
        )
        self.inplanes = inchannel*2
        self.res4 = self._make_layer(BasicBlock,inchannel*2,1,"B")
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(inchannel*2, inchannel, 2, stride=2, padding=0),
            nn.BatchNorm3d(inchannel),
            self.relu
        )

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    self.downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        tmp1 = x
        out = self.down1(x)
        out = self.res1(out)
        tmp2 = out
        out = self.down2(out)
        out = self.res2(out)

        out = self.res3(out)
        out = self.up1(out)
        out += tmp2

        out = self.res4(out)
        out = self.up2(out)

        out += tmp1
        return out


class V2VNet(nn.Module):
    def get_basic_block(self, ichannel, outchannel, kernel):
        return nn.Sequential(
            nn.Conv3d(ichannel, outchannel,
                      kernel_size=kernel, stride=1, padding=int(kernel/2),bias=False),
            nn.BatchNorm3d(outchannel),
            self.relu
        )

    def downsample_basic_block(x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.Tensor(
            out.size(0), planes - out.size(1), out.size(2), out.size(3),
            out.size(4)).zero_()
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = Variable(torch.cat([out.data, zero_pads], dim=1))

        return out


    def __init__(self):
        super(V2VNet, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.basic1 = self.get_basic_block(1,32,7)
        self.downsample = nn.MaxPool3d(2,2)
        self.inplanes = 32
        self.reslayer = self._make_layer(BasicBlock, 32, 3,'B')
        self.encodecoder = EncoDecoder(32)
        self.res2 = self._make_layer(BasicBlock,32,1,'B')
        self.basic2 = self.get_basic_block(32, 32, 1)
        self.basic3 = self.get_basic_block(32, JOINT_LEN, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    self.downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.basic1(x)
        x = self.downsample(x)
        x = self.reslayer(x)
        x = self.encodecoder(x)
        x = self.res2(x)
        x = self.basic2(x)
        x = self.basic3(x)
        return x

# a = Variable(torch.FloatTensor(1,16,16,16,16))
# en = EncoDecoder(16)
# print en(a)
# print a.size()
# conv = nn.ConvTranspose2d(2,4,2,stride=2,padding=0)
# b = conv(a)
# print b.size()