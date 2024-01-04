import torch
import torch.nn as nn
import torch.nn.functional as F
from .Prototype import *
from .layers import *
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder_flow(nn.Module):
    def __init__(self):
        super(Encoder_flow, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.res_block1 = ResidualBlock(16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.res_block2 = ResidualBlock(32, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.res_block3 = ResidualBlock(64, 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x1 = F.relu(self.bn2(self.conv2(x)))
        x2 = self.res_block2(x1)
        skip = x2
        x = F.relu(self.bn3(self.conv3(x2)))
        x = self.res_block3(x)
        return x, skip


class Decoder_flow(nn.Module):
    def __init__(self):
        super(Decoder_flow, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
            )

        def CoordAttention(in_channel):
            return CoordAtt(in_channel, in_channel)


        self.moduleconv3 = Basic(64, 64)
        self.moduleAtt3 = CoordAttention(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.moduleconv2 = Basic(64, 32)
        self.moduleAtt2 = CoordAttention(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.moduleconv1 = Basic(16, 16)
        self.moduleAtt1 = CoordAttention(16)
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.final_deconv = nn.ConvTranspose2d(8, 2, kernel_size=1)


    def forward(self, flow_feas, skips):
        output_flow_list = []

        for i in range(len(flow_feas)):
            x = flow_feas[i]
            skip2 = skips[i]

            tensorconv3 = self.moduleconv3(x)
            tensoratt3 = self.moduleAtt3(tensorconv3)
            tensorconv3 = tensoratt3 * tensorconv3
            tensor3 = self.deconv3(tensorconv3)
            tensor3 = self.bn3(tensor3)
            tensor3 = torch.cat((skip2, tensor3), dim=1)

            tensorconv2 = self.moduleconv2(tensor3)
            tensoratt2 = self.moduleAtt2(tensorconv2)
            tensorconv2 = tensoratt2 * tensorconv2
            tensor2 = self.deconv2(tensorconv2)
            tensor2 = self.bn2(tensor2)

            tensorconv1 = self.moduleconv1(tensor2)
            tensoratt1 = self.moduleAtt1(tensorconv1)
            tensorconv1 = tensoratt1 * tensorconv1
            tensor1 = self.deconv1(tensorconv1)
            tensor1 = self.bn1(tensor1)

            x = self.final_deconv(tensor1)
            output_flow_list.append(x)

        return tuple(output_flow_list)

class flow_net(nn.Module):
    def __init__(self):
        super(flow_net, self).__init__()
        self.Encoder_flow = Encoder_flow()
        self.Decoder_flow = Decoder_flow()


    def forward(self, flow_tuple, mode="train"):
        flow_feas = []
        skips = []
        for flow in flow_tuple:
            flow_fea, skip = self.Encoder_flow(flow)
            flow_feas.append(flow_fea)
            skips.append(skip)


        if mode == "train":
            output_flow_list = self.Decoder_flow(flow_feas, skips)
            out = dict(output_flow_tuple=output_flow_list)

            return out

        else:
            output_flow_list = self.Decoder_flow(flow_feas, skips)

            out = dict(output_flow_tuple=output_flow_list)

            return out






