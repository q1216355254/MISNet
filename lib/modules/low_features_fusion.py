import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Conv3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)


        return x


class ConvBNR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class LFM(nn.Module):
    def __init__(self):
        super(LFM, self).__init__()
        # Conv1x1第一个参数分别对应x0_rfb、x1_rfb的输出通道数,第二个参数待斟酌
        self.reduce1 = Conv1x1(32, 64)
        self.reduce2 = Conv3x3(64, 64)
        self.reduce4 = Conv1x1(32, 256)
        self.reduce5 = Conv3x3(256, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3)
            )
        # nn.Conv2d(256, 1, 1)

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x1 = self.reduce2(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x2 = self.reduce5(x4)
        out = torch.cat((x2, x1), dim=1)
        out = self.block(out)

        return out