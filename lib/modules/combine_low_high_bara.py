import torch
import torch.nn as nn

from .high_features_fusion import BasicConv2d


class global_module(nn.Module):
    def __init__(self, channels=96, r=6):
        super(global_module, self).__init__()
        out_channels = int(channels // r)
        # local_att

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xg  = self.global_att(x)
        out = self.sig(xg)

        return out               # 96*11
class BWM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel=64):
        super(BWM, self).__init__()

        self.relu = nn.ReLU(True)

        self.global_att = global_module(channel)

        self.conv_layer_1 = BasicConv2d(channel * 2, channel * 2, 3, padding=1)
        self.conv_layer_2 = BasicConv2d(channel * 3, channel, 3, padding=1)

    def forward(self, x, x_boun_atten, x_low_feature_fusion):
        out1 = self.conv_layer_1(torch.cat((x, x_boun_atten), dim=1))
        out1 = self.conv_layer_2(torch.cat((out1, x_low_feature_fusion), dim=1))

        out2 = self.global_att(out1)
        out3 = out1.mul(out2)

        out = x + out3

        return out