import torch
import torch.nn as nn
from .high_features_fusion import BasicConv2d


class SSFM(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel, r=16, L=32, M=2):
        self.init__ = super(SSFM, self).__init__()

        act_fn = nn.ReLU(inplace=True)
        d = max(int(16 / r), L)

        self.layer0 = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv2d(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )

        self.layer5_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )

        self.layer_out = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            act_fn
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, 16))
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x0, x1):

        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)

        x_3_1 = self.layer3_1(torch.cat((x0_1, x1_1), dim=1))
        x_5_1 = self.layer5_1(torch.cat((x1_1, x0_1), dim=1))

        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))

        x_3_2 = x_3_2.unsqueeze(dim=1)
        x_5_2 = x_5_2.unsqueeze(dim=1)


        x_1 = torch.cat([x_3_2, x_5_2], dim=1)
        x_2 = torch.sum(x_1, dim=1)
        x_3 = self.global_pool(x_2)
        x_4 = x_3.mean(-1).mean(-1)
        x = self.fc(x_4)
        for i, fc in enumerate(self.fcs):

            vector = fc(x).unsqueeze(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        x0_1 = x0_1.unsqueeze(dim=1)
        x1_1 = x1_1.unsqueeze(dim=1)
        x_fuse = torch.cat([x0_1, x1_1], dim=1)
        out = (x_fuse * attention_vectors).sum(dim=1)

        return out
