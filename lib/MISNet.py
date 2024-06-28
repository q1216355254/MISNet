import torch
import torch.nn as nn
import torch.nn.functional as F


from Res2Net_v1b import res2net50_v1b_26w_4s
from modules.combine_low_high import SSFM
from modules.combine_low_high_bara import BWM
from modules.high_features_fusion import BasicConv2d, HFM
from modules.low_features_cbam import CBAM
from modules.low_features_fusion import LFM, Conv1x1
from modules.low_features_fusion_at import PAA_e
from optim.losses import *


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MISNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(MISNet, self).__init__()

        act_fn = nn.ReLU(inplace=True)
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like modules ----
        self.rfb0_1 = RFB_modified(64, channel)
        self.rfb1_1 = RFB_modified(256, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        # ---- Low-level Feature Fusion ----
        self.lfm = LFM()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- High-level Feature Fusion ----
        self.agg = HFM(channel)

        # ---- Selectively Shared Fusion Module ----
        self.cross = SSFM(256, 512, channel)
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.reduce = Conv1x1(16, 1)

        # ---- low-feature fusion:CBAM ----
        self.cba = CBAM()
        self.cat = BWM()

        # ---- branch 4 ----
        self.context4 = PAA_e(2048)
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 64, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(64, 1, kernel_size=1)
        self.x4_conv = BasicConv2d(2048, 64, kernel_size=1)
        self.lfd_conv = BasicConv2d(256, 64, kernel_size=1)

        self.bara_conv = nn.Sequential(
            nn.Conv2d(128, 64,3,1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.layer_hig01 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64), act_fn)
        self.layer_hig11 = nn.Sequential(nn.Conv2d(64, 64,  kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),act_fn)
        self.layer_hig21 = nn.Sequential(nn.Conv2d(64, 64,  kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64),act_fn)
        self.layer_hig31 = nn.Sequential(nn.Conv2d(64, 1,  kernel_size=1))

        # ---- branch 3 ----
        self.context3 = PAA_e(1024)
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.x3_conv = BasicConv2d(1024, 64, kernel_size=1)

        # ---- branch 2 ----
        self.context2 = PAA_e(512)
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.x2_conv = BasicConv2d(512, 64, kernel_size=1)
        self.layer0 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            act_fn
        )
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.loss_fn = bce_iou_loss


    def forward(self, x):
        # x = sample['image']
        # if 'gt' in sample.keys():
        #     y = sample['gt']
        # else:
        #     y = None


        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)        # 64,176,176
        x = self.resnet.maxpool(x0)

        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        high_feature_fusion = self.agg(x4_rfb, x3_rfb, x2_rfb)            # 512,44,44

        x0_rfb = self.rfb0_1(x0)        # channel -> 32  x:32, 88, 88  x0:32, 176, 176
        x1_rfb = self.rfb1_1(x1)        # channel -> 32  32,88,88

        low_feature_fusion = self.lfm(x1_rfb, x0_rfb)
        low_feature_fusion = self.maxpool(low_feature_fusion)          # 256,88,88
        low_feature_fusion_att = torch.sigmoid(low_feature_fusion)

        feature1 = self.downSample(low_feature_fusion_att)
        feature2 = high_feature_fusion

        cross_feature = self.cross(feature1, feature2)
        cross_feature = self.reduce(cross_feature)

        global_map = cross_feature

        lateral_map_5 = F.interpolate(global_map, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)


        low_feature_fusion_guide = self.cba(low_feature_fusion_att)

        # ---- branch_4 ----
        x4_at = self.context4(x4)          # 2048*11
        x4_conv = self.x4_conv(x4)
        x4_conv = self.up_4(x4_conv)          # 64*44
        crop_4 = F.interpolate(global_map, scale_factor=0.25, mode='bilinear')        # 1*11
        score4 = torch.sigmoid(crop_4)
        dist1 = torch.abs(score4 - 0.5)
        boundary_att_4 = 1 - (dist1 / 0.5)
        boundary_x_4 = x4_at * boundary_att_4
        boundary_x_4 = boundary_x_4 + x4  # bad结果2048*11
        boundary_x_4 = self.ra4_conv1(boundary_x_4)
        boundary_x_4 = F.relu(self.ra4_conv2(boundary_x_4))
        boundary_x_4 = F.relu(self.ra4_conv3(boundary_x_4))
        boundary_x_4 = F.relu(self.ra4_conv4(boundary_x_4))         # 64*11
        boundary_x_4 = self.up_4(boundary_x_4)           # 64*44
        dist2 = -1 * (torch.sigmoid(crop_4)) + 1
        reverse_x_4 = x4_at * dist2          # 2048*11
        reverse_x_4 = reverse_x_4 + x4
        reverse_x_4 = self.ra4_conv1(reverse_x_4)
        reverse_x_4 = F.relu(self.ra4_conv2(reverse_x_4))
        reverse_x_4 = F.relu(self.ra4_conv3(reverse_x_4))
        reverse_x_4 = F.relu(self.ra4_conv4(reverse_x_4))          # 64*11
        reverse_x_4= self.up_4(reverse_x_4)          # 64*44
        bara4 = torch.cat((boundary_x_4, reverse_x_4), dim=1)  # 128*44
        bara4_conv = self.bara_conv(bara4)              # 64*44
        lfd = self.layer0(low_feature_fusion_guide)          # 256*44
        b4_combine = self.cat(x4_conv, bara4_conv, lfd)         # 64*44
        crop_4 = global_map        # 1*44
        lateral_map_4 = b4_combine + crop_4
        lateral_map_4_guide = self.ra4_conv5(lateral_map_4)
        lateral_map_4 = self.layer_hig01(self.up_2(lateral_map_4))
        lateral_map_4 = self.layer_hig11(self.up_2(lateral_map_4))
        lateral_map_4 = self.layer_hig21(self.up_2(lateral_map_4))
        lateral_map_4 = self.layer_hig31(lateral_map_4)

        # ---- branch_3 ----
        x3_at = self.context3(x3)  # 轴注意力         1024*22
        x3_conv = self.x3_conv(x3)
        x3_conv = self.up_2(x3_conv)
        crop_3 = F.interpolate(lateral_map_4_guide, scale_factor=0.5, mode='bilinear')
        score = torch.sigmoid(crop_3)
        dist1 = torch.abs(score - 0.5)
        boundary_att_3 = 1 - (dist1 / 0.5)
        boundary_x_3 = x3_at * boundary_att_3
        boundary_x_3 = boundary_x_3 + x3  # bad结果
        boundary_x_3 = self.ra3_conv1(boundary_x_3)
        boundary_x_3 = F.relu(self.ra3_conv2(boundary_x_3))
        boundary_x_3 = F.relu(self.ra3_conv3(boundary_x_3))
        boundary_x_3 = self.up_2(boundary_x_3)
        dist2 = -1 * (torch.sigmoid(crop_3)) + 1
        reverse_x_3 = x3_at * dist2
        reverse_x_3 = reverse_x_3 + x3
        reverse_x_3 = self.ra3_conv1(reverse_x_3)
        reverse_x_3 = F.relu(self.ra3_conv2(reverse_x_3))
        reverse_x_3 = F.relu((self.ra3_conv3(reverse_x_3)))
        reverse_x_3 = self.up_2(reverse_x_3)
        bara3 = torch.cat((boundary_x_3, reverse_x_3), dim=1)
        bara3_conv = self.bara_conv(bara3)
        b3_combine = self.cat(x3_conv, bara3_conv, lfd)
        crop_3 = lateral_map_4_guide
        lateral_map_3 = b3_combine + crop_3
        lateral_map_3_guide = self.ra3_conv4(lateral_map_3)
        lateral_map_3 = self.layer_hig01(self.up_2(lateral_map_3))
        lateral_map_3 = self.layer_hig11(self.up_2(lateral_map_3))
        lateral_map_3 = self.layer_hig21(self.up_2(lateral_map_3))
        lateral_map_3 = self.layer_hig31(lateral_map_3)

        # ----  branch_2 ----
        x2_at = self.context2(x2)  # 轴注意力         512*44
        x2_conv = self.x2_conv(x2)
        crop_2 = lateral_map_3_guide             # 44
        score = torch.sigmoid(crop_2)
        dist1 = torch.abs(score - 0.5)
        boundary_att_2 = 1 - (dist1 / 0.5)
        boundary_x_2 = x2_at * boundary_att_2
        boundary_x_2 = boundary_x_2 + x2  # bad结果      512*44
        boundary_x_2 = self.ra2_conv1(boundary_x_2)
        boundary_x_2 = F.relu(self.ra2_conv2(boundary_x_2))
        boundary_x_2 = F.relu(self.ra2_conv3(boundary_x_2))
        dist2 = -1 * (torch.sigmoid(crop_2)) + 1
        reverse_x_2 = x2_at * dist2
        reverse_x_2 = reverse_x_2 + x2
        reverse_x_2 = self.ra2_conv1(reverse_x_2)
        reverse_x_2 = F.relu(self.ra2_conv2(reverse_x_2))
        reverse_x_2 = F.relu(self.ra2_conv3(reverse_x_2))       # 64*44
        bara2 = torch.cat((boundary_x_2, reverse_x_2), dim=1)
        bara2_conv = self.bara_conv(bara2)              # 64*44
        b2_combine = self.cat(x2_conv, bara2_conv, lfd)
        lateral_map_2 = b2_combine + crop_2
        lateral_map_2_guide = self.ra2_conv4(lateral_map_2)
        lateral_map_2 = self.layer_hig01(self.up_2(lateral_map_2))
        lateral_map_2 = self.layer_hig11(self.up_2(lateral_map_2))
        lateral_map_2 = self.layer_hig21(self.up_2(lateral_map_2))
        lateral_map_2 = self.layer_hig31(lateral_map_2)


        # if y is not None:
        #     loss5 = self.loss_fn(lateral_map_5, y)
        #     loss4 = self.loss_fn(lateral_map_4, y)
        #     loss3 = self.loss_fn(lateral_map_3, y)
        #     loss2 = self.loss_fn(lateral_map_2, y)
        #     loss = loss2 + loss3 + loss4 + loss5
        #     debug = [lateral_map_5, lateral_map_4, lateral_map_3]
        # else:
        #     loss = 0
        #     debug = []
        # return {'pred2': lateral_map_2,
        #         'pred3': lateral_map_3,
        #         'pred4': lateral_map_4,
        #         'pred5': lateral_map_5,
        #         'loss2': loss2,
        #         'loss3': loss3,
        #         'loss4': loss4,
        #         'loss5': loss5,
        #         'loss': loss,
        #         'debug': debug}

        return lateral_map_2


if __name__ == '__main__':
    # -- coding: utf-8 --
    # import torch
    # import torchvision
    # from thop import profile

    # # Model
    # print('==> Building model..')
    # model = PraNet_v3b_4_250().cuda()

    # dummy_input = torch.randn(1, 3, 224, 224).cuda()
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # -- coding: utf-8 --
    # import torch
    # import torchvision
    # from thop import profile

    # # Model
    # print('==> Building model..')
    # model = PolypPVT().cuda()

    # dummy_input = torch.randn(1, 3, 352, 352).cuda()
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    import time
    import torch

    input_tensor = torch.randn(1, 3, 32, 32).cuda()

    # 推断的次数
    num_inferences = 100  # 假设进行100次推断

    # 加载模型到 GPU
    model = MISNet().cuda()

    # 确保 CUDA 操作的同步
    torch.cuda.synchronize()

    # 开始计时
    start = time.time()

    # 进行多次推断
    for _ in range(num_inferences):
        result = model(input_tensor).cuda()  # 重新加载模型以模拟多次推断
        # print(result)

    # result = model(input_tensor).cuda()

    # 确保 CUDA 操作的同步
    torch.cuda.synchronize()

    # 结束计时
    end = time.time()

    # 计算总时间
    total_time = end - start

    # 计算每秒钟的推断次数
    fps = num_inferences / total_time

    print('FPS:', fps)

#     ras = HarDMSEG().cuda()
#     input_tensor = torch.randn(1, 3, 352, 352).cuda()
#
#     out = ras(input_tensor)


# if __name__ == '__main__':
#     ras = caranet().cuda()
#     input_tensor = torch.randn(1, 3, 352, 352).cuda()

#     out = ras(input_tensor)
#     print(out)


