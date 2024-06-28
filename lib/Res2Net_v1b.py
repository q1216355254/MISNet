import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    # expansion对应卷积层中残差结构的主分支的卷积核个数有没有发生变化
    # 比如在ResNet18和34中，每一个卷积层的残差结构的卷积核大小没有发生变化
    # 但是在ResNet50和101中，每一个卷积层的残差结构中第3个卷积核的个数是前两个的4倍
    expansion = 4


    # 定义残差结构中使用到的一系列参数
    # output=(input-kernel_size+2*padding)/stride+1 如果得到的是一个小数就向下取整
    # 在使用batchnormalization时不使用偏置bias
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    # 这个block就是刚才定义的Bottle2Neck
    # layers对应bottle2neck的数目，是一个列表参数
    # basewidth是什么
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        # 在经过bottle2Neck之前的池化层后得到的特征图的通道数作为输入bottle2neck的通道数为64
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        # 在输入到bottle2neck之前的操作
        # 经过了三个卷积层
        self.conv1 = nn.Sequential(
            # conv1是3*3的卷积层对通道数是3的输入图片（RGB图像）先进行卷积操作
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # 之后又经过了一个最大池化层，stride=2使得特征图缩减为了原来的一半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 然后经过4个bottle2neck,一系列的残差结构是通过_make_layer生成的
        # 第一个bottle2Neck只需要调整特征图的通道数，所以stride=1
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # _make_layer函数
    # block就是刚才定义的bottle2neck
    # planes定义的是每一个bottle2neck中第一个卷积核的个数
    # blocks就是每一个block2neck会重复几次
    def _make_layer(self, block, planes, blocks, stride=1):
        # 对于18层和34层的resnet不进行下面的操作
        downsample = None
        # 在ResNet中，对于50.101.150层的网络来说，会存在一个虚线的下采样的操作，要进行维度的匹配
        # 所以会进行下面的操作
        # or的后半部分是对第一个bottle2neck进行操作，前半部分是对后三个bottle2neck进行操作
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # 加了一个全局平均池化
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # 定义一个空的列表
        layers = []
        # 对每一个bottle2neck的第一块进行搭建，这一块会进行循环
        # 感觉应该是层吧，第一层
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        # 将输入的通道数扩大为原来的4倍。虚线残差结构
        self.inplanes = planes * block.expansion


        # 每一个bottle2neck中第二层开始都是实线的残差结构，循环压入网络结构中，循环压入的是块，不是层
        # 到底是块还是层
        for i in range(1, blocks):
            # planes是主线分支上第一个卷积层的卷积核个数
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))


        # 通过非关键字参数的形式传入到nn.Sequential中
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b lib.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('D:/uncertainty/PraNet_v4b/lib/weights/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state)
        # model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('/home/22031212472/uncertainty/ON/lib/weights/res2net101_v1b_26w_4s-0812c246.pth')
        model.load_state_dict(model_state)
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net50_v1b_26w_4s(pretrained=True)
    model = model.cuda(0)
    # print(model(images).size())
    print(model)
