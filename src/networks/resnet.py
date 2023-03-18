import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class FEATURES:
    conv_features = []
    layer_featrues = []


class FedNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        FEATURES.conv_features.append(x)
        return x


class NoNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InstanceNorm2d(nn.Module):
    def __init__(self, planes, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(planes, affine=True)
        self.norm.weight.data.fill_(1)
        self.norm.bias.data.zero_()

    def forward(self, x):
        return x if (x.shape[2] == 1 and x.shape[3] == 1) else self.norm(x)


def Norm2d(planes, norm='bn'):
    """
    Separate C channels into C groups (equivalent with InstanceNorm), use InstanceNorm2d instead, to avoid the case where height=1 and width=1
    
    Put all C channels into 1 single group (equivalent with LayerNorm)
    """
    if norm == 'gn':
        return nn.GroupNorm(planes // 16, planes)
    elif norm == 'in':
        return InstanceNorm2d(planes, affine=True)
    elif norm == 'ln':
        return nn.GroupNorm(1, planes)
    elif norm == 'bn':
        return nn.BatchNorm2d(planes)
    elif norm == 'sbn':
        return nn.BatchNorm2d(planes, momentum=None)
    elif norm == 'fbn':
        return nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
    elif norm == 'no':
        return NoNorm()
    elif norm == 'ours':
        return FedNorm()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm='bn'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes, norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes, norm)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        FEATURES.layer_featrues.append(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm='bn'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm2d(planes, norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4, norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, norm='bn', restriction='mean+std'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = Norm2d(64, norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       norm=norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       norm=norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       norm=norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       norm=norm)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)

        self.conv_list = []
        self.layer_list = []
        self.restriction = restriction
        self.mse = nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                self.conv_list.append(m)
            elif isinstance(m, nn.BatchNorm2d) and norm != 'fbn':
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
                self.layer_list.append(m)
            if isinstance(m, BasicBlock):
                self.layer_list.append(m)
                if not (m.bn2, nn.InstanceNorm2d):
                    m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, norm='bn'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(planes * block.expansion, norm),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x, similarity=False):
        FEATURES.conv_features = []
        FEATURES.layer_featrues = []
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
        out = self.fc(x)
        if similarity:
            x_norm = x.norm(dim=1, p=2).unsqueeze(1).expand(out.shape[0], out.shape[1])
            w_norm = self.fc.weight.norm(dim=1, p=2).unsqueeze(0).expand(out.shape[0], out.shape[1])
            div_tens = torch.maximum(torch.mul(x_norm, w_norm), 1e-8 * torch.ones([out.shape[0], out.shape[1]]).cuda())
            out = torch.div(out, div_tens)
        return out


def resnet10(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet10']))
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    resnet = resnet18(pretrained=False)
    print(resnet.layer_list)
