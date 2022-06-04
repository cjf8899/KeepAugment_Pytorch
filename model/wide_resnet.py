# From https://github.com/xternalz/WideResNet-pytorch

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                                padding=0, bias=False) or None
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#         else:
#             out = self.relu1(self.bn1(x))
#         out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         out = self.conv2(out)
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)

# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)

# class WideResNet_Early(nn.Module):
#     def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
#         super(WideResNet_Early, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(160, 64, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(64)
#         )
        
#         self.aux_linear = nn.Linear(1024, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#     def forward(self, x, early):
#         out = self.conv1(x)
#         out = self.block1(out)
        
#         aux = F.adaptive_avg_pool2d(out, (4,4))
#         aux = F.relu(self.conv(aux), inplace=True)
#         aux = torch.flatten(aux, 1)
#         aux = self.aux_linear(aux)
# #         aux = F.relu(self.aux_linear(aux), inplace=True)
#         if early:
#             return aux
        
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.adaptive_avg_pool2d(out, (1,1))
#         out = out.view(-1, self.nChannels)
#         out = self.fc(out)
#         return out, aux
    
    
# class WideResNet(nn.Module):
#     def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
#         super(WideResNet, self).__init__()
#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) / 6
#         block = BasicBlock
#         # 1st conv before any network block
#         self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
#         # 2nd block
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
#         # 3rd block
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.nChannels = nChannels[3]

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.adaptive_avg_pool2d(out, (1,1))
#         out = out.view(-1, self.nChannels)
#         out = self.fc(out)
#         return out

    
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet_Early(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet_Early, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.aux_linear = nn.Linear(1024, num_classes)
        

        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, early):
        out = self.conv1(x)
        out = self.layer1(out)
        
        aux = F.adaptive_avg_pool2d(out, (4,4))
        aux = F.relu(self.conv(aux), inplace=True)
        aux = torch.flatten(aux, 1)
        aux = self.aux_linear(aux)
#         aux = F.relu(self.aux_linear(aux), inplace=True)
        if early:
            return aux
        
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, aux

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out