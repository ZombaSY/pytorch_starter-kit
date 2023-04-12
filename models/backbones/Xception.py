"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, channel_factor=1):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        channel_factor = 2 ** channel_factor
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32 // channel_factor, 3, 1, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(32 // channel_factor)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32 // channel_factor, 64 // channel_factor, 3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(64 // channel_factor)
        self.relu2 = nn.ReLU(inplace=True)

        self.block1 = Block(64 // channel_factor, 128 // channel_factor, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128 // channel_factor, 256 // channel_factor, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256 // channel_factor, 512 // channel_factor, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(512 // channel_factor, 512 // channel_factor, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(512 // channel_factor, 512 // channel_factor, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(512 // channel_factor, 512 // channel_factor, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(512 // channel_factor)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(512 // channel_factor, 512 // channel_factor, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512 // channel_factor)

    def features(self, _input):
        x = self.conv1(_input)
        x = self.bn1(x)
        x = self.relu1(x)   # (w)

        x = self.conv2(x)
        x = self.bn2(x)
        s1 = self.relu2(x)   # (w)

        s2 = self.block1(s1)  # (w / 2)
        s3 = self.block2(s2)  # (w / 4)
        x = self.block3(s3)  # (w / 8)
        x = self.block4(x)  # (w / 8)
        x = self.block5(x)  # (w / 8)
        x = self.block6(x)  # (w / 8)
        x = self.block7(x)  # (w / 8)
        x = self.block8(x)  # (w / 8)
        x = self.block9(x)  # (w / 8)
        x = self.block10(x)  # (w / 8)
        s4 = self.block11(x)  # (w / 8)
        x = self.block12(s4)  # (w / 16)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        s5 = self.bn4(x)

        return s1, s2, s3, s4, s5

    def forward(self, _input):
        s1, s2, s3, s4, s5= self.features(_input)
        return s1, s2, s3, s4, s5


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    model.last_linear = model.fc
    del model.fc
    return model