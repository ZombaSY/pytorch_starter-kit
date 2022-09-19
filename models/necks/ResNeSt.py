import torch
import torch.nn as nn

from torch.nn import functional as F


"""
https://github.com/zhanghang1989/ResNeSt/
https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/models/splat.py
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
"""


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        assert radix > 0
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttention(nn.Module):
    def __init__(self, channels, radix, cardinality, reduction_factor=4):
        super(SplitAttention, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels
        inter_channels = max(channels*radix//reduction_factor, 32)
        self.fc1 = nn.Conv2d(channels//radix, inter_channels, 1, groups=cardinality)
        # self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=cardinality)
        self.rsoftmax = rSoftMax(radix, cardinality)

    def forward(self, x):
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        # gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)

            # out = sum([att*split for (att, split) in zip(attens, splited)])   # original version

            out = [att*split for (att, split) in zip(attens, splited)]
            out = torch.cat(out, dim=1)

        else:
            out = atten * x

        return out.contiguous()


def main():
    import time

    channel_size = 96
    radix = 16
    cardinality = 2

    x = torch.rand(8, channel_size, 38, 28).cuda()

    split_attention = SplitAttention(channel_size, radix, cardinality).cuda()

    tt = time.time()

    z = split_attention(x)
    print(z.shape)


if __name__ == '__main__':
    main()
