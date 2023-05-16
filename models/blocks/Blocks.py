from torch.nn import functional as F
import torch.nn as nn


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channel_1, in_channel_2, inter_channel):
        super(CrossAttentionBlock, self).__init__()

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channel_1, inter_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channel)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channel_2, inter_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channel)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
