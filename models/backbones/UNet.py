import torch
import torch.nn as nn


# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=3, padding=1, bias=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # set align_corners=False in tflite version
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, padding=padding, bias=bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, base_c=64, kernel_size=3, padding=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_c * 1, kernel_size=kernel_size, padding=padding)
        self.down1 = Down(base_c * 1, base_c * 2, kernel_size=kernel_size, padding=padding)
        self.down2 = Down(base_c * 2, base_c * 4, kernel_size=kernel_size, padding=padding)
        self.down3 = Down(base_c * 4, base_c * 8, kernel_size=kernel_size, padding=padding)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, kernel_size=kernel_size, padding=padding)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, kernel_size=kernel_size, padding=padding)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, kernel_size=kernel_size, padding=padding)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, kernel_size=kernel_size, padding=padding)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear, kernel_size=kernel_size, padding=padding)
        self.outc = OutConv(base_c * 1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        out_dict = {'seg': logits,
                    'feats': [x2, x3, x4, x5]}

        return out_dict


class UNet_dsv(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, base_c=64, kernel_size=3, padding=1):
        super(UNet_dsv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_c * 1, kernel_size=kernel_size, padding=padding)
        self.down1 = Down(base_c * 1, base_c * 2, kernel_size=kernel_size, padding=padding)
        self.down2 = Down(base_c * 2, base_c * 4, kernel_size=kernel_size, padding=padding)
        self.down3 = Down(base_c * 4, base_c * 8, kernel_size=kernel_size, padding=padding)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, kernel_size=kernel_size, padding=padding)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, kernel_size=kernel_size, padding=padding)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, kernel_size=kernel_size, padding=padding)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, kernel_size=kernel_size, padding=padding)
        self.up4 = Up(base_c * 2, base_c * 1, bilinear, kernel_size=kernel_size, padding=padding)
        self.outc = OutConv(base_c * 1, n_classes)
        self.outc_aux1 = OutConv(base_c * 1, n_classes)
        self.outc_aux2 = OutConv(base_c * 2, n_classes)
        self.outc_aux3 = OutConv(base_c * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        logits = self.outc(x_up4)
        logits_aux1 = self.outc_aux1(x_up3)
        logits_aux2 = self.outc_aux2(x_up2)
        logits_aux3 = self.outc_aux3(x_up1)

        out_dict = {'seg': logits,
                    'feats': [x2, x3, x4, x5],
                    'seg_aux': [logits_aux1, logits_aux2, logits_aux3]}

        return out_dict

