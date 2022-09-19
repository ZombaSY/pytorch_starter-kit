import torch
import torch.nn as nn
import torch.nn.functional as F
import fastseg
import copy

from models.backbones import Resnet
from models.backbones import Unet_part
from models.backbones.Swin import SwinTransformer
from models.blocks.Blocks import Upsample
from models.heads.UPerHead import M_UPerHead


def initialize_weights(layer, activation='relu'):

    for module in layer.modules():
        module_name = module.__class__.__name__

        if activation in ('relu', 'leaky_relu'):
            layer_init_func = nn.init.kaiming_uniform_
        elif activation == 'tanh':
            layer_init_func = nn.init.xavier_uniform_
        else:
            raise Exception('Please specify your activation function name')

        if hasattr(module, 'weight'):
            if module_name.find('Conv2') != -1:
                layer_init_func(module.weight)
            elif module_name.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
            elif module_name.find('Linear') != -1:
                layer_init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.1)
            else:
                # print('Cannot initialize the layer :', module_name)
                pass
        else:
            pass


class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Unet_part.DoubleConv(n_channels, 64)
        self.down1 = Unet_part.Down(64, 128)
        self.down2 = Unet_part.Down(128, 256)
        self.down3 = Unet_part.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Unet_part.Down(512, 1024 // factor)
        self.up1 = Unet_part.Up(1024, 512 // factor, bilinear)
        self.up2 = Unet_part.Up(512, 256 // factor, bilinear)
        self.up3 = Unet_part.Up(256, 128 // factor, bilinear)
        self.up4 = Unet_part.Up(128, 64, bilinear)
        self.outc = Unet_part.OutConv(64, n_classes)

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

        return logits


class Swin(nn.Module):
    def __init__(self, num_classes=2, in_channel=3):
        super(Swin, self).__init__()

        self.swin_transformer = SwinTransformer(in_chans=in_channel,
                                                embed_dim=96,
                                                depths=[2, 2, 6, 2],
                                                num_heads=[3, 6, 12, 24],
                                                window_size=7,
                                                mlp_ratio=4.,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=0.,
                                                attn_drop_rate=0.,
                                                drop_path_rate=0.3,
                                                ape=False,
                                                patch_norm=True,
                                                out_indices=(0, 1, 2, 3),
                                                use_checkpoint=False)

        self.uper_head = M_UPerHead(in_channels=[96, 192, 384, 768],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=num_classes,
                                    align_corners=False,)

    def forward(self, x):
        x_size = x.shape[2:]

        feat = self.swin_transformer(x)     # list of feature pyramid
        feat = self.uper_head(feat)
        feat = Upsample(feat, x_size)

        return feat


class ResNet18_multihead(nn.Module):
    def __init__(self, num_classes=6, sub_classes=4):
        super(ResNet18_multihead, self).__init__()

        self.num_classes = num_classes
        self.sub_classes = sub_classes

        self.resnet = Resnet.ResNet18()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifiers = nn.ModuleList([nn.Sequential(*[
            nn.Linear(512 * 4, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, sub_classes)
        ]) for _ in range(num_classes)])

    def forward(self, x):
        x = x.contiguous()
        s1_feature, s2_feature, s3_feature, final_feature = self.resnet(x)

        final_feature = self.avgpool(final_feature)
        final_feature = torch.flatten(final_feature, 1)
        outputs = [self.classifiers[i](final_feature) for i in range(self.num_classes)]

        output = torch.cat([output.unsqueeze(1) for output in outputs], dim=1)
        output = output.view([-1, self.sub_classes, self.num_classes])  # reshape for CE loss

        return output
