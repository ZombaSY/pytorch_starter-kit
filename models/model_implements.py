import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.Swin import SwinTransformer
from models.heads.UPerHead import M_UPerHead
from models.backbones import UNet as UNet_part
from models.backbones.timm_backbone import BackboneLoader
from models.heads import MLP

from collections import OrderedDict


class Swin_t(nn.Module):
    def __init__(self, in_channel=3, base_c=96, **kwargs):
        super().__init__()

        self.swin_transformer = SwinTransformer(in_chans=in_channel,
                                                embed_dim=base_c,
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

    def load_pretrained_imagenet(self, dst):
        pretrained_states = torch.load(dst)['model']
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if 'head.weight' == item or 'head.bias' == item or 'norm.weight' == item or 'norm.bias' == item or 'layers.0.blocks.1.attn_mask' == item or 'layers.1.blocks.1.attn_mask' == item or 'layers.2.blocks.1.attn_mask' == item or 'layers.2.blocks.3.attn_mask' == item or 'layers.2.blocks.5.attn_mask' == item:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.swin_transformer.remove_fpn_norm_layers()  # temporally remove fpn norm layers that not included on public-release model
        self.swin_transformer.load_state_dict(pretrained_states_backbone)
        self.swin_transformer.add_fpn_norm_layers()

    def forward(self, x):
        feat1, feat2, feat3, feat4 = self.swin_transformer(x)
        out_dict = {'feats': [feat1, feat2, feat3, feat4]}

        return out_dict


class Swin_t_SemanticSegmentation(Swin_t):
    def __init__(self, num_class=2, in_channel=3, base_c=96, **kwargs):
        super(Swin_t_SemanticSegmentation, self).__init__(in_channel, base_c)
        self.uper_head = M_UPerHead(in_channels=[base_c, base_c * 2, base_c * 4, base_c * 8],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_class=num_class,
                                    align_corners=False,)

    def forward(self, x):
        x_size = x.shape[2:]

        # get segmentation map
        feat1, feat2, feat3, feat4 = self.swin_transformer(x)
        out_dict = {'feats': [feat1, feat2, feat3, feat4]}

        feat = self.uper_head(out_dict['feats'])
        feat = F.interpolate(feat, x_size, mode='bilinear', align_corners=False)
        out_dict['seg'] = feat

        return out_dict


class UNet(nn.Module):
    def __init__(self, num_class=2, in_channel=3, bilinear=True, kernel_size=3, padding=1, base_c=64, **kwargs):
        super(UNet, self).__init__()
        self.u_net = UNet_part.UNet(n_channels=in_channel,
                                    n_classes=num_class,
                                    base_c=base_c,
                                    bilinear=bilinear,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        out= self.u_net(x)

        return out


class ConvNext_l_classification(nn.Module):
    def __init__(self, hidden_dims, num_class, freeze_backbone=False, normalization='BatchNorm1d', activation='ReLU', dropblock=True, **kwargs):
        super().__init__()
        self.backbone = BackboneLoader('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', exportable=True, pretrained=True)
        self.classifier = MLP.SimpleClassifier(in_features=1536, hidden_dims=hidden_dims, num_class=num_class, normalization=normalization, activation=activation, dropblock=dropblock)

        if freeze_backbone:
            for m in self.backbone.parameters():
                m.requires_grad = False

    def load_pretrained(self, dst):
        pretrained_weights = torch.load(dst)
        taget_weights = OrderedDict()
        for key in pretrained_weights.keys():
            if 'backbone.backbone' in key:
                target_key = key.replace('backbone.backbone', 'backbone')
                taget_weights[target_key] = pretrained_weights[key]
        self.backbone.load_state_dict(taget_weights)

    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        out = self.classifier(feat)

        out_dict['class'] = out
        out_dict['feat'] = feat

        return out_dict


class ConvNext_xxl_classification(nn.Module):
    def __init__(self, hidden_dims, num_class, freeze_backbone=False, normalization='BatchNorm1d', activation='ReLU', dropblock=True, **kwargs):
        super().__init__()
        self.backbone = BackboneLoader('convnext_xxlarge.clip_laion2b_soup_ft_in1k', exportable=True, pretrained=True)
        self.classifier = MLP.SimpleClassifier(in_features=3072, hidden_dims=hidden_dims, num_class=num_class, normalization=normalization, activation=activation, dropblock=dropblock)

        if freeze_backbone:
            for m in self.backbone.parameters():
                m.requires_grad = False

    def load_pretrained(self, dst):
        pretrained_weights = torch.load(dst)
        taget_weights = OrderedDict()
        for key in pretrained_weights.keys():
            if 'backbone.backbone' in key:
                target_key = key.replace('backbone.backbone', 'backbone')
                taget_weights[target_key] = pretrained_weights[key]
        self.backbone.load_state_dict(taget_weights)

    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        out = self.classifier(feat)

        out_dict['class'] = out
        out_dict['feat'] = feat

        return out_dict


class Swin_l_classification(nn.Module):
    def __init__(self, hidden_dims, num_class, freeze_backbone=False, normalization='BatchNorm1d', activation='ReLU', dropblock=True, **kwargs):
        super().__init__()
        self.backbone = BackboneLoader('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', exportable=True, pretrained=True)
        self.classifier = MLP.SimpleClassifier(in_features=1536, hidden_dims=hidden_dims, num_class=num_class, normalization=normalization, activation=activation, dropblock=dropblock)

        if freeze_backbone:
            for m in self.backbone.parameters():
                m.requires_grad = False

    def load_pretrained(self, dst):
        pretrained_weights = torch.load(dst)
        taget_weights = OrderedDict()
        for key in pretrained_weights.keys():
            if 'backbone.backbone' in key:
                target_key = key.replace('backbone.backbone', 'backbone')
                taget_weights[target_key] = pretrained_weights[key]
        self.backbone.load_state_dict(taget_weights)

    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        out = self.classifier(feat.permute(0, 3, 1, 2))

        out_dict['class'] = out
        out_dict['feat'] = feat

        return out_dict