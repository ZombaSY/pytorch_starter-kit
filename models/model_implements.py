import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.Swin import SwinTransformer
from models.heads.UPerHead import M_UPerHead
from models.backbones import timm_backbone
from models.backbones import MobileOne
from models.heads import MLP
from collections import OrderedDict


class Swin_T(nn.Module):
    def __init__(self, conf_model, base_c=96):
        super().__init__()

        self.swin_transformer = SwinTransformer(in_chans=conf_model['in_channel'],
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


class Swin_T_semanticSegmentation(Swin_T):
    def __init__(self, conf_model, base_c=96):
        super().__init__(conf_model, base_c)

        self.uper_head = M_UPerHead(in_channels=[base_c, base_c * 2, base_c * 4, base_c * 8],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_class=conf_model['num_class'],
                                    align_corners=False,)

    def forward(self, x):
        out_dict = {}
        x_size = x.shape[2:]

        feats = self.swin_transformer(x)
        seg_map = self.uper_head(feats)
        out_dict['seg'] = F.interpolate(seg_map, x_size, mode='bilinear', align_corners=False)

        return out_dict


class ConvNextV2_l_regression(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = timm_backbone.BackboneLoader('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', exportable=True, pretrained=True)
        self.classifier1 = MLP.SimpleClassifier(in_features=1536, num_class=conf_model['num_class'], normalization=conf_model['normalization'], activation=conf_model['activation'], dropblock=conf_model['dropblock'])
        self.classifier2 = MLP.SimpleClassifier(in_features=1536, num_class=conf_model['num_class'], normalization=conf_model['normalization'], activation=conf_model['activation'], dropblock=conf_model['dropblock'])


    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        score1 = self.classifier1(feat)
        score2 = self.classifier2(feat)

        out_dict['vec'] = torch.cat([score1, score2], dim=1)
        out_dict['feat'] = feat

        return out_dict



class Mobileone_s0_landmark(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = MobileOne.mobileone(variant='s0')
        self.classifier = MLP.SimpleClassifier(in_features=1024, num_class=conf_model['num_class'], normalization=conf_model['normalization'], activation=conf_model['activation'], dropblock=conf_model['dropblock'])

    def forward(self, x):
        out_dict = {}

        x = self.backbone(x)
        x_reg = self.classifier(x)
        out_dict['vec'] = x_reg * 0.5 + 0.5

        return out_dict


class Mobileone_s0_regression(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = MobileOne.mobileone(variant='s0')
        self.classifier1 = MLP.SimpleClassifier(in_features=1024, num_class=conf_model['num_class'], normalization=conf_model['normalization'], activation=conf_model['activation'], dropblock=conf_model['dropblock'])
        self.classifier2 = MLP.SimpleClassifier(in_features=1024, num_class=conf_model['num_class'], normalization=conf_model['normalization'], activation=conf_model['activation'], dropblock=conf_model['dropblock'])

    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        score1 = self.classifier1(feat)
        score2 = self.classifier2(feat)

        out_dict['vec'] = torch.cat([score1, score2], dim=1)
        out_dict['feat'] = feat

        return out_dict
