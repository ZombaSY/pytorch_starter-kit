import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.NeighborhoodTransformer import DiNAT_s
from models.blocks import Blocks as Blocks
from models.heads.UPerHead import M_UPerHead
from models.necks import perturbations

from collections import OrderedDict


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


class DiNAT_s_T(nn.Module):
    def __init__(self, num_classes=2, in_channel=3):
        super(DiNAT_s_T, self).__init__()

        self.neighborhood_transformer = DiNAT_s(in_chans=in_channel,
                                                embed_dim=96,
                                                depths=[2, 2, 6, 2],
                                                num_heads=[3, 6, 12, 24],
                                                drop_path_rate=0.3,
                                                patch_norm=True,
                                                kernel_size=7,
                                                dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4], [1, 2]])

        self.uper_head = M_UPerHead(in_channels=[96, 192, 384, 768],
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=num_classes,
                                    align_corners=False,)

    def load_pretrained(self, dst):
        pretrained_states = torch.load(dst)
        pretrained_states_new = OrderedDict()
        for item in pretrained_states.keys():
            if 'neighborhood_transformer' in item:
                key = item.replace('module.', '')
                key = key.replace('neighborhood_transformer.', '')
                if key[:4] == 'norm':
                    key = key.replace('norm', '')
                    key = 'norm_layer_list.' + key
                pretrained_states_new[key] = pretrained_states[item]

        self.neighborhood_transformer.load_state_dict(pretrained_states_new)

    def load_pretrained_imagenet(self, dst, device):
        pretrained_states = torch.load(dst)
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if 'head.weight' == item or 'head.bias' == item or 'norm.weight' == item or 'norm.bias' == item or 'layers.0.blocks.1.attn_mask' == item or 'layers.1.blocks.1.attn_mask' == item or 'layers.2.blocks.1.attn_mask' == item or 'layers.2.blocks.3.attn_mask' == item or 'layers.2.blocks.5.attn_mask' == item:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.neighborhood_transformer.remove_fpn_norm_layers()
        self.neighborhood_transformer.load_state_dict(pretrained_states_backbone)
        self.neighborhood_transformer.add_fpn_norm_layers()

        self.to(device)

    def forward(self, x):
        x_size = x.shape[2:]

        feat1, feat2, feat3, feat4 = self.neighborhood_transformer(x)

        feat = self.uper_head(feat1, feat2, feat3, feat4)
        feat = F.interpolate(feat, x_size, mode='bilinear', align_corners=False)

        return feat, [feat1, feat2, feat3, feat4]


class DiNAT_s_T_segMap_score_multi_stem_repr(DiNAT_s_T):
    def __init__(self, num_classes=2, in_channel=3):
        super(DiNAT_s_T_segMap_score_multi_stem_repr, self).__init__(num_classes, in_channel)

        self.classifier = Blocks.Classifier_map_score_multi_stem()

        self.perturbations_pos = nn.Sequential(*[
            perturbations.FeatureNoise(),
            perturbations.VATDecoder(16, 768, num_classes),
            perturbations.DropOut(),
        ])

        self.perturbations_neg = nn.Sequential(*[
            perturbations.VATDecoderNegative(16, 768, num_classes),
        ])

        self.repr = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def train_score_callback(self):
        self.uper_head.eval()

    def forward(self, x, is_val=False):
        x_h, x_w, = x.shape[-2:]

        feats = self.neighborhood_transformer(x)

        out = {}
        if not is_val:
            # merge perturbations with batch
            feat_perturbation_pos = torch.cat([perturb(feats) for perturb in self.perturbations_pos], dim=0)
            feat_perturbation_neg = torch.cat([perturb(feats) for perturb in self.perturbations_neg], dim=0)

            feat_repr_pos = self.repr(feat_perturbation_pos)
            feat_repr_neg = self.repr(feat_perturbation_neg)
            feats_repr = self.repr(feats[-1])

            out['feats_repr'] = feats_repr
            out['feat_repr_pos'] = feat_repr_pos
            out['feat_repr_neg'] = feat_repr_neg

        seg_map = self.uper_head(feats)
        seg_map = F.interpolate(seg_map, (x_h, x_w), mode='bilinear', align_corners=False)

        seg_map_score_1 = torch.sum(torch.softmax(seg_map, dim=1)[:, 1], dim=(1, 2)).unsqueeze(-1) / (x_h * x_w)
        seg_map_score_2 = torch.sum(torch.softmax(seg_map, dim=1)[:, 2], dim=(1, 2)).unsqueeze(-1) / (x_h * x_w)
        seg_map_score = torch.cat([seg_map_score_1, seg_map_score_2], dim=1)

        score = self.classifier(feats[-1], seg_map_score)

        out['seg_map'] = seg_map
        out['score'] = score

        return out
