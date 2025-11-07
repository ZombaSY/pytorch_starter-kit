from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import MobileOne, timm_backbone
from models.backbones.Swin import SwinTransformer
from models.heads import MLP
from models.heads.UPerHead import M_UPerHead_dsv
from tools import utils_tool


class Swin_t(nn.Module):
    def __init__(self, conf_model, base_c=96):
        super().__init__()

        self.swin_transformer = SwinTransformer(
            in_chans=conf_model["in_channel"],
            embed_dim=base_c,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=False,
        )

    def load_pretrained_imagenet(self, dst):
        pretrained_states = torch.load(dst)["model"]
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if (
                "head.weight" == item
                or "head.bias" == item
                or "norm.weight" == item
                or "norm.bias" == item
                or "layers.0.blocks.1.attn_mask" == item
                or "layers.1.blocks.1.attn_mask" == item
                or "layers.2.blocks.1.attn_mask" == item
                or "layers.2.blocks.3.attn_mask" == item
                or "layers.2.blocks.5.attn_mask" == item
            ):
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.swin_transformer.remove_fpn_norm_layers()  # temporally remove fpn norm layers that not included on public-release model
        self.swin_transformer.load_state_dict(pretrained_states_backbone)
        self.swin_transformer.add_fpn_norm_layers()

    def forward(self, x):
        feat1, feat2, feat3, feat4 = self.swin_transformer(x)
        out_dict = {"feats": [feat1, feat2, feat3, feat4]}

        return out_dict


class Swin_t_semanticSegmentation_dsv(Swin_t):
    def __init__(self, conf_model, base_c=96):
        super().__init__(conf_model, base_c)

        self.uper_head = M_UPerHead_dsv(
            in_channels=[base_c, base_c * 2, base_c * 4, base_c * 8],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_class=conf_model["num_class"],
            align_corners=False,
        )

    def forward(self, x):
        x_size = x.shape[2:]

        # get segmentation map
        feats = self.swin_transformer(x)

        out_dict = self.uper_head(*feats)
        out_dict["seg"] = F.interpolate(out_dict["seg"], x_size, mode="bilinear", align_corners=False)
        for i in range(len(out_dict["seg_aux"])):
            out_dict["seg_aux"][i] = F.interpolate(out_dict["seg_aux"][i], x_size, mode="bilinear", align_corners=False)
        out_dict["feats"] = feats

        return out_dict


class ConvNextV2_l_regression(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = timm_backbone.BackboneLoader(
            "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384", exportable=True, pretrained=True
        )
        self.classifier1 = MLP.SimpleClassifier(
            in_features=1536,
            num_class=conf_model["num_class"],
            normalization=conf_model["normalization"],
            activation=conf_model["activation"],
            dropblock=conf_model["dropblock"],
        )
        self.classifier2 = MLP.SimpleClassifier(
            in_features=1536,
            num_class=conf_model["num_class"],
            normalization=conf_model["normalization"],
            activation=conf_model["activation"],
            dropblock=conf_model["dropblock"],
        )

    def forward(self, x, is_perturb=False):
        if not is_perturb:
            out_dict = {}

            feat = self.backbone(x)
            score1 = self.classifier1(feat)
            score2 = self.classifier2(feat)

            out_dict["vec"] = torch.cat([score1, score2], dim=1)
            out_dict["feat"] = feat

            return out_dict
        else:
            feat = x["feat"]
            out_dict = self.backbone.forward_perturb(feat)

            out_dicts = []
            for i in range(len(out_dict)):
                score1 = self.classifier1(out_dict[i])
                score2 = self.classifier2(out_dict[i])
                out_dicts.append(torch.cat([score1, score2], dim=1))

            return out_dicts


class Mobileone_s0_landmark(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = MobileOne.mobileone(variant="s0")
        self.classifier = MLP.SimpleLandmarker(
            channel_in=1024, num_points=conf_model["num_class"], input_size=conf_model["input_size"]
        )

    def load_pretrained_imagenet(self, dst):
        pretrained_states = torch.load(dst)
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if item in ["linear.weight", "linear.bias"]:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.backbone.load_state_dict(pretrained_states_backbone)

    def forward(self, x):
        out_dict = {}

        x = self.backbone(x)
        x_reg = self.classifier(x)
        out_dict["vec"] = x_reg * 0.5 + 0.5

        return out_dict


class Mobileone_s0_regression(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = MobileOne.mobileone(variant="s0")
        self.classifier1 = MLP.SimpleClassifier(
            in_features=1024,
            num_class=conf_model["num_class"],
            normalization=conf_model["normalization"],
            activation=conf_model["activation"],
            dropblock=conf_model["dropblock"],
        )
        self.classifier2 = MLP.SimpleClassifier(
            in_features=1024,
            num_class=conf_model["num_class"],
            normalization=conf_model["normalization"],
            activation=conf_model["activation"],
            dropblock=conf_model["dropblock"],
        )

    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        score1 = self.classifier1(feat)
        score2 = self.classifier2(feat)

        out_dict["vec"] = torch.cat([score1, score2], dim=1)
        out_dict["feat"] = feat

        return out_dict


class Mobileone_s0_classification(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.backbone = MobileOne.mobileone(variant="s0")
        self.classifier = MLP.SimpleClassifier(
            in_features=1024,
            num_class=conf_model["num_class"],
            normalization=conf_model["normalization"],
            activation=conf_model["activation"],
            dropblock=conf_model["dropblock"],
        )

    def forward(self, x):
        out_dict = {}

        feat = self.backbone(x)
        logits = self.classifier(feat)

        out_dict["vec"] = logits
        out_dict["feat"] = feat

        return out_dict


class SegmentatorRegressor(nn.Module):
    def __init__(self, conf_model):
        super().__init__()
        self.segmentator = utils_tool.init_model(conf_model, "cpu")  # recursive initialization
        if conf_model["model"]["saved_ckpt"] != "":  # for ineference
            self.segmentator.module.load_state_dict(torch.load(conf_model["model"]["saved_ckpt"]))
        self.backbone = MobileOne.mobileone(variant="s0")
        self.classifier = MLP.SimpleRegressor(channel_in=128, num_class=conf_model['num_class'])
        self.train_callback()

    def set_bn_eval(self, m):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    def train_callback(self):
        for p in self.segmentator.parameters():
            p.requires_grad = False
        self.segmentator.eval()
        self.apply(self.set_bn_eval)

    def forward(self, x):
        out = self.segmentator(x)
        out_dict = {}

        seg_map = torch.softmax(out['seg'], dim=1)
        seg_mask = sum(torch.unbind(seg_map, dim=1)[1:])    # get the RoI mask from segmentation result
        seg_mask = torch.clamp(seg_mask, 0, 1).unsqueeze(1) / 2
        mask = torch.zeros_like(seg_mask) + 0.5
        mask = mask + seg_mask
        x_mask = x * mask

        feat = self.backbone(x_mask)
        score = self.classifier(feat[-1])

        out_dict['seg'] = out['seg']
        out_dict['vec'] = score
        out_dict['feat'] = feat[-1]

        return out_dict
