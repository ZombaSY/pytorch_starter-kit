import torch
import torch.nn as nn

from collections import OrderedDict
from models.blocks.Blocks import Upsample
from models import model_implements
from models.heads.UPerHead import M_UPerHead
from models.necks.ResNeSt import SplitAttention


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


# Unet + Unet-tiny + swin + U-DeepLab-Resnet14
class Ensemble5(nn.Module):
    def __init__(self, pre_trained_path=None, n_channels=3, n_classes=2, freeze_extractor=True):
        super(Ensemble5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.unet = nn.DataParallel(model_implements.Unet())
        self.unet_tiny = nn.DataParallel(model_implements.Unet_Tiny())
        self.swin = nn.DataParallel(model_implements.Swin(num_classes=self.n_classes))
        self.u_deeplab_resnet14 = nn.DataParallel(model_implements.U_DeepLab_14(num_classes=self.n_classes))

        self.uper_head = M_UPerHead(in_channels=[320, 576, 1152, 1664],     # vary by ensemble model
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=self.n_classes,
                                    align_corners=False, )

        if pre_trained_path is not None:
            self.unet.load_state_dict(torch.load(pre_trained_path[0]))
            self.unet_tiny.load_state_dict(torch.load(pre_trained_path[1]))
            self.swin.load_state_dict(torch.load(pre_trained_path[2]))
            self.u_deeplab_resnet14.load_state_dict(torch.load(pre_trained_path[3]))

        self.__remove_decoder()
        if freeze_extractor:
            self.__freeze_feature_extractor()

    @staticmethod
    def __get_blocks_to_be_concat(model, x):
        shapes = set()
        blocks = OrderedDict()
        hooks = []
        count = 0

        def register_hook(module):

            def hook(module, input, output):
                try:
                    nonlocal count
                    if module.name == f'blocks_{count}_output_batch_norm':
                        count += 1
                        shape = output.size()[-2:]
                        if shape not in shapes:
                            shapes.add(shape)
                            blocks[module.name] = output

                    elif module.name == 'head_swish':
                        # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                        # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                        # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                        # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                        # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                        # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                        blocks.popitem()
                        blocks[module.name] = output

                except AttributeError:
                    pass

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        # register hook
        model.apply(register_hook)

        # make a forward pass to trigger the hooks
        model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return blocks

    def __remove_decoder(self):
        # remove useless heads, decoders
        del self.unet.module.up1
        del self.unet.module.up2
        del self.unet.module.up3
        del self.unet.module.up4
        del self.unet.module.outc

        del self.unet_tiny.module.up1
        del self.unet_tiny.module.up2
        del self.unet_tiny.module.up3
        del self.unet_tiny.module.up4
        del self.unet_tiny.module.outc_5

        del self.swin.module.uper_head

        del self.u_deeplab_resnet14.module.aspp
        del self.u_deeplab_resnet14.module.final

    def __freeze_feature_extractor(self):
        # freeze the feature extractor
        self.unet_tiny.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.swin.module.requires_grad_(False)
        self.u_deeplab_resnet14.requires_grad_(False)

    def forward(self, x):
        _, _, h, w = x.shape

        x_unet = self.unet.module.inc(x)
        x_unet_down = []
        x_unet_down.append(self.unet.module.down1(x_unet))
        x_unet_down.append(self.unet.module.down2(x_unet_down[0]))
        x_unet_down.append(self.unet.module.down3(x_unet_down[1]))
        x_unet_down.append(self.unet.module.down4(x_unet_down[2]))

        x_unet_tiny = self.unet_tiny.module.inc(x)
        x_unet_tiny_down = []
        x_unet_tiny_down.append(self.unet_tiny.module.down1(x_unet_tiny))
        x_unet_tiny_down.append(self.unet_tiny.module.down2(x_unet_tiny_down[0]))
        x_unet_tiny_down.append(self.unet_tiny.module.down3(x_unet_tiny_down[1]))
        x_unet_tiny_down.append(self.unet_tiny.module.down4(x_unet_tiny_down[2]))

        x_swin_down = self.swin.module.swin_transformer(x)

        x_resnet14_down = list(self.u_deeplab_resnet14.module.resnet(x))
        x_resnet14_down[0] = Upsample(x_resnet14_down[0], (h // 2, w // 2))       # specific case in Resnet

        x_downs = []

        for i in range(1, 5):  # concatenate feature per depth
            scale = 32 // (2 ** i)
            x_unet = x_unet_down[4 - i]
            x_unet_tiny = x_unet_tiny_down[4 - i]
            x_swin = Upsample(x_swin_down[4 - i], (h // scale, w // scale))  # (h, w) // 32 to (h, w) // 16
            x_resnet14 = x_resnet14_down[4 - i]

            x_feat = torch.cat((x_unet, x_unet_tiny, x_swin, x_resnet14), dim=1)
            x_downs.append(x_feat)

        x_downs.reverse()

        out = self.uper_head(x_downs)
        out = Upsample(out, (h, w))

        return out


# Unet + Unet-tiny + swin + U-DeepLab-Resnet14
class Ensemble5_Attention(nn.Module):
    def __init__(self, pre_trained_path=None, n_channels=3, n_classes=2, freeze_extractor=True):
        super(Ensemble5_Attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.unet = nn.DataParallel(model_implements.Unet())
        self.unet_tiny = nn.DataParallel(model_implements.Unet_Tiny())
        self.swin = nn.DataParallel(model_implements.Swin(num_classes=self.n_classes))
        self.u_deeplab_resnet14 = nn.DataParallel(model_implements.U_DeepLab_14(num_classes=self.n_classes))

        self.split_attention = SplitAttention(channels=1664, cardinality=8, radix=16)
        self.uper_head = M_UPerHead(in_channels=[320, 576, 1152, 1664],     # vary by ensemble model
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=self.n_classes,
                                    align_corners=False, )

        if pre_trained_path is not None:
            self.unet.load_state_dict(torch.load(pre_trained_path[0]))
            self.unet_tiny.load_state_dict(torch.load(pre_trained_path[1]))
            self.swin.load_state_dict(torch.load(pre_trained_path[2]))
            self.u_deeplab_resnet14.load_state_dict(torch.load(pre_trained_path[3]))

        self.__remove_decoder()
        if freeze_extractor:
            self.__freeze_feature_extractor()

    @staticmethod
    def __get_blocks_to_be_concat(model, x):
        shapes = set()
        blocks = OrderedDict()
        hooks = []
        count = 0

        def register_hook(module):

            def hook(module, input, output):
                try:
                    nonlocal count
                    if module.name == f'blocks_{count}_output_batch_norm':
                        count += 1
                        shape = output.size()[-2:]
                        if shape not in shapes:
                            shapes.add(shape)
                            blocks[module.name] = output

                    elif module.name == 'head_swish':
                        # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                        # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                        # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                        # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                        # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                        # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                        blocks.popitem()
                        blocks[module.name] = output

                except AttributeError:
                    pass

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        # register hook
        model.apply(register_hook)

        # make a forward pass to trigger the hooks
        model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return blocks

    def __remove_decoder(self):
        # remove useless heads, decoders
        del self.unet.module.up1
        del self.unet.module.up2
        del self.unet.module.up3
        del self.unet.module.up4
        del self.unet.module.outc

        del self.unet_tiny.module.up1
        del self.unet_tiny.module.up2
        del self.unet_tiny.module.up3
        del self.unet_tiny.module.up4
        del self.unet_tiny.module.outc_5

        del self.swin.module.uper_head

        del self.u_deeplab_resnet14.module.aspp
        del self.u_deeplab_resnet14.module.final

    def __freeze_feature_extractor(self):
        # freeze the feature extractor
        self.unet_tiny.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.swin.module.requires_grad_(False)
        self.u_deeplab_resnet14.requires_grad_(False)

    def forward(self, x):
        _, _, h, w = x.shape

        x_unet = self.unet.module.inc(x)
        x_unet_down = []
        x_unet_down.append(self.unet.module.down1(x_unet))
        x_unet_down.append(self.unet.module.down2(x_unet_down[0]))
        x_unet_down.append(self.unet.module.down3(x_unet_down[1]))
        x_unet_down.append(self.unet.module.down4(x_unet_down[2]))

        x_unet_tiny = self.unet_tiny.module.inc(x)
        x_unet_tiny_down = []
        x_unet_tiny_down.append(self.unet_tiny.module.down1(x_unet_tiny))
        x_unet_tiny_down.append(self.unet_tiny.module.down2(x_unet_tiny_down[0]))
        x_unet_tiny_down.append(self.unet_tiny.module.down3(x_unet_tiny_down[1]))
        x_unet_tiny_down.append(self.unet_tiny.module.down4(x_unet_tiny_down[2]))

        x_swin_down = self.swin.module.swin_transformer(x)

        x_resnet14_down = list(self.u_deeplab_resnet14.module.resnet(x))
        x_resnet14_down[0] = Upsample(x_resnet14_down[0], (h // 2, w // 2))       # specific case in Resnet

        x_downs = []

        for i in range(1, 5):  # concatenate feature per depth
            scale = 32 // (2 ** i)
            x_unet = x_unet_down[4 - i]
            x_unet_tiny = x_unet_tiny_down[4 - i]
            x_swin = Upsample(x_swin_down[4 - i], (h // scale, w // scale))  # (h, w) // 32 to (h, w) // 16
            x_resnet14 = x_resnet14_down[4 - i]

            x_feat = torch.cat((x_unet, x_unet_tiny, x_swin, x_resnet14), dim=1)
            x_downs.append(x_feat)

        x_downs.reverse()
        x_downs[3] = self.split_attention(x_downs[3])

        out = self.uper_head(x_downs)
        out = Upsample(out, (h, w))

        return out


# Unet + Unet-tiny + swin + U-DeepLab-Resnet14 -> Decision convolution
class Ensemble7(nn.Module):
    def __init__(self, pre_trained_path=None, n_channels=3, n_classes=2, freeze_extractor=True):
        super(Ensemble7, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.unet = nn.DataParallel(model_implements.Unet())
        self.unet_tiny = nn.DataParallel(model_implements.Unet_Tiny())
        self.swin = nn.DataParallel(model_implements.Swin(num_classes=self.n_classes))
        self.u_deeplab_resnet14 = nn.DataParallel(model_implements.U_DeepLab_14(num_classes=self.n_classes))

        self.conv_seg = nn.Conv2d(8, 2, kernel_size=1)
        if pre_trained_path is not None:
            self.unet.load_state_dict(torch.load(pre_trained_path[0]))
            self.unet_tiny.load_state_dict(torch.load(pre_trained_path[1]))
            self.swin.load_state_dict(torch.load(pre_trained_path[2]))
            self.u_deeplab_resnet14.load_state_dict(torch.load(pre_trained_path[3]))

        if freeze_extractor:
            self.__freeze_feature_extractor()

    @staticmethod
    def __get_blocks_to_be_concat(model, x):
        shapes = set()
        blocks = OrderedDict()
        hooks = []
        count = 0

        def register_hook(module):

            def hook(module, input, output):
                try:
                    nonlocal count
                    if module.name == f'blocks_{count}_output_batch_norm':
                        count += 1
                        shape = output.size()[-2:]
                        if shape not in shapes:
                            shapes.add(shape)
                            blocks[module.name] = output

                    elif module.name == 'head_swish':
                        # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                        # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                        # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                        # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                        # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                        # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                        blocks.popitem()
                        blocks[module.name] = output

                except AttributeError:
                    pass

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        # register hook
        model.apply(register_hook)

        # make a forward pass to trigger the hooks
        model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return blocks

    def __remove_decoder(self):
        # remove useless heads, decoders
        del self.unet.module.up1
        del self.unet.module.up2
        del self.unet.module.up3
        del self.unet.module.up4
        del self.unet.module.outc

        del self.unet_tiny.module.up1
        del self.unet_tiny.module.up2
        del self.unet_tiny.module.up3
        del self.unet_tiny.module.up4
        del self.unet_tiny.module.outc_5

        del self.swin.module.uper_head

        del self.u_deeplab_resnet14.module.aspp
        del self.u_deeplab_resnet14.module.final

    def __freeze_feature_extractor(self):
        # freeze the feature extractor
        self.unet_tiny.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.swin.module.requires_grad_(False)
        self.u_deeplab_resnet14.requires_grad_(False)

    def forward(self, x):

        x_unet = self.unet(x)
        x_unet_tiny = self.unet_tiny(x)
        x_swin = self.swin(x)
        x_u_deeplab_resnet14 = self.u_deeplab_resnet14(x)

        x_cat = torch.cat((x_unet, x_unet_tiny, x_swin, x_u_deeplab_resnet14), dim=1)
        out = self.conv_seg(x_cat)

        return out


# Unet + Unet-tiny + swin + U-DeepLab-Resnet14 -> Feature convolution
class Ensemble8(nn.Module):
    def __init__(self, pre_trained_path=None, n_channels=3, n_classes=2, freeze_extractor=True):
        super(Ensemble8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.unet_no_seg = nn.DataParallel(model_implements.Unet_no_seg())
        self.unet_tiny_no_seg = nn.DataParallel(model_implements.Unet_Tiny_no_seg())
        self.swin_no_seg = nn.DataParallel(model_implements.Swin_no_seg(num_classes=self.n_classes))
        self.u_deeplab_resnet14_no_seg = nn.DataParallel(model_implements.U_DeepLab_14_no_seg(num_classes=self.n_classes))

        self.conv_seg = nn.Conv2d(64 + 16 + 512 + 320, 2, kernel_size=1)
        if pre_trained_path is not None:
            self.unet_no_seg.load_state_dict(torch.load(pre_trained_path[0]))
            self.unet_tiny_no_seg.load_state_dict(torch.load(pre_trained_path[1]))
            self.swin_no_seg.load_state_dict(torch.load(pre_trained_path[2]))
            self.u_deeplab_resnet14_no_seg.load_state_dict(torch.load(pre_trained_path[3]))

        if freeze_extractor:
            self.__freeze_feature_extractor()

    def __remove_decoder(self):
        # remove useless heads, decoders
        del self.unet.module.outc

        del self.unet_tiny.module.outc_5

        del self.swin.module.uper_head.conv_seg

        del self.u_deeplab_resnet14.module.final

    def __freeze_feature_extractor(self):
        # freeze the feature extractor
        self.unet_no_seg.requires_grad_(False)
        self.unet_tiny_no_seg.requires_grad_(False)
        self.swin_no_seg.module.requires_grad_(False)
        self.u_deeplab_resnet14_no_seg.requires_grad_(False)

    def forward(self, x):

        x_unet = self.unet_no_seg(x)
        x_unet_tiny = self.unet_tiny_no_seg(x)
        x_swin = self.swin_no_seg(x)
        x_u_deeplab_resnet14 = self.u_deeplab_resnet14_no_seg(x)
        x_cat = torch.cat((x_unet, x_unet_tiny, x_swin, x_u_deeplab_resnet14), dim=1)
        out = self.conv_seg(x_cat)

        return out


# Swin + SwinV2_L + Swin_Attention
class Ensemble_S_SAtt(nn.Module):
    def __init__(self, pre_trained_path=None, n_channels=3, n_classes=2, freeze_extractor=True):
        super(Ensemble_S_SAtt, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.swin = nn.DataParallel(model_implements.Swin())
        self.swin_att = nn.DataParallel(model_implements.Swin_Attention())

        self.uper_head = M_UPerHead(in_channels=[192, 384, 768, 1536],     # vary by ensemble model
                                    in_index=[0, 1, 2, 3],
                                    pool_scales=(1, 2, 3, 6),
                                    channels=512,
                                    dropout_ratio=0.1,
                                    num_classes=self.n_classes,
                                    align_corners=False, )

        if pre_trained_path is not None:
            self.swin.load_state_dict(torch.load(pre_trained_path[0]))
            self.swin_att.load_state_dict(torch.load(pre_trained_path[1]))

        self.__remove_decoder()
        if freeze_extractor:
            self.__freeze_feature_extractor()

    def __remove_decoder(self):
        # remove useless heads, decoders

        del self.swin.module.uper_head
        del self.swin_att.module.uper_head

    def __freeze_feature_extractor(self):
        # freeze the feature extractor)
        self.swin.module.requires_grad_(False)
        self.swin_att.module.requires_grad_(False)

    def forward(self, x):
        _, _, h, w = x.shape

        x_swin_down = self.swin.module.swin_transformer(x)
        x_swin_down = list(x_swin_down)

        x_swin_att_down = self.swin_att.module.swin_transformer(x)
        x_swin_att_down = list(x_swin_att_down)
        for idx, att in enumerate(self.swin_att.module.attentions):
            x_swin_att_down[idx] = att(x_swin_att_down[idx])

        # concatenate feature pyramid
        x_downs = []
        for i in range(1, 5):  # concatenate feature per depth
            # scale = 32 // (2 ** i)

            x_swin = x_swin_down[i-1]
            x_swin_att = x_swin_att_down[i-1]

            x_feat = torch.cat((x_swin, x_swin_att), dim=1)
            x_downs.append(x_feat)

        feat = self.uper_head(x_downs)

        out = Upsample(feat, (h, w))

        return out
