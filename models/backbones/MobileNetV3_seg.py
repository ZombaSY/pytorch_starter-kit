"""Modified MobileNetV3 for use as semantic segmentation feature extractors."""

# https://github.com/ekzhang/fastseg/blob/master/fastseg/model/mobilenetv3.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from geffnet import tf_mobilenetv3_large_100, tf_mobilenetv3_small_100, tf_mobilenetv3_small_minimal_100
from geffnet.efficientnet_builder import InvertedResidual, Conv2dSame, Conv2dSameExport


MODEL_WEIGHTS_URL = {
    ('mobilev3large-lraspp', 256): 'https://github.com/ekzhang/fastseg/releases/download/v0.1-weights/mobilev3large-lraspp-f256-9b613ffd.pt',
    ('mobilev3large-lraspp', 128): 'https://github.com/ekzhang/fastseg/releases/download/v0.1-weights/mobilev3large-lraspp-f128-9cbabfde.pt',
    ('mobilev3small-lraspp', 256): 'https://github.com/ekzhang/fastseg/releases/download/v0.1-weights/mobilev3small-lraspp-f256-d853f901.pt',
    ('mobilev3small-lraspp', 128): 'https://github.com/ekzhang/fastseg/releases/download/v0.1-weights/mobilev3small-lraspp-f128-a39a1e4b.pt',
    ('mobilev3small-lraspp', 64): 'https://github.com/ekzhang/fastseg/releases/download/v0.1-weights/mobilev3small-lraspp-f64-114fc23b.pt',
}


def get_trunk(trunk_name):
    """Retrieve the pretrained network trunk and channel counts"""
    if trunk_name == 'mobilenetv3_large':
        backbone = MobileNetV3_Large(pretrained=True)
        s2_ch = 16
        s4_ch = 24
        high_level_ch = 960
    elif trunk_name == 'mobilenetv3_small':
        backbone = MobileNetV3_Small(pretrained=True)
        s2_ch = 16
        s4_ch = 16
        high_level_ch = 576
    elif trunk_name == 'mobilenetv3_tiny':
        backbone = MobileNetV3_Tiny(pretrained=True)
        s2_ch = 16
        s4_ch = 16
        high_level_ch = 576
    else:
        raise ValueError('unknown backbone {}'.format(trunk_name))
    return backbone, s2_ch, s4_ch, high_level_ch


class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.

    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BaseSegmentation(nn.Module):
    """Module subclass providing useful convenience functions for inference."""

    @classmethod
    def from_pretrained(cls, filename=None, num_filters=128, **kwargs):
        """Load a pretrained model from a .pth checkpoint given by `filename`."""
        if filename is None:
            # Pull default pretrained model from internet
            name = (cls.model_name, num_filters)
            if name in MODEL_WEIGHTS_URL:
                weights_url = MODEL_WEIGHTS_URL[name]
                print(f'Loading pretrained model {name[0]} with F={name[1]}...')
                checkpoint = torch.hub.load_state_dict_from_url(weights_url, map_location='cpu')
            else:
                raise ValueError(f'pretrained weights not found for model {name}, please specify a checkpoint')
        else:
            checkpoint = torch.load(filename, map_location='cpu')
        net = cls(checkpoint['num_classes'], num_filters=num_filters, **kwargs)
        net.load_checkpoint(checkpoint)
        return net

    def load_checkpoint(self, checkpoint):
        """Load weights given a checkpoint object from training."""
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = v
        self.load_state_dict(state_dict)

    def predict_one(self, image, return_prob=False, device=None):
        """Generate and return segmentation for a single image.

        See the documentation of the `predict()` function for more details. This function
        is a convenience wrapper that only returns predictions for a single image, rather
        than an entire batch.
        """
        return self.predict([image], return_prob, device)[0]

    def predict(self, images, return_prob=False, device=None):
        """Generate and return segmentations for a batch of images.

        Keyword arguments:
        images -- a list of PIL images or NumPy arrays to run segmentation on
        return_prob -- whether to return the output probabilities (default False)
        device -- the device to use when running evaluation, defaults to 'cuda' or 'cpu'
            (this must match the device that the model is currently on)

        Returns:
        if `return_prob == False`, a NumPy array of shape (len(images), height, width)
            containing the predicted classes
        if `return_prob == True`, a NumPy array of shape (len(images), num_classes, height, width)
            containing the log-probabilities of each class
        """
        # Determine the device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        # Preprocess images by normalizing and turning into `torch.tensor`s
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ipt = torch.stack([tfms(im) for im in images]).to(device)

        # Run inference
        with torch.no_grad():
            out = self.forward(ipt)

        # Return the output as a `np.ndarray` on the CPU
        if not return_prob:
            out = out.argmax(dim=1)
        return out.cpu().numpy()


class LRASPP(BaseSegmentation):
    """Lite R-ASPP style segmentation network."""
    def __init__(self, num_classes, trunk, use_aspp=False, num_filters=128):
        """Initialize a new segmentation model.

        Keyword arguments:
        num_classes -- number of output classes (e.g., 19 for Cityscapes)
        trunk -- the name of the trunk to use ('mobilenetv3_large', 'mobilenetv3_small')
        use_aspp -- whether to use DeepLabV3+ style ASPP (True) or Lite R-ASPP (False)
            (setting this to True may yield better results, at the cost of latency)
        num_filters -- the number of filters in the segmentation head
        """
        super(LRASPP, self).__init__()

        self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(trunk_name=trunk)
        self.use_aspp = use_aspp

        # Reduced atrous spatial pyramid pooling
        if self.use_aspp:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=12, padding=12),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv3 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=36, padding=36),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            aspp_out_ch = num_filters * 4
        else:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Sigmoid(),
            )
            aspp_out_ch = num_filters

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1)
        self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        s2, s4, final = self.trunk(x)
        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
            ], 1)
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


class MobileV3Large(LRASPP):
    """MobileNetV3-Large segmentation network."""
    model_name = 'mobilev3large-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Large, self).__init__(
            num_classes,
            trunk='mobilenetv3_large',
            **kwargs
        )


class MobileV3Small(LRASPP):
    """MobileNetV3-Small segmentation network."""
    model_name = 'mobilev3small-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Small, self).__init__(
            num_classes,
            trunk='mobilenetv3_small',
            **kwargs
        )


class MobileV3Tiny(LRASPP):
    """MobileNetV3-Small segmentation network."""
    model_name = 'mobilev3tiny-lraspp'

    def __init__(self, num_classes, **kwargs):
        super(MobileV3Tiny, self).__init__(
            num_classes,
            trunk='mobilenetv3_tiny',
            **kwargs
        )


class MobileNetV3_Large(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_large_100, pretrained=False):
        super(MobileNetV3_Large, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[3][0].conv_dw.stride = (1, 1)
        net.blocks[5][0].conv_dw.stride = (1, 1)

        for block_num in (3, 4, 5, 6):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 5:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]

    def forward(self, x):
        x = self.early(x) # 2x
        x = self.block0(x)
        s2 = x
        x = self.block1(x) # 4x
        s4 = x
        x = self.block2(x) # 8x
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return s2, s4, x


class MobileNetV3_Small(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_small_100, pretrained=False):
        super(MobileNetV3_Small, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[2][0].conv_dw.stride = (1, 1)
        net.blocks[4][0].conv_dw.stride = (1, 1)

        for block_num in (2, 3, 4, 5):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 4:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]

    def forward(self, x):
        x = self.early(x) # 2x
        s2 = x
        x = self.block0(x) # 4x
        s4 = x
        x = self.block1(x) # 8x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return s2, s4, x


class MobileNetV3_Tiny(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_small_minimal_100, pretrained=False):
        super(MobileNetV3_Tiny, self).__init__()
        net = trunk(pretrained=pretrained,
                    norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[2][0].conv_dw.stride = (1, 1)
        net.blocks[4][0].conv_dw.stride = (1, 1)

        for block_num in (2, 3, 4, 5):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 4:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(m, Conv2dSameExport):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]

    def forward(self, x):
        x = self.early(x) # 2x
        s2 = x
        x = self.block0(x) # 4x
        s4 = x
        x = self.block1(x) # 8x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return s2, s4, x


if __name__ == '__main__':
    model = MobileNetV3_Tiny(pretrained=False)
    input = torch.rand(1, 3, 512, 512)
    low, mid, x = model(input)
    print(model)
    print(sum(p.numel() for p in model.parameters()), ' parameters')
    print(x.size())
    print(low.size())
    print(mid.size())
