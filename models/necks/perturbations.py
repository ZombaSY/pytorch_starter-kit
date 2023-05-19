import torch
import torch.nn.functional as F
import math

from models import losses
from torch import nn
from torch.distributions.uniform import Uniform


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def forward(self, x):
        x = x[-1]
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise


class DropOut(nn.Module):
    def __init__(self, drop_rate=0.2, spatial_dropout=True):
        super(DropOut, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x):
        x = x[-1]
        return self.dropout(x)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * (scale ** 2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = x[-1]
        r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        return r_adv


class VATDecoderNegative(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoderNegative, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = x[-1]
        r_adv = -get_r_adv(x, self.upsample, self.it, self.xi, self.eps)    # to negative direction
        return r_adv


# https://github.com/yaoyugua/ProjectedGradientDescent/blob/main/PGD-pytorch/PGD.ipynb
class ProjectedGradientDescent(nn.Module):
    def __init__(self, model_head, eps=0.3, alpha=2 / 255, iters=7, targeted=True):
        super(ProjectedGradientDescent, self).__init__()
        self.model_head = model_head
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.targeted = targeted

    def forward(self, feat):
        criterion = losses.CrossEntropy().to('cuda')

        feat_origin = feat[-1].data
        feat_copy = feat[-1].data

        for i in range(self.iters):
            feat_copy.requires_grad = True
            feat[-1] = feat_copy
            outputs = self.model_head(feat)
            _, y = torch.max(outputs, 1)
            outputs = F.interpolate(outputs, y.shape[-2:])

            self.model_head.zero_grad()
            loss = criterion(outputs, y)
            loss.backward(retain_graph=True)

            # here we do x(t+1) = x(t) +  alpha sign(grad_over_image(L))
            feat_adv = feat_copy + self.alpha * feat_copy.grad.sign()  # core

            eta = torch.clamp(feat_adv - feat_origin, min=-self.eps, max=self.eps)
            if self.targeted:
                eta = -eta

            feat_copy = (feat_origin + eta).detach()

        return feat_copy
