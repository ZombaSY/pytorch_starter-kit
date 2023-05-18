import torch
import torch.nn.functional as F
import math
import numpy as np

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


def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation


def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    x_feat = x[-1]
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x_feat, torch.tensor(clip_min, device=x_feat.device, dtype=x_feat.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x_feat, torch.tensor(clip_max, device=x_feat.device, dtype=x_feat.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x_feat = x_feat.clone().detach().to(torch.float).requires_grad_(True)
    x[-1] = x_feat

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()

    x_pred = F.interpolate(model_fn(x), y.shape[-2:])
    loss = loss_fn(x_pred, y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x_feat.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x_feat + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


class ProjectedGradientDescent(nn.Module):
    def __init__(self, model_fn, eps=0.3, eps_iter=0.01, nb_iter=7, norm=np.inf, targeted=False):
        super(ProjectedGradientDescent, self).__init__()
        self.model_fn = model_fn
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.norm = norm
        self.targeted = targeted
        self.projected_gradient_descent = self.projected_gradient_descent

    def projected_gradient_descent(
            self,
            x,
            clip_min=None,
            clip_max=None,
            y=None,
            rand_init=True,
            rand_minmax=None,
            sanity_checks=True,
    ):
        """
        This class implements either the Basic Iterative Method
        (Kurakin et al. 2016) when rand_init is set to False. or the
        Madry et al. (2017) method if rand_init is set to True.
        Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
        Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
        :param model_fn: a callable that takes an input tensor and returns the model logits.
        :param x: input tensor.
        :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
        :param eps_iter: step size for each attack iteration
        :param nb_iter: Number of attack iterations.
        :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional) float. Minimum float value for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param y: (optional) Tensor with true labels. If targeted is true, then provide the
                  target label. Otherwise, only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used
                  as labels to avoid the "label leaking" effect (explained in this paper:
                  https://arxiv.org/abs/1611.01236). Default is None.
        :param targeted: (optional) bool. Is the attack targeted or untargeted?
                  Untargeted, the default, will try to make the label incorrect.
                  Targeted will instead try to move in the direction of being more like y.
        :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
        :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
                  which the random perturbation on x was drawn. Effective only when rand_init is
                  True. Default equals to eps.
        :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
                  memory or for unit tests that intentionally pass strange input)
        :return: a tensor for the adversarial example
        """
        x_feat = x[-1]
        if self.norm == 1:
            raise NotImplementedError(
                "It's not clear that FGM is a good inner loop"
                " step for PGD when norm=1, because norm=1 FGM "
                " changes only one pixel at a time. We need "
                " to rigorously test a strong norm=1 PGD "
                "before enabling this feature."
            )
        if self.norm not in [np.inf, 2]:
            raise ValueError("Norm order must be either np.inf or 2.")
        if self.eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(self.eps)
            )
        if self.eps == 0:
            return x
        if self.eps_iter < 0:
            raise ValueError(
                "eps_iter must be greater than or equal to 0, got {} instead".format(
                    self.eps_iter
                )
            )
        if self.eps_iter == 0:
            return x

        assert self.eps_iter <= self.eps, (self.eps_iter, self.eps)
        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            assert_ge = torch.all(
                torch.ge(x_feat, torch.tensor(clip_min, device=x_feat.device, dtype=x_feat.dtype))
            )
            asserts.append(assert_ge)

        if clip_max is not None:
            assert_le = torch.all(
                torch.le(x_feat, torch.tensor(clip_max, device=x_feat.device, dtype=x_feat.dtype))
            )
            asserts.append(assert_le)

        # Initialize loop variables
        if rand_init:
            if rand_minmax is None:
                rand_minmax = self.eps
            eta = torch.zeros_like(x_feat).uniform_(-rand_minmax, rand_minmax)
        else:
            eta = torch.zeros_like(x_feat)

        # Clip eta
        eta = clip_eta(eta, self.norm, self.eps)
        adv_x = x_feat + eta
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

        x_adv_feats = x[:-1]
        x_adv_feats.append(adv_x)
        i = 0
        while i < self.nb_iter:
            x_adv_feats[-1] = adv_x
            adv_x = fast_gradient_method(
                self.model_fn,
                x_adv_feats,
                self.eps_iter,
                self.norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y,
                targeted=self.targeted,
            )

            # Clipping perturbation eta to norm norm ball
            eta = adv_x - x_feat
            eta = clip_eta(eta, self.norm, self.eps)
            adv_x = x_feat + eta

            # Redo the clipping.
            # FGM already did it, but subtracting and re-adding eta can add some
            # small numerical error.
            if clip_min is not None or clip_max is not None:
                adv_x = torch.clamp(adv_x, clip_min, clip_max)
            i += 1

        asserts.append(self.eps_iter <= self.eps)
        if self.norm == np.inf and clip_min is not None:
            # TODO necessary to cast clip_min and clip_max to x.dtype?
            asserts.append(self.eps + clip_min <= clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return adv_x

    def forward(self, x, y):
        x_adv = self.projected_gradient_descent(x, y=y)
        return x_adv
