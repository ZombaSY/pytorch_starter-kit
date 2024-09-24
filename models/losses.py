import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import math

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve


class CrossEntropyLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        y = y.squeeze(1)

        return self.loss(x, y)


class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    """
    def __init__(self, conf):
        super().__init__()
        self.gamma = conf['gamma']
        self.weight = conf['weight']

    def forward(self, y_pred, y):
        """
        input: [N, C]
        target: [N, 1]
        """
        y = y.squeeze(1)
        log_pt = F.log_softmax(y_pred, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = F.nll_loss(log_pt, y, self.weight)

        return loss


class KLDivergenceLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction=conf['reduction'])
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.eps = 1e-16
        self.temperature = conf['temperature']

    def forward(self, x, y, alpha=1):
        x = (1 - alpha) * y + alpha * x
        x = self.log_softmax(x / self.temperature + self.eps)
        y = self.softmax(y / self.temperature + self.eps)

        return self.loss(x, y)


# Distance Transform MSE Loss
class DTMSELoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, conf):
        super().__init__()
        self.threshold = 0.5
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            img_mask = img[batch] > self.threshold
            img_mask_dt = edt(img_mask)

            field[batch] = img_mask_dt

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()

        loss = self.mse(pred_dt.squeeze(), target_dt.squeeze())

        return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, conf):
        super().__init__()
        self.alpha = torch.tensor(conf['alpha'])

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error.cpu() * distance.cpu()
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, conf):
        super().__init__()
        self.alpha = conf['alpha']
        self.erosions = conf['erosions']
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().detach().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class L1loss(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, x, y):
        y = y.float()

        return self.loss(x.squeeze(), y.squeeze())


class MSELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, x, y):
        y = y.float()

        return self.loss(x.squeeze(), y.squeeze())


class BCELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, x, y):
        y = y.float()
        BCE = F.binary_cross_entropy(x, y, reduction='mean')

        return BCE


class DiceLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        targets = targets.float()
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class JaccardLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalBCELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        targets = targets.float()
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce

        return focal_loss


class FocalCELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.ce = CrossEntropyLoss()

    def forward(self, x, y, alpha=0.8, gamma=2, smooth=1):
        ce = self.ce(x, y)
        ce_exp = torch.exp(-ce)
        focal_loss = alpha * (1 - ce_exp) ** gamma * ce

        return focal_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, x, y, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        x = x.view(-1)
        y = y.view(-1)

        intersection = (x * y).sum()
        dice = 1 - (2. * intersection + smooth) / (x.sum() + y.sum() + smooth)

        ce_exp = torch.exp(-dice)
        focal_loss = alpha * (1 - ce_exp) ** gamma * dice

        return focal_loss


class FocalMSELoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.mse = MSELoss()

    def forward(self, x, y, alpha=0.8, gamma=2, smooth=1):
        mse = self.mse(x.squeeze(), y.squeeze())
        mse_exp = torch.exp(-mse)
        focal_loss = alpha * (1 - mse_exp) ** gamma * mse

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class JSDivergenceLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.kld = KLDivergenceLoss(reduction=conf['reduction'])

    def forward(self, x, y):
        m = (x + y) / 2
        p = self.kld(x, m) / 2
        q = self.kld(y, m) / 2

        return p + q


class MSELoss_SSL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def softmax_mse_loss(self, inputs, targets):
        assert inputs.size() == targets.size(), f'{inputs.size()} {targets.size()}'
        inputs = F.softmax(inputs, dim=0)
        targets = F.softmax(targets, dim=0)

        return self.mse_loss(inputs.squeeze(), targets.squeeze())

    def forward(self, x, y):
        assert y.shape[0] == 1, 'target batch size should be 1.'
        x_b, _, _, _ = x.shape
        y = y.float().squeeze(0)

        for idx in range(x_b):
            losses = [self.softmax_mse_loss(x[idx], y)]

        return sum(losses) / x_b


class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://github.com/RElbers/info-nce-pytorch
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    conf:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, conf):
        super().__init__()
        self.temperature = conf['temperature']
        self.reduction = conf['reduction']
        self.negative_mode = conf['negative_mode']

    def forward(self, query, positive_key, negative_keys=None):
        return self.info_nce(query, positive_key, negative_keys,
                             temperature=self.temperature,
                             reduction=self.reduction,
                             negative_mode=self.negative_mode)

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ self.transpose(positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class CorrelationCoefficientLoss(nn.Module):
    """
    Notice: output range is [0, 1]
            The closer to 0 indicates the higher pearson correlation coefficient.
    """
    def __init__(self, conf):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)

        pearson = self.cos(x - x.mean(dim=1, keepdim=True), y - y.mean(dim=1, keepdim=True))

        return -((pearson.mean() - 1) / 2)


class MSECorrelationCoefficientLoss(nn.Module):
    """
    Notice: output range is [0, 1]
            The closer to 0 indicates the higher pearson correlation coefficient.
    """
    def __init__(self, conf):
        super().__init__()
        self.mse = MSELoss()
        self.corr = CorrelationCoefficientLoss()

    def forward(self, x, y):
        mse = self.mse(x.squeeze(), y.squeeze())
        corr = self.corr(x, y)

        return mse + corr


class TanHLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.temperature = conf['temperature']

    def forward(self, x, y):

        return torch.tanh(F.mse_loss(x, y) ** self.temperature)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.epsilon = conf['epsilon']
        self.reduction = conf['reduction']
        self.loss = nn.CrossEntropyLoss()

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        log_loss = self.reduce_loss(-log_preds.sum(dim=-1))
        loss = self.loss(preds, target.squeeze())

        return self.linear_combination(log_loss / n, loss)


# https://github.com/ZhenglinZhou/STAR/tree/master/lib/loss
class WingLoss(nn.Module):
    def __init__(self, conf):
        super(  ).__init__()
        # omega=0.1, epsilon=2
        self.omega = conf['omega']
        self.epsilon = conf['epsilon']

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_2 = (y - y_hat).pow(2).sum(dim=-1, keepdim=False)
        delta = delta_2.clamp(min=1e-6).sqrt()
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - C
        )
        return loss.mean()


# https://github.com/ZhenglinZhou/STAR/tree/master/lib/loss
class SmoothL1Loss(nn.Module):
    def __init__(self, conf):
        super(SmoothL1Loss, self).__init__()
        self.scale = conf['scale']
        self.EPSILON = 1e-10

    def __repr__(self):
        return "SmoothL1Loss()"

    def forward(self, output: torch.Tensor, groundtruth: torch.Tensor, reduction='mean'):
        """
            input:  b x n x 2
            output: b x n x 1 => 1
        """
        if output.dim() == 4:
            shape = output.shape
            groundtruth = groundtruth.reshape(shape[0], shape[1], 1, shape[3])

        delta_2 = (output - groundtruth).pow(2).sum(dim=-1, keepdim=False)
        delta = delta_2.clamp(min=1e-6).sqrt()
        # delta = torch.sqrt(delta_2 + self.EPSILON)
        loss = torch.where( \
            delta_2 < self.scale * self.scale, \
            0.5 / self.scale * delta_2, \
            delta - 0.5 * self.scale)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss
