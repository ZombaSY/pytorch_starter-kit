import torch
import torch.nn.functional as F
import math
import numpy as np
import itertools

from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from scipy.ndimage.morphology import distance_transform_edt as edt
from sklearn.metrics import auc, roc_curve, confusion_matrix


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = SSIM.create_window(window_size, self.channel)

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def __ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = SSIM.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return SSIM.__ssim(img1, img2, window, self.window_size, channel, self.size_average)


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))


# Metric instead of IoU
class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()

        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics_segmentation(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.metric_dict = {
            "Overall Acc": 0,
            "Mean Acc": 0,
            "FreqW Acc": 0,
            "Mean IoU": 0,
            "Class IoU": 0
        }

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean iou
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        cls_iou = dict(zip(range(self.n_classes), iou))

        self.metric_dict['Overall Acc'] = acc
        self.metric_dict['Mean Acc'] = acc_cls
        self.metric_dict['FreqW Acc'] = fwavacc
        self.metric_dict['Mean IoU'] = mean_iou
        self.metric_dict['Class IoU'] = cls_iou

        return self.metric_dict

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class StreamSegMetrics_classification:
    def __init__(self, n_classes):
        self.metric_dict = {'kappa_score': -1,
                            'acc': -1,
                            'f1': -1}
        self.pred_list = []
        self.target_list = []
        self.n_classes = n_classes

    def update(self, pred, target):
        self.pred_list.append(pred.transpose())
        self.target_list.append(target.transpose())

    def get_results(self):
        pred_np_flatten = np.array([item for item in itertools.chain.from_iterable(self.pred_list)])
        target_np_flatten = np.array([item for item in itertools.chain.from_iterable(self.target_list)])

        self.metric_dict['kappa_score'] = cohen_kappa_score(pred_np_flatten, target_np_flatten)
        self.metric_dict['acc'] = accuracy_score(pred_np_flatten, target_np_flatten)
        self.metric_dict['f1'] = f1_score(pred_np_flatten, target_np_flatten, average='macro')

        return self.metric_dict

    def get_pred_flatten(self):
        return np.array([item for item in itertools.chain.from_iterable(self.pred_list)])

    def reset(self):
        self.pred_list = []
        self.target_list = []


def metrics_np(np_res, np_gnd, b_auc=False):
    f1m = []
    accm = []
    aucm = []
    specificitym = []
    precisionm = []
    sensitivitym = []
    ioum = []
    mccm = []

    epsilon = 2.22045e-16
    for i in range(np_res.shape[0]):
        label = np_gnd[i, ...]
        pred = np_res[i,...]
        label = label.flatten()
        pred = pred.flatten()
        # assert label.max() == 1 and (pred).max() <= 1
        # assert label.min() == 0 and (pred).min() >= 0

        y_pred = np.zeros_like(pred)
        y_pred[pred > 0.5] = 1

        try:
            tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=pred).ravel()  # for binary
        except ValueError as e:
            tn, fp, fn, tp = 0, 0, 0, 0
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)  #
        sensitivity = tp / (tp + fn + epsilon)  # Recall
        precision = tp / (tp + fp + epsilon)
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision + epsilon)
        iou = tp + (tp + fp + fn + epsilon)

        tp_tmp, tn_tmp, fp_tmp, fn_tmp = tp / 1000, tn / 1000, fp / 1000, fn / 1000     # to prevent overflowing
        mcc = (tp_tmp * tn_tmp - fp_tmp * fn_tmp) / math.sqrt((tp_tmp + fp_tmp) * (tp_tmp + fn_tmp) * (tn_tmp + fp_tmp) * (tn_tmp + fn_tmp) + epsilon)  # Matthews correlation coefficient

        f1m.append(f1_score)
        accm.append(accuracy)
        specificitym.append(specificity)
        precisionm.append(precision)
        sensitivitym.append(sensitivity)
        ioum.append(iou)
        mccm.append(mcc)
        if b_auc:
            fpr, tpr, thresholds = roc_curve(sorted(y_pred), sorted(label))
            AUC = auc(fpr, tpr)
            aucm.append(AUC)

    output = dict()
    output['f1'] = np.array(f1m).mean()
    output['acc'] = np.array(accm).mean()
    output['spe'] = np.array(specificitym).mean()
    output['sen'] = np.array(sensitivitym).mean()
    output['iou'] = np.array(ioum).mean()
    output['pre'] = np.array(precisionm).mean()
    output['mcc'] = np.array(mccm).mean()

    if b_auc:
        output['auc'] = np.array(aucm).mean()

    return output
