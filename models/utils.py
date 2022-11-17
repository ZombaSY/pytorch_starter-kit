import cv2
import numpy as np
import torch
import math
import random
import time

from torch.autograd import Variable
from sklearn.metrics import auc, roc_curve, confusion_matrix
from matplotlib.image import imread
from PIL import Image


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


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

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
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


class FunctionTimer:
    def __init__(self, _func):
        self.__func = _func

    def __call__(self, *args, **kwargs):
        tt = time.time()
        self.__func(*args, **kwargs)
        print(f'\"{self.__func.__name__}\" play time: {time.time() - tt}')

    def __enter__(self):
        return self


class ImageProcessing(object):
    '''
    @issue
    'hsv_to_rgb' and 'rgb_to_hsv' convert the image with H 180 value to 0, resulting blue color to red color

    '''

    @staticmethod
    def rgb_to_lab(img, is_training=True):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
            0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169,
             0.950227],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad=False).cuda())

        epsilon = 6 / 29

        img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + \
              (torch.clamp(img, min=0.0001) **
               (1.0 / 3.0) * img.gt(epsilon ** 3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],  # fx
                                                    # fy
                                                    [116.0, -500.0, 200.0],
                                                    # fz
                                                    [0.0, 0.0, -200.0],
                                                    ]), requires_grad=False).cuda()

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).cuda()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[0, :, :] = img[0, :, :] / 100
        img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
        img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

        img[(img != img).detach()] = 0

        img = img.contiguous()

        return img.cuda()

    @staticmethod
    def lab_to_rgb(img, is_training=True):
        """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor
        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)
        img_copy = img.clone()

        img_copy[:, 0] = img[:, 0] * 100
        img_copy[:, 1] = ((img[:, 1] * 2) - 1) * 110
        img_copy[:, 2] = ((img[:, 2] * 2) - 1) * 110

        img = img_copy.clone().cuda()
        del img_copy

        lab_to_fxfyfz = Variable(torch.FloatTensor([  # X Y Z
            [1 / 116.0, 1 / 116.0, 1 / 116.0],  # R
            [1 / 500.0, 0, 0],  # G
            [0, 0, -1 / 200.0],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(
            img + Variable(torch.cuda.FloatTensor([16.0, 0.0, 0.0])), lab_to_fxfyfz)

        epsilon = 6.0 / 29.0

        img = (((3.0 * epsilon ** 2 * (img - 4.0 / 29.0)) * img.le(epsilon).float()) +
               ((torch.clamp(img, min=0.0001) ** 3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, Variable(
            torch.cuda.FloatTensor([0.950456, 1.0, 1.088754])))

        xyz_to_rgb = Variable(torch.FloatTensor([  # X Y Z
            [3.2404542, -0.9692660, 0.0556434],  # R
            [-1.5371385, 1.8760108, -0.2040259],  # G
            [-0.4985314, 0.0415560, 1.0572252],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(img, xyz_to_rgb)

        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                        min=0.0001) ** (
                                                                        1 / 2.4) * 1.055) - 0.055) * img.gt(
            0.0031308).float()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        img = img.contiguous()
        img[(img != img).detach()] = 0

        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = ImageProcessing.normalise_image(
            imread(img_filepath), normaliser)  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                        np.log10(max_intensity ** 2 /
                                 ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def hsv_to_rgb(img):
        """Converts a HSV image to RGB
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: HSV image
        :returns: RGB image
        :rtype: Tensor

        """
        img = torch.clamp(img, 0, 1)
        img = img.permute(2, 1, 0)

        m1 = 0
        m2 = (img[:, :, 2] * (1 - img[:, :, 1]) - img[:, :, 2]) / 60
        m3 = 0
        m4 = -1 * m2
        m5 = 0

        r = img[:, :, 2] + torch.clamp(img[:, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(img[:, :, 0] * 360 - 60, 0,
                                                                                         60) * m2 + torch.clamp(
            img[:, :, 0] * 360 - 120, 0, 120) * m3 + torch.clamp(img[:, :, 0] * 360 - 240, 0, 60) * m4 + torch.clamp(
            img[:, :, 0] * 360 - 300, 0, 60) * m5

        m1 = (img[:, :, 2] - img[:, :, 2] * (1 - img[:, :, 1])) / 60
        m2 = 0
        m3 = -1 * m1
        m4 = 0

        g = img[:, :, 2] * (1 - img[:, :, 1]) + torch.clamp(img[:, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(
            img[:, :, 0] * 360 - 60,
            0, 120) * m2 + torch.clamp(img[:, :, 0] * 360 - 180, 0, 60) * m3 + torch.clamp(img[:, :, 0] * 360 - 240, 0,
                                                                                           120) * m4

        m1 = 0
        m2 = (img[:, :, 2] - img[:, :, 2] * (1 - img[:, :, 1])) / 60
        m3 = 0
        m4 = -1 * m2

        b = img[:, :, 2] * (1 - img[:, :, 1]) + torch.clamp(img[:, :, 0] * 360 - 0, 0, 120) * m1 + torch.clamp(
            img[:, :, 0] * 360 -
            120, 0, 60) * m2 + torch.clamp(img[:, :, 0] * 360 - 180, 0, 120) * m3 + torch.clamp(
            img[:, :, 0] * 360 - 300, 0, 60) * m4

        img = torch.stack((r, g, b), 2)
        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()
        img = torch.clamp(img, 0, 1)

        return img

    @staticmethod
    def rgb_to_hsv(img):
        """Converts an RGB image to HSV
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: RGB image
        :returns: HSV image
        :rtype: Tensor

        """
        img = torch.clamp(img, 0.000000001, 1)

        img = img.permute(2, 1, 0)
        # 3, H, W
        shape = img.shape

        img = img.contiguous()
        img = img.view(-1, 3)

        mx = torch.max(img, 1)[0]
        mn = torch.min(img, 1)[0]

        ones = Variable(torch.FloatTensor(
            torch.ones((img.shape[0])))).cuda()
        zero = Variable(torch.FloatTensor(torch.zeros(shape[0:2]))).cuda()

        img = img.view(shape)

        ones1 = ones[0:math.floor((ones.shape[0] / 2))]
        ones2 = ones[math.floor(ones.shape[0] / 2):(ones.shape[0])]

        mx1 = mx[0:math.floor((ones.shape[0] / 2))]
        mx2 = mx[math.floor(ones.shape[0] / 2):(ones.shape[0])]
        mn1 = mn[0:math.floor((ones.shape[0] / 2))]
        mn2 = mn[math.floor(ones.shape[0] / 2):(ones.shape[0])]

        df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
        df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))

        df = torch.cat((df1, df2), 0)
        del df1, df2
        df = df.view(shape[0:2]) + 1e-10
        mx = mx.view(shape[0:2])

        img = img.cuda()
        df = df.cuda()
        mx = mx.cuda()

        g = img[:, :, 1].clone().cuda()
        b = img[:, :, 2].clone().cuda()
        r = img[:, :, 0].clone().cuda()

        img_copy = img.clone()

        img_copy[:, :, 0] = (((g - b) / df) * r.eq(mx).float() + (2.0 + (b - r) / df)
                             * g.eq(mx).float() + (4.0 + (r - g) / df) * b.eq(mx).float())
        img_copy[:, :, 0] = img_copy[:, :, 0] * 60.0

        zero = zero.cuda()
        img_copy2 = img_copy.clone()

        img_copy2[:, :, 0] = img_copy[:, :, 0].lt(zero).float(
        ) * (img_copy[:, :, 0] + 360) + img_copy[:, :, 0].ge(zero).float() * (img_copy[:, :, 0])

        img_copy2[:, :, 0] = img_copy2[:, :, 0] / 360

        del img, r, g, b

        img_copy2[:, :, 1] = mx.ne(zero).float() * (df / mx) + \
                             mx.eq(zero).float() * (zero)
        img_copy2[:, :, 2] = mx

        img_copy2[(img_copy2 != img_copy2).detach()] = 0

        img = img_copy2.clone()

        img = img.permute(2, 1, 0)
        img = torch.clamp(img, 0.000000001, 1)

        return img

    @staticmethod
    def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out,
                    clamp=True, same_channel=True):
        """Applies a peicewise linear curve defined by a set of knot points to
        an image channel

        :param img: image to be adjusted
        :param C: predicted knot points of curve
        :returns: adjusted image
        :rtype: Tensor

        """
        slope = Variable(torch.zeros((C.shape[0] - 1))).cuda()
        curve_steps = C.shape[0] - 1
        '''
        Compute the slope of the line segments
        '''
        for i in range(0, C.shape[0] - 1):
            slope[i] = C[i + 1] - C[i]

        '''
        Compute the squared difference between slopes
        '''
        for i in range(0, slope.shape[0] - 1):
            slope_sqr_diff += (slope[i + 1] - slope[i]) * (slope[i + 1] - slope[i])

        '''
        Use predicted line segments to compute scaling factors for the channel
        '''
        scale = float(C[0])
        for i in range(0, slope.shape[0] - 1):
            if clamp:
                scale += float(slope[i]) * (torch.clamp(img[:, :, channel_in] * curve_steps - i, 0, 1))
                # scale += float(slope[i]) * (torch.clamp(img[:, :, channel_in], 0, 1))
            else:
                scale += float(slope[i]) * (img[:, :, channel_in] * curve_steps - i)
        img_copy = img.clone()

        if same_channel:
            # channel in and channel out are the same channel
            img_copy[:, :, channel_out] = img[:, :, channel_in] * scale
        else:
            # otherwise
            img_copy[:, :, channel_out] = img[:, :, channel_out] * scale

        img_copy = torch.clamp(img_copy, 0, 1)

        return img_copy, slope_sqr_diff

    @staticmethod
    def adjust_hsv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        S1 = torch.exp(S[0:int(S.shape[0] / 4)])
        S2 = torch.exp(S[(int(S.shape[0] / 4)):(int(S.shape[0] / 4) * 2)])
        S3 = torch.exp(S[(int(S.shape[0] / 4) * 2):(int(S.shape[0] / 4) * 3)])
        S4 = torch.exp(S[(int(S.shape[0] / 4) * 3):(int(S.shape[0] / 4) * 4)])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        '''
        Adjust Hue channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Adjust Saturation channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1, same_channel=False)

        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_sv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        img = img.contiguous()

        S3 = torch.exp(S[(int(S.shape[0] / 2) * 0):(int(S.shape[0] / 2) * 1)])
        S4 = torch.exp(S[(int(S.shape[0] / 2) * 1):(int(S.shape[0] / 2) * 2)])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S3, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_rgb(img, R):
        """Adjust the RGB channels of a RGB image using learnt curves

        :param img: image to be adjusted
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        R1 = torch.exp(R[0:int(R.shape[0] / 3)])
        R2 = torch.exp(R[(int(R.shape[0] / 3)):(int(R.shape[0] / 3) * 2)])
        R3 = torch.exp(R[(int(R.shape[0] / 3) * 2):(int(R.shape[0] / 3) * 3)])

        '''
        Apply the curve to the R channel 
        '''
        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Apply the curve to the G channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Apply the curve to the B channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_lab(img, L):
        """Adjusts the image in LAB space using the predicted curves

        :param img: Image tensor
        :param L: Predicited curve parameters for LAB channels
        :returns: adjust image, and regularisation parameter
        :rtype: Tensor, float

        """
        img = img.permute(2, 1, 0)

        shape = img.shape
        img = img.contiguous()

        '''
        Extract predicted parameters for each L,a,b curve
        '''
        L1 = torch.exp(L[0:int(L.shape[0] / 3)])
        L2 = torch.exp(L[(int(L.shape[0] / 3)):(int(L.shape[0] / 3) * 2)])
        L3 = torch.exp(L[(int(L.shape[0] / 3) * 2):(int(L.shape[0] / 3) * 3)])

        slope_sqr_diff = Variable(torch.zeros(1) * 0.0).cuda()

        '''
        Apply the curve to the L channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Now do the same for the a channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Now do the same for the b channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff


def cut_mix(_input, mask_1, _refer, mask_2) -> (Image, Image):
    """
    :param _input: PIL.Image
    :param mask_1: PIL.Image
    :param _refer: PIL.Image
    :param mask_2: PIL.Image

    :returns: cut-mixed image
    """
    random_gen = random.Random()

    _input_np = np.array(_input)
    mask_1_np = np.array(mask_1)
    _refer_np = np.array(_refer)
    mask_2_np = np.array(mask_2)

    h1, w1, _ = _input_np.shape
    h2, w2, _ = _refer_np.shape

    # cutout positions
    rand_x = random_gen.random() * 0.75
    rand_y = random_gen.random() * 0.75
    rand_w = random_gen.random() * 0.5
    rand_h = random_gen.random() * 0.5

    cx_1 = int(rand_x * w1)  # range of [0, 0.5]
    cy_1 = int(rand_y * h1)
    cw_1 = int((rand_w + 0.25) * w1)  # range of [0.25, 0.75]
    ch_1 = int((rand_h + 0.25) * h1)

    cx_2 = int(rand_x * w2)
    cy_2 = int(rand_y * h2)
    cw_2 = int((rand_w + 0.25) * w2)
    ch_2 = int((rand_h + 0.25) * h2)

    if cy_1 + ch_1 > h1: ch_1 = h1 - cy_1  # push overflowing area
    if cx_1 + cw_1 > w1: cw_1 = w1 - cx_1

    cutout_img = _refer_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]
    cutout_mask = mask_2_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]

    cutout_img = cv2.resize(cutout_img, (cw_1, ch_1))
    cutout_mask = cv2.resize(cutout_mask, (cw_1, ch_1), interpolation=cv2.INTER_NEAREST)

    _input_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_img
    mask_1_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_mask

    return Image.fromarray(_input_np.astype(np.uint8)), Image.fromarray(mask_1_np.astype(np.uint8))


def grey_to_heatmap(img):
    """
    img: numpy.ndarray, or [0, 255] range of integer

    return: numpy.ndarray
    """

    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


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
        label = np_gnd[i, :, :]
        pred = np_res[i, :, :]
        label = label.flatten()
        pred = pred.flatten()
        # assert label.max() == 1 and (pred).max() <= 1
        # assert label.min() == 0 and (pred).min() >= 0

        y_pred = np.zeros_like(pred)
        y_pred[pred > 0.5] = 1

        try:
            tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=y_pred).ravel()  # for binary
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

