import cv2
import numpy as np
import torch
import math
import random
import time
import os
import wandb
import logging

from timm.models.layers import trunc_normal_
from PIL import Image
from sklearn.neighbors import KernelDensity

SEED = 3407


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


def singleton(obj):
    instances = {}

    def wrapper(*args, **kwargs):
        if obj not in instances:
            instances[obj] = obj(*args, **kwargs)
        return instances[obj]

    return wrapper


@singleton
class Logger(logging.Logger):
    def __init__(self, dst, level=logging.INFO):
        super().__init__('logger')
        self.setLevel(level)

        # string formatting
        print_formatter = logging.Formatter(f'%(message)s')
        write_formatter = logging.Formatter(f'[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')

        # file writing handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(print_formatter)
        self.addHandler(stream_handler)

        # file name handler
        if self.level is not logging.DEBUG:
            file_handler = logging.FileHandler(filename=dst)
            file_handler.setFormatter(write_formatter)
            self.addHandler(file_handler)


class FunctionTimer:
    def __init__(self, _func):
        self.__func = _func

    def __call__(self, *conf, **kwargs):
        tt = time.time()
        self.__func(*conf, **kwargs)
        Logger().info(f'\"{self.__func.__name__}\" play time: {time.time() - tt}')

    def __enter__(self):
        return self


# https://arxiv.org/abs/1905.04899
def cut_mix(_input, _refer, _input_label=None, _refer_label=None):
    """
    :param _input: PIL.Image or ndarray
    :param _refer: PIL.Image or ndarray
    :param _input_label: PIL.Image or ndarray
    :param _refer_label: PIL.Image or ndarray

    :returns: cut-mixed image
    """
    _is_pil = isinstance(_input, Image.Image)
    random_gen = random.Random()

    if _is_pil:
        _input = np.array(_input)
        _refer = np.array(_refer)

    h1, w1 = _input.shape[:2]
    h2, w2 = _refer.shape[:2]

    # cutout positions
    rand_x = random_gen.random() * 0.75
    rand_y = random_gen.random() * 0.75
    rand_w = random_gen.random() * 0.5
    rand_h = random_gen.random() * 0.5

    cx_1 = int(rand_x * w1)  # range [0, 0.5]
    cy_1 = int(rand_y * h1)
    cw_1 = int((rand_w + 0.25) * w1)  # range [0.25, 0.75]
    ch_1 = int((rand_h + 0.25) * h1)

    cx_2 = int(rand_x * w2)
    cy_2 = int(rand_y * h2)
    cw_2 = int((rand_w + 0.25) * w2)
    ch_2 = int((rand_h + 0.25) * h2)

    if cy_1 + ch_1 > h1: ch_1 = h1 - cy_1  # push overflowing area
    if cx_1 + cw_1 > w1: cw_1 = w1 - cx_1

    # generate cutout_mask for post-process
    cutout_mask = np.zeros(_input.shape[:2], dtype=np.bool_)
    cutout_mask[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = True

    cutout_img = _refer[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]
    cutout_img = cv2.resize(cutout_img, (cw_1, ch_1))

    _input[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_img

    if _input_label is not None and _refer_label is not None:
        _input_label = np.array(_input_label)
        _refer_label = np.array(_refer_label)

        cutout_mask = _refer_label[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]
        cutout_mask = cv2.resize(cutout_mask, (cw_1, ch_1), interpolation=cv2.INTER_NEAREST)
        _input_label[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_mask

        if _is_pil:
            return Image.fromarray(_input.astype(np.uint8)), Image.fromarray(_input_label.astype(np.uint8)), Image.fromarray(cutout_mask)
        else:
            return _input.astype(np.uint8), _input_label.astype(np.uint8), cutout_mask

    if _is_pil:
        return Image.fromarray(_input.astype(np.uint8)), cutout_mask
    else:
        return _input.astype(np.uint8), cutout_mask


def grey_to_heatmap(img, is_bgr=True):
    """
    img: numpy.ndarray, or [0, 255] range of integer

    return: numpy.ndarray
    """

    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    if not is_bgr:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


# function for non-utf-8 string
def cv2_imread(fns_img, color=cv2.IMREAD_UNCHANGED):
    img_array = np.fromfile(fns_img, np.uint8)
    img = cv2.imdecode(img_array, color)
    if img.shape[-1] == 3 and color in [cv2.IMREAD_UNCHANGED, cv2.IMREAD_COLOR]:
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)  # convert color space for albumentaitons
    return img.astype(np.uint8)


def cv2_imwrite(fns_img, img):
    extension = os.path.splitext(fns_img)[1]
    result, encoded_img = cv2.imencode(extension, img)

    if result:
        with open(fns_img, mode='w+b') as f:
            encoded_img.tofile(f)


def denormalize_img(img, mean, std):
    img_t = img.permute(0, 2, 3, 1)
    img_t = (img_t * std + mean) * 255

    return img_t


def draw_image(x_img, output_prob, img_save_dir, img_id, n_class):
    img_fn = os.path.splitext(os.path.split(img_id)[-1])[0]
    output_grey = (output_prob * 255).astype(np.uint8)
    # output_grey = np.where(output_grey > 128, output_grey, 0)

    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)

    cv2_imwrite(os.path.join(img_save_dir, img_fn) + '.png', cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR))
    for i in range(1, n_class):
        cv2_imwrite(os.path.join(img_save_dir, img_fn) + f'_map_class_{i}.png', output_grey[i])


# https://arxiv.org/abs/2210.05775
def get_mixup_sample_rate(y_list):

    def stats_values(targets):
        mean = np.mean(targets)
        min = np.min(targets)
        max = np.max(targets)
        std = np.std(targets)
        Logger().info(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
        return mean, min, max, std

    mix_idx = []
    is_np = isinstance(y_list, np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    data_len = len(data_list)

    for i in range(data_len):
        data_i = data_list[i]
        data_i = data_i.reshape(-1, data_i.shape[0])  # get 2Dn

        # KDE sample rate
        kd = KernelDensity(kernel='gaussian', bandwidth=1.75).fit(data_i)  # should be 2D
        each_rate = np.exp(kd.score_samples(data_list))
        each_rate /= np.sum(each_rate)

        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    return mix_idx


def init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        torch.nn.init.constant_(m.bias, 0)


def random_hflip_landmark(image, target, points_flip):
    image = cv2.flip(image, 1)  # 1: horizontal
    target = np.array(target).reshape(-1, 2)
    target = target[points_flip, :]
    target[:, 0] = 1 - target[:, 0]

    return image, target


def random_vflip_landmark(image, target, points_flip):
    image = cv2.flip(image, 0)  # 0: vertical
    target = np.array(target).reshape(-1, 2)
    target = target[points_flip, :]
    target[:, 1] = 1 - target[:, 1]

    return image, target


def random_rotate_landmark(image, target, theta):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, math.degrees(theta), 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    center_x = 0.5
    center_y = 0.5
    landmark_num = int(len(target) / 2)
    target_center = np.array(target) - np.array([center_x, center_y]*landmark_num)
    target_center = target_center.reshape(landmark_num, 2)

    c, s = np.cos(theta), np.sin(theta)
    rot = np.array(((c, -s), (s, c)))
    target_center_rot = np.matmul(target_center, rot)
    target_rot = target_center_rot.reshape(landmark_num * 2) + np.array([center_x, center_y] * landmark_num)

    return image, target_rot


def get_landmark_label(root_path, label_file, task_type=None):
    label_path = os.path.join(root_path, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0]) == 1:
        return labels

    labels_new = []
    labels = sorted(labels)
    for label in labels:
        image_name = os.path.join(root_path, label[0])
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])

    return labels_new


def draw_landmark(img, lmk, save_dir, img_fn, put_index=True):
    tmp_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lmk = np.reshape(lmk, (-1, 2))
    for index in range(lmk.shape[0]):
        tmp_img = cv2.resize(tmp_img, (512, 512))
        draw_pos = (int(lmk[index][0] * tmp_img.shape[0]), int(lmk[index][1] * tmp_img.shape[1]))
        cv2.circle(tmp_img, draw_pos, 2, (0, 255, 0), 3)
        if put_index:
            cv2.putText(tmp_img, str(index), (draw_pos[0] + 3, draw_pos[1] + 3), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

    img_fn = os.path.split(img_fn)[-1]
    fn = os.path.join(save_dir, img_fn)
    cv2_imwrite(fn, tmp_img)
    np.save(fn.replace('.png', '.npy').replace('.jpg', '.npy'), lmk)


def multiprocessing_wrapper(conf):
    return conf[0](*conf[1:])


def log_epoch(mode, epoch, metric_dict, use_wandb=False):
    if mode == 'train':
        log_color = Colors.LIGHT_GREEN
    elif mode == 'validation':
        log_color = Colors.LIGHT_CYAN
    else:
        log_color = Colors.BOLD

    for key in metric_dict.keys():
        log_str = f'{mode} {key}: {metric_dict[key]}'
        Logger().info(f'{log_color} {epoch} epoch / {log_str} {Colors.END}')

        if use_wandb:
            wandb.log({f'{mode} {key}': metric_dict[key]},
                      step=epoch)


def append_data_stats(data_stats, key, value):
    if key in data_stats.keys():
        data_stats[key].append(value)
    else:
        data_stats[key] = [value]


class TrainerCallBack:

    def on_train_start(self):
        pass
