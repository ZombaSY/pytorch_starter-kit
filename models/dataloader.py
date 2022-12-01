import os
import torch
import torchvision.transforms.functional as tf
import random
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from models import utils
from multiprocessing import set_start_method


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# function for non-utf-8 string
def is_image(src):
    ext = os.path.splitext(src)[1]
    return True if ext in ['.jpg', '.png', '.JPG', '.PNG'] else False


# https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class Image2ImageLoader(Dataset):

    def __init__(self,
                 x_path,
                 y_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = os.listdir(x_path)
        y_img_name = os.listdir(y_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)

        self.x_img_path = []
        self.y_img_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)

        img_paths = zip(x_img_name, y_img_name)
        for item in img_paths:
            self.x_img_path.append(x_path + os.sep + item[0])
            self.y_img_path.append(y_path + os.sep + item[1])

        assert len(self.x_img_path) == len(self.y_img_path), 'Images in directory must have same file indices!!'

        self.len = len(x_img_name)

        del x_img_name
        del y_img_name

    def transform(self, image, target):
        if hasattr(self.args, 'input_size'):
            image = tf.resize(image, [int(self.args.input_size[0]), int(self.args.input_size[1])])
            target = tf.resize(target, [int(self.args.input_size[0]), int(self.args.input_size[1])], interpolation=InterpolationMode.NEAREST)

        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.8) and self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.len - 1)     # randomly generates reference image on dataset
                image_refer = Image.open(self.x_img_path[rand_n]).convert('RGB')
                target_refer = Image.open(self.y_img_path[rand_n]).convert('L')
                image, target = utils.cut_mix(image, target, image_refer, target_refer)

            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 1.5) + 0.5  # [0.5, 2.0]
                rand_w = (random_gen.random() * 1.5) + 0.5
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])
                target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)
                target = tf.crop(target, i, j, h, w)

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)
                target = tf.hflip(target)

            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
                target = tf.perspective(target, start_p, end_p, interpolation=InterpolationMode.NEAREST)

        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        if self.args.num_class == 2:  # for binary label
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1
        target_tensor = target_tensor.unsqueeze(0)    # expand 'grey channel' for loss function dependency

        if self.args.input_space == 'HSV':
            try:
                set_start_method('spawn')
                image_tensor = utils.ImageProcessing.rgb_to_hsv(image_tensor)
            except RuntimeError:
                pass

        return image_tensor, target_tensor

    def __getitem__(self, index):
        x_path = self.x_img_path[index]
        y_path = self.y_img_path[index]

        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')

        img_x_tr, img_y_tr = self.transform(img_x, img_y)

        return (img_x_tr, x_path), (img_y_tr, y_path)

    def __len__(self):
        return self.len


class Image2VectorLoader(Dataset):

    def __init__(self,
                 csv_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        if hasattr(self.args, 'input_size'):
            h, w = self.args.input_size[0], self.args.input_size[1]
            self.size_1x = [int(h), int(w)]

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        self.data_root_path = os.path.split(csv_path)[0]
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df['value_1'])

    def transform(self, image):

        if hasattr(self.args, 'input_size'):
            image = tf.resize(image, [int(self.args.input_size[0]), int(self.args.input_size[1])])

        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random

            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 1.5) + 0.5  # [0.5, 2.0]
                rand_w = (random_gen.random() * 1.5) + 0.5
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])

            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(
                int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)

            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)

            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 8 + 3).__round__())
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end.
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)

        image_tensor = tf.to_tensor(image)

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        if self.args.input_space == 'HSV':
            try:
                set_start_method('spawn')
                image_tensor = utils.ImageProcessing.rgb_to_hsv(image_tensor)
            except RuntimeError:
                pass

        return image_tensor

    def __getitem__(self, index):
        x_path = os.path.join(*[self.data_root_path, self.df['sub_path'][index], self.df['image_file_name'][index]])

        img_x = Image.open(x_path).convert('RGB')
        img_x = self.transform(img_x)

        # example for csv column
        vec_y = torch.tensor([self.df['value_1'][index],
                              self.df['value_2'][index],
                              self.df['value_3'][index],
                              self.df['value_4'][index],
                              self.df['value_5'][index],
                              self.df['value_6'][index]])

        return (img_x, x_path), (vec_y, torch.tensor(0))

    def __len__(self):
        return self.len


class Image2ImageDataLoader:

    def __init__(self,
                 x_path,
                 y_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2ImageLoader(x_path,
                                              y_path,
                                              mode=mode,
                                              **kwargs)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.Loader.__len__()


class Image2VectorDataLoader:

    def __init__(self,
                 csv_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2VectorLoader(csv_path,
                                               mode=mode,
                                               **kwargs)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.Loader.__len__()
