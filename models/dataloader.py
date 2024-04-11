import os
import torch
import random
import numpy as np
import pandas as pd
import cv2
import multiprocessing
import albumentations
import itertools

from torch.utils.data import Dataset
from models import utils


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_image(src):
    ext = os.path.splitext(src)[1]
    return True if ext.lower() in ['.jpeg', '.jpg', '.png', '.gif'] else False


def mount_data_on_memory_wrapper(args):
    return mount_data_on_memory(*args)


def mount_data_on_memory(img_path, CV_COLOR):
    img = utils.cv2_imread(img_path, CV_COLOR)
    return {'data': img, 'path': img_path}


def augmentations(args, target_size):
    return [albumentations.RandomScale(interpolation=cv2.INTER_NEAREST, p=args.transform_rand_resize),
            albumentations.RandomCrop(height=target_size, width=target_size, p=1.0),
            albumentations.HorizontalFlip(p=args.transform_hflip),
            albumentations.VerticalFlip(p=args.transform_vflip),
            albumentations.ImageCompression(quality_lower=80, quality_upper=100, p=args.transform_jpeg),
            albumentations.GaussianBlur(p=args.transform_blur),
            albumentations.CLAHE(p=args.transform_clahe),
            albumentations.RandomRain(p=args.transform_rain),
            albumentations.RandomFog(p=args.transform_fog),
            albumentations.GaussNoise(p=args.transform_g_noise),
            albumentations.FancyPCA(p=args.transform_fancyPCA),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=args.transform_jitter),
            albumentations.Perspective(interpolation=cv2.INTER_NEAREST, p=args.transform_perspective)]


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

        if hasattr(self.args, 'input_size'):
            h, w = self.args.input_size[0], self.args.input_size[1]
            self.size_1x = [int(h), int(w)]

        if hasattr(self.args, 'transform_rand_crop'):
            self.crop_factor = int(self.args.transform_rand_crop)

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

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

        # mount_data_on_memory
        if self.args.data_cache:
            print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(mount_data_on_memory_wrapper, zip(self.x_img_path, itertools.repeat(cv2.IMREAD_COLOR)))
                self.memory_data_y = pools.map(mount_data_on_memory_wrapper, zip(self.y_img_path, itertools.repeat(cv2.IMREAD_GRAYSCALE)))

        self.update_transform()

    def update_transform(self, scaler=1):
        excludings = ['transform_rand_crop', 'transform_landmark_rotate', 'transform_landmark_hflip']

        self.transform_resize = albumentations.Compose([
            albumentations.Resize(height=self.size_1x[0], width=self.size_1x[1], p=1),
        ])
        if self.mode == 'train':
            # progressively update augmentation scale
            for param in vars(self.args):
                if 'transform_' in param:
                    if param not in excludings:
                        setattr(self.args, param, min(getattr(self.args, param) * scaler, 1))
                    print(f'{utils.Colors.LIGHT_WHITE}{param}: {getattr(self.args, param)}{utils.Colors.END}')
            self.transform2 = albumentations.Compose([
                *augmentations(self.args, self.crop_factor)
            ])
        self.transforms_normalize = albumentations.Compose([
            albumentations.Normalize(mean=self.image_mean, std=self.image_std)
        ])

    def transform(self, _input, _label):
        random_gen = random.Random()

        transform = self.transform_resize(image=_input, mask=_label)
        _input = transform['image']
        _label = transform['mask']

        if self.mode == 'train':
            if random_gen.random() < self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.len - 1)
                if self.args.data_cache:
                    _input_refer = self.memory_data_x[rand_n]['data']
                    _label_refer = self.memory_data_y[rand_n]['data']
                else:
                    _input_refer = utils.cv2_imread(self.x_img_path[rand_n], cv2.IMREAD_COLOR)
                    _label_refer = utils.cv2_imread(self.y_img_path[rand_n], cv2.IMREAD_GRAYSCALE)
                transform_ref = self.transform_resize(image=_input_refer, mask=_label_refer)
                _input_refer = transform_ref['image']
                _label_refer = transform_ref['mask']
                _input, _label = utils.cut_mix(_input, _label, _input_refer, _label_refer)

            _input = _input.astype(np.uint8)
            _label = _label.astype(np.uint8)
            transform = self.transform2(image=_input, mask=_label)
            _input = transform['image']

        norm = self.transforms_normalize(image=_input)
        _input = norm['image']
        _label = transform['mask']

        if self.args.num_class == 2:  # for binary label
            _label[_label < 128] = 0
            _label[_label >= 128] = 1

        _input = np.transpose(_input, [2, 0, 1])

        _input = torch.from_numpy(_input.astype(np.float32))  # (3, 640, 480)
        _label = torch.from_numpy(_label)  # (640, 480)
        _label = _label.unsqueeze(0)    # expand 'grey channel' for loss function dependency

        return _input, _label

    def __getitem__(self, index):
        if self.args.data_cache:
            img_x = self.memory_data_x[index]['data']
            img_y = self.memory_data_y[index]['data']
            x_path = self.memory_data_x[index]['path']
            y_path = self.memory_data_y[index]['path']

        else:
            x_path = self.x_img_path[index]
            y_path = self.y_img_path[index]

            img_x = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
            img_y = utils.cv2_imread(y_path, cv2.IMREAD_GRAYSCALE)

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

        if hasattr(self.args, 'transform_rand_crop'):
            self.crop_factor = int(self.args.transform_rand_crop)

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

        self.data_root_path = os.path.split(csv_path)[0]
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df['FILENAME'])

        # mount_data_on_memory
        if self.args.data_cache:
            print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
            x_img_path = []
            self.memory_data_y = []
            for idx in range(self.len):
                x_img_path.append(os.path.join(*[self.data_root_path, self.df['label'][idx]]) + '.png')
                self.memory_data_y.append(torch.tensor([self.df['label'][idx]]))
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(mount_data_on_memory_wrapper, zip(x_img_path, itertools.repeat(cv2.IMREAD_COLOR)))

        self.transform_resize = albumentations.Compose([
            albumentations.Resize(height=self.size_1x[0], width=self.size_1x[1], p=1),
        ])
        self.transform2 = albumentations.Compose([
            *augmentations(self.args, self.crop_factor)
        ])
        self.transforms_normalize = albumentations.Compose([
            albumentations.Normalize(mean=self.image_mean, std=self.image_std)
        ])

        if self.mode == 'train' and hasattr(self.args, 'transform_mixup'):
            if self.args.transform_mixup > 0:
                self.mixup_sample_1 = np.array(utils.get_mixup_sample_rate(np.expand_dims(np.array(self.df['col1']), -1)))
                self.mixup_sample_2 = np.array(utils.get_mixup_sample_rate(np.expand_dims(np.array(self.df['col2']), -1)))
                self.mixup_sample = (self.mixup_sample_1 + self.mixup_sample_2) / 2     # get the average of sample

    def transform(self, _input, _label, idx_1):
        random_gen = random.Random()

        if self.mode == 'train':
            transform = self.transform_resize(image=_input)
            _input = transform['image']

            # MixUp
            if random_gen.random() < self.args.transform_mixup:
                # select index from pre-defined sampler
                idx_2 = np.random.choice(np.arange(self.len), p=self.mixup_sample[idx_1])

                # load the pair of X and Y
                x_path = os.path.join(*[self.data_root_path, self.df['label'][idx_2]]) + '.png'
                x_img = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
                transform = self.transform_resize(image=x_img)
                _input_2 = transform['image']
                _label_2 =  torch.tensor([self.df['label'][idx_2]])

                lam = np.random.beta(2, 2)

                _input = _input * lam + _input_2 * (1 - lam)
                _label = _label * lam + _label_2 * (1 - lam)

            _input = _input.astype(np.uint8)
            transform = self.transform2(image=_input)
        else:
            transform = self.transform_resize(image=_input)

        norm = self.transforms_normalize(image=transform['image'])
        _input = norm['image']
        _input = np.transpose(_input, [2, 0, 1])

        _input = torch.from_numpy(_input.astype(np.float32))  # (3, 640, 480)

        return _input, _label.float()

    def __getitem__(self, index):
        if self.args.data_cache:
            x_img = self.memory_data_x[index]['data']
            y_vec = self.memory_data_y[index]
            x_path = self.memory_data_x[index]['path']
        else:
            x_path = os.path.join(*[self.data_root_path, self.df['label'][index]]) + '.png'
            x_img = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
            y_vec = torch.tensor([self.df['label'][index]])

        x_img_tr, y_vec = self.transform(x_img, y_vec, index)

        return (x_img_tr, x_path), (y_vec, x_path)

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
                                            shuffle=(mode == 'train'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()


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

        if kwargs['args'].weighted_sampler == True and mode == 'train':
            sampler1 = utils.get_frequency_in_list(np.array(self.image_loader.df['col1']).tolist(), reciprocal=True)
            sampler2 = utils.get_frequency_in_list(np.array(self.image_loader.df['col2']).tolist(), reciprocal=True)
            sampler = torch.utils.data.WeightedRandomSampler(sampler1 + sampler2, self.image_loader.__len__(), replacement=True)
            self.Loader = MultiEpochsDataLoader(self.image_loader,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                worker_init_fn=seed_worker,
                                                generator=g,
                                                pin_memory=pin_memory,
                                                sampler=sampler)

        else:
            self.Loader = MultiEpochsDataLoader(self.image_loader,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=(mode == 'train'),
                                                worker_init_fn=seed_worker,
                                                generator=g,
                                                pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()
