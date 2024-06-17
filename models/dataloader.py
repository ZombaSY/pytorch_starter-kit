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
from torch.nn import functional as F
from models import utils


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_image(src):
    ext = os.path.splitext(src)[1]
    return True if ext.lower() in ['.jpeg', '.jpg', '.png', '.gif'] else False


def read_image_data(img_path, CV_COLOR):
    img = utils.cv2_imread(img_path, CV_COLOR)
    return {'data': img, 'path': img_path}


def augmentations(conf):
    target_size = conf['transform_rand_crop']
    return [albumentations.RandomScale(interpolation=cv2.INTER_NEAREST, p=conf['transform_rand_resize']),
            albumentations.RandomCrop(height=target_size, width=target_size, p=1.0),
            albumentations.CoarseDropout(max_holes=4, max_height=target_size // 8, max_width=target_size // 8, min_height=target_size // 32, min_width=target_size // 32, min_holes=1, p=conf['transform_coarse_dropout']),
            albumentations.HorizontalFlip(p=conf['transform_hflip']),
            albumentations.VerticalFlip(p=conf['transform_vflip']),
            albumentations.ImageCompression(quality_lower=70, quality_upper=100, p=conf['transform_jpeg']),
            albumentations.GaussianBlur(p=conf['transform_blur']),
            albumentations.CLAHE(p=conf['transform_clahe']),
            albumentations.RandomRain(p=conf['transform_rain']),
            albumentations.RandomFog(p=conf['transform_fog']),
            albumentations.GaussNoise(p=conf['transform_g_noise']),
            albumentations.FancyPCA(p=conf['transform_fancyPCA']),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=conf['transform_jitter']),
            albumentations.Perspective(interpolation=cv2.INTER_NEAREST, p=conf['transform_perspective']),
            ]


# https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *conf, **kwargs):
        super().__init__(*conf, **kwargs)
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
    conf:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class ImageLoader(Dataset):

    def __init__(self, conf, conf_dataloader):
        self.conf = conf
        self.conf_dataloader = conf_dataloader

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

        self.df = pd.read_csv(conf_dataloader['data_path'])
        self.len = len(self.df)

        if self.conf_dataloader['data_cache']:
            print(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['input'], itertools.repeat(cv2.IMREAD_COLOR)))

        self.update_transform()

    def update_transform(self):
        self.transform_resize = albumentations.Resize(height=self.conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        if self.conf_dataloader['mode'] == 'train':
            self.transform_augmentation = albumentations.Compose([*augmentations(self.conf_dataloader['augmentations'])])
        self.transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

    def transform(self, _input):

        if self.conf_dataloader['mode'] == 'train':
            transform = self.transform_resize(image=_input)
            _input = transform['image']

            _input = _input.astype(np.uint8)
            transform = self.transform_augmentation(image=_input)
        else:
            transform = self.transform_resize(image=_input)

        norm = self.transforms_normalize(image=transform['image'])
        _input = norm['image']
        _input = np.transpose(_input, [2, 0, 1])

        _input = torch.from_numpy(_input.astype(np.float32))

        return _input

    def __getitem__(self, index):
        if self.conf_dataloader['data_cache']:
            x_img = self.memory_data_x[index]['data']
            x_path = self.memory_data_x[index]['path']
        else:
            x_path = self.df['input'][index]
            x_img = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)

        x_img_tr = self.transform(x_img)

        return (x_img_tr, x_path), (x_img_tr, x_path)

    def __len__(self):
        return self.len


class Image2ImageLoader(Dataset):

    def __init__(self, conf, conf_dataloader):
        self.conf = conf
        self.conf_dataloader = conf_dataloader

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

        self.df = pd.read_csv(conf_dataloader['data_path'])
        self.len = len(self.df)

        # mount_data_on_memory
        if self.conf_dataloader['data_cache']:
            print(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['input'], itertools.repeat(cv2.IMREAD_COLOR)))
                self.memory_data_y = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['label'], itertools.repeat(cv2.IMREAD_GRAYSCALE)))

        self.update_transform()

    def update_transform(self):
        self.transform_resize = albumentations.Resize(height=self.conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        if self.conf_dataloader['mode'] == 'train':
            self.transform_augmentation = albumentations.Compose([*augmentations(self.conf_dataloader['augmentations'])])
        self.transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

    def transform(self, _input, _label):
        random_gen = random.Random()

        if self.conf_dataloader['mode'] == 'train':
            transform = self.transform_resize(image=_input, mask=_label)
            _input = transform['image']
            _label = transform['mask']

            if random_gen.random() < self.conf_dataloader['augmentations']['transform_cutmix']:
                rand_n = random_gen.randint(0, self.len - 1)
                if self.conf_dataloader['data_cache']:
                    _input_refer = self.memory_data_x[rand_n]['data']
                    _label_refer = self.memory_data_y[rand_n]['data']
                else:
                    _input_refer = utils.cv2_imread(self.df['input'][rand_n], cv2.IMREAD_COLOR)
                    _label_refer = utils.cv2_imread(self.df['label'][rand_n], cv2.IMREAD_GRAYSCALE)
                transform_ref = self.transform_resize(image=_input_refer, mask=_label_refer)
                _input_refer = transform_ref['image']
                _label_refer = transform_ref['mask']
                _input, _label = utils.cut_mix(_input, _input_refer, _label, _label_refer)

            _input = _input.astype(np.uint8)
            _label = _label.astype(np.uint8)
            transform = self.transform_augmentation(image=_input, mask=_label)
        else:
            transform = self.transform_resize(image=_input, mask=_label)

        norm = self.transforms_normalize(image=transform['image'])
        _input = norm['image']
        _label = transform['mask']

        if self.conf['model']['num_class'] == 2:  # for binary label
            _label = np.where(_label >= 128, 1, 0)

        _input = np.transpose(_input, [2, 0, 1])

        _input = torch.from_numpy(_input.astype(np.float32))
        _label = torch.from_numpy(_label)
        _label = _label.unsqueeze(0)    # expand 'grey channel' for loss function dependency

        return _input, _label

    def __getitem__(self, index):
        if self.conf_dataloader['data_cache']:
            img_x = self.memory_data_x[index]['data']
            img_y = self.memory_data_y[index]['data']
            x_path = self.memory_data_x[index]['path']
            y_path = self.memory_data_y[index]['path']

        else:
            x_path = self.df['input'][index]
            y_path = self.df['label'][index]

            img_x = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
            img_y = utils.cv2_imread(y_path, cv2.IMREAD_GRAYSCALE)

        img_x_tr, img_y_tr = self.transform(img_x, img_y)

        return (img_x_tr, x_path), (img_y_tr, y_path)

    def __len__(self):
        return self.len


class Image2VectorLoader(Dataset):

    def __init__(self, conf, conf_dataloader):
        self.conf = conf
        self.conf_dataloader = conf_dataloader

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

        self.df = pd.read_csv(conf_dataloader['data_path'])
        self.len = len(self.df)

        # mount_data_on_memory
        if self.conf_dataloader['data_cache']:
            print(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
            x_img_path = []
            self.memory_data_y = []
            for idx in range(self.len):
                x_img_path.append(self.df['input'][idx])
                label = F.one_hot(torch.tensor(self.df[self.conf_dataloader['label_cols']].values[idx]), self.conf['model']['num_class']) if self.conf['env']['task'] == 'classification' else torch.tensor(self.df[self.conf_dataloader['label_cols']].values[idx])
                self.memory_data_y.append(label)
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), x_img_path, itertools.repeat(cv2.IMREAD_COLOR)))

        if self.conf_dataloader['mode'] == 'train':
            if self.conf_dataloader['augmentations']['transform_mixup'] > 0:
                # C-mixup
                # self.mixup_sample_1 = np.array(utils.get_mixup_sample_rate(np.expand_dims(np.array(self.df['col1']), -1)))
                # self.mixup_sample_2 = np.array(utils.get_mixup_sample_rate(np.expand_dims(np.array(self.df['col2']), -1)))
                # self.mixup_sample = (self.mixup_sample_1 + self.mixup_sample_2) / 2

                # mix-up
                self.mixup_sample = np.ones(self.len) / self.len

        self.update_transform()

    def update_transform(self):
        self.transform_resize = albumentations.Resize(height=self.conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        if self.conf_dataloader['mode'] == 'train':
            self.transform_augmentation = albumentations.Compose([*augmentations(self.conf_dataloader['augmentations'])])
        self.transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

    def transform(self, _input, _label):
        random_gen = random.Random()

        if self.conf_dataloader['mode'] == 'train':
            transform = self.transform_resize(image=_input)
            _input = transform['image']

            # MixUp
            if random_gen.random() < self.conf_dataloader['augmentations']['transform_mixup']:
                # select index from pre-defined sampler
                idx_2 = np.random.choice(np.arange(self.len), p=self.mixup_sample)

                # load the pair of X and Y
                x_path = self.df['input'][idx_2]
                x_img = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
                transform = self.transform_resize(image=x_img)
                _input_2 = transform['image']
                _label_2 = torch.tensor(self.df[self.conf_dataloader['label_cols']].values[idx_2])

                if self.conf['env']['task'] == 'classification':
                    _label_2 = F.one_hot(_label_2, num_classes=self.conf['model']['num_class'])

                lam = np.random.beta(2, 2)

                _input = _input * lam + _input_2 * (1 - lam)
                _label = _label * lam + _label_2 * (1 - lam)

                del x_img
                del x_path

            _input = _input.astype(np.uint8)
            transform = self.transform_augmentation(image=_input)
        else:
            transform = self.transform_resize(image=_input)

        norm = self.transforms_normalize(image=transform['image'])
        _input = norm['image']
        _input = np.transpose(_input, [2, 0, 1])

        _input = torch.from_numpy(_input.astype(np.float32))

        return _input, _label.float()

    def __getitem__(self, index):
        if self.conf_dataloader['data_cache']:
            x_img = self.memory_data_x[index]['data']
            y_vec = self.memory_data_y[index]
            x_path = self.memory_data_x[index]['path']
        else:
            x_path = self.df['input'][index]
            x_img = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
            y_vec = torch.tensor(self.df[self.conf_dataloader['label_cols']].values[index])

            if self.conf['env']['task'] == 'classification' and self.conf_dataloader['mode'] != 'test':
                y_vec = F.one_hot(y_vec, num_classes=self.conf['model']['num_class'])

        x_img_tr, y_vec = self.transform(x_img, y_vec)

        return (x_img_tr, x_path), (y_vec, x_path)

    def __len__(self):
        return self.len


class Image2LandmarkLoader(Dataset):

    def __init__(self, conf, conf_dataloader):
        self.conf = conf
        self.conf_dataloader = conf_dataloader

        self.points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                            42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
                            43, 44, 45, 46,
                            51, 50, 49, 48, 47,
                            61, 60, 59, 58, 63, 62, 55, 54, 53, 52, 57, 56,
                            71, 70, 69, 68, 67, 66, 65, 64,
                            75, 76, 77, 72, 73, 74,
                            79, 78, 81, 80, 83, 82,
                            90, 89, 88, 87, 86, 85, 84, 95, 94, 93, 92, 91, 100, 99, 98, 97, 96, 103, 102, 101, 105, 104
                            ]
        self.points_flip = np.array(self.points_flip).tolist()
        if 'train' in os.path.split(self.conf_dataloader['data_path'])[-1]:
            sub_dir = 'images_train'
        elif 'valid' in os.path.split(self.conf_dataloader['data_path'])[-1]:
            sub_dir = 'images_valid'
        else:
            sub_dir = 'images_test'
        root_path = os.path.join(os.path.split(self.conf_dataloader['data_path'])[0], sub_dir)
        self.xy = utils.get_landmark_label(root_path, self.conf_dataloader['data_path'])

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

        self.update_transform()

    def update_transform(self):
        self.transform_resize = albumentations.Resize(height=self.conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        if self.conf_dataloader['mode'] == 'train':
            self.transform_augmentation = albumentations.Compose([*augmentations(self.conf_dataloader['augmentations'])])
        self.transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

    def transform(self, _input, _label):
        random_gen = random.Random()
        transform = self.transform_resize(image=_input)
        _input = transform['image']

        if self.conf_dataloader['mode'] == 'train':
            if random_gen.random() < self.conf_dataloader['augmentations']['transform_landmark_hflip']:
                _input, _label = utils.random_hflip(_input, _label, self.points_flip)

            if random_gen.random() < self.conf_dataloader['augmentations']['transform_landmark_rotate']:
                angle_max = 10
                theta_max = np.radians(angle_max)
                theta = random.uniform(-theta_max, theta_max)
                _input, _label = utils.random_rotate(_input, _label, theta)

            _input = _input.astype(np.uint8)
            transform = self.transform_augmentation(image=_input)
            _input = transform['image']

        norm = self.transforms_normalize(image=_input)
        _input = norm['image']
        _input = np.transpose(_input, [2, 0, 1])
        _input = torch.from_numpy(_input.astype(np.float32))
        _label = torch.from_numpy(_label.astype(np.float32))

        return _input, _label

    def __getitem__(self, index):
        if self.conf_dataloader['data_cache']:
            x_input = self.memory_data_x[index]['data']
            y_label = self.memory_data_y[index]['data']

            x_path = self.memory_data_x[index]['path']
        else:
            x_path = self.xy[index][0]

            x_input = utils.cv2_imread(x_path, cv2.IMREAD_COLOR)
            y_label = self.xy[index][1]

        x_input, y_label = self.transform(x_input, y_label)

        return (x_input, x_path), (y_label, x_path)

    def __len__(self):
        return len(self.xy)


class ImageDataLoader:

    def __init__(self, conf, conf_dataloader):
        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = ImageLoader(conf, conf_dataloader)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=conf_dataloader['batch_size'],
                                            num_workers=conf_dataloader['workers'],
                                            shuffle=(conf_dataloader['mode']=='train'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=True)

    def __len__(self):
        return self.image_loader.__len__()


class Image2ImageDataLoader:

    def __init__(self, conf, conf_dataloader):
        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2ImageLoader(conf, conf_dataloader)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=conf_dataloader['batch_size'],
                                            num_workers=conf_dataloader['workers'],
                                            shuffle=(conf_dataloader['mode']=='train'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=True)

    def __len__(self):
        return self.image_loader.__len__()


class Image2VectorDataLoader:

    def __init__(self, conf, conf_dataloader):
        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2VectorLoader(conf, conf_dataloader)

        if conf_dataloader['weighted_sampler'] == True and conf_dataloader['mode'] == 'train':
            sampler1 = utils.get_frequency_in_list(np.array(self.image_loader.df['label1']).tolist(), reciprocal=True)
            sampler2 = utils.get_frequency_in_list(np.array(self.image_loader.df['label']).tolist(), reciprocal=True)
            sampler = torch.utils.data.WeightedRandomSampler(sampler1 + sampler2, self.image_loader.__len__(), replacement=True)
            self.Loader = MultiEpochsDataLoader(self.image_loader,
                                                batch_size=conf_dataloader['batch_size'],
                                                num_workers=conf_dataloader['workers'],
                                                shuffle=(conf_dataloader['mode']=='train'),
                                                generator=g,
                                                pin_memory=True,
                                                sampler=sampler)

        else:
            self.Loader = MultiEpochsDataLoader(self.image_loader,
                                                batch_size=conf_dataloader['batch_size'],
                                                num_workers=conf_dataloader['workers'],
                                                shuffle=(conf_dataloader['mode']=='train'),
                                                worker_init_fn=seed_worker,
                                                generator=g,
                                                pin_memory=True)

    def __len__(self):
        return self.image_loader.__len__()


class Image2LandmarkDataLoader:

    def __init__(self, conf, conf_dataloader):
        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2LandmarkLoader(conf, conf_dataloader)

        # use your own data loader
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=conf_dataloader['batch_size'],
                                            num_workers=conf_dataloader['workers'],
                                            shuffle=(conf_dataloader['mode']=='train'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=True)

    def __len__(self):
        return self.image_loader.__len__()
