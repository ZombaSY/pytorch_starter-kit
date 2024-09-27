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
from main import SEED


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_image(src):
    ext = os.path.splitext(src)[1]
    return True if ext.lower() in ['.jpeg', '.jpg', '.png', '.gif'] else False


def read_image_data(path, CV_COLOR):
    img = utils.cv2_imread(path, CV_COLOR)
    return {'data': img, 'path': path}


def read_numpy_data(path):
    data = np.load(path)
    return {'data': data, 'path': path}


def augmentations(conf):
    target_size = conf['transform_rand_crop']
    return [albumentations.RandomScale(interpolation=cv2.INTER_NEAREST, p=conf['transform_rand_resize']),
            albumentations.RandomCrop(height=target_size, width=target_size, p=1.0),
            albumentations.CoarseDropout(max_holes=4, max_height=target_size // 4, max_width=target_size // 4, min_height=target_size // 16, min_width=target_size // 16, min_holes=1, p=conf['transform_coarse_dropout']),
            albumentations.HorizontalFlip(p=conf['transform_hflip']),
            albumentations.VerticalFlip(p=conf['transform_vflip']),
            albumentations.ImageCompression(quality_lower=70, quality_upper=100, p=conf['transform_jpeg']),
            albumentations.GaussianBlur(p=conf['transform_blur']),
            albumentations.CLAHE(p=conf['transform_clahe']),
            albumentations.RandomRain(p=conf['transform_rain']),
            albumentations.RandomFog(p=conf['transform_fog']),
            albumentations.GaussNoise(p=conf['transform_g_noise']),
            albumentations.FancyPCA(p=conf['transform_fancyPCA']),
            albumentations.ColorJitter(brightness=(0.7, 1.0), contrast=(0.7, 1.0), saturation=(0.7, 1.0), hue=(-0.05, 0.05), p=conf['transform_jitter']),
            albumentations.Rotate(limit=(-30, 30), p=conf['transform_rotate']),
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
            utils.Logger().info(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['input'], itertools.repeat(cv2.IMREAD_COLOR)))

        self.transform_resize, self.transform_augmentation, self.transforms_normalize = self.init_transform(self.conf_dataloader)

    def init_transform(self, conf_dataloader):
        transform_resize = albumentations.Resize(height=conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        transform_augmentation = albumentations.Compose(augmentations(conf_dataloader['augmentations']))
        transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

        return transform_resize, transform_augmentation, transforms_normalize

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

        return {'input': x_img_tr,
                'input_path': x_path,
                'label': torch.empty(0)}

    def __len__(self):
        return self.len


class ImageSSLLoader(ImageLoader):

    def __init__(self, conf, conf_dataloader):
        super(ImageSSLLoader, self).__init__(conf, conf_dataloader)

        # overwrite inherited transforms
        self.transform_resize, self.transform_augmentation, self.transforms_normalize = self.init_transform(self.conf['dataloader_train'])
        _, self.transform_augmentation_ssl, _ = self.init_transform(self.conf_dataloader)

    def transform_ssl(self, _input):
        if self.conf_dataloader['mode'] == 'train':
            transform = self.transform_resize(image=_input)
            _input = transform['image']

            _input = _input.astype(np.uint8)
            transform = self.transform_augmentation_ssl(image=_input)
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

        if self.conf_dataloader['mode'] == 'train':
            x_img_target = self.transform(x_img)  # canonical target is once transformed by `dataloader_train.augmentation`

            x_img_perturbs = torch.empty([0, *x_img_target.shape])
            x_img_target_np = (torch.permute(x_img_target, [1, 2, 0]).numpy() * self.image_std + self.image_mean) * 255
            for i in range(self.conf_dataloader['perturbation_nums']):
                x_img_tr = self.transform_ssl(x_img_target_np).unsqueeze(0)
                x_img_perturbs = torch.cat([x_img_perturbs, x_img_tr], dim=0)
        else:
           x_img_target = self.transform(x_img)

        return {'input': x_img_target,
                'input_perturb': x_img_perturbs,
                'input_path': x_path}


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
            utils.Logger().info(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['input'], itertools.repeat(cv2.IMREAD_COLOR)))
                self.memory_data_y = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['label'], itertools.repeat(cv2.IMREAD_GRAYSCALE)))

        self.transform_resize, self.transform_augmentation, self.transforms_normalize = self.init_transform(self.conf_dataloader)

    def init_transform(self, conf_dataloader):
        transform_resize = albumentations.Resize(height=conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        transform_augmentation = albumentations.Compose(augmentations(conf_dataloader['augmentations']))
        transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

        return transform_resize, transform_augmentation, transforms_normalize

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
                _input, _label, cutout_mask = utils.cut_mix(_input, _input_refer, _label, _label_refer)

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

        return {'input': img_x_tr,
                'label': img_y_tr,
                'input_path': x_path,
                'label_path': y_path}

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
            utils.Logger().info(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
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

        self.transform_resize, self.transform_augmentation, self.transforms_normalize = self.init_transform(self.conf_dataloader)

    def init_transform(self, conf_dataloader):
        transform_resize = albumentations.Resize(height=conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        transform_augmentation = albumentations.Compose(augmentations(conf_dataloader['augmentations']))
        transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

        return transform_resize, transform_augmentation, transforms_normalize

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

        return {'input': x_img_tr,
                'label': y_vec,
                'input_path': x_path}

    def __len__(self):
        return self.len


class Image2LandmarkLoader(Dataset):

    def __init__(self, conf, conf_dataloader):
        self.conf = conf
        self.conf_dataloader = conf_dataloader

        '''
        examples)

        origin landmark indices:
        [0,  1,  2,  3,  4,  5,
         6,  7,  8,  9,  10, 11,
         12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23]

        horizontal flip landmark indices:
        [5,  4,  3,  2,  1,  0,
         11, 10, 9,  8,  7,  6,
         17, 16, 15, 14, 13, 12,
         23, 22, 21, 20, 19, 18]

        verical flip landmark indices:
        [18, 19, 20, 21, 22, 23
         12, 13, 14, 15, 16, 17,
         6,  7,  8,  9,  10, 11,
         0,  1,  2,  3,  4,  5,]
        '''
        self.points_hflip = [5,  4,  3,  2,  1,  0,
                             11, 10, 9,  8,  7,  6,
                             17, 16, 15, 14, 13, 12,
                             23, 22, 21, 20, 19, 18]

        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.25, 0.25, 0.25]

        self.df = pd.read_csv(conf_dataloader['data_path'])
        self.len = len(self.df)

        # mount_data_on_memory
        if self.conf_dataloader['data_cache']:
            utils.Logger().info(f"{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.conf_dataloader['mode']}{utils.Colors.END}")
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                self.memory_data_x = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_image_data), self.df['input'], itertools.repeat(cv2.IMREAD_COLOR)))
                self.memory_data_y = pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(read_numpy_data), self.df['label']))

        self.transform_resize, self.transform_augmentation, self.transforms_normalize = self.init_transform(self.conf_dataloader)

    def init_transform(self, conf_dataloader):
        transform_resize = albumentations.Resize(height=conf_dataloader['input_size'][0], width=self.conf_dataloader['input_size'][1], p=1)
        transform_augmentation = albumentations.Compose(
                augmentations(self.conf_dataloader['augmentations']),
                keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False)
                )
        transforms_normalize = albumentations.Compose([albumentations.Normalize(mean=self.image_mean, std=self.image_std)])

        return transform_resize, transform_augmentation, transforms_normalize

    def transform(self, _input, _label):
        random_gen = random.Random()
        transform = self.transform_resize(image=_input)
        _input = transform['image']

        if self.conf_dataloader['mode'] == 'train':

            if random_gen.random() < self.conf_dataloader['augmentations']['transform_landmark_hflip']:
                _input, _label = utils.random_hflip_landmark(_input, _label, self.points_hflip)

            _input = _input.astype(np.uint8)
            _label[..., 0] = _label[..., 0] * self.conf_dataloader['input_size'][0]
            _label[..., 1] = _label[..., 1] * self.conf_dataloader['input_size'][1]
            transform = self.transform_augmentation(image=_input, keypoints=_label)

            _input, _label = transform['image'], np.array(transform['keypoints']).astype(np.float32)
            _label[..., 0] = _label[..., 0] / self.conf_dataloader['input_size'][0]
            _label[..., 1] = _label[..., 1] / self.conf_dataloader['input_size'][1]

        norm = self.transforms_normalize(image=_input)
        _input = norm['image']
        _input = np.transpose(_input, [2, 0, 1])
        _input = torch.from_numpy(_input.astype(np.float32))
        _label = torch.from_numpy(_label.astype(np.float32))
        _label = _label.reshape(self.conf['model']['num_class'])

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

            img_x = read_image_data(x_path, cv2.IMREAD_COLOR)['data']
            img_y = read_numpy_data(y_path)['data']

        img_x_tr, img_y_tr = self.transform(img_x, img_y)

        return {'input': img_x_tr,
                'label': img_y_tr,
                'input_path': x_path,
                'label_path': y_path}

    def __len__(self):
        return self.len


class Image:

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


class ImageSSL:

    def __init__(self, conf, conf_dataloader):
        g = torch.Generator()
        g.manual_seed(SEED)

        self.image_loader = ImageSSLLoader(conf, conf_dataloader)

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


class Image2Image:

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


class Image2Vector:

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


class Image2Landmark:
    def __init__(self, conf, conf_dataloader):
        g = torch.Generator()
        g.manual_seed(SEED)

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
