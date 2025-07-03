import abc
import torch
import os
import wandb
import timm
import shutil
import logging

from models import utils
from tools import utils_tool
from datetime import datetime
from accelerate import Accelerator


class TrainerBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, conf, now=None, k_fold=0):
        self.conf = conf
        self.accelerator = Accelerator()    # `$accelearte config` to set configuration at first
        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = os.path.join(self.conf['env']['saved_model_directory'], now_time)
        self.k_fold = k_fold

        # save hyper-parameters
        if not self.conf['env']['debug']:
            # make direcotry
            if not os.path.exists(self.saved_model_directory):
                os.makedirs(self.saved_model_directory)

            # save configuration
            shutil.copy(self.conf['config_path'], os.path.join(self.saved_model_directory, os.path.split(self.conf['config_path'])[-1]))

            # init logger file
            utils.Logger(os.path.join(self.saved_model_directory, 'log.txt'), level=logging.DEBUG if self.conf['env']['debug'] else logging.INFO)
            utils.Logger().info(f'{utils.Colors.BOLD}Running {self.k_fold}th fold...{utils.Colors.END}')
        else:
            utils.Logger('log.txt', level=logging.DEBUG if self.conf['env']['debug'] else logging.INFO)

        # Check cuda available and assign to device
        use_cuda = self.conf['env']['cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        utils.Logger().info(f'device is located to {self.device}')

        # init model
        self.model = utils_tool.init_model(self.conf, self.device)
        if self.conf['model']['saved_ckpt'] != '':
            if 'imagenet' in self.conf['model']['saved_ckpt'].lower():
                self.model.module.load_pretrained_imagenet(self.conf['model']['saved_ckpt'])
                utils.Logger().info(f'{utils.Colors.LIGHT_RED}Model loaded successfully!!! (ImageNet){utils.Colors.END}')
            else:
                self.model.module.load_state_dict(torch.load(self.conf['model']['saved_ckpt']))
                utils.Logger().info(f'{utils.Colors.LIGHT_RED}Model loaded successfully!!! (Custom){utils.Colors.END}')
        if self.conf['model']['use_ema']:
            self.model_ema = timm.utils.ModelEmaV2(self.model, decay=0.9999, device=self.device)

        # init dataloader
        self.loader_train = utils_tool.init_data_loader(conf=self.conf,
                                                  conf_dataloader=self.conf['dataloader_train'])
        self.loader_valid = utils_tool.init_data_loader(conf=self.conf,
                                                  conf_dataloader=self.conf['dataloader_valid'])

        # init optimizer
        self.optimizer = utils_tool.init_optimizer(self.conf['optimizer'], self.model)

        # init scheduler
        self.scheduler = utils_tool.set_scheduler(self.conf, self.conf['dataloader_train'], self.optimizer, self.loader_train)

        # init criterion
        self.criterion = utils_tool.init_criterion(self.conf['criterion'], self.device)

        # init metrics
        self.metric_train = utils_tool.init_metric(self.conf['env']['task'], self.conf['model']['num_class'])
        self.metric_val = utils_tool.init_metric(self.conf['env']['task'], self.conf['model']['num_class'])

        if self.conf['env']['wandb']:
            wandb.init(project='{}'.format(self.conf['env']['project_name']), config=conf, name=now_time,
                       settings=wandb.Settings(start_method="fork"))
            wandb.watch(self.model)

        self.model_post_path_dict = {}
        self.last_saved_epoch = 0
        self.callback = utils.TrainerCallBack()
        if hasattr(self.model.module, 'train_callback'):
            self.callback.train_callback = self.model.module.train_callback

        self._validate_interval = max(1, (len(self.loader_train.Loader) // self.conf['env']['train_fold']) + 1)

        # set acceleration
        self.model, self.optimizer, self.scheduler, self.loader_train, self.loader_valid = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.loader_train, self.loader_valid)

    @abc.abstractmethod
    def _train(self, epoch):
        pass

    @abc.abstractmethod
    def _validate(self, model, epoch):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def save_model(self, epoch, metric=None, metric_name='metric'):
        if not self.conf['env']['debug']:
            file_path = self.saved_model_directory

            file_format = os.path.join(file_path, self.conf['model']['name'] + '-folds_' + str(self.k_fold) + '-' + metric_name + '_' + str(metric)[:6] + '-Epoch_' + str(epoch) + '.pt')

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            if metric_name in self.model_post_path_dict.keys():
                os.remove(self.model_post_path_dict[metric_name])
            self.model_post_path_dict[metric_name] = file_format

            torch.save(self.model.module.state_dict(), file_format)

            utils.Logger().info(f'{utils.Colors.LIGHT_RED}model saved -> {file_format}{utils.Colors.END}')
            self.last_saved_epoch = epoch

    def check_metric(self, epoch, metric_dict):
        if not hasattr(self, 'metric_best'):
            self.metric_best = {}
            for key in metric_dict.keys():
                self.metric_best[key] = metric_dict[key]
                self.save_model(epoch, metric_dict[key], metric_name=key)
        else:
            for key in metric_dict.keys():
                if ('loss' in key and metric_dict[key] < self.metric_best[key]) or ('loss' not in key and metric_dict[key] > self.metric_best[key]):
                    self.metric_best[key] = metric_dict[key]
                    self.save_model(epoch, metric_dict[key], metric_name=key)
