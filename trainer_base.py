import abc
import torch
import os
import wandb
import math

from models import dataloader
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import utils
from models import metrics

from datetime import datetime


class TrainerBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, conf, now=None, k_fold=0):
        self.conf = conf
        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = self.conf['env']['saved_model_directory'] + '/' + now_time
        self.k_fold = k_fold

        # save hyper-parameters
        if not self.conf['env']['debug']:
            with open(self.conf['config_path'], 'r') as f_r:
                file_path = os.path.join(self.conf['env']['saved_model_directory'], now_time)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path, os.path.split(self.conf['config_path'])[-1]), 'w') as f_w:
                    f_w.write(f_r.read())

        # Check cuda available and assign to device
        use_cuda = self.conf['env']['cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # init model
        self.model = self.init_model(self.conf, self.device)
        if self.conf['model']['saved_ckpt'] != '':
            if 'imagenet' in self.conf['model']['saved_ckpt'].lower():
                self.model.module.load_pretrained_imagenet(self.conf['model']['saved_ckpt'])
                print(f'{utils.Colors.LIGHT_RED}Model loaded successfully!!! (ImageNet){utils.Colors.END}')
            else:
                self.model.module.load_state_dict(torch.load(self.conf['model']['saved_ckpt']))
                print(f'{utils.Colors.LIGHT_RED}Model loaded successfully!!! (Custom){utils.Colors.END}')
            self.model.to(self.device)

        # init dataloader
        self.loader_train = self.init_data_loader(conf=self.conf,
                                                  conf_dataloader=self.conf['dataloader_train'])
        self.loader_valid = self.init_data_loader(conf=self.conf,
                                                  conf_dataloader=self.conf['dataloader_valid'])

        # init optimizer
        self.optimizer = self.init_optimizer(self.conf['optimizer'], self.model)

        # init scheduler
        self.scheduler = self.set_scheduler(self.conf, self.conf['dataloader_train'], self.optimizer, self.loader_train)

        # init criterion
        self.criterion = self.init_criterion(self.conf['criterion'], self.device)

        # init metrics
        self.metric_train = self.init_metric(self.conf['env']['task'], self.conf['model']['num_class'])
        self.metric_val = self.init_metric(self.conf['env']['task'], self.conf['model']['num_class'])

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
        file_path = self.saved_model_directory + '/'

        file_format = file_path + self.conf['model']['name'] + '-folds_' + str(self.k_fold) + '-' + metric_name + '_' + str(metric)[:6] + '-Epoch_' + str(epoch) + '.pt'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if metric_name in self.model_post_path_dict.keys():
            os.remove(self.model_post_path_dict[metric_name])
        self.model_post_path_dict[metric_name] = file_format

        torch.save(self.model.module.state_dict(), file_format)

        print(f'{utils.Colors.LIGHT_RED}{file_format} model saved!!{utils.Colors.END}')
        self.last_saved_epoch = epoch

    def check_metric(self, epoch, metric_dict):
        if not hasattr(self, 'metric_best'):
            self.metric_best = {}
            for key in metric_dict.keys():
                self.metric_best[key] = metric_dict[key]
                self.save_model(epoch, metric_dict[key], metric_name=key)
        else:
            for key in metric_dict.keys():
                if (key == 'loss' and metric_dict[key] < self.metric_best[key]) or (key != 'loss' and metric_dict[key] > self.metric_best[key]):
                    self.metric_best[key] = metric_dict[key]
                    self.save_model(epoch, metric_dict[key], metric_name=key)

    @staticmethod
    def init_data_loader(conf,
                         conf_dataloader):
        loader = getattr(dataloader, conf_dataloader['name'])(conf, conf_dataloader)

        return loader

    @staticmethod
    def set_scheduler(conf, conf_dataloader, optimizer, data_loader):
        scheduler = None
        steps_per_epoch = math.ceil((data_loader.__len__() / conf_dataloader['batch_size']))

        if hasattr(conf, 'scheduler'):
            if conf['scheduler']['name'] == 'WarmupCosine':
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                              warmup_steps=steps_per_epoch * conf['scheduler']['warmup_epoch'],
                                                              t_total=conf['env']['epoch'] * steps_per_epoch,
                                                              cycles=conf['env']['epoch'] / conf['scheduler']['cycles'],
                                                              last_epoch=-1)
            elif conf['scheduler']['name'] == 'CosineAnnealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf['scheduler']['cycles'], eta_min=conf['scheduler']['lr_min'])
            elif conf['scheduler']['name'] == 'Constant':
                scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
            elif conf['scheduler']['name'] == 'WarmupConstant':
                scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * conf['scheduler']['warmup_epoch'])
            else:
                print(f"{utils.Colors.LIGHT_PURPLE}No scheduler found --> {conf['scheduler']['name']}{utils.Colors.END}")
        else:
            pass

        return scheduler

    @staticmethod
    def init_model(conf, device):
        model = getattr(model_implements, conf['model']['name'])(conf['model']).to(device)

        return torch.nn.DataParallel(model)

    @staticmethod
    def init_criterion(conf_criterion, device):
        criterion = getattr(loss_hub, conf_criterion['name'])(conf_criterion).to(device)

        return criterion

    @staticmethod
    def init_optimizer(conf_optimizer, model):
        optimizer = None

        if conf_optimizer['name'] == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=conf_optimizer['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=conf_optimizer['weight_decay'])

        return optimizer

    @staticmethod
    def init_metric(task_name, num_class):
        if 'segmentation' in task_name:
            metric = metrics.StreamSegMetrics_segmentation(num_class)
        elif 'classification' or 'regression' or 'landmark' in task_name:
            metric = metrics.StreamSegMetrics_classification(num_class)
        else:
            raise Exception('No task named', task_name)

        return metric
