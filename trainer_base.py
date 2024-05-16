import abc
import torch
import os
import wandb
import math

from models import dataloader as dataloader_hub
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import utils
from models import metrics

from datetime import datetime


class TrainerBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, now=None, k_fold=0):
        self.args = args
        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = self.args.saved_model_directory + '/' + now_time
        self.k_fold = k_fold

        # save hyper-parameters
        if not self.args.debug:
            with open(self.args.config_path, 'r') as f_r:
                file_path = self.args.saved_model_directory + '/' + now_time
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path, self.args.config_path.split('/')[-1]), 'w') as f_w:
                    f_w.write(f_r.read())

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.init_model(self.args, self.device)
        self.optimizer = self.init_optimizer(self.args, self.model)
        self.criterion = self.init_criterion(self.args, self.device)
        self.metric_train = self.init_metric(self.args.task, self.args.num_class)
        self.metric_val = self.init_metric(self.args.task, self.args.num_class)

        if hasattr(self.args, 'model_path'):
            if self.args.model_path != '':
                if 'imagenet' in self.args.model_path.lower():
                    self.model.module.load_pretrained_imagenet(self.args.model_path)
                    print('Model loaded successfully!!! (ImageNet)')
                else:
                    self.model.module.load_pretrained(self.args.model_path)
                    print('Model loaded successfully!!! (Custom)')
                self.model.to(self.device)

        if self.args.wandb:
            wandb.init(project='{}'.format(args.project_name), config=args, name=now_time,
                       settings=wandb.Settings(start_method="fork"))
            wandb.watch(self.model)

        self.model_post_path_dict = {}
        self.last_saved_epoch = 0
        self.callback = utils.TrainerCallBack()
        if hasattr(self.model.module, 'train_callback'):
            self.callback.train_callback = self.model.module.train_callback

    @abc.abstractmethod
    def _train(self, epoch):
        pass

    @abc.abstractmethod
    def _validate(self, model, epoch):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def save_model(self, model, model_name, epoch, metric=None, metric_name='metric'):
        file_path = self.saved_model_directory + '/'

        file_format = file_path + model_name + '-Epoch_' + str(epoch) + '-' + metric_name + '_' + str(metric)[:6] + '-folds_' + str(self.k_fold) + '.pt'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if metric_name in self.model_post_path_dict.keys():
            os.remove(self.model_post_path_dict[metric_name])
        self.model_post_path_dict[metric_name] = file_format

        torch.save(model.state_dict(), file_format)

        print(f'{utils.Colors.LIGHT_RED}{file_format} model saved!!{utils.Colors.END}')
        self.last_saved_epoch = epoch

    @staticmethod
    def init_data_loader(args,
                         x_path=None,
                         y_path=None,
                         csv_path=None):

        if args.dataloader == 'Image2Image':
            loader = dataloader_hub.Image2ImageDataLoader(x_path=x_path,
                                                          y_path=y_path,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.worker,
                                                          mode=args.mode,
                                                          args=args)
        elif args.dataloader == 'Image2Vector':
            loader = dataloader_hub.Image2VectorDataLoader(csv_path=csv_path,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.worker,
                                                           mode=args.mode,
                                                           args=args)
        elif args.dataloader == 'Image2Landmark':
            loader = dataloader_hub.Image2LandmarkDataLoader(data_path=csv_path,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.worker,
                                                             mode=args.mode,
                                                             args=args)
        else:
            raise Exception('No dataloader named', args.dataloader)

        return loader

    @staticmethod
    def set_scheduler(args, optimizer, data_loader):
        scheduler = None
        steps_per_epoch = math.ceil((data_loader.__len__() / args.batch_size))

        if hasattr(args, 'scheduler'):
            if args.scheduler == 'WarmupCosine':
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                              warmup_steps=steps_per_epoch * args.warmup_epoch,
                                                              t_total=args.epoch * steps_per_epoch,
                                                              cycles=args.epoch / args.cycles,
                                                              last_epoch=-1)
            elif args.scheduler == 'CosineAnnealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cycles, eta_min=args.lr_min)
            elif args.scheduler == 'Constant':
                scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
            elif args.scheduler == 'WarmupConstant':
                scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * args.warmup_epoch)
            else:
                print(f'{utils.Colors.LIGHT_PURPLE}No scheduler found --> {args.scheduler}{utils.Colors.END}')
        else:
            pass

        return scheduler

    @staticmethod
    def init_model(args, device):
        model = getattr(model_implements, args.model_name)(**vars(args)).to(device)

        return torch.nn.DataParallel(model)

    @staticmethod
    def init_criterion(args, device):
        criterion = getattr(loss_hub, args.criterion)().to(device)

        return criterion

    @staticmethod
    def init_optimizer(args, model):
        optimizer = None

        if args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

        return optimizer

    @staticmethod
    def init_metric(task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class)
        elif task_name == 'classification':
            metric = metrics.StreamSegMetrics_classification(num_class)
        elif task_name == 'regression' or 'landmark':
            metric = metrics.StreamSegMetrics_classification(num_class)
        else:
            raise Exception('No task named', task_name)

        return metric
