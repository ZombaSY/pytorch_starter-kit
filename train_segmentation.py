import torch
import time
import os
import wandb
import numpy as np
import sys

from models import dataloader as dataloader_hub
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import metrics

from datetime import datetime
from timm.utils import ModelEmaV2, get_state_dict


class Trainer_seg:
    def __init__(self, args, now=None):
        self.start_time = time.time()
        self.args = args

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available of being set, not must.
        self.loader_train = self.__init_data_loader(self.args.train_x_path,
                                                    self.args.train_y_path,
                                                    self.args.batch_size,
                                                    mode='train')
        self.loader_val = self.__init_data_loader(self.args.val_x_path,
                                                  self.args.val_y_path,
                                                  batch_size=1,
                                                  mode='validation')

        self.model = self.__init_model(self.args.model_name)
        self.optimizer = self._init_optimizer(self.model, self.args.lr)
        self.scheduler = self._set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)

        if hasattr(self.args, 'model_path'):
            if self.args.model_path != '':
                if 'imagenet' in self.args.model_path.lower():
                    self.model.module.load_pretrained_imagenet(self.args.model_path)
                    print('Model loaded successfully!!! (ImageNet)')
                else:
                    self.model.module.load_pretrained(self.args.model_path)    # TODO: define "load_pretrained" abstract method to all models
                    print('Model loaded successfully!!! (Custom)')
                self.model.to(self.device)

        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        if self.args.ema_decay != 0:
            self.model_ema = ModelEmaV2(self.model, decay=self.args.ema_decay, device=self.device)

        self.criterion = self._init_criterion(self.args.criterion)

        if self.args.wandb:
            if self.args.mode == 'train':
                wandb.watch(self.model)

        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = self.args.saved_model_directory + '/' + now_time

        self.metric_train = metrics.StreamSegMetrics_segmentation(self.args.num_class)
        self.metric_val = metrics.StreamSegMetrics_segmentation(self.args.num_class)
        self.metric_best = {'cIoU': 0, 'mIoU': 0}
        self.model_post_path_dict = {}
        self.last_saved_epoch = 0

        self.__validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold

    def _train(self, epoch):
        self.model.train()
        batch_losses = []
        print('Start Train')
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            if (x_in[0].shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

            output = self.model(x_in)

            # compute metric
            output_argmax = torch.argmax(output, dim=1).cpu()
            self.metric_train.update(target.cpu().detach().numpy(), output_argmax.numpy())

            # compute loss
            loss = self.criterion(output, target)

            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.args.ema_decay != 0:
                self.model_ema.update(self.model)

            batch_losses.append(loss.item())

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self.__validate_interval) == 0:
                    if self.args.ema_decay != 0:
                        self._validate(self.model_ema.module, epoch)
                    else:
                        self._validate(self.model, epoch)

            if (batch_idx != 0) and (batch_idx % (self.args.log_interval // self.args.batch_size) == 0):
                loss_mean = np.mean(batch_losses)
                print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                                    self.args.criterion,
                                                                    loss_mean,
                                                                    self.optimizer.param_groups[0]['lr']))

            torch.cuda.empty_cache()

        loss_mean = np.mean(batch_losses)
        metrics = self.metric_train.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class

        print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                            self.args.criterion,
                                                            loss_mean,
                                                            self.optimizer.param_groups[0]['lr']))

        print('{} epoch / Train mIoU: {}'.format(epoch, mIoU))
        for i in range(self.args.num_class):
            print(f'{epoch} epoch / Train Class {i} IoU: {cIoU[i]}')

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean,
                       'Train mIoU': mIoU})
            for i in range(self.args.num_class):
                wandb.log({f'Train Class {i} IoU': cIoU[i]})

        self.metric_train.reset()

    def _validate(self, model, epoch):
        model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target
                x_in = x_in.to(self.device)

                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = model(x_in)

                # compute metric
                output_argmax = torch.argmax(output, dim=1).cpu()
                self.metric_val.update(target.cpu().detach().numpy(), output_argmax.numpy())

                # Log Image on WandB
                # if (batch_idx == 0) and self.args.wandb and (epoch % self.args.save_interval == 0):
                #     print('logging image on wandb...')
                #     target = target[0]
                #
                #     # resize image for the log
                #     x_in = F.interpolate(x_in, scale_factor=0.5)
                #     output = F.interpolate(output, scale_factor=0.5)
                #
                #     x_in = x_in[0]    # detach batch
                #     output = output[0]
                #
                #     target = target.float()
                #     target = target.repeat(3, 1, 1)  # repeat 3 channel on grey scale image
                #
                #     wandb.log({'input': [wandb.Image(x_in.cpu().detach())]})
                #     wandb.log({'target': [wandb.Image(target.cpu().detach())]})
                #     wandb.log({'output': [wandb.Image(output.cpu().detach())]})

        metrics = self.metric_val.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class

        print('{} epoch / Val mIoU: {}'.format(epoch, mIoU))
        for i in range(self.args.num_class):
            print(f'{epoch} epoch / Val Class {i} IoU: {cIoU[i]}')

        if self.args.wandb:
            wandb.log({'Val mIoU': mIoU})
            for i in range(self.args.num_class):
                wandb.log({f'Val Class {i} IoU': cIoU[i]})

        model_metrics = {'cIoU': cIoU[1], 'mIoU': mIoU}     # cIoU

        for key in model_metrics.keys():
            if model_metrics[key] > self.metric_best[key]:
                self.metric_best[key] = model_metrics[key]
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=True, metric_name=key)

        self.metric_val.reset()

        if (epoch - self.last_saved_epoch) > self.args.cycles * 2:
            print('The model seems to be converged. Early stop training.')
            print(f'Best mIoU -----> {self.metric_best["mIoU"]}')
            wandb.log({f'Best mIoU': self.metric_best['mIoU']})
            sys.exit()  # safe exit

    def start_train(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            if self.args.ema_decay != 0:
                self._validate(self.model_ema.module, epoch)
            else:
                self._validate(self.model, epoch)

            print('### {} / {} epoch ended###'.format(epoch, self.args.epoch))

    def save_model(self, model, model_name, epoch, metric=None, best_flag=False, metric_name='metric'):
        file_path = self.saved_model_directory + '/'

        file_format = file_path + model_name + '_Epoch_' + str(epoch) + '_' + metric_name + '_' + str(metric) + '.pt'

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        if best_flag:
            if metric_name in self.model_post_path_dict.keys():
                os.remove(self.model_post_path_dict[metric_name])
            self.model_post_path_dict[metric_name] = file_format

        if self.args.ema_decay != 0:
            torch.save(get_state_dict(model), file_format)
        else:
            torch.save(model.state_dict(), file_format)

        print(file_format + '\t model saved!!')
        self.last_saved_epoch = epoch

    def __init_data_loader(self,
                           x_path,
                           y_path,
                           batch_size,
                           mode):

        if self.args.dataloader == 'Image2Image':
            loader = dataloader_hub.Image2ImageDataLoader(x_path=x_path,
                                                          y_path=y_path,
                                                          batch_size=batch_size,
                                                          num_workers=self.args.worker,
                                                          pin_memory=self.args.pin_memory,
                                                          mode=mode,
                                                          args=self.args)
        else:
            raise Exception('No datalodaer named', self.args.dataloader)

        return loader

    def __init_model(self, model_name):
        if model_name == 'Unet':
            model = model_implements.Unet(n_channels=self.args.input_channel, n_classes=self.args.num_class).to(self.device)
        elif model_name == 'Swin':
            model = model_implements.Swin(num_classes=self.args.num_class,
                                          in_channel=self.args.input_channel).to(self.device)
        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

    def _init_criterion(self, criterion_name):
        if criterion_name == 'CE':
            criterion = loss_hub.CrossEntropy().to(self.device)
        else:
            raise Exception('No criterion named', criterion_name)

        return criterion

    def _init_optimizer(self, optimizer_name, model, lr):
        optimizer = None

        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)

        return optimizer

    def _set_scheduler(self, optimizer, scheduler_name, data_loader, batch_size):
        scheduler = None
        steps_per_epoch = (data_loader.__len__() // batch_size) + 1

        if hasattr(self.args, 'scheduler'):
            if scheduler_name == 'WarmupCosine':
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                              warmup_steps=steps_per_epoch * self.args.warmup_epoch,
                                                              t_total=self.args.epoch * steps_per_epoch,
                                                              cycles=self.args.epoch / self.args.cycles,
                                                              last_epoch=-1)
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.cycles, eta_min=0)
            elif scheduler_name == 'ConstantLRSchedule':
                scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
            elif scheduler_name == 'WarmupConstantSchedule':
                scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * self.args.warmup_epoch)
            else:
                raise Exception('No scheduler named', scheduler_name)
        else:
            pass

        return scheduler
