import torch
import wandb
import numpy as np
import sys

from models import utils
from trainer_base import TrainerBase


class TrainerClassification(TrainerBase):
    def __init__(self, args, now=None, k_fold=0):
        super(TrainerClassification, self).__init__(args, now=now, k_fold=k_fold)

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available to being set, not must.
        self.loader_train = self.init_data_loader(batch_size=self.args.batch_size,
                                                  mode='train',
                                                  dataloader_name=self.args.dataloader,
                                                  csv_path=self.args.train_csv_path)
        self.loader_val = self.init_data_loader(batch_size=self.args.batch_size,
                                                mode='validation',
                                                dataloader_name=self.args.dataloader,
                                                csv_path=self.args.valid_csv_path)

        self.scheduler = self.set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold // self.args.batch_size

    def _train(self, epoch):
        self.model.train()
        metric_list_mean = {'acc': [],
                            'f1': []}
        batch_losses = 0
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            output = self.model(x_in)

            # compute loss
            loss = self.criterion(output['class'], target)
            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            # ----- backward ----- #
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_losses += loss.item()

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self._validate_interval) == 0 and batch_idx < (self.loader_train.__len__() // self.args.batch_size) - self._validate_interval:
                    self._validate(epoch)

            # compute metric
            output_argmax = torch.argmax(output['class'], dim=1).detach().cpu().numpy()
            target_argmax = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
            self.metric_train.update(output_argmax, target_argmax)

        metric_result = self.metric_train.get_results()
        metric_list_mean['acc'] = metric_result['acc']
        metric_list_mean['f1'] = metric_result['f1']

        for key in metric_list_mean.keys():
            metric_list_mean[key] = np.mean(metric_list_mean[key])

        for key in metric_list_mean.keys():
            log_str = f'train {key}: {metric_list_mean[key]}'
            print(f'{utils.Colors.LIGHT_GREEN} {epoch} epoch / {log_str} {utils.Colors.END}')

            if self.args.wandb:
                wandb.log({f'train {key}': metric_list_mean[key]},
                          step=epoch)

        loss_mean = batch_losses / self.loader_train.Loader.__len__()

        print('{}{} epoch / train Loss {}: {:.4f}, lr {:.7f}{}'.format(utils.Colors.LIGHT_CYAN,
                                                                       epoch,
                                                                       self.args.criterion,
                                                                       loss_mean,
                                                                       self.optimizer.param_groups[0]['lr'],
                                                                       utils.Colors.END))

        if self.args.wandb:
            wandb.log({'train Loss {}'.format(self.args.criterion): loss_mean}, step=epoch)

        self.metric_train.reset()

    def _validate(self, epoch):
        self.model.eval()
        metric_list_mean = {'acc': [],
                            'f1': []}

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                # compute metric
                output_argmax = torch.argmax(output['class'], dim=1).detach().cpu().numpy()
                target_argmax = torch.argmax(target.squeeze() if (self.args.batch_size // 4) != 1 else target, dim=1).detach().cpu().numpy()
                self.metric_val.update(output_argmax, target_argmax)

        metric_result = self.metric_val.get_results()
        metric_list_mean['acc'] = metric_result['acc']
        metric_list_mean['f1'] = metric_result['f1']

        for key in metric_list_mean.keys():
            metric_list_mean[key] = np.mean(metric_list_mean[key])

        for key in metric_list_mean.keys():
            log_str = f'validation {key}: {metric_list_mean[key]}'
            print(f'{utils.Colors.LIGHT_GREEN} {epoch} epoch / {log_str} {utils.Colors.END}')

            if self.args.wandb:
                wandb.log({f'validation {key}': metric_list_mean[key]},
                          step=epoch)

        if epoch == 1:  # initialize value
            if hasattr(self, 'metric_best'):
                pass
            else:
                self.metric_best = {}
                for key in metric_list_mean.keys():
                    self.metric_best[key] = metric_list_mean[key]

        for key in metric_list_mean.keys():
            if metric_list_mean[key] > self.metric_best[key] or epoch % self.args.save_interval == 0:
                best_flag = True
                if epoch % self.args.save_interval == 0:
                    best_flag = False
                self.metric_best[key] = metric_list_mean[key]
                self.save_model(self.model.module, self.args.model_name, epoch, metric_list_mean[key], best_flag=best_flag, metric_name=key)

        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print('The model seems to be converged. Early stop training.')
                print(f'Best acc -----> {self.metric_best["f1"]}')
                if self.self.args.wandb:
                    wandb.log({f'Best f1': self.metric_best['f1']})
