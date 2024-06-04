import torch
import wandb

from models import utils
from trainer_base import TrainerBase


class TrainerRegression(TrainerBase):
    def __init__(self, args, now=None, k_fold=0):
        super(TrainerRegression, self).__init__(args, now=now, k_fold=k_fold)

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available to being set, not must.
        self.loader_train = self.init_data_loader(args=self.args,
                                                  mode='train',
                                                  csv_path=self.args.train_csv_path)
        self.loader_val = self.init_data_loader(args=self.args,
                                                mode='inference',
                                                csv_path=self.args.valid_csv_path)

        self.scheduler = self.set_scheduler(self.args, self.optimizer, self.loader_train)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold // self.args.batch_size

    def _train(self, epoch):
        self.model.train()

        metric_list_mean = {}
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

        loss_mean = batch_losses / self.loader_train.Loader.__len__()
        metric_list_mean['loss'] = loss_mean
        for key in metric_list_mean.keys():
            log_str = f'train {key}: {metric_list_mean[key]}'
            print(f'{utils.Colors.LIGHT_GREEN} {epoch} epoch / {log_str} {utils.Colors.END}')

            if self.args.wandb:
                wandb.log({f'train {key}': metric_list_mean[key]},
                          step=epoch)

        print('{}{} epoch / train {}: {:.4f}, lr {:.7f}{}'.format(utils.Colors.LIGHT_CYAN,
                                                                       epoch,
                                                                       self.args.criterion,
                                                                       loss_mean,
                                                                       self.optimizer.param_groups[0]['lr'],
                                                                       utils.Colors.END))

        self.metric_train.reset()

    def _validate(self, epoch):
        self.model.eval()
        metric_list_mean = {'loss': []}
        batch_losses = 0

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                # compute loss
                loss = self.criterion(output['class'], target)
                if not torch.isfinite(loss):
                    raise Exception('Loss is NAN. End training.')

                batch_losses += loss.item()

        loss_mean = batch_losses / self.loader_val.Loader.__len__()
        metric_list_mean['loss'] = loss_mean
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
            if (key == 'loss' and metric_list_mean[key] < self.metric_best[key]) or (key != 'loss' and metric_list_mean[key] > self.metric_best[key]):
                self.metric_best[key] = metric_list_mean[key]
                self.save_model(self.model.module, self.args.model_name, epoch, metric_list_mean[key], metric_name=key)

        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print('The model seems to be converged. Early stop training.')
                print(f'Best loss -----> {self.metric_best["loss"]}')
                if self.args.wandb:
                    wandb.log({f'Best f1': self.metric_best['loss']})
                break
