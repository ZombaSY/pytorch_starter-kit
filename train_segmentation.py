import torch
import wandb
import numpy as np
import sys

from models import utils
from trainer_base import TrainerBase


class TrainerSegmentation(TrainerBase):
    def __init__(self, args, now=None):
        super(TrainerSegmentation, self).__init__(args, now=now)

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available to being set, not must.
        self.loader_train = self.init_data_loader(batch_size=self.args.batch_size,
                                                  mode='train',
                                                  dataloader_name=self.args.dataloader,
                                                  x_path=self.args.train_x_path,
                                                  y_path=self.args.train_y_path)
        self.loader_val = self.init_data_loader(batch_size=self.args.batch_size // 4,
                                                mode='validation',
                                                dataloader_name=self.args.dataloader,
                                                x_path=self.args.val_x_path,
                                                y_path=self.args.val_y_path)

        self.scheduler = self.set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold

    def _train(self, epoch):
        self.model.train()
        self.callback.train_callback()

        batch_losses = []
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            output, _ = self.model(x_in)

            # compute metric
            output_argmax = torch.argmax(output, dim=1)
            for b in range(output.shape[0]):
                self.metric_train.update(target[b][0].cpu().detach().numpy(), output_argmax[b].cpu().detach().numpy())

            # compute loss
            loss = self.criterion(output, target)

            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            # ----- backward ----- #
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.args.ema_decay != 0:
                self.model_ema.update(self.model)
            if self.scheduler is not None:
                self.scheduler.step()

            batch_losses.append(loss.item())

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self._validate_interval) == 0 and not (batch_idx != len(self.loader_train) - 1):
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

        loss_mean = np.mean(batch_losses)
        metrics = self.metric_train.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class

        print('{}{} epoch / Train Loss {}: {:.4f}, lr {:.7f} \n Train mIoU: {:.4f}{}'.format(utils.Colors.LIGHT_CYAN,
                                                                                             epoch,
                                                                                             self.args.criterion,
                                                                                             loss_mean,
                                                                                             self.optimizer.param_groups[0]['lr'],
                                                                                             mIoU, utils.Colors.END))
        for i in range(self.args.num_class):
            print(f'{utils.Colors.LIGHT_CYAN}{epoch} epoch / Train Class {i} IoU: {cIoU[i]}{utils.Colors.END}')

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean,
                       'Train mIoU': mIoU},
                      step=epoch)
            for i in range(self.args.num_class):
                wandb.log({f'Train Class {i} IoU': cIoU[i]}, step=epoch)

        self.metric_train.reset()

    def _validate(self, model, epoch):
        model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output, _ = model(x_in)

                # compute metric
                output_argmax = torch.argmax(output, dim=1).cpu()
                for b in range(output.shape[0]):
                    self.metric_val.update(target[b][0].cpu().detach().numpy(), output_argmax[b].cpu().detach().numpy())

        metrics_out = self.metric_val.get_results()
        cIoU = [metrics_out['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class

        print('{}{} epoch / Val mIoU: {}{}'.format(utils.Colors.LIGHT_GREEN, epoch, mIoU, utils.Colors.END))
        for i in range(self.args.num_class):
            print(f'{utils.Colors.LIGHT_GREEN}{epoch} epoch / Val Segmentation Class {i} IoU: {cIoU[i]}{utils.Colors.END}')

        if self.args.wandb:
            wandb.log({'Val Segmentation mIoU': mIoU},
                      step=epoch)
            for i in range(self.args.num_class):
                wandb.log({f'Val Segmentation Class {i} IoU': cIoU[i]},
                          step=epoch)

        if epoch == 1:  # initialize value
            if hasattr(self, 'metric_best'):
                self.metric_best['mIoU'] = mIoU
            else:
                self.metric_best = {'mIoU': mIoU}
        model_metrics = {'mIoU': mIoU}

        for key in model_metrics.keys():
            if model_metrics[key] > self.metric_best[key] or epoch % self.args.save_interval == 0:
                best_flag = True
                if epoch % self.args.save_interval == 0:
                    best_flag = False
                self.metric_best[key] = model_metrics[key]
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=best_flag, metric_name=key)

        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            if self.args.ema_decay != 0:
                self._validate(self.model_ema.module, epoch)
            else:
                self._validate(self.model, epoch)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print('The model seems to be converged. Early stop training.')
                print(f'Best mIoU -----> {self.metric_best["mIoU"]}')
                wandb.log({f'Best mIoU': self.metric_best['mIoU']})
                sys.exit()  # safe exit
