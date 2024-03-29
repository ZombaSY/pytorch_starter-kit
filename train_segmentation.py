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
        self.loader_val = self.init_data_loader(batch_size=batch_size=self.args.batch_size // 4,
                                                mode='validation',
                                                dataloader_name=self.args.dataloader,
                                                x_path=self.args.val_x_path,
                                                y_path=self.args.val_y_path)

        self.scheduler = self.set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold

    def _train(self, epoch):
        self.model.train()

        batch_losses = 0
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            output = self.model(x_in)
            
            # compute loss
            loss = self.criterion(output['seg'], target)
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
                if batch_idx != 0 and (batch_idx % self._validate_interval) == 0 and not (batch_idx != len(self.loader_train) - 1):
                    self._validate(epoch)
                    
        loss_mean = batch_losses / self.loader_train.Loader.__len__()

        print('{}{} epoch / Train Loss {}: {:.4f}, lr {:.7f}{}'.format(utils.Colors.LIGHT_CYAN,
                                                                       epoch,
                                                                       self.args.criterion,
                                                                       loss_mean,
                                                                       self.optimizer.param_groups[0]['lr'],
                                                                       utils.Colors.END))

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean}, step=epoch)

    def _validate(self, epoch):
        self.model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                # compute metric
                output_argmax = torch.argmax(output['seg'], dim=1).cpu()
                for b in range(output['seg'].shape[0]):
                    self.metric_val.update(target[b][0].cpu().detach().numpy(), output_argmax[b].cpu().detach().numpy())

        metrics_out = self.metric_val.get_results()
        c_iou = [metrics_out['Class IoU'][i] for i in range(self.args.num_class)]
        m_iou = sum(c_iou) / self.args.num_class

        print('{}{} epoch / Val mIoU: {}{}'.format(utils.Colors.LIGHT_GREEN, epoch, m_iou, utils.Colors.END))
        for i in range(self.args.num_class):
            print(f'{utils.Colors.LIGHT_GREEN}{epoch} epoch / Val Segmentation Class {i} IoU: {c_iou[i]}{utils.Colors.END}')

        if self.args.wandb:
            wandb.log({'Val Segmentation mIoU': m_iou},
                      step=epoch)
            for i in range(self.args.num_class):
                wandb.log({f'Val Segmentation Class {i} IoU': c_iou[i]},
                          step=epoch)

        if epoch == 1:  # initialize value
            if hasattr(self, 'metric_best'):
                self.metric_best['mIoU'] = m_iou
            else:
                self.metric_best = {'mIoU': m_iou}
        model_metrics = {'mIoU': m_iou}

        for key in model_metrics.keys():
            if model_metrics[key] > self.metric_best[key] or epoch % self.args.save_interval == 0:
                best_flag = True
                if epoch % self.args.save_interval == 0:
                    best_flag = False
                self.metric_best[key] = model_metrics[key]
                self.save_model(self.model, self.args.model_name, epoch, model_metrics[key], best_flag=best_flag, metric_name=key)

        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print('The model seems to be converged. Early stop training.')
                print(f'Best mIoU -----> {self.metric_best["mIoU"]}')
                wandb.log({f'Best mIoU': self.metric_best['mIoU']})
                sys.exit()  # safe exit
