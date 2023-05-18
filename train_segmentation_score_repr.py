from abc import ABC

import torch
import wandb
import numpy as np
import sys

from models import utils
from models import metrics
from trainer_base import TrainerBase


class Trainer_segScoreRepr(TrainerBase):
    def __init__(self, args, now=None):
        super(Trainer_segScoreRepr, self).__init__(args, now)

        self.loader_train = self.init_data_loader(batch_size=self.args.batch_size,
                                                  mode='train',
                                                  dataloader_name=self.args.dataloader,
                                                  x_path=self.args.train_x_path,
                                                  y_path=self.args.train_y_path)
        self.loader_val = self.init_data_loader(batch_size=1,
                                                mode='validation',
                                                dataloader_name=self.args.dataloader,
                                                x_path=self.args.val_x_path,
                                                y_path=self.args.val_y_path)

        self.loader_train_score = self.init_data_loader(batch_size=self.args.batch_size,
                                                        mode='train',
                                                        dataloader_name=self.args.dataloader_score,
                                                        csv_path=self.args.train_csv_path)
        self.loader_val_score = self.init_data_loader(batch_size=self.args.batch_size // 4,
                                                      mode='validation',
                                                      dataloader_name=self.args.dataloader_score,
                                                      csv_path=self.args.val_csv_path)

        self.loader_train_unsup = self.init_data_loader(batch_size=self.args.batch_size,
                                                        mode='train',
                                                        dataloader_name=self.args.dataloader_unsup,
                                                        x_path=self.args.train_x_path)

        self.criterion_score = self.init_criterion(self.args.criterion_score)
        self.criterion_unsup = self.init_criterion(self.args.criterion_unsup)
        self.metric_val_score = metrics.StreamSegMetrics_classification(self.args.num_class - 1)

        self.scheduler = self.set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold

        if hasattr(self.model.module, 'train_score_callback'):
            self.callback.train_score_callback = self.model.module.train_score_callback

    def _train_segmentation(self, epoch):
        self.model.train()

        batch_losses = []
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            output = self.model(x_in)

            # compute loss
            loss = self.criterion(output['seg_map'], target)
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

        print('{}{} epoch / Train Loss {}: {:.4f}, lr {:.7f}{}'.format(utils.Colors.LIGHT_CYAN,
                                                                       epoch,
                                                                       self.args.criterion,
                                                                       loss_mean,
                                                                       self.optimizer.param_groups[0]['lr'],
                                                                       utils.Colors.END))

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean}, step=epoch)

    def _train_score(self, epoch):
        self.model.train()
        self.callback.train_score_callback()

        batch_losses = []
        for batch_idx, (x_in, target) in enumerate(self.loader_train_score.Loader):
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            output = self.model(x_in)
            loss = self.criterion_score(output['score'], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.args.ema_decay != 0:
                self.model_ema.update(self.model)

            batch_losses.append(loss.item())

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self._validate_interval) == 0 and not (batch_idx != len(self.loader_train_score) - 1):
                    self._validate_score(self.model, epoch)

        loss_mean = np.mean(batch_losses)
        print('{}{} epoch / Train Loss score {}: {:.5f}, lr {:.5f} {}'.format(utils.Colors.LIGHT_CYAN,
                                                                              epoch,
                                                                              self.args.criterion_score,
                                                                              loss_mean,
                                                                              self.optimizer.param_groups[0]['lr'],
                                                                              utils.Colors.END))

        if self.args.wandb:
            wandb.log({'Train score Loss {}'.format(self.args.criterion_score): loss_mean},
                      step=epoch)

    def _train_unsup(self, epoch):
        self.model.train()

        batch_losses = []
        for batch_idx, (x_in, target) in enumerate(self.loader_train_unsup.Loader):
            x_in, _ = x_in
            x_in = x_in.to(self.device)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            output = self.model(x_in)

            # compute loss
            batch_size = x_in.shape[0]
            perturb_num = output['feat_repr_pos'].shape[0] // batch_size
            loss = torch.zeros(1).to(self.device)
            for i in range(perturb_num):
                loss += self.criterion_unsup(output['feats_repr'], output['feat_repr_pos'][i * batch_size: (i + 1) * batch_size], output['feat_repr_neg'])

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

            if (batch_idx != 0) and (batch_idx % (self.args.log_interval // self.args.batch_size) == 0):
                loss_mean = np.mean(batch_losses)
                print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                                    self.args.criterion_unsup,
                                                                    loss_mean,
                                                                    self.optimizer.param_groups[0]['lr']))

        loss_mean = np.mean(batch_losses)

        print('{}{} epoch / Train Loss {}: {:.4f}, lr {:.7f}{}'.format(utils.Colors.LIGHT_CYAN,
                                                                       epoch,
                                                                       self.args.criterion_unsup,
                                                                       loss_mean,
                                                                       self.optimizer.param_groups[0]['lr'],
                                                                       utils.Colors.END))

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion_unsup): loss_mean}, step=epoch)

    def _validate_segmentation(self, model, epoch):
        model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = model(x_in, is_val=True)

                # compute metric
                output_argmax = torch.argmax(output['seg_map'], dim=1).cpu()
                for b in range(output['seg_map'].shape[0]):
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
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=best_flag, metric_name=key)

        self.metric_val.reset()

    def _validate_score(self, model, epoch):
        model.eval()
        list_score = []
        list_target = []

        for batch_idx, (x_in, target) in enumerate(self.loader_val_score.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)

                output = model(x_in, is_val=True)

                target_item = target.cpu().numpy()
                score_item = output['score'].detach().cpu().numpy()

                for b in range(output['seg_map'].shape[0]):
                    self.metric_val_score.update(score_item[b].round(0), target_item[b].round(0))
                    list_score.append(score_item[b].tolist())
                    list_target.append(target_item[b].tolist())

        # Calculate the correlation between the two lists
        correlation1 = np.corrcoef(np.array(list_score).T[0], np.array(list_target).T[0])[0, 1]
        correlation2 = np.corrcoef(np.array(list_score).T[1], np.array(list_target).T[1])[0, 1]
        correlation = (correlation1 + correlation2) / 2

        print('{}{} epoch / Val score Correlation {:.5f}{} \n'.format(utils.Colors.LIGHT_GREEN,
                                                                      epoch,
                                                                      correlation,
                                                                      utils.Colors.END,))

        if self.args.wandb:
            for i in range(self.args.num_class - 1):    # ignore 0 class
                wandb.log({f'Val score correlation': correlation},
                          step=epoch)

        if epoch == 1:  # initialize value
            if hasattr(self, 'metric_best'):
                self.metric_best['Correl'] = correlation
            else:
                self.metric_best = {'Correl': correlation}
        model_metrics = {'Correl': correlation}
        for key in model_metrics.keys():
            if key == 'MSE' and model_metrics[key] < self.metric_best[key]:
                self.metric_best[key] = model_metrics[key]
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=True, metric_name=key)
            elif key != 'MSE' and model_metrics[key] > self.metric_best[key]:
                self.metric_best[key] = model_metrics[key]
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=True, metric_name=key)

        self.metric_val_score.reset()

    def _train(self, epoch):
        self._train_segmentation(epoch)
        self._train_score(epoch)
        self._train_unsup(epoch)

    def _validate(self, model, epoch):
        if self.args.ema_decay != 0:
            self._validate_segmentation(self.model_ema.module, epoch)
        else:
            self._validate_segmentation(self.model, epoch)
            self._validate_score(self.model, epoch)

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(self.model, epoch)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print(f'{utils.Colors.CYAN}The model seems to be converged. Early stop training.')
                print(f'Best mIoU -----> {self.metric_best["mIoU"]}')
                print(f'Best Correl -----> {self.metric_best["Correl"]}{utils.Colors.END}')
                wandb.log({f'Best Correl': self.metric_best['Correl'],
                           f'Best mIoU': self.metric_best['mIoU']})
                sys.exit()  # safe exit
