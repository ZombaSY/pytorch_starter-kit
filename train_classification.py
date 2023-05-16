import torch
import wandb
import numpy as np
import sys

from models import utils
from trainer_base import TrainerBase


class TrainerClassification(TrainerBase):
    def __init__(self, args, now=None):
        super(TrainerClassification, self).__init__(args, now=now)

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available to being set, not must.
        self.loader_train = self.init_data_loader(batch_size=self.args.batch_size,
                                                  mode='train',
                                                  dataloader_name=self.args.dataloader,
                                                  csv_path=self.args.train_csv_path)
        self.loader_val = self.init_data_loader(batch_size=1,
                                                mode='validation',
                                                dataloader_name=self.args.dataloader,
                                                csv_path=self.args.val_csv_path)

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
            target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            score, _ = self.model(x_in)
            loss = self.criterion(score, target)

            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            for b in range(score.shape[0]):
                self.metric_train.update(score[b].detach().cpu().numpy().round(0), target[b].cpu().numpy().round(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_losses.append(loss.item())

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self._validate_interval) == 0 and not (batch_idx != len(self.loader_train) - 1):
                    self._validate(self.model, epoch)

        loss_mean = np.mean(batch_losses)
        print('{}{} epoch / Train Loss {}: {:.5f}, lr {:.5f} {}'.format(utils.Colors.LIGHT_CYAN,
                                                                        epoch,
                                                                        self.args.criterion,
                                                                        loss_mean,
                                                                        self.optimizer.param_groups[0]['lr'],
                                                                        utils.Colors.END))

        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean},
                      step=epoch)

        self.metric_train.reset()

    def _validate(self, model, epoch):
        model.eval()
        batch_losses = []
        list_score = []
        list_target = []

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

                score, _ = self.model(x_in)
                loss = self.criterion(score, target)
                batch_losses.append(loss.item())

                target_item = target.cpu().numpy()
                score_item = score.detach().cpu().numpy()

                for b in range(score.shape[0]):
                    self.metric_val.update(score_item[b].round(0), target_item[b].round(0))
                    list_score.append(score_item[b].tolist())
                    list_target.append(target_item[b].tolist())

        # Calculate the correlation between the two lists
        correlation1 = np.corrcoef(np.array(list_score).T[0], np.array(list_target).T[0])[0, 1]
        correlation2 = np.corrcoef(np.array(list_score).T[1], np.array(list_target).T[1])[0, 1]
        correlation = (correlation1 + correlation2) / 2

        loss_mean = np.mean(batch_losses)
        print('{}{} epoch / Val Loss {}: {:.5f} Correlation {:.5f}{} \n'.format(utils.Colors.LIGHT_GREEN,
                                                                                epoch,
                                                                                self.args.criterion,
                                                                                loss_mean,
                                                                                correlation,
                                                                                utils.Colors.END,))

        if self.args.wandb:
            wandb.log({'Val Loss {}'.format(self.args.criterion): loss_mean},
                      step=epoch)
            for i in range(self.args.num_class - 1):    # ignore 0 class
                wandb.log({f'Val correlation': correlation},
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
