import torch
import numpy as np
import wandb

from models import utils
from trainer_base import TrainerBase


class TrainerRegression(TrainerBase):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerRegression, self).__init__(conf, now=now, k_fold=k_fold)

    def _train(self, epoch):
        self.model.train()

        batch_losses = 0
        for iteration, data in enumerate(self.loader_train.Loader):
            x_in = data['input']
            target = data['label']

            x_in = x_in.to(self.device)
            target = target.to(self.device)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():
                break   # avoid BN issue

            output = self.model(x_in)

            '''
            For visualize input data

            image_mean = torch.tensor(self.loader_train.image_loader.image_mean).to(self.device)
            image_std = torch.tensor(self.loader_train.image_loader.image_std).to(self.device)
            x_raw = utils.denormalize_img(x_in, image_mean, image_std).detach().cpu().numpy()
            for i in range(len(x_raw)):
                tmp1 = x_raw[i]
                tmp2 = target[i].detach().cpu().numpy()
                utils.draw_landmark(tmp1, tmp2, 'input-batch', str(i).zfill(3) + '.png')
            '''

            # compute loss
            loss = self.criterion(output['vec'], target)
            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            # ----- backward ----- #
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_losses += loss.item()

            if (iteration + 1) % self._validate_interval == 0:
                self._validate(epoch)

        loss_mean = batch_losses / self.loader_train.Loader.__len__()

        metric_dict = {}
        metric_dict['loss'] = loss_mean

        utils.log_epoch('train', epoch, metric_dict, self.conf['env']['wandb'])
        self.metric_train.reset()

    def _validate(self, epoch):
        self.model.eval()

        list_score = []
        list_target = []
        batch_losses = 0
        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data['input']
                target = data['label']

                x_in = x_in.to(self.device)
                target = target.to(self.device)

                output = self.model(x_in)

                # compute loss
                loss = self.criterion(output['vec'], target)
                if not torch.isfinite(loss):
                    raise Exception('Loss is NAN. End training.')

                batch_losses += loss.item()

                target_item = target.cpu().numpy()
                score_item = output['vec'].detach().cpu().numpy()
                for b in range(output['vec'].shape[0]):
                    list_score.append(score_item[b].tolist())
                    list_target.append(target_item[b].tolist())

        # Calculate the correlation between the two lists
        correlation1 = np.corrcoef(np.array(list_score).T[0], np.array(list_target).T[0])[0, 1]
        correlation2 = np.corrcoef(np.array(list_score).T[1], np.array(list_target).T[1])[0, 1]
        correlation = (correlation1 + correlation2) / 2

        loss_mean = batch_losses / self.loader_valid.Loader.__len__()

        metric_dict = {}
        metric_dict['loss'] = loss_mean
        metric_dict['corr'] = correlation

        utils.log_epoch('validation', epoch, metric_dict, self.conf['env']['wandb'])
        self.check_metric(epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.conf['env']['epoch'] + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.conf['env']['early_stop_epoch']:
                utils.Logger().info('The model seems to be converged. Early stop training.')
                utils.Logger().info(f'Best loss -----> {self.metric_best["loss"]}')
                if self.conf['env']['wandb']:
                    wandb.log({f'Best loss': self.metric_best['loss']})
                break
