import torch
import numpy as np
import wandb

from models import utils
from tools.trainer_base import TrainerBase
from tools import utils_tool


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
            aa = torch.tensor(self.loader_train.image_loader.image_mean).to(self.device)
            bb = torch.tensor(self.loader_train.image_loader.image_std).to(self.device)
            x_img = utils.denormalize_img(x_in, aa, bb).detach().cpu().numpy()
            for i in range(len(x_img)):
                tmp1 = x_img[i]
                tmp2 = target[i].detach().cpu().numpy()
                utils.draw_landmark(tmp1, tmp2, 'tmp', str(i) + '.png')
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
        metric_dict['lr'] = utils_tool.get_learning_rate(self.optimizer)
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

                score_item = output['vec'].detach().squeeze().cpu().numpy().tolist()
                target_item = target.squeeze().cpu().numpy().tolist()

                list_score.extend(score_item)
                list_target.extend(target_item)

        correlation = np.corrcoef(np.array(list_score), np.array(list_target))[0, 1]

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
