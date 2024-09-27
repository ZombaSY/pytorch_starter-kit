import torch
import wandb
import einops

from models import utils
from tools.train_regression import TrainerRegression


class TrainerRegressionSSL(TrainerRegression):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerRegressionSSL, self).__init__(conf, now=now, k_fold=k_fold)

        self.loader_train_ssl = self.init_data_loader(conf=self.conf,
                                                      conf_dataloader=self.conf['dataloader_ssl'])
        self.criterion_ssl = self.init_criterion(self.conf['criterion_ssl'], self.device)
        self.optimizer_ssl = self.init_optimizer(self.conf['optimizer_ssl'], self.model)

    def _train_ssl(self, epoch):
        self.model.train()

        batch_losses = 0
        for iteration, data in enumerate(self.loader_train_ssl.Loader):
            x_in = data['input']
            x_in_perturb = data['input_perturb']

            x_in = x_in.to(self.device)
            x_in_perturb = x_in_perturb.to(self.device)
            x_in_perturb = einops.rearrange(x_in_perturb, 'b p c w h -> p b c w h')

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():
                break   # avoid BN issue

            output = self.model(x_in)
            output_perturb = []
            for perturb in x_in_perturb:
                output_perturb.append(self.model(perturb))

            # compute loss
            loss = 0
            for perturb in output_perturb:
                # apply adaptive pooling on bottle-neck feature
                output_feat = torch._adaptive_avg_pool2d(output['feat'], 1)
                perturb_feat = torch._adaptive_avg_pool2d(perturb['feat'], 1)
                loss += self.criterion_ssl(output_feat, perturb_feat)
            loss = loss * self.conf['criterion_ssl']['weight_lambda'] / len(output_perturb)

            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            # ----- backward ----- #
            self.optimizer_ssl.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer_ssl.step()

            batch_losses += loss.item()

        loss_mean = batch_losses / self.loader_train.Loader.__len__()

        metric_dict = {}
        metric_dict['loss_ssl'] = loss_mean

        utils.log_epoch('train', epoch, metric_dict, self.conf['env']['wandb'])
        self.metric_train.reset()

    def run(self):
        for epoch in range(1, self.conf['env']['epoch'] + 1):
            self._train(epoch)
            self._validate(epoch)
            if epoch >= self.conf['dataloader_ssl']['train_warmup_epoch']:
                self._train_ssl(epoch)
                self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.conf['env']['early_stop_epoch' ]:
                utils.Logger().info('The model seems to be converged. Early stop training.')
                utils.Logger().info(f'Best mIoU -----> {self.metric_best["mIoU"]}')
                if self.conf['env']['wandb']:
                    wandb.log({f'Best mIoU': self.metric_best['mIoU']})
                break
