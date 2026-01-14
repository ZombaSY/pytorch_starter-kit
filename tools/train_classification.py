import torch
import wandb

from models import utils
from tools.trainer_base import TrainerBase


class TrainerClassification(TrainerBase):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerClassification, self).__init__(conf, now=now, k_fold=k_fold)

    def _train(self, epoch):
        self.model.train()
        self.callback.on_train_start()

        batch_losses = 0
        for iteration, data in enumerate(self.loader_train.Loader):
            x_in = data['input']
            target = data['label']

            x_in = x_in.to(self.device)
            target = target.to(self.device)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():
                break   # avoid BN issue

            output = self.model(x_in)

            # compute loss
            loss = self.criterion(output['vec'], target)
            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            # ----- backward ----- #
            self._backward_and_update_weight(self.optimizer, loss, scheduler=self.scheduler)

            batch_losses += loss.item()

            if (iteration + 1) % self._validate_interval == 0:
                self._validate(self.model, epoch)
                if self.model_ema is not None:
                    self._validate(self.model_ema.module, epoch, log_prefix='[EMA]')

            # compute metric
            output_argmax = torch.argmax(output['vec'], dim=1).detach().cpu().numpy()
            target_argmax = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy()
            self.metric_train.update(output_argmax, target_argmax)

        metric_result = self.metric_train.get_results()

        loss_mean = batch_losses / self.loader_train.Loader.__len__()

        metric_dict = {}
        metric_dict['acc'] = metric_result['acc']
        metric_dict['f1'] = metric_result['f1']
        metric_dict['loss'] = loss_mean

        utils.log_epoch('train', epoch, metric_dict, self.conf['env']['wandb'])
        self.metric_train.reset()

    def _validate(self, model, epoch, log_prefix=''):
        model.eval()

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data['input']
                target = data['label']

                x_in = x_in.to(self.device)
                target = target.to(self.device)

                output = model(x_in)

                # compute metric
                output_argmax = torch.argmax(output['vec'], dim=1).detach().cpu().numpy()
                target_argmax = torch.argmax(target.squeeze() if (self.conf['dataloader']['batch_size'] // 4) != 1 else target, dim=1).detach().cpu().numpy()
                self.metric_val.update(output_argmax, target_argmax)

        metric_result = self.metric_val.get_results()

        metric_dict = {}
        metric_dict[log_prefix + 'acc'] = metric_result['acc']
        metric_dict[log_prefix + 'f1'] = metric_result['f1']

        utils.log_epoch('validation', epoch, metric_dict, self.conf['env']['wandb'], prefix=log_prefix)
        self.evaluate_model(model, epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.conf['env']['epoch'] + 1):
            self._train(epoch)
            self._validate(self.model, epoch)
            if self.model_ema is not None:
                self._validate(self.model_ema.module, epoch, log_prefix='[EMA]')

            if (epoch - self.last_saved_epoch) > self.conf['env']['early_stop_epoch']:
                self.end_train()
                break
