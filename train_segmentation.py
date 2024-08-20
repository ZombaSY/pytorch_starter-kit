import torch
import wandb

from models import utils
from trainer_base import TrainerBase


class TrainerSegmentation(TrainerBase):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerSegmentation, self).__init__(conf, now=now, k_fold=k_fold)

    def _train(self, epoch):
        self.model.train()

        batch_losses = 0
        for iteration, data in enumerate(self.loader_train.Loader):
            x_in = data['input']
            target = data['label']

            x_in = x_in.to(self.device)
            target = target.long().to(self.device)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():
                break   # avoid BN issue

            output = self.model(x_in)

            # compute loss
            loss = self.criterion(output['seg'], target)
            if 'seg_aux' in output:
                loss_aux = 0
                for aux in output['seg_aux']:
                    loss_aux += self.criterion(aux, target)
                loss_aux /= len(output['seg_aux'])
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

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data['input']
                target = data['label']

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)

                output = self.model(x_in)

                # compute metric
                output_argmax = torch.argmax(output['seg'], dim=1).cpu()
                for b in range(output['seg'].shape[0]):
                    self.metric_val.update(target[b][0].cpu().detach().numpy(), output_argmax[b].cpu().detach().numpy())


        metrics_out = self.metric_val.get_results()
        c_iou = [metrics_out['Class IoU'][i] for i in range(self.conf['model']['num_class'])]
        m_iou = sum(c_iou) / self.conf['model']['num_class']

        metric_dict = {}
        metric_dict['mIoU'] = m_iou
        for i in range(len(c_iou)):
            metric_dict[f'cIoU_{i}'] = c_iou[i]

        utils.log_epoch('validation', epoch, metric_dict, self.conf['env']['wandb'])
        self.check_metric(epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.conf['env']['epoch'] + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.conf['env']['early_stop_epoch']:
                utils.Logger().info('The model seems to be converged. Early stop training.')
                utils.Logger().info(f'Best mIoU -----> {self.metric_best["mIoU"]}')
                if self.conf['env']['wandb']:
                    wandb.log({f'Best mIoU': self.metric_best['mIoU']})
                break
