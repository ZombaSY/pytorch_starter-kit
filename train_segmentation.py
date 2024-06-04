import torch
import wandb

from models import utils
from trainer_base import TrainerBase


class TrainerSegmentation(TrainerBase):
    def __init__(self, args, now=None, k_fold=0):
        super(TrainerSegmentation, self).__init__(args, now=now, k_fold=k_fold)

        self.loader_train = self.init_data_loader(args=self.args,
                                                  mode='train',
                                                  csv_path=self.args.train_csv_path)
        self.loader_val = self.init_data_loader(args=self.args,
                                                mode='inference',
                                                csv_path=self.args.valid_csv_path)

        self.scheduler = self.set_scheduler(self.args, self.optimizer, self.loader_train)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold // self.args.batch_size) == 0 else self.loader_train.__len__() // self.args.train_fold // self.args.batch_size

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
            if 'seg_dsv' in output:
                for aux in output['seg_dsv']:
                    loss += self.criterion(aux, target)
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

        metric_dict = {}
        metric_dict['loss'] = loss_mean

        utils.log_epoch('train', epoch, metric_dict, self.args.wandb)
        self.metric_train.reset()

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

        metric_dict = {}
        metric_dict['mIoU'] = m_iou
        for i in range(len(c_iou)):
            metric_dict[f'cIoU_{i}'] = c_iou[i]

        utils.log_epoch('validation', epoch, metric_dict, self.args.wandb)
        self.check_metric(epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(epoch)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print('The model seems to be converged. Early stop training.')
                print(f'Best mIoU -----> {self.metric_best["mIoU"]}')
                if self.args.wandb:
                    wandb.log({f'Best mIoU': self.metric_best['mIoU']})
                break
