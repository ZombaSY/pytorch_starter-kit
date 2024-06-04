import torch
import wandb
import numpy as np
import sys
import einops

from models import utils
from models import metrics
from trainer_base import TrainerBase


class TrainerRgressionPerturb(TrainerBase):
    def __init__(self, args, now=None, k_fold=0):
        super().__init__(args, now=now, k_fold=k_fold)

        self.loader_train = self.init_data_loader(args=self.args,
                                                  mode='train',
                                                  csv_path=self.args.train_csv_path)
        self.loader_val = self.init_data_loader(args=self.args,
                                                mode='inference',
                                                csv_path=self.args.valid_csv_path)

        self.criterion_perturb = self.init_criterion(self.args.criterion_perturb, self.device)

        self.scheduler = self.set_scheduler(self.args, self.optimizer, self.loader_train)
        self._validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold // self.args.batch_size

        if hasattr(self.model.module, 'train_score_callback'):
            self.callback.train_callback = self.model.module.train_score_callback

    def _train(self, epoch):
        self.model.train()
        self.callback.train_callback()

        batch_losses = []
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            x_in, _ = x_in
            target, _ = target

            x_in = x_in.to(self.device)
            target = target.to(self.device)  # shape = (batch_size, img_h, img_w)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            x_in = einops.rearrange(x_in, 'b a c h w -> (b a) c h w')
            target = einops.rearrange(target, 'b a c -> (b a) c')

            out_dict = self.model(x_in)
            out_perturb = self.model(out_dict, is_perturb=True)

            # ----- loss computation ----- #
            loss_perturb = []
            for i in range(len(out_perturb)):
                loss_perturb.append(self.criterion_perturb(out_perturb[i], out_dict['vec'].detach()))
            loss_perturb = sum(loss_perturb)
            loss = self.criterion(out_dict['vec'], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_losses.append(loss.item())

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self._validate_interval) == 0 and not (batch_idx != len(self.loader_train) - 1):
                    self._validate_score(self.model, epoch)

        loss_mean = np.mean(batch_losses)

        metric_dict = {}
        metric_dict['loss'] = loss_mean

        utils.log_epoch('train', epoch, metric_dict, self.args.wandb)
        self.metric_train.reset()

    def _validate(self, model, epoch, dataloader):
        model.eval()

        list_score = []
        list_target = []
        batch_losses = 0
        for batch_idx, (x_in, target) in enumerate(dataloader.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)

                out_dict = model(x_in)

                target_item = target.cpu().numpy()
                score_item = out_dict['vec'].detach().cpu().numpy()

                loss = self.criterion(out_dict['vec'], target)
                batch_losses += loss.item()

                for b in range(out_dict['vec'].shape[0]):
                    self.metric_val.update(score_item[b].round(0), target_item[b].round(0))
                    list_score.append(score_item[b].tolist())
                    list_target.append(target_item[b].tolist())

        # Calculate the correlation between the two lists
        correlation1 = np.corrcoef(np.array(list_score).T[0], np.array(list_target).T[0])[0, 1]
        correlation2 = np.corrcoef(np.array(list_score).T[1], np.array(list_target).T[1])[0, 1]
        correlation = (correlation1 + correlation2) / 2

        # from scipy.stats import spearmanr
        # spearmanr(np.array(list_score).T[0], np.array(list_target).T[0]).statistic

        loss_mean = batch_losses / self.loader_val.Loader.__len__()

        metric_dict = {}
        metric_dict['loss'] = loss_mean
        metric_dict['corr'] = correlation

        utils.log_epoch('validation', epoch, metric_dict, self.args.wandb)
        self.check_metric(epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(self.model, epoch, self.loader_val)

            if (epoch - self.last_saved_epoch) > self.args.early_stop_epoch:
                print(f'{utils.Colors.CYAN}The model seems to be converged. Early stop training.')
                print(f'Best Correl -----> {self.metric_best["Correl"]}{utils.Colors.END}')
                wandb.log({f'Best Correl': self.metric_best['Correl']})
                sys.exit()  # safe exit
