import numpy as np
import torch

from models import utils
from tools.trainer_base import TrainerBase


class TrainerRegression(TrainerBase):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerRegression, self).__init__(conf, now=now, k_fold=k_fold)

    def _train(self, epoch):
        self.model.train()
        self.callback.on_train_start()

        batch_losses = 0

        for iteration, data in enumerate(self.loader_train.Loader):
            x_in = data["input"]
            target = data["label"]

            x_in = x_in.to(self.device)
            target = target.to(self.device)

            if (x_in.shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():
                break  # avoid BN issue

            output = self.model(x_in)

            # compute loss
            loss = self.criterion(output["vec"], target)
            if not torch.isfinite(loss):
                raise Exception("Loss is NAN. End training.")

            # ----- backward ----- #
            self._backward_and_update_weight(self.optimizer, loss, scheduler=self.scheduler)

            batch_losses += loss.item()

            if (iteration + 1) % self._validate_interval == 0:
                self._validate(self.model, epoch)
                if self.model_ema is not None:
                    self._validate(self.model_ema.module, epoch, log_prefix="[EMA]")

        loss_mean = batch_losses / self.loader_train.Loader.__len__()

        metric_dict = {}
        metric_dict["loss"] = loss_mean

        utils.log_epoch("train", epoch, metric_dict, self.conf["env"]["wandb"])
        self.metric_train.reset()

    def _validate(self, model, epoch, log_prefix=""):
        model.eval()

        list_score = []
        list_target = []
        batch_losses = 0
        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data["input"]
                target = data["label"]

                x_in = x_in.to(self.device)
                target = target.to(self.device)

                output = model(x_in)

                # compute loss
                loss = self.criterion(output["vec"], target)
                if not torch.isfinite(loss):
                    raise Exception("Loss is NAN. End training.")

                batch_losses += loss.item()

                target_item = target.cpu().numpy()
                score_item = output["vec"].detach().cpu().numpy()
                for b in range(output["vec"].shape[0]):
                    list_score.append(score_item[b].tolist())
                    list_target.append(target_item[b].tolist())

        # Calculate the correlation between the two lists. TODO: Modify torch-based code later
        correlation1 = np.corrcoef(np.array(list_score).T[0], np.array(list_target).T[0])[0, 1]
        correlation2 = np.corrcoef(np.array(list_score).T[1], np.array(list_target).T[1])[0, 1]
        correlation = (correlation1 + correlation2) / 2

        loss_mean = batch_losses / self.loader_valid.Loader.__len__()

        metric_dict = {}
        metric_dict[log_prefix + "loss"] = loss_mean
        metric_dict[log_prefix + "corr"] = correlation

        utils.log_epoch("validation", epoch, metric_dict, self.conf["env"]["wandb"], prefix=log_prefix)
        self.evaluate_model(model, epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.conf["env"]["epoch"] + 1):
            # suspects = ["segmentator.module.u_net.inc.0.weight", "segmentator.module.u_net.outc.conv.weight"]
            # suspects = ["segmentator.module.swin_transformer.patch_embed.proj.weight", "segmentator.module.uper_head.fpn_bottleneck.weight"]
            # before = utils.snapshot(self.model.module, suspects)

            self._train(epoch)
            self._validate(self.model, epoch)
            if self.model_ema is not None:
                self._validate(self.model_ema.module, epoch, log_prefix="[EMA]")

            # after = utils.snapshot(self.model.module, suspects)
            # for n in suspects:
            #     utils.Logger().info(f'{n} {torch.allclose(before[n], after[n]), (before[n] - after[n]).abs().max().item()}')

            if (epoch - self.last_saved_epoch) > self.conf["env"]["early_stop_epoch"]:
                self.end_train()
                break
