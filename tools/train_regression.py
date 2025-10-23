import torch
import wandb

from models import utils
from tools.trainer_base import TrainerBase


class TrainerRegression(TrainerBase):
    def __init__(self, conf, now=None, k_fold=0):
        super(TrainerRegression, self).__init__(conf, now=now, k_fold=k_fold)

    def _train(self, epoch):
        self.callback.on_train_start()
        self.model.train()

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

        loss_mean = batch_losses / self.loader_valid.Loader.__len__()

        metric_dict = {}
        metric_dict[log_prefix + "loss"] = loss_mean

        utils.log_epoch("validation", epoch, metric_dict, self.conf["env"]["wandb"], prefix=log_prefix)
        self.evaluate_model(model, epoch, metric_dict)
        self.metric_val.reset()

    def run(self):
        for epoch in range(1, self.conf["env"]["epoch"] + 1):
            self._train(epoch)
            self._validate(self.model, epoch)
            if self.model_ema is not None:
                self._validate(self.model_ema.module, epoch, log_prefix="[EMA]")

            if (epoch - self.last_saved_epoch) > self.conf["env"]["early_stop_epoch"]:
                utils.Logger().info("The model seems to be converged. Early stop training.")
                utils.Logger().info(f'Best loss -----> {self.metric_best["loss"]}')
                if self.conf["env"]["wandb"]:
                    wandb.log({"Best loss": self.metric_best["loss"]})
                break
