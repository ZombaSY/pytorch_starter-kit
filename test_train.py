import importlib
import os
import random

import cv2
import numpy as np
import ptflops
import torch
from accelerate import Accelerator

from models import lr_scheduler, utils
from tools import utils_tool
from tools.trainer_base import TrainerBase

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ModelMini:
    def __init__(self):
        self.device = "cuda"
        self.accelerator = Accelerator()

        self.x = torch.rand([4, 3, 128, 128]).to(self.device)
        self.y = torch.rand([4, 1, 128, 128]).to(self.device).long()

        # select model configuraiton and load from configs for test
        spec = importlib.util.spec_from_file_location("conf", "configs/train_segmentation.py")
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)
        self.conf = imported_module.conf

        self.model = TrainerBase.init_model(self.conf, self.device)
        self.model.to(self.device)
        self.criterion = TrainerBase.init_criterion(self.conf["criterion"], self.device)
        self.optimizer = TrainerBase.init_optimizer(self.conf["optimizer"], self.model)

        self.epochs = 3
        self.steps = 10
        self.t_max = 20
        self.cycles = self.epochs / self.t_max
        self.scheduler = lr_scheduler.WarmupCosineSchedule(
            optimizer=self.optimizer,
            warmup_steps=self.steps * 20,
            t_total=self.epochs * self.steps,
            cycles=self.cycles,
            last_epoch=-1,
        )

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # torch.save(self.model.get_saved_weight(), 'unet.pt')
        self.train_mini()
        # self.inference()
        # self.measure_computations()
        # self.measure_inference_time()
        # self.measure_memory_consumption()
        # self.save_jit()

    def load_image(self, src):
        img = cv2.imread(src, cv2.IMREAD_COLOR) / 255
        img = cv2.resize(img, (512, 512))
        img = (img - 0.5) / 0.25

        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    def train_mini(self):
        self.model.train()
        for epoch in range(self.epochs):
            for step in range(self.steps):
                out = self.model(self.x)
                loss = self.criterion(out["seg"], self.y)
                self.accelerator.backward(loss)
                self.optimizer.zero_grad()
                self.optimizer.step()
                self.scheduler.step()
                print(loss.item())

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            self.model(self.x)

    def measure_inference_time(self):
        # measure inference time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((self.steps - 1, 1))
        self.model.eval()
        with torch.no_grad():
            for step in range(self.steps):
                starter.record()
                _ = self.model(self.x)
                if step != 0:
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[step - 1] = curr_time
        mean_syn = np.sum(timings) / (self.steps - 1)
        print(f"{mean_syn}ms")

    def measure_computations(self):
        self.model.eval()
        with torch.cuda.device(0):
            macs, params = ptflops.get_model_complexity_info(
                self.model, tuple(self.x.shape[1:]), as_strings=True, print_per_layer_stat=True, verbose=True
            )
            print("{:<30}  {:<8}".format("Computational complexity: ", macs))
            print("{:<30}  {:<8}".format("Number of parameters: ", params))

    def measure_memory_consumption(self):
        self.model.eval()
        with torch.no_grad():
            print("Max Memory Allocated: ", torch.cuda.max_memory_allocated())

    def save_jit(self):
        with torch.no_grad():
            self.model.eval()
            m = torch.jit.script(self.model)
            torch.jit.save(m, "model.torchscript")

    def stress_test(self):
        import time

        conf = {"model": {"name": "SimpleCNN", "base_c": 64, "in_channels": 3, "num_layer": 40}}
        x = torch.rand([4, 3, 128, 128]).to(self.device)
        model = utils_tool.init_model(conf, self.device)
        tt = time.time()
        print(f"{utils.Colors.CYAN}Stress test started...{utils.Colors.END}")
        while time.time() - tt < 600:  # run for 10 minutes
            model(x)
        print(f"{utils.Colors.CYAN}Stress test passed!{utils.Colors.END}")


def main():
    seed = 3407
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    ModelMini()


if __name__ == "__main__":
    main()
