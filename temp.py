import os
import torch
import random
import numpy as np
import ptflops
import cv2

from models import losses as loss_hub
from models import model_implements
from models import lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class ModelMini:
    def __init__(self):
        self.device = 'cuda'
        self.x = torch.rand([4, 3, 256, 256]).to(self.device)
        self.y = torch.zeros([4, 256, 256]).to(self.device).long()

        self.model = model_implements.DiNAT_s_T_segMap_score_multi_stem_repr(3, 3).to(self.device)
        self.criterion = loss_hub.CrossEntropy()
        self.criterion2 = loss_hub.InfoNCE()
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, amsgrad=False)

        self.epochs = 2
        self.steps = 5
        self.t_max = 10
        self.cycles = self.epochs / self.t_max
        self.scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer,
                                                           warmup_steps=self.steps * 20,
                                                           t_total=self.epochs * self.steps,
                                                           cycles=self.cycles,
                                                           last_epoch=-1)

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
        for epoch in range(self.epochs):
            for step in range(self.steps):
                out = self.model(self.x, self.y)
                self.criterion2(out['feats_repr'], out['feat_repr_pos'][:4], out['feat_repr_neg'])
                batch_size = self.x.shape[0]
                perturb_num = out['feat_repr_pos'].shape[0] // batch_size
                for i in range(perturb_num):
                    self.criterion2(out['feats_repr'], out['feat_repr_pos'][i * batch_size: (i + 1) * batch_size], out['feat_repr_neg'])
                loss = self.criterion(out['seg_map'], self.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                print(f'{epoch}: {step}')

    def inference(self):
        with torch.no_grad():
            self.model.eval()
            out = self.model(self.x, self.y)
            print(out)

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
        print(f'{mean_syn}ms')

    def measure_computations(self):
        with torch.cuda.device(0):
            macs, params = ptflops.get_model_complexity_info(self.model, tuple(self.x.shape[1:]), as_strings=True,
                                                             print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    def measure_memory_consumption(self):
        self.model.eval()
        with torch.no_grad():
            print('Max Memory Allocated: ', torch.cuda.max_memory_allocated())

    def save_jit(self):
        with torch.no_grad():
            self.model.eval()
            m = torch.jit.script(self.model)
            torch.jit.save(m, 'model.torchscript')


def main():
    seed = 3407
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    ModelMini()


if __name__ == '__main__':
    main()
