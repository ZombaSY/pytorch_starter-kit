import torch
import time
import os

from trainer_base import TrainerBase


class Exportor:
    def __init__(self, conf):
        self.start_time = time.time()
        self.conf = conf

        use_cuda = self.conf['env']['cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = TrainerBase.init_model(self.conf, self.device).module
        self.model.load_state_dict(torch.load(self.conf['model']['saved_ckpt']))
        self.model.eval()

        dir_path, fn = os.path.split(self.conf['model']['saved_ckpt'])
        fn, ext = os.path.splitext(fn)
        self.save_dir = os.path.join(dir_path, fn)


    def torch2onnx(self):
        dummy_input = torch.autograd.Variable(torch.randn(self.conf['export']['input_shape'])).to(self.device)
        torch.onnx.export(self.model, dummy_input, self.save_dir + '.onnx', opset_version=self.conf['export']['opset_version'])


    def export(self):
        self.torch2onnx()
