import torch
import time
import os
import onnx

from trainer_base import TrainerBase
from onnxsim import simplify


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

    def onnx_simplify(self):
        onnx_model = onnx.load(self.save_dir + '.onnx')
        onnx_model, check = simplify(onnx_model)
        onnx.save(onnx_model, self.save_dir + '.onnx')
        print('onnx simplified!')

    def export(self):
        # execution squence
        self.torch2onnx()
        self.onnx_simplify()
