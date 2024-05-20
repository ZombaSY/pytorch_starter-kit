import torch
import time
import os

from trainer_base import TrainerBase


class Exportor:

    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = TrainerBase.init_model(self.args, self.device).module
        self.model.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        self.save_dir = os.path.join(dir_path, fn)


    def torch2onnx(self):
        dummy_input = torch.autograd.Variable(torch.randn(self.args.input_shape)).to(self.device)
        torch.onnx.export(self.model, dummy_input, self.save_dir + '.onnx', opset_version=self.args.opset_version)


    def export(self):
        self.torch2onnx()
