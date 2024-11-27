# import os
# import tensorflow as tf
# import onnx
# import torch
# import time
# import tensorflowjs as tfjs

# from tools.trainer_base import TrainerBase
# from onnxsim import simplify
# from onnx_tf.backend import prepare


class Exportor:

    def __init__(self, conf):
        self.start_time = time.time()
        self.conf = conf

        use_cuda = self.conf['env']['cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = TrainerBase.init_model(self.conf, self.device).module
        self.model.load_state_dict(torch.load(self.conf['model']['saved_ckpt']))
        self.model.eval()

        from models.backbones.MobileOne import reparameterize_model
        self.model.backbone = reparameterize_model(self.model.backbone)

        dir_path, fn = os.path.split(self.conf['model']['saved_ckpt'])
        fn, ext = os.path.splitext(fn)
        self.fn = fn
        self.save_dir = os.path.join(dir_path, fn)

    def torch2onnx(self):
        dummy_input = torch.autograd.Variable(torch.randn(self.conf['export']['input_shape'])).to(self.device)
        torch.onnx.export(self.model, dummy_input, self.save_dir + '.onnx', opset_version=self.conf['export']['opset_version'])

    def torch2torchscript(self):
        with torch.no_grad():
            self.model.eval()
            m = torch.jit.script(self.model)
            torch.jit.save(m, self.save_dir + '.torchscript')

    def onnx_simplify(self):
        onnx_model = onnx.load(self.save_dir + '.onnx')
        onnx_model, check = simplify(onnx_model)
        onnx.save(onnx_model, self.save_dir + '.onnx')
        print('onnx simplified!')

    def onnx2pb(self):
        onnx_model = onnx.load(self.save_dir + '.onnx')

        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"

        # ==================== if raise KeyError, unannotate below ==================== #
        # https://stackoverflow.com/questions/76839366/tf-rep-export-graphtf-model-path-keyerror-input-1
        name_map = {"input.1": "input_1"}
        new_inputs = []

        # Iterate over the inputs and change their names if needed
        for inp in onnx_model.graph.input:
            if inp.name in name_map:
                # Create a new ValueInfoProto with the new name
                new_inp = onnx.helper.make_tensor_value_info(name_map[inp.name],
                                                            inp.type.tensor_type.elem_type,
                                                            [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
                new_inputs.append(new_inp)
            else:
                new_inputs.append(inp)

        # Clear the old inputs and add the new ones
        onnx_model.graph.ClearField("input")
        onnx_model.graph.input.extend(new_inputs)

        # Go through all nodes in the model and replace the old input name with the new one
        for node in onnx_model.graph.node:
            for i, input_name in enumerate(node.input):
                if input_name in name_map:
                    node.input[i] = name_map[input_name]

        # ============================================================================= #

        # Save the renamed ONNX model
        onnx.save(onnx_model, self.save_dir + '.onnx')
        onnx_model = onnx.load(self.save_dir + '.onnx')

        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(self.save_dir)

    def pb2tflite(self):
        # float32 settings
        converter = tf.lite.TFLiteConverter.from_saved_model(self.save_dir, signature_keys=['serving_default'])
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(self.save_dir + '-float32-NCHW.tflite', 'wb') as f_w:
            f_w.write(tflite_model)

        # float16 settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(self.save_dir + '-float16-NCHW.tflite', 'wb') as f_w:
            f_w.write(tflite_model)

        # dynamic-int8 settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = set()   # set default for dynamic-quantization
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(self.save_dir + '-dynamic_int8-NCHW.tflite', 'wb') as f_w:
            f_w.write(tflite_model)

    def pb2tfjs(self):
        tfjs.converters.convert_tf_saved_model(self.save_dir, os.path.join(self.save_dir, 'tfjs'))


    def export(self):
        self.torch2onnx()
        self.onnx_simplify()

        if self.conf['export']['target'] == 'tflite':
            self.onnx2pb()
            self.pb2tflite()

        if self.conf['export']['target'] == 'tfjs':
            self.onnx2pb()
            self.pb2tfjs()
