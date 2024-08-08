import torch
import os
import argparse
import numpy as np
import random
import ast
import sys
import importlib

from export import Exportor
from train_segmentation import TrainerSegmentation
from train_regression import TrainerRegression
from train_classification import TrainerClassification
from inference import Inferencer
from datetime import datetime
from models import utils


# fix seed for reproducibility
seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)  # raise error if CUDA >= 10.2
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def init_trainer(conf, now_time, k_fold=0):
    if conf['env']['task'] == 'segmentation':
        trainer = TrainerSegmentation(conf, now_time, k_fold=k_fold)
    elif conf['env']['task'] == 'classification':
        trainer = TrainerClassification(conf, now_time, k_fold=k_fold)
    elif conf['env']['task'] == 'regression':
        trainer = TrainerRegression(conf, now_time, k_fold=k_fold)
    else:
        raise ValueError('No trainer found.')

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    arg, unknown_arg = parser.parse_known_args()

    conf = {}
    if arg.config_path is not None:
        # Create a module spec using the directory path
        spec = importlib.util.spec_from_file_location("conf", arg.config_path)
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)

        conf = imported_module.conf
        conf['config_path'] = arg.config_path
    else:
        # for sweeper case in WandB
        conf = {'config_path': 'configs/sweep_config.yaml'}
        for item in unknown_arg:
            item = item.strip('--')
            key, value = item.split('=')
            if key != 'CUDA_VISIBLE_DEVICES':
                try:
                    if value == 'true' or value == 'false':
                        value = value.title()
                    value = ast.literal_eval(value)
                except ValueError:
                    if value.isalpha(): pass
                except SyntaxError as e:
                    if '/' in value: pass
                    else: raise e
            conf[key] = value

    now_time = datetime.now().strftime("%Y-%m-%d %H%M%S")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf['env']['CUDA_VISIBLE_DEVICES']

    print('Use CUDA :', conf['env']['cuda'] and torch.cuda.is_available())

    if conf['env']['debug']:
        conf['env']['wandb'] = False
        for key in conf:
            if 'dataloader' in key:
                conf[key]['data_cache'] = False

    if conf['env']['mode'] == 'train':
        # check K-folds
        if 'data_path_folds' in conf['dataloader_train'].keys():
            k_fold = len(conf['dataloader_train']['data_path_folds'])

            for idx in range(k_fold):
                print(f'{utils.Colors.BOLD}Running {idx}th fold...{utils.Colors.END}')
                conf['dataloader_train']['data_path'] = conf['dataloader_train']['data_path_folds'][idx]
                conf['dataloader_valid']['data_path'] = conf['dataloader_valid']['data_path_folds'][idx]

                trainer = init_trainer(conf, now_time, k_fold=idx)
                trainer.run()
                del trainer # release memory

        else:
            trainer = init_trainer(conf, now_time)
            trainer.run()

    elif conf['env']['mode'] in ['valid', 'test']:
        inferencer = Inferencer(conf)
        inferencer.inference()

    elif conf['env']['mode'] == 'export':
        exportor = Exportor(conf)

        if conf['export']['target'] == 'onnx':
            exportor.torch2onnx()
        else:
            raise ValueError(f"unsupported target: {conf['export']['target']}")

    else:
        print('No mode supported.')

    sys.exit()  # safe exit


if __name__ == "__main__":
    main()
