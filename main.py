import torch
import os
import argparse
import yaml
import numpy as np
import random
import ast

from train_segmentation import TrainerSegmentation
from train_segmentation_score_repr import Trainer_segScoreRepr
from inference import Inferencer
from torch.cuda import is_available
from datetime import datetime


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


def conf_to_args(args, **kwargs):    # pass in variable numbers of args
    var = vars(args)

    for key, value in kwargs.items():
        var[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    arg, unknown_arg = parser.parse_known_args()

    if arg.config_path is not None:
        with open(arg.config_path, 'rb') as f:
            conf = yaml.load(f.read(), Loader=yaml.Loader)  # load the config file
            conf['config_path'] = arg.config_path
    else:
        # make unrecognized args to dict
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

    args = argparse.Namespace()
    conf_to_args(args, **conf)  # pass in keyword args

    now_time = datetime.now().strftime("%Y-%m-%d %H%M%S")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    print('Use CUDA :', args.cuda and is_available())

    if args.mode == 'train':
        if args.task == 'segmentation':
            trainer = TrainerSegmentation(args, now_time)
        elif args.task == 'segmentation_score_cycle_repr':
            args.task = 'segmentation'
            trainer = Trainer_segScoreRepr(args, now_time)
        else:
            raise ValueError('')

        trainer.run()

    elif args.mode in 'inference':

        if args.task == 'segmentation':
            inferencer = Inferencer(args)
        elif args.task == 'segmentation_score_cycle':
            inferencer = Inferencer(args)
        else:
            raise ValueError('Please select correct inference_mode !!!')

        inferencer.inference()
    else:
        print('No mode supported.')


if __name__ == "__main__":
    main()
