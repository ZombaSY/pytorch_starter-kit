import argparse
import ast
import importlib
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from models.utils import SEED
from tools.export import Exportor
from tools.inference import Inferencer
from tools.train_classification import TrainerClassification
from tools.train_regression import TrainerRegression
from tools.train_regression_ssl import TrainerRegressionSSL
from tools.train_segmentation import TrainerSegmentation
from tools.train_segmentation_ssl import TrainerSegmentationSSL

# fix seed for reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)  # raise error if CUDA >= 10.2
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def init_trainer(conf, now_time, k_fold=0):
    if conf["env"]["task"] == "segmentation":
        trainer = TrainerSegmentation(conf, now_time, k_fold=k_fold)
    elif conf["env"]["task"] == "segmentation-ssl":
        trainer = TrainerSegmentationSSL(conf, now_time, k_fold=k_fold)
    elif conf["env"]["task"] == "classification":
        trainer = TrainerClassification(conf, now_time, k_fold=k_fold)
    elif conf["env"]["task"] in ["regression", "landmark"]:
        trainer = TrainerRegression(conf, now_time, k_fold=k_fold)
    elif conf["env"]["task"] in ["regression-ssl", "landmark-ssl"]:
        trainer = TrainerRegressionSSL(conf, now_time, k_fold=k_fold)
    else:
        raise ValueError("No trainer found.")

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    arg, unknown_arg = parser.parse_known_args()

    conf = {}
    if arg.config_path is not None:
        # Create a module spec using the directory path
        spec = importlib.util.spec_from_file_location("conf", arg.config_path)
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)

        conf = imported_module.conf
        conf["config_path"] = arg.config_path
    else:
        # for sweeper case in WandB
        conf = {"config_path": "configs/sweep_config.yaml"}
        for item in unknown_arg:
            item = item.strip("--")
            key, value = item.split("=")
            if key != "CUDA_VISIBLE_DEVICES":
                try:
                    if value == "true" or value == "false":
                        value = value.title()
                    value = ast.literal_eval(value)
                except ValueError:
                    if value.isalpha():
                        pass
                except SyntaxError as e:
                    if "/" in value:
                        pass
                    else:
                        raise e
            conf[key] = value

    now_time = datetime.now().strftime("%Y-%m-%d %H%M%S")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["env"]["CUDA_VISIBLE_DEVICES"]

    if conf["env"]["debug"]:
        conf["env"]["wandb"] = False
        for key in conf:
            if "dataloader" in key:
                conf[key]["data_cache"] = False
        # conf.transform_mixup = 0

    if conf["env"]["mode"] == "train":
        # check K-folds
        if "data_path_folds" in conf["dataloader_train"].keys():
            k_fold = len(conf["dataloader_train"]["data_path_folds"])

            for idx in range(k_fold):
                conf["dataloader_train"]["data_path"] = conf["dataloader_train"]["data_path_folds"][idx]
                conf["dataloader_valid"]["data_path"] = conf["dataloader_valid"]["data_path_folds"][idx]

                trainer = init_trainer(conf, now_time, k_fold=idx)
                trainer.run()
                del trainer  # release memory

        else:
            trainer = init_trainer(conf, now_time)
            trainer.run()

    elif conf["env"]["mode"] in ["valid", "test"]:
        inferencer = Inferencer(conf)
        inferencer.inference()

    elif conf["env"]["mode"] == "export":
        exportor = Exportor(conf)

        if conf["export"]["target"] == "onnx":
            exportor.export()
        else:
            raise ValueError(f"unsupported target: {conf['export']['target']}")

    else:
        raise ValueError(f"unsupported mode: {conf['env']['mode']}")

    sys.exit()


if __name__ == "__main__":
    main()
