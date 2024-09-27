# Pytorch starter-kit poject for newers


## Environments

- OS: Ubuntu 22.040. LTS
- GPU: Tesla H100 80GB
- GPU Driver version: 550.54.14
- CUDA: 12.4

## Dataset
The dataset should be listed in the `[train].csv` with each columns `[input_paths]` and `[label_path]`.

## Configuration
All the configurations of train, inference, and export can be easily modified with "configs/[task_name].py"<br>
For example, to run "segmentation", edit "configs/train_segmentation.py" script and excute directly in command line:
```
python3 main.py --config_path "configs/train_segmentation.py"
```


## Train
For <b>train</b>, modify `bash_train.sh` with your task belonging to your configs, or execute directly in command line:
```
# for classification
python3 main.py --config_path "configs/train_classification.py"

# for regression
python3 main.py --config_path "configs/train_regression.py"

# for segmentation
python3 main.py --config_path "configs/train_segmentation.py"

# for landmark detection
python3 main.py --config_path "configs/train_landmark.py"

# for segmentation semi-supervised learning
python3 main.py --config_path "configs/train_segmentation-ssl.py"
```


## Inference
For <b>inference</b>, execute directly in command line:
```
bash bash_inference.sh
```


## Export
For <b>train</b>, execute directly in command line:
```
bash bash_export.sh
```
