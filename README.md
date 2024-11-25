# Pytorch starter-kit for beginners


## Environments

- OS: Ubuntu 22.040. LTS
- GPU: Tesla H100 80GB
- GPU Driver version: 550.54.14
- CUDA: 12.4

## Dataset
The dataset should be listed in the `[train].csv` and `[valid].csv` with each columns `[input_paths]` and `[label_paths]`.<br>
`[input_paths]` and `[label_paths]` can differ from tasks.

```
# [train].csv
input_paths             | label_paths
/path/to/input01.jpg    | /path/to/label01.jpg
/path/to/input03.jpg    | /path/to/label03.jpg
/path/to/input04.jpg    | /path/to/label04.jpg
...                     | ...
/path/to/input99.jpg    | /path/to/label99.jpg
```
```
# [valid].csv
input_paths             | label_paths
/path/to/input02.jpg    | /path/to/label02.jpg
/path/to/input05.jpg    | /path/to/label05.jpg
...                     | ...
/path/to/input98.jpg    | /path/to/label98.jpg
```

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
