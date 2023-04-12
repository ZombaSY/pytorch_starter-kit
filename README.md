# Pytorch Start-kit Project
There are so enormous pytorch environments in research and projects. <br/>
To minimize your efforts on setting own scripts, this repository introduces the skeletons for pytorch-beginners.


## Dataset
The dataset recommend following hierarchy

```
[root_path]
├── [train]
│   ├── [input]
│   └── [label]
└── [val]
    ├── [input]
    └── [label]
```

## 🚅🚅 To train

If you have installed 'WandB', login your ID in command line.<br>
If not, update 'wandb: false,' in the "./configs/train_***.yml"

For <b>Segmentation</b>, update the "configs/train_segmentation.yml", "bash_train_segmentation.sh" and execute command
```
bash bash_train_segmentation.sh
```

For <b>Classification</b>, update the "configs/train_classification.yml", "bash_train_classification.sh" and execute command
```
bash bash_train_classification.sh
```


## 🚓🚓 To inference

For <b>Inference</b>, fix the 'configs/inference.yml' and execute below command
```
bash bash_inference.sh
```

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)