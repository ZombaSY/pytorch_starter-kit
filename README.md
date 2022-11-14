# Pytorch Start-kit Project


## Environments

- OS: Ubuntu 16.04
- GPU: Tesla V100 32GB
- GPU Driver version: 460.106.00
- CUDA: 11.2
- Pytorch 1.8.1

## Dataset
The dataset recommend following below hierarchy

```
[root_path]
├── [train]
│   ├── [input]
│   └── [label]
└── [val]
    ├── [input]
    └── [label]
```

## Train

If you have installed 'WandB', login your ID in command line.<br>
If not, fix to 'wandb: false,' in "./hyper_parameters/train_***.yml"

For <b>Segmentation</b>, fix the "hyper_parameters/train_segmentation.yml", "bash_train_segmentation.sh" and execute command
```
bash bash_train_segmentation.sh
```

For <b>Classification</b>, fix the "hyper_parameters/train_classification.yml", "bash_train_classification.sh" and execute command
```
bash bash_train_classification.sh
```


## Inference

For <b>Inference</b>, fix the 'hyper_parameters/inference.yml' and execute below command
```
bash bash_inference.sh
```
