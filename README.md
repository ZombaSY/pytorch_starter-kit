# PyTorch Starter Kit

A beginner-friendly framework for training, inference, and exporting deep learning models for various computer vision tasks.

## 📋 System Requirements

- **OS**: Ubuntu 22.04 LTS
- **GPU Driver**: Version 550.54.14
- **CUDA**: Version 12.4

---

## 📊 Dataset Setup

Your dataset should be organized in CSV files with two columns: `input_paths` and `label_paths`.

> **Note:** The exact meaning of `input_paths` and `label_paths` depends on your task type.

### Training Dataset (`train.csv`)
```csv
input_paths             | label_paths
/path/to/input01.jpg    | /path/to/label01.jpg
/path/to/input03.jpg    | /path/to/label03.jpg
/path/to/input04.jpg    | /path/to/label04.jpg
...                     | ...
/path/to/input99.jpg    | /path/to/label99.jpg
```

### Validation Dataset (`valid.csv`)
```csv
input_paths             | label_paths
/path/to/input02.jpg    | /path/to/label02.jpg
/path/to/input05.jpg    | /path/to/label05.jpg
...                     | ...
/path/to/input98.jpg    | /path/to/label98.jpg
```

---

## ⚙️ Configuration

All training, inference, and export settings are configured in Python files located in the `configs/` directory. Simply edit the relevant configuration file for your task:

```bash
python3 main.py --config_path "configs/train_segmentation.py"
```

---

## 🎯 Training

Run training for your specific task:

### Classification
```bash
python3 main.py --config_path "configs/train_classification.py"
```

### Regression
```bash
python3 main.py --config_path "configs/train_regression.py"
```

### Semantic Segmentation
```bash
python3 main.py --config_path "configs/train_segmentation.py"
```

### Landmark Detection
```bash
python3 main.py --config_path "configs/train_landmark.py"
```

### Semi-Supervised Learning (Segmentation)
```bash
python3 main.py --config_path "configs/train_segmentation_regression.py"
```

Alternatively, edit `bash_train.sh` and run it directly.

---

## 🔍 Inference

Run inference on your trained model:

```bash
bash bash_inference.sh
```

---

## 📤 Model Export

Export your trained model:

```bash
bash bash_export.sh
```

---

## 📂 Project Structure

```
├── configs/             # Task-specific configurations
├── models/              # Model implementations and utilities
│   ├── backbones/       # Pre-trained backbone networks
│   ├── heads/           # Task-specific model heads
│   ├── losses.py        # Loss functions
│   ├── metrics.py       # Evaluation metrics
│   └── dataloader.py    # Data loading utilities
├── tools/               # Training, inference, and export scripts
├── main.py              # Entry point
├── bash_train.sh        # Training script template
├── bash_inference.sh    # Inference script template
└── bash_export.sh       # Export script template
```

---

## 🚀 Quick Start

1. **Prepare your dataset** in CSV format (see Dataset Setup section)
2. **Edit configuration** file for your task in `configs/`
3. **Run training** using the appropriate command
4. **Run inference** with `bash bash_inference.sh`
5. **Export model** with `bash bash_export.sh`
