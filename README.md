# MD-ViSCo: A Unified Model for Multi-Directional Vital Sign Waveform Conversion

## Overview

MD-ViSCo (Multi-Directional Vital-Sign Converter) is a unified deep learning framework that converts any vital sign waveform (ECG, PPG, ABP) into another using a single model. It combines a 1D U-Net with a Swin Transformer using AdaIN for waveform style adaptation, and integrates patient demographic information via text embeddings.

## Project Structure
```bash
├── src/                    # Main source code directory
│   ├── model/              # Core MD-ViSCo model implementations
│   │   ├── base_model.py   # Base class with common extraction logic
│   │   ├── MDViSCo.py      # Main MD-ViSCo model (UNet + Swin Transformer)
│   │   ├── NABNet.py       # NABNet baseline model
│   │   ├── PatchTST.py     # PatchTST baseline model
│   │   ├── PPG2ABP.py      # PPG2ABP baseline models
│   │   └── P2E_WGAN.py     # P2E-WGAN baseline model
│   │
│   ├── conf/               # Hydra-based configuration system
│   │   ├── config.yaml     # Main configuration entry point
│   │   ├── model/          # Model-specific configurations
│   │   ├── trainer/        # Trainer configurations
│   │   ├── dataset/        # Dataset configurations
│   │   └── directions/     # Direction-specific configurations
│   │
│   ├── dataset/            # Dataset handling and management
│   │   ├── base_dataset.py # Base dataset classes
│   │   ├── pulsedb_dataset.py # PulseDB dataset implementation
│   │   └── uci_dataset.py  # UCI dataset implementation
│   │
│   ├── trainers/           # Training framework
│   │   ├── trainer.py      # Base trainer classes
│   │   ├── approximation_trainer.py # Approximation model training
│   │   └── refinement_trainer.py # Refinement model training
│   │
│   ├── script/             # Standalone scripts
│   │   └── preprocess/     # Data preprocessing utilities
│   │
│   ├── criterions/         # Loss functions and criteria
│   ├── utils/              # Utility functions
│   └── train.py            # Main training script
└── 
```

## Installation

### Requirements
- Python ≥ 3.10
- CUDA 12.1+ for GPU support (optional)
- Conda or Miniconda

### Setup Environment

1. Create the environment from the provided YAML file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate mdvisco
   ```

3. Verify installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Dataset Preprocessing

This project supports preprocessing of two datasets:

- **[PulseDB (v1)](https://github.com/pulselabteam/PulseDB/tree/v1_0)**
- **[UCI (as preprocessed in the NABNet paper)](https://github.com/Sakib1263/NABNet)**

Raw datasets must be manually downloaded.

---

### Preprocessing Script

The project uses a **preprocessing pipeline** that converts raw datasets into standardized HDF5 formats.

#### PulseDB Dataset Preprocessing

```bash
python -m src.script.preprocess.preprocess \
    dataset=pulsedb \
    input_file=path/to/PulseDB_data.mat \
    output_file=path/to/output/directory
```

#### UCI Dataset Preprocessing

```bash
python -m src.script.preprocess.preprocess \
    dataset=uci \
    input_file=path/to/UCI_data.h5 \
    output_file=path/to/output/directory
```

## Training the Model

Model training is divided into two stages:
1. **Approximation Model** — generates normalized waveform
2. **Refinement Model** — improves ABP waveform by rescaling to real unit (mmHg) using patient information

---

### Supported Architectures

- **MD-ViSCo** (proposed)
- **NABNet**
- **PPG2ABP**
- **PatchTST**
- **P2E-WGAN**

---

### Approximation Model Training

The approximation model must be trained for each of the six directional pairs:

- `PPG2ABP`
- `ECG2ABP`
- `ABP2PPG`
- `ABP2ECG`
- `PPG2ECG`
- `ECG2PPG`

All baseline models and **MD-ViSCo** support these directions.  
*MD-ViSCo* can also be trained **multi-directionally**.

---

#### Approximation Model Training Commands

```bash
# All approximation training options in one command:
torchrun --standalone --nproc_per_node=1 src/train.py [train_dataset=train_pulsedb|train_uci] [trainer=approximation_trainer]

# Examples:
# Basic (uses defaults): 
torchrun --standalone --nproc_per_node=1 src/train.py

# PulseDB dataset:
torchrun --standalone --nproc_per_node=1 src/train.py train_dataset=train_pulsedb

# UCI dataset:
torchrun --standalone --nproc_per_node=1 src/train.py train_dataset=train_uci

# Multi-GPU:
torchrun --standalone --nproc_per_node=2 src/train.py train_dataset=train_pulsedb

# With specific trainer:
torchrun --standalone --nproc_per_node=1 src/train.py trainer=approximation_trainer

# Multiple overrides:
torchrun --standalone --nproc_per_node=1 src/train.py \
    train_dataset=train_pulsedb \
    trainer=approximation_trainer
```

---

### Refinement Model Training

The **refinement model** is applied **only to ABP waveform prediction**, as it is the only waveform that can be scaled to real units (mmHg). Therefore, training is restricted to the following directional pairs:

- `PPG2ABP`
- `ECG2ABP`

---

#### Behavior by Model

- **MD-ViSCo** uses a **shared text encoder** for PPG and ECG inputs. Thus, it jointly optimizes both directions.
- **PatchTST** and **MD-ViSCo** can utilize **patient demographic information** on **PulseDB**. This feature is **not available for UCI** due to missing data.
- **MD-ViSCo** supports a **Weighted Contrastive Loss (WCL)**.
---

#### Refinement Model Training Commands

```bash
# All refinement training options in one command:
torchrun --standalone --nproc_per_node=1 src/train.py trainer=refinement_trainer [train_dataset=train_pulsedb|train_uci]

# Examples:
# Basic refinement training:
torchrun --standalone --nproc_per_node=1 src/train.py trainer=refinement_trainer

# Multi-GPU refinement training:
torchrun --standalone --nproc_per_node=2 src/train.py trainer=refinement_trainer

# With PulseDB dataset:
torchrun --standalone --nproc_per_node=1 src/train.py \
    train_dataset=train_pulsedb \
    trainer=refinement_trainer

# With UCI dataset:
torchrun --standalone --nproc_per_node=1 src/train.py \
    train_dataset=train_uci \
    trainer=refinement_trainer
```