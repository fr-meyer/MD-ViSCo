# MD-ViSCo: A Unified Model for Multi-Directional Vital Sign Waveform Conversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Hydra Config](https://img.shields.io/badge/config-Hydra-1f4b99)](https://github.com/facebookresearch/hydra)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MD-ViSCo is a unified deep learning framework for converting vital sign waveforms (ECG, PPG, ABP) using a single model. It combines a 1D U-Net with a Swin Transformer using AdaIN for waveform style adaptation, and integrates patient demographic information via text embeddings for enhanced predictions.

**Published in IEEE Journal of Biomedical and Health Informatics (2026).**

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Tutorials](#tutorials)
  - [Quick Start](#quick-start)
- [How-To Guides](#how-to-guides)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
  - [Using Demographics](#using-demographics)
  - [Feature Extraction](#feature-extraction)
  - [Custom Experiments](#custom-experiments)
    - [Configuration Structure](#configuration-structure)
- [Reference](#reference)
  - [Supported Models](#supported-models)
  - [Datasets](#datasets)
- [Explanation](#explanation)
  - [Architecture Overview](#architecture-overview)
  - [Two-Stage vs Single-Stage Models](#two-stage-vs-single-stage-models)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Multi-directional conversion** between ECG, PPG, ABP, and IMP waveforms
- **Unified model architecture** for all conversion tasks
- **Demographic integration** for enhanced predictions (PulseDB only)
- **Multiple baseline implementations** (MD-ViSCo, NABNet, PatchTST, PPG2ABP, P2E-WGAN, WaveNet)
- **Blood pressure estimation** from PPG/ECG signals
- **Feature extraction capabilities** for physiological analysis
- **Atrial fibrillation classification** from ECG using MIMIC PERform AF dataset

## Requirements

- Python 3.10
- CUDA 12.1+ (optional, for GPU support)
- Conda or Miniconda

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fr-meyer/MD-ViSCo
   cd MD-ViSCo
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate environment:**
   ```bash
   conda activate mdvisco
   ```

4. **Verify installation:**
   
   **Check CUDA/GPU support (optional):**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   For CPU-only training, unset `CUDA_VISIBLE_DEVICES` and use `--nproc_per_node=1`.
   
   **Verify dataset paths:**
   - Check `dataset_path`, `dataset_folder`, and `file_name` in `train_dataset` / `test_dataset` configs under `src/conf/`
   - Ensure preprocessing was run and the output file exists at the configured location

## Tutorials

This section provides a minimal end-to-end flow to get you from raw data to a trained and evaluated model. For more detailed and task-specific instructions, see [How-To Guides](#how-to-guides).

### Quick Start

### 1. Preprocess Dataset

Prepare the dataset once by converting the raw files into the HDF5 format expected by the training and evaluation pipelines:

```bash
# PulseDB: preprocessor=pulsedb_preprocessing, input_file=path/to/PulseDB_data.mat
# UCI: preprocessor=uci_preprocessing, input_file=path/to/UCI_data.h5
python -m src.script.preprocess.preprocess \
    preprocessor=pulsedb_preprocessing \
    preprocessor.input_file=path/to/PulseDB_data.mat \
    preprocessor.output_file=path/to/output/directory/processed_pulsedb.h5
```

### 2. Train a Model

Train a baseline model on PulseDB. You can later swap `train_dataset` / `test_dataset`, `model`, or `trainer` for other experiments:

```bash
# Single GPU: --nproc_per_node=1
# Multi-GPU: --nproc_per_node=2 (or number of GPUs)
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    train_dataset=train_pulsedb \
    test_dataset=test_pulsedb \
    model=patchtst \
    trainer=approximation_trainer_patchtst
```

### 3. Evaluate

Run evaluation on the test split using the saved checkpoint.

```bash
torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator=waveform_reconstruction_evaluator \
    test_dataset=test_pulsedb \
    model=patchtst \
    evaluator.checkpoint_epoch=100
```

## How-To Guides

This section focuses on task-oriented recipes: pick the guide that matches the problem you want to solve.

### Training Models

#### Waveform Reconstruction

Train models to convert between waveforms (e.g., PPG→ABP, ECG→PPG):

```bash
# MD-ViSCo: model=mdvisco_approximation, trainer=approximation_trainer_mdvisco
# PatchTST: model=patchtst, trainer=approximation_trainer_patchtst
# NABNet: model=nabnet_approximation, trainer=approximation_trainer_nabnet
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    train_dataset=train_pulsedb \
    test_dataset=test_pulsedb \
    model=patchtst \
    trainer=approximation_trainer_patchtst
```

#### Blood Pressure Prediction

Train models to predict BP scalars (SBP/DBP) from waveforms:

```bash
# PPG→BP: trainer.directions=ppg2bp
# ECG→BP: trainer.directions=ecg2bp
# PPG+ECG→BP: trainer.directions=ppg2bp_ecg2bp, trainer=refinement_trainer_mdvisco
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=refinement_trainer_nabnet \
    trainer.directions=ppg2bp \
    train_dataset=train_pulsedb
```

#### AF Classification

Train AF classification models on the MIMIC PERform AF dataset:

```bash
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=classification_trainer \
    train_dataset=train_mimic_perform_af_1024
```

### Evaluating Models

Evaluate trained models on test datasets:

```bash
# Waveform reconstruction: evaluator=waveform_reconstruction_evaluator
# Blood pressure: evaluator=blood_pressure_evaluator, test_dataset=test_pulsedb_refinement_bp
torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator=waveform_reconstruction_evaluator \
    test_dataset=test_pulsedb \
    model=patchtst \
    evaluator.checkpoint_epoch=100
```

**Common overrides:** `evaluator.direction_mode=single`, `evaluator.input_preprocessing.source.vital=ppg`, `model@evaluator.model=...`

**Note:** Checkpoint paths include training parameters (`batch_size`, `num_epochs`, `learning_rate`, `seed`). When evaluating, these must match the training configuration used to create the checkpoint.

### Using Demographics

Demographic information (age, gender, height, weight, BMI) is available only for PulseDB dataset and can improve predictions:

```bash
# With demographics: model.use_demographics=true, model.num_demographic_channels=5
# Without demographics: model.use_demographics=false
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    model=patchtst \
    model.use_demographics=true \
    model.num_demographic_channels=5 \
    trainer=refinement_trainer_patchtst \
    train_dataset=train_pulsedb_refinement_bp
```

### Feature Extraction

Extract physiological features from waveforms:

```bash
# ECG features: processor=waveform_processor_ecg_features, evaluator.directions=ppg2ecg
# PPG features: processor=waveform_processor_ppg_features, evaluator.directions=ecg2ppg
torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator=feature_extraction_evaluator \
    processor=waveform_processor_ecg_features \
    test_dataset=test_uci \
    evaluator.directions=ppg2ecg \
    evaluator.checkpoint_epoch=100

# Feature analysis
python -m src.script.features.feature_analysis \
    feature_analysis.gt_features_file=path/to/features/DATASET/ground_truth/seed_SEED/features_DIRECTION.h5 \
    feature_analysis.model_name=mdvisco \
    feature_analysis.seed=42 \
    feature_analysis.direction=PPG2ECG \
    feature_analysis.dataset_name=PulseDB
```


### Custom Experiments

#### CLI Overrides (Quick Experiments)

Use CLI overrides for quick parameter tuning:

```bash
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    train_dataset=train_uci \
    test_dataset=test_uci \
    model=patchtst \
    model.d_model=256 \
    model.num_encoder_layers=6 \
    trainer=approximation_trainer_patchtst \
    trainer.optimizer.lr=0.0005 \
    trainer.num_epochs=150
```

#### Configuration Structure

MD-ViSCo uses Hydra with ConfigStore for type-safe, composable configuration management. Configurations in `src/conf/` are organized by concern: `model/`, `processor/`, `train_dataset/`, `test_dataset/`, `trainer/`, `criterion/`, `optimizer/`, `scheduler/`, `early_stopping/`, `directions/`.

#### Common Configuration Issues

- **InterpolationKeyError**: A referenced key like `${train_dataset.input_size}` cannot be resolved. **Solution:** Always specify `train_dataset` and `test_dataset` when training.
- **Shape Mismatch Errors**: Occur when `input_length` or `num_targets` are manually overridden. **Solution:** Let Hydra interpolation handle these values automatically from dataset configs; don't manually set `model.input_length` or `model.num_targets`.

## Reference

### Supported Models

- **MD-ViSCo** (proposed): Unified model combining 1D U-Net with Swin Transformer and AdaIN for multi-directional vital sign waveform conversion. Based on: [Swin Transformer](https://ieeexplore.ieee.org/document/9710580/) | [Swin-Unet](https://link.springer.com/10.1007/978-3-031-25066-8_9) | [PatchTST Time Series](https://arxiv.org/abs/2211.14730) | [PatchTST BP Estimation](https://ieeexplore.ieee.org/document/10445970/)

- **NABNet**: Baseline model for vital sign conversion. **Note:** On MIMIC PERform Large, NABNet requires overrides: `model.model_depth=5`, `model.model_width=32`, `model.attention_type=lstm`. [Paper](https://linkinghub.elsevier.com/retrieve/pii/S1746809422007017) | [Code](https://github.com/Sakib1263/NABNet)

- **PatchTST**: Time series transformer baseline with optional demographic fusion. [Paper](https://arxiv.org/abs/2211.14730) | [PatchTST BP Estimation](https://ieeexplore.ieee.org/document/10445970/) | [Code](https://github.com/yuqinie98/PatchTST) | [Docs](https://huggingface.co/docs/transformers/main/en/model_doc/patchtst)

- **PPG2ABP**: Baseline models (UNetDS64, MultiResUNet1D). [Paper](https://www.mdpi.com/2306-5354/9/11/692) | [Code](https://github.com/nibtehaz/PPG2ABP)

- **P2E-WGAN**: Generative adversarial network baseline. [Paper](https://dl.acm.org/doi/10.1145/3412841.3441979) | [Code](https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction)

- **WaveNet**: WaveNet architecture for waveform generation. [Paper](https://arxiv.org/abs/1609.03499) | [Code](https://github.com/vincentherrmann/pytorch-wavenet)

### Datasets

The same model config can be reused across datasets via Hydra variable interpolation. Changing `train_dataset` automatically adapts `input_length`, BP normalization bounds, and processor configuration.

- **PulseDB**: Large, cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods. Includes demographics (`age_raw`, `gender_raw`, `height_raw`, `weight_raw`, `bmi_raw`). Input length: 1280, BP bounds: `dbp_min=2.34`, `sbp_max=286.58`. [Paper](https://www.frontiersin.org/articles/10.3389/fdgth.2022.1090854/full) | [Repository](https://github.com/pulselabteam/PulseDB/tree/v1_0)

- **UCI**: Cuff-Less Blood Pressure Estimation dataset from UCI Machine Learning Repository. Input length: 1024, BP bounds: `dbp_min=50.0`, `sbp_max≈189.98`. [Repository](https://archive.ics.uci.edu/dataset/340) | [Preprocessing](https://github.com/Sakib1263/NABNet)

- **MIMIC PERform AF**: Dataset for atrial fibrillation classification tasks. Contains AF labels required for AF Classifier models. [Paper](https://iopscience.iop.org/article/10.1088/1361-6579/ac826d) | [Repository](https://zenodo.org/records/15906524)

- **MIMIC PERform Large**: Large-scale dataset for vital sign analysis. **Note:** Lacks ABP waveforms, so refinement models (which require ABP for BP prediction) cannot be used. Approximation models can convert between ECG, PPG, and IMP waveforms. [Paper](https://iopscience.iop.org/article/10.1088/1361-6579/ac826d) | [Repository](https://zenodo.org/records/15906524)

## Explanation

### Architecture Overview

MD-ViSCo combines:
- **1D U-Net**: Encoder-decoder architecture for waveform processing
- **Swin Transformer**: Hierarchical vision transformer for feature extraction
- **AdaIN**: Adaptive instance normalization for waveform style adaptation
- **Text Embeddings**: Patient demographic information integration (PulseDB only)

### Two-Stage vs Single-Stage Models

#### Two-Stage Scaling Models

**MD-ViSCo, PatchTST, NABNet:**
- **Stage 1**: Produces normalized ABP waveform from source signals (PPG/ECG)
- **Stage 2**: Predicts SBP/DBP scalars for unscaling the normalized waveform to mmHg
- **Checkpoint loading**: Separate managers for stage1 (approximation) and stage2 (refinement)

#### Single-Stage Models

**PPG2ABP, WaveNet, P2E-WGAN:**
- **Direct ABP waveform output**: No separate scaling step required
- **Cascade architecture**: Stage1 approximation → Stage2 refinement
- **Checkpoint loading**: Standard single-model manager

## Contributing

### Code Quality

Before submitting changes, ensure your code passes formatting, linting, and type checks:

```bash
ruff format src/
ruff check src/ --fix
pyright src/
```

### How to Contribute

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/fr-meyer/MD-ViSCo/issues)
- **Discussions**: Ask questions and share ideas in [GitHub Discussions](https://github.com/fr-meyer/MD-ViSCo/discussions)
- **Pull requests**: Open a pull request against the appropriate branch. Ensure changes pass formatting and type checks

## Citation

If you use MD-ViSCo in your research, please cite:

```bibtex
@ARTICLE{11366001,
  author={Meyer, Franck and Hur, Kyunghoon and Choi, Edward},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={MD-ViSCo: A Unified Model for Multi-Directional Vital Sign Waveform Conversion},
  year={2026},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/JBHI.2025.3639315},
  ISSN={2168-2208},
  url={https://ieeexplore.ieee.org/document/11366001}
}
```

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for the full text.

## Acknowledgments

- **GenHPF Framework**: The code architecture and organization is based on [GenHPF: General Healthcare Predictive Framework for Multi-Task Multi-Source Learning](https://ieeexplore.ieee.org/document/10298642/) ([code](https://github.com/hoon9405/GenHPF))

- **pyPPG**: PPG feature extraction uses [pyPPG: A Python toolbox for comprehensive photoplethysmography signal analysis](https://iopscience.iop.org/article/10.1088/1361-6579/ad33a2) ([code](https://github.com/godamartonaron/GODA_pyPPG))

- **NeuroKit2**: ECG feature extraction uses [NeuroKit2: A Python toolbox for neurophysiological signal processing](https://link.springer.com/article/10.3758/s13428-020-01516-y) ([code](https://github.com/neuropsychology/NeuroKit))
