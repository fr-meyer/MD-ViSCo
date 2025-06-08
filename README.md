# MD-ViSCo: A Unified Model for Multi-Directional Vital Sign Waveform Conversion

## Overview

MD-ViSCo (Multi-Directional Vital-Sign Converter) is a unified deep learning framework that converts any vital sign waveform (ECG, PPG, ABP) into another using a single model. It combines a 1D U-Net with a Swin Transformer using AdaIN for waveform style adaptation, and integrates patient demographic information via text embeddings.

## Project Structure
```bash
├── model/                  # Core MD-ViSCo model implementations
│
├── features/               # Feature extraction and analysis
│
├── preprocessing/          # Data preprocessing utilities
│
├── datasets/               # Dataset storage and management
│
├── config/                 # Configuration files
│
├── baseline/               # Baseline model implementations
│
├── train/                  # Training scripts
│
├── test/                   # Testing and evaluation scripts
│
├── utils/                  # Utility functions
│
└── dataload/               # Data loading utilities
```

## Installation

### Requirements

- Python ≥ 3.10
- CUDA 12.1+ for GPU support (optional)
- PyTorch, Transformers, NeuroKit2, etc.
- See `environment.yml` or `requirements.txt` for full details

You can set up the environment using either **Conda** (recommended) or **pip**, depending on your setup.

---

### Option 1: Using Conda (Recommended)

1. Create the environment from the provided YAML file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate mdvisco
   ```
---

### Option 2: Using pip

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preprocessing

This project supports preprocessing of two datasets:

- **[PulseDB (v1)](https://github.com/pulselabteam/PulseDB/tree/v1_0)**
- **[UCI (as preprocessed in the NABNet paper)](https://github.com/Sakib1263/NABNet)**

Raw datasets must be manually downloaded and placed in the `datasets/` directory.
[See datasets instructions for more details](datasets/datasets_README.md)

---

### Preprocessing Script

To run preprocessing from the project root:

```bash
python -m preprocessing.DataPreprocessing
```

The script will process the data and save the preprocessed output in the `datasets/Preprocessed/` directory.


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

Each model is configured with equivalent complexity to ensure fair comparisons.

> Training is implemented using **PyTorch Distributed Data Parallel (DDP)** and requires **at least two GPUs**.

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
*MD-ViSCo* can also be trained **multi-directionally** by omitting the `--direction` argument.

---

#### Unidirectional Training Command

```bash
python -m train.ddp_trainer \
    --gpu_ids "0,1" \                          # Comma-separated GPU IDs (e.g., "0,1,2") 
    --dataset {pulsedb,uci} \                  # Choose dataset
    --model_type approximation \              
    --model_name {nabnet,ppg2abp,p2ewgan,patchtst,mdvisco} \  # Model architecture
    --direction {PPG2ABP,ECG2ABP,...} \        # Conversion direction
    --checkpoint_name CHECKPOINT_NAME \        # Checkpoint prefix
    --batch_size_approximation_model 256 \     # Training batch size
    --test_batch_size 256 \                    # Testing batch size
    --learning_rate 1e-3 \                     # Learning rate
    --scheduler_patience 3 \                   # LR scheduler patience
    --early_stopping_patience 5 \              # Early stopping patience
    --num_epochs_approximation_model 100 \     # Total training epochs
    --seed 42                                  # Reproducibility seed
```

---

#### Multi-Directional Training (MD-ViSCo only)

To train **MD-ViSCo** across all direction pairs simultaneously, **omit** the `--direction` flag:

```bash
python -m train.ddp_trainer \
    --gpu_ids "0,1" \
    --dataset {pulsedb,uci} \
    --model_type approximation \
    --model_name mdvisco \
    --checkpoint_name mdvisco_multidir \
    --batch_size_approximation_model 256 \
    --test_batch_size 256 \
    --learning_rate 1e-3 \
    --scheduler_patience 3 \
    --early_stopping_patience 5 \
    --num_epochs_approximation_model 100 \
    --seed 42
```

---

### Refinement Model Training

The **refinement model** is applied **only to ABP waveform prediction**, as it is the only waveform that can be scaled to real units (mmHg). Therefore, training is restricted to the following directional pairs:

- `PPG2ABP`
- `ECG2ABP`

---

#### Behavior by Model

- **MD-ViSCo** uses a **shared text encoder** for PPG and ECG inputs. Thus, it jointly optimizes both directions and allows **omitting the `--direction` argument**.
- **PatchTST** and **MD-ViSCo** can utilize **patient demographic information** on **PulseDB** using the `--pi` flag. This feature is **not available for UCI** due to missing data.
- **MD-ViSCo** supports a **Weighted Contrastive Loss (WCL)** enabled with `--wcl`.
- **MD-ViSCo** training involves two phases:
  - **Pretraining**: `--is_pretraining`
  - **Finetuning**: `--is_finetuning`

---

#### Sample Command (MD-ViSCo - Refinement Model)

```bash
python -m train.ddp_trainer \
    --gpu_ids "0,1" \                      # Comma-separated list of GPU IDs to use
    --dataset pulsedb \                    # Dataset to use: pulsedb or uci
    --model_type refinement \              # Type of model to train: 'refinement' here
    --model_name mdvisco \                 # Model architecture: use 'mdvisco' for our proposed method
    --checkpoint_name checkpoint \         # Prefix name for checkpoint files
    --batch_size 256 \                     # Batch size used for training
    --test_batch_size 256 \                # Batch size used for testing
    --learning_rate 1e-3 \                 # Learning rate for optimizer
    --scheduler_patience 3 \               # Patience (in epochs) before LR is reduced
    --early_stopping_patience 5 \          # Patience (in epochs) before early stopping triggers
    --num_epochs 100 \                     # Total number of training epochs
    --seed 42 \                            # Seed value for reproducibility
    --wcl \                                # Use Weighted Contrastive Loss (WCL)
    --pi \                                 # Include Patient Information (only on PulseDB)
    --bp_norm \                            # Apply BP waveform normalization
    --use_patient_split \                  # Enable patient-wise data splitting
    --is_pretraining \                     # Flag for pretraining phase (MD-ViSCo only)
    --is_finetuning                        # Flag for finetuning phase (MD-ViSCo only)
```

## Evaluation

Model evaluation is conducted **on a single GPU**.

---

### Approximation Model Evaluation

- Evaluates waveform similarity using:
  - **Mean Absolute Error (MAE)**
  - **Pearson Correlation (PC)**
- Targets are **normalized waveforms** (e.g., PPG or ECG)
- Specific flags for approximation model:
  - `--calculate_bpm`: Extract beats-per-minute (BPM)
  - `--extract_waveform_features`: Compute waveform-derived biomarkers
  - These are only valid when the **target waveform is PPG or ECG**:
    - `ABP2PPG`, `ABP2ECG`, `PPG2ECG`, `ECG2PPG`

---

### Refinement Model Evaluation

- Applies **only to ABP waveform prediction**
- Outputs are evaluated in **real-valued units (mmHg)**:
  - ABP waveform
  - Systolic BP (SBP), Diastolic BP (DBP), Mean Arterial Pressure (MAP)
- Includes evaluation for:
  - **MAE**, **PC**
  - **AAMI** and **BHS** standard compliance

Specific flags for refinement model:
- `--bp_norm`: Apply BP-specific normalization
- `--use_patient_split`: Use patient-wise split for testing
- `--is_finetuning`: Evaluate model in fine-tuned mode
- `--wcl`: Use Weighted Contrastive Loss (only for MD-ViSCo)
- `--pi`: Include patient demographic information (only supported on PulseDB)

---

### Multi-Directional Evaluation (MD-ViSCo only)

- For **MD-ViSCo**, the `--direction` flag can be **omitted** during evaluation
- This will evaluate **all directions** jointly in a single run

---

### Sample Evaluation Command

```bash
python test/test_runner.py \
    --dataset pulsedb \                          # Dataset: 'pulsedb' or 'uci' (required)
    --model_type refinement \                    # Model type: 'approximation' or 'refinement' (required)
    --checkpoint_name your_checkpoint_name \     # Name of checkpoint file (required)

    # Optional Model Configuration
    --model_name mdvisco \                       # Model architecture
    # --direction PPG2ABP \                      # Omit for MD-ViSCo multi-directional eval

    # Testing Options
    --gpu_ids 0 \                                # Single GPU ID (evaluation runs on 1 GPU only)
    --checkpoint_epoch 100 \                     # Specific checkpoint epoch to use
    --batch_size 256 \                           # Batch size for refinement model
    --test_batch_size 256 \                      # Test batch size

    # Approximation model only
    --calculate_bpm \                            # [Approximation] Compute beats per minute
    --extract_waveform_features \                # [Approximation] Extract waveform-derived features

    # Refinement model only
    --bp_norm \                                  # [Refinement] Normalize ABP outputs
    --use_patient_split \                        # [Refinement] Patient-based evaluation split
    --is_finetuning \                            # [Refinement, MD-ViSCo only] Evaluate fine-tuned model
    --wcl \                                      # [Refinement, MD-ViSCo only] Use WCL loss
    --pi \                                       # [Refinement, PulseDB only, MD-ViSCo or PatchTST only] Include patient information

    # Misc
    --seed 42 \                                  # Random seed
```

---


### Physiological Feature Extraction (Approximation Model Level)

This section focuses on the extraction and analysis of **physiological features** from **converted PPG/ECG waveforms**.  
> **Note**: ABP-based features are evaluated during **refinement model evaluation**.

---

#### Step 1: Extract Ground Truth Features

Before evaluating the converted waveforms, extract ground truth physiological features from the original datasets:

```bash
python -m features.feature_extraction \
    --output_dir ./datasets \            # Directory to save extracted features
    --sampling_rate 125 \                # Signal sampling rate in Hz
    --dataset {pulsedb,uci}              # Choose between 'pulsedb' or 'uci'
```

**Output Location**  
This generates `.h5` files with extracted features saved in:

```
datasets/
├── PulseDB/
│   └── Features/
│       └── AAMI_Test_Subset_PulseDB_Features.h5
└── UCI/
    └── Features/
        └── UCI_Dataset_Preprocessed_Features.h5
```

---

#### Step 2: Extract Features from Converted Signals

Use the `--extract_waveform_features` flag during **approximation model evaluation** to extract physiological features from the **converted PPG/ECG waveforms**.

**Output Location**  
The features are saved in `.h5` files following this structure:

```
./results/features_extraction/
└── {Dataset}/
    └── {Model_Name}/
        └── seed_{SEED}/
            └── features_{DIRECTION}.h5
```

- `{Dataset}`: either `UCI` or `PulseDB`
- `{Model_Name}`: e.g., `MD-ViSCo`, `patchtst`, `nabnet`, etc.
- `{SEED}`: the seed value used for model training/evaluation
- `{DIRECTION}`: one of `PPG2ECG`, `ECG2PPG`, `ABP2PPG`, `ABP2ECG`

See the [Evaluation](#evaluation) section for how to run evaluation with `--extract_waveform_features`.

---

#### Step 3: Analyze Feature Accuracy

Compare the extracted features from converted waveforms against the ground truth:

```bash
python -m features.feature_analysis \
    --dataset_type {PulseDB,UCI} \                     # Dataset type
    --sample_file SAMPLE_FILE_PATH \                   # Path to GT feature file
    --model_name {mdvisco,patchtst,p2ewgan,ppg2abp,nabnet} \  # Model to analyze
    --seed SEED \                                      # Model seed
    --direction {PPG2ECG,ECG2PPG,ABP2PPG,ABP2ECG} \     # Conversion direction
    --features_base_path FEATURES_PATH \               # Directory of extracted features
    --dataset {0,1,2} \                                # Data split: 0=train, 1=val, 2=test (default: 2)
    --sample_length SAMPLE_LENGTH \                    # Sample length (default: 1280)
    --output_csv OUTPUT_CSV_PATH                       # Output CSV file path for results
```

This analysis reports feature-wise errors and comparisons, providing insight into the morphological fidelity of the reconstructed waveforms.

---

