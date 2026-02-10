"""Internal protocol constants for MD-ViSCo codebase.

These constants define the contracts between dataset → model → processor →
trainer/evaluator.
They are NOT Hydra-configurable because changing them is a compatibility
break, not a parameter change.

Categories:
1. Batch dictionary keys: How data flows from dataset to model
2. Processor output keys: How processor returns results to trainer/evaluator
3. Training stage names: How trainer/evaluator distinguish train/val/test
   phases
4. Checkpoint payload keys: How checkpoints store model/optimizer/scheduler
   state
5. Metric keys: How metrics are named in aggregation/logging
6. DDP prefix: How to handle DistributedDataParallel wrapping

Usage:
    >>> from src.utils.constants import BATCH_KEY_INPUT, STAGE_TRAIN
    >>> input_tensor = batch[BATCH_KEY_INPUT]
    >>> if stage == STAGE_TRAIN:
    ...     # Training logic

See Also:
    - src.model.mdvisco: Uses UNetSwinUnetConfig (Hydra-configurable
      architecture)
    - src.trainers.trainer: Uses these constants for batch/stage/checkpoint
      handling
    - src.evaluators.base_evaluator: Uses these constants for
      processor/checkpoint handling

Note:
    Architecture parameters (init_features, kernel_size, etc.) are
    Hydra-configurable via UNetSwinUnetConfig; do not duplicate them as
    Python constants.
"""

# ============================================================================
# Batch Dictionary Keys
# ============================================================================
# How data flows from dataset collate function to model forward pass

BATCH_KEY_INPUT = "x"  # [B, C, T] input waveform
BATCH_KEY_TARGET = "y"  # [B, C, T] target waveform
BATCH_KEY_TARGET_INDICES = "tgt_idxs"  # [B] target channel indices per sample
# Direction enum or string for multi-directional models
BATCH_KEY_DIRECTION = "direction"

# ============================================================================
# Processor Output Keys
# ============================================================================
# How processor returns results to trainer/evaluator

PROCESSOR_KEY_WAVEFORM = "waveform"  # [B, C, T] processed waveform
PROCESSOR_KEY_PREDICTIONS = "predictions"  # Stage-aware (trimmed for val/test)
PROCESSOR_KEY_PREDICTIONS_RAW = "predictions_raw"  # Always-padded from model
PROCESSOR_KEY_METRICS = "metrics"  # Nested dict of processor-computed metrics
PROCESSOR_KEY_PADDING_METADATA = "padding_metadata"  # padding_length, original_length
PROCESSOR_KEY_EXTRAS = "extras"  # May contain padding_metadata

# ============================================================================
# Training Stage Names
# ============================================================================
# How trainer/evaluator distinguish phases

STAGE_TRAIN = "train"  # Uses padded predictions
STAGE_VAL = "val"  # Uses trimmed predictions
STAGE_TEST = "test"  # Uses trimmed predictions

# ============================================================================
# Checkpoint Payload Keys
# ============================================================================
# How checkpoints store model/optimizer/scheduler/early_stopping state
# Note: checkpoint_key (e.g., 'model_state_dict' vs 'generator_state_dict')
# is Hydra-configurable per evaluator config, so we only define the common
# default here.

CHECKPOINT_MODEL_STATE_KEY = "model_state_dict"
CHECKPOINT_OPTIMIZER_KEY = "optimizer"
CHECKPOINT_SCHEDULER_KEY = "scheduler"
CHECKPOINT_EARLY_STOPPING_KEY = "early_stopping"

# Legacy keys used in payloads (optimizer_state_dict, etc.) - keep for
# compatibility
CHECKPOINT_OPTIMIZER_STATE_KEY = "optimizer_state_dict"
CHECKPOINT_SCHEDULER_STATE_KEY = "scheduler_state_dict"
CHECKPOINT_EARLY_STOPPING_STATE_KEY = "early_stopping_state"

# ============================================================================
# Metric Keys
# ============================================================================
# How metrics are named in aggregation/logging

METRIC_KEY_LOSS = "loss"
METRIC_KEY_MAE = "mae"
METRIC_KEY_CORRELATION = "correlation"
METRIC_KEY_BASIC_MSE = "basic_mse"
METRIC_KEY_BASIC_MAE = "basic_mae"

# ============================================================================
# DDP Prefix
# ============================================================================
# How to handle DistributedDataParallel wrapping

DDP_MODULE_PREFIX = "module."  # Added by DDP to state dict keys
