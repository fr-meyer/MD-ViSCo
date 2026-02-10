"""Waveform reconstruction trainer for vital signal transformation tasks.

This module provides WaveformReconstructionTrainer, a trainer specialized
for models that output waveform predictions (e.g., PPG->ABP, ECG->PPG
transformations). It handles the complete training, validation, and
testing workflow for waveform reconstruction models.

**Supported Models:**
- NABNet: Multi-scale waveform reconstruction
- MDViSCo: Multi-directional vital signal conversion
- WaveNetModel: Temporal convolution-based waveform generation
- PPG2ABP: Direct PPG to ABP waveform mapping
- P2E-WGAN: Adversarial waveform generation

**Key Features:**
- Stage-aware loss computation (padded for training, trimmed for
  validation/test)
- Processor-based output parsing with automatic padding handling
- Support for both single-direction and multi-directional training
- Distributed training support via PyTorch DDP

**Exports:**
- WaveformReconstructionTrainer: Main trainer class
- WaveformReconstructionTrainerConfig: Hydra-compatible configuration
  dataclass
"""

import logging

# Standard library imports
from dataclasses import dataclass
from typing import Any
from typing import cast

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.processors import OutputProcessor
from src.trainers.trainer import BaseTrainer
from src.trainers.trainer import DirectionMode
from src.trainers.trainer import TrainerBaseConfig
from src.utils.constants import BATCH_KEY_TARGET
from src.utils.constants import BATCH_KEY_TARGET_INDICES
from src.utils.constants import METRIC_KEY_BASIC_MAE
from src.utils.constants import METRIC_KEY_BASIC_MSE
from src.utils.constants import METRIC_KEY_LOSS
from src.utils.constants import PROCESSOR_KEY_EXTRAS
from src.utils.constants import PROCESSOR_KEY_METRICS
from src.utils.constants import PROCESSOR_KEY_PADDING_METADATA
from src.utils.constants import PROCESSOR_KEY_PREDICTIONS
from src.utils.constants import PROCESSOR_KEY_PREDICTIONS_RAW
from src.utils.constants import STAGE_TEST
from src.utils.constants import STAGE_TRAIN
from src.utils.constants import STAGE_VAL
from src.utils.flatten_metrics import flatten_nested_metrics
from src.utils.validation_utils import validate_tensor_shapes_match
from src.utils.waveform_utils import compute_trimmed_length
from src.utils.waveform_utils import extract_padding_length
from src.utils.waveform_utils import trim_target_padding

logger = logging.getLogger(__name__)


@dataclass
class WaveformReconstructionTrainerConfig(TrainerBaseConfig):
    """Configuration for WaveformReconstructionTrainer with Hydra-compatible defaults.

    This trainer works with any model that outputs waveform predictions.
    Compatible models: NABNet, MDViSCo, WaveNetModel, PPG2ABP, P2E-WGAN
    """

    _target_: str = (
        "src.trainers.waveform_reconstruction_trainer.WaveformReconstructionTrainer"
    )
    trainer_name: str = "WaveRec"

    # Data preprocessing configuration (inherits from TrainerBaseConfig)
    # Override default preprocessing if needed for waveform reconstruction
    # tasks
    input_preprocessing: dict[str, Any] = MISSING


class WaveformReconstructionTrainer(BaseTrainer):
    """Trainer for waveform reconstruction tasks (PPG->ABP, ECG->PPG, etc.).

    Works with any model that outputs waveform predictions.
    Compatible models: NABNet, MDViSCo, WaveNetModel, PPG2ABP, P2E-WGAN

    Supports both single-direction and multi-directional training modes.
    - MDViSCo can be trained uni-directionally (direction_mode="single") or
      multi-directionally (direction_mode="multi")
    - NABNet, PatchTST, PPG2ABP can only be trained uni-directionally
      (direction_mode="single")
    - Users must explicitly specify direction_mode="single" or "multi" - no
      automatic detection

    Loss semantics:
    - Training epochs compute loss on padded predictions/targets so
      gradients cover the full receptive field.
    - Validation/test stages compute loss on trimmed (unpadded) regions to
      match the scope used by waveform metrics, ensuring consistency with
      reported MSE/MAE.

    Processor is inherited from BaseTrainer and validated in __init__.
    All stages now call ``processor.process(...)`` exactly once, relying on
    the processor to perform stage-aware trimming, scalar extraction, and
    metric computation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize trainer with config-driven processor and model.

        Raises:
            TypeError: If processor is passed positionally instead of via
                config or keyword.
            ValueError: If processor is None (processor is required).
        """
        super().__init__(*args, **kwargs)

        # Defensive check for legacy positional processor argument
        if not isinstance(self.trainer_name, str) or (
            len(args) > 0 and isinstance(args[0], OutputProcessor)
        ):
            raise TypeError(
                "Processor must be passed via config (trainer.processor) or "
                "as a keyword argument. Positional processor arguments are "
                "not supported. Please update your code to use: "
                "trainer.processor=<processor_instance> in config, or pass "
                "processor=<processor_instance> as a keyword argument."
            )

        if self.processor is None:
            raise ValueError(
                "WaveformReconstructionTrainer requires a processor. "
                "Specify via config: trainer.processor"
            )
        logger.info(
            "WaveformReconstructionTrainer initialized with processor: %s",
            type(self.processor).__name__,
        )

        logger.info(
            "Waveform reconstruction trainer initialized with model_name: "
            "%s, direction_mode: %s",
            self._get_model_name(),
            self.direction_mode.value,
        )

    def _get_main_output(self, outputs):
        """Handle different model output formats (legacy helper)."""
        if not isinstance(outputs, (tuple, list)):
            return outputs
        return outputs[0]

    def _extract_target_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract target per sample based on direction selection using tgt_idxs.

        This method extracts the correct target vital for each sample based
        on the direction assigned to that sample. Works with multi-channel
        batch["y"].

        Args:
            batch: Batch dictionary with:
                - "y": All target channels [B, C_targets, T]
                - "tgt_idxs": Target channel indices [B]

        Returns:
            torch.Tensor: Correct target for each sample [B, 1, T]
        """
        # [B, C_targets, T] - multi-channel targets
        y_targets = batch[BATCH_KEY_TARGET]
        # [B] - which channel is target for each sample
        tgt_idxs = batch[BATCH_KEY_TARGET_INDICES]
        batch_arange = torch.arange(y_targets.size(0), device=y_targets.device)

        # Extract correct target per sample from multi-channel tensor
        y_target = y_targets[batch_arange, tgt_idxs].unsqueeze(1)  # [B, 1, T]

        return y_target

    def _prepare_predictions_for_loss(
        self, processed: dict[str, Any], stage: str
    ) -> torch.Tensor:
        """Handle stage-aware prediction selection (raw vs trimmed).

        Args:
            processed: Processor output dict with predictions_raw and
                predictions.
            stage: Training stage (train/val/test).

        Returns:
            predictions_for_loss: Tensor to use for loss computation.

        Raises:
            KeyError: If required key ('predictions_raw' for train or
                'predictions' for val/test) is missing.
        """
        predictions_trimmed = processed.get(PROCESSOR_KEY_PREDICTIONS)
        predictions_raw = processed.get(PROCESSOR_KEY_PREDICTIONS_RAW)
        if stage == STAGE_TRAIN:
            if predictions_raw is None:
                raise KeyError(
                    "Training requires processor outputs['predictions_raw']."
                )
            return predictions_raw
        if predictions_trimmed is None:
            raise KeyError(f"{stage} stage requires processor outputs['predictions'].")
        return predictions_trimmed

    def _prepare_targets_for_loss(
        self, batch: dict[str, Any], processed: dict[str, Any], stage: str
    ) -> torch.Tensor:
        """Handle target extraction and trimming.

        Args:
            batch: Batch dict with y and tgt_idxs.
            processed: Processor output dict with padding_metadata.
            stage: Training stage (train/val/test).

        Returns:
            target_for_loss: Tensor to use for loss computation.

        Raises:
            KeyError: If batch lacks required keys (e.g. 'y', 'tgt_idxs').
            ValueError: If padding metadata or target shape is invalid.
        """
        y_target = self._extract_target_from_batch(batch)
        padding_metadata = processed.get(PROCESSOR_KEY_PADDING_METADATA)
        if not padding_metadata:
            extras = processed.get(PROCESSOR_KEY_EXTRAS)
            if isinstance(extras, dict):
                padding_metadata = extras.get(PROCESSOR_KEY_PADDING_METADATA)
        padding_length = extract_padding_length(padding_metadata, default=0)
        target_for_loss = y_target
        should_trim = stage != STAGE_TRAIN and padding_length > 0
        if should_trim:
            target_for_loss = trim_target_padding(target_for_loss, padding_length)
        return target_for_loss

    def _align_predictions_to_target(
        self,
        predictions_for_loss: torch.Tensor,
        target_for_loss: torch.Tensor,
        processed: dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """Trim predictions to target length when stage is val/test and padding used.

        Returns:
            predictions_for_loss: Tensor (possibly trimmed) matching target
                length.

        Raises:
            ValueError: If prediction length does not match padded or trimmed
                target length.
        """
        if stage == STAGE_TRAIN:
            return predictions_for_loss
        padding_metadata = processed.get(PROCESSOR_KEY_PADDING_METADATA)
        if not padding_metadata:
            extras = processed.get(PROCESSOR_KEY_EXTRAS)
            if isinstance(extras, dict):
                padding_metadata = extras.get(PROCESSOR_KEY_PADDING_METADATA)
        padding_length = extract_padding_length(padding_metadata, default=0)
        if padding_length <= 0:
            return predictions_for_loss
        original_target_length = target_for_loss.shape[-1] + (2 * padding_length)
        if predictions_for_loss.shape[-1] == original_target_length:
            return trim_target_padding(predictions_for_loss, padding_length)
        if predictions_for_loss.shape[-1] != target_for_loss.shape[-1]:
            expected_trimmed = compute_trimmed_length(
                original_target_length, padding_length
            )
            raise ValueError(
                "Mismatch between prediction length and padding metadata. "
                f"Received predictions with length "
                f"{predictions_for_loss.shape[-1]}, "
                f"expected {original_target_length} (padded) or "
                f"{expected_trimmed} (trimmed) "
                f"for padding_length={padding_length}."
            )
        return predictions_for_loss

    def _validate_prediction_target_shapes(
        self,
        predictions_for_loss: torch.Tensor,
        target_for_loss: torch.Tensor,
    ) -> None:
        """Validate shape compatibility between predictions and targets.

        Raises:
            ValueError: If prediction and target tensors have incompatible
                shapes.
        """
        validate_tensor_shapes_match(
            predictions_for_loss,
            target_for_loss,
            tensor1_name="predictions",
            tensor2_name="targets",
            error_context=(
                "Prediction and target tensors must share the same shape "
                "for loss computation."
            ),
        )

    def _compute_loss_and_metrics(
        self,
        predictions_for_loss: torch.Tensor,
        target_for_loss: torch.Tensor,
        processed: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and aggregate metrics.

        Args:
            predictions_for_loss: Predictions tensor for loss.
            target_for_loss: Target tensor for loss.
            processed: Processor output dict with metrics.

        Returns:
            loss: Scalar loss tensor.
            metrics: Dict of scalar metrics (loss + flattened processor
                metrics).
        """
        if self.criterion is None:
            raise RuntimeError("criterion must be set before _compute_loss_and_metrics")
        raw_loss = self.criterion(predictions_for_loss, target_for_loss)
        if isinstance(raw_loss, dict):
            loss = raw_loss["total_loss"]
        elif isinstance(raw_loss, tuple):
            loss = raw_loss[0]
        else:
            loss = raw_loss
        metrics = {METRIC_KEY_LOSS: float(loss.detach())}
        processor_metrics = processed.get(PROCESSOR_KEY_METRICS) or {}
        if processor_metrics:
            metrics.update(flatten_nested_metrics(processor_metrics))
        return loss, metrics

    def _step_core(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, float], Any]:
        """Pure compute: forward + processor.process() + loss + metrics.

        This method orchestrates the core computation pipeline by delegating
        to specialized helper methods for each stage.

        See Also:
            - _prepare_predictions_for_loss: Stage-aware prediction
              selection
            - _prepare_targets_for_loss: Target extraction and trimming
            - _align_predictions_to_target: Trim predictions to match target
              when val/test
            - _validate_prediction_target_shapes: Shape compatibility checks
            - _compute_loss_and_metrics: Loss computation and metric
              aggregation
        """
        if stage == STAGE_TRAIN and self.compute_full_metrics_during_train:
            cast("dict[str, Any]", batch)["_compute_full_metrics_during_train"] = True

        outputs = model(batch)
        processed = (
            self.processor.process(outputs, batch, stage=stage)
            if self.processor is not None
            else outputs
        )

        predictions_for_loss = self._prepare_predictions_for_loss(processed, stage)
        target_for_loss = self._prepare_targets_for_loss(batch, processed, stage)
        predictions_for_loss = self._align_predictions_to_target(
            predictions_for_loss, target_for_loss, processed, stage
        )
        self._validate_prediction_target_shapes(predictions_for_loss, target_for_loss)
        loss, metrics = self._compute_loss_and_metrics(
            predictions_for_loss, target_for_loss, processed
        )

        return loss, metrics, outputs

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prediction step leveraging processor.process(stage='test').

        Args:
            batch: Input batch dictionary

        Returns:
            Dict[str, Tensor]: Processed outputs with 'waveform' key and
                optional scalars

        Examples:
            >>> processed = trainer.predict_step(batch)
            >>> print(processed.keys())
            ... # ['waveform'] or ['waveform', 'sbp', 'dbp']
        """
        return super().predict_step(batch)

    # ============================================================================
    # EPOCH METHODS - Using shared implementation from BaseTrainer
    # ============================================================================

    def _train_epoch(self, epoch, model, train_loader, optim, device, master_process):
        """Training epoch using the shared runner from BaseTrainer."""
        return self._run_epoch(
            epoch,
            model,
            train_loader,
            device,
            master_process,
            STAGE_TRAIN,
            optim,
        )

    def _validate_epoch(self, epoch, model, val_loader, device, master_process):
        """Validate epoch using the shared runner from BaseTrainer."""
        return self._run_epoch(
            epoch, model, val_loader, device, master_process, STAGE_VAL
        )

    def _test_epoch(self, epoch, model, test_loader, device, master_process):
        """Test epoch using the shared runner from BaseTrainer."""
        return self._run_epoch(
            epoch, model, test_loader, device, master_process, STAGE_TEST
        )

    def _setup_training_metadata(self) -> None:
        """Initialize training state and log configuration."""
        self._vital_channel_mapping = self._build_vital_channel_mapping()
        mapping_display = {
            v.name: idx for v, idx in self._vital_channel_mapping.items()
        }
        logger.info(f"Using preprocessing-defined channel mapping: {mapping_display}")
        if self.direction_mode == DirectionMode.MULTI:
            if self.is_rank0:
                logger.info(
                    f"Training multi-directional model with directions: "
                    f"{[d.key() for d in self.directions]}"
                )
        else:
            if self.is_rank0:
                logger.info(
                    f"Training single-direction "
                    f"{[d.key() for d in self.directions]} model..."
                )
        current_epoch = self.get_epoch()
        if self.get_best_loss() is not None:
            logger.info(
                f"Resuming training from epoch {current_epoch} with "
                f"best_loss {self.get_best_loss()}"
            )

    def _run_training_epoch_cycle(
        self, epoch: int, train_loader, val_loader, test_loader
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], bool | None]:
        """Execute single epoch (train + validate + test).

        Returns train_metrics, val_metrics, test_metrics, loss_improved.
        """
        self.set_sampler_epoch(epoch)
        train_metrics = self._train_epoch(
            epoch,
            self.model,
            train_loader,
            self.optimizer,
            self.device,
            self.is_rank0,
        )
        loss_improved = None
        val_metric = None
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self._validate_epoch(
                epoch, self.model, val_loader, self.device, self.is_rank0
            )
            val_metric = val_metrics.get(METRIC_KEY_LOSS, 0.0)
        if val_metric is not None:
            self.step_scheduler(val_metric)
            if self.early_stopping:
                self.early_stopping(val_metric)
            loss_improved = self.set_best_loss(val_metric)
            if loss_improved:
                logger.info(f"New best loss: {val_metric:.6f}")
            self.update_training_state(epoch)
            if self.is_rank0:
                final_loss = val_metrics.get(METRIC_KEY_LOSS, 0.0)
                final_mse = val_metrics.get(METRIC_KEY_BASIC_MSE, 0.0)
                final_mae = val_metrics.get(METRIC_KEY_BASIC_MAE, 0.0)
                logger.info(
                    f"\nValidation Loss: {final_loss:.4f}, Basic MSE: "
                    f"{final_mse:.4f}, Basic MAE: {final_mae:.4f}"
                )
        test_metrics = {}
        if test_loader is not None:
            test_metrics = self._test_epoch(
                epoch, self.model, test_loader, self.device, self.is_rank0
            )
            if self.is_rank0:
                final_test_loss = test_metrics.get(METRIC_KEY_LOSS, 0.0)
                final_test_mse = test_metrics.get(METRIC_KEY_BASIC_MSE, 0.0)
                final_test_mae = test_metrics.get(METRIC_KEY_BASIC_MAE, 0.0)
                logger.info(
                    f"\nTest Loss: {final_test_loss:.4f}, Basic MSE: "
                    f"{final_test_mse:.4f}, Basic MAE: {final_test_mae:.4f}"
                )
        return train_metrics, val_metrics, test_metrics, loss_improved

    def _handle_epoch_checkpointing(
        self, epoch: int, val_metric: float | None, train_loader
    ) -> None:
        """Manage checkpoint saving logic (best model and periodic)."""
        if (
            val_metric is not None
            and self.get_best_loss() is not None
            and val_metric == self.get_best_loss()
        ):
            logger.info(
                f"Saving best model checkpoint with loss: {self.get_best_loss():.6f}"
            )
            opt = self.optimizer
            sched = self.scheduler
            es = self.early_stopping
            self.save_checkpoint(
                epoch=None,
                dataset=train_loader,
                optimizer_state_dict=(
                    {"optimizer": opt.state_dict()} if opt is not None else None
                ),
                scheduler_state_dict=(
                    sched.state_dict() if sched is not None else None
                ),
                early_stopping_state=(es.state_dict() if es is not None else None),
            )
        if (
            self.save_checkpoint_frequency is not None
            and epoch % self.save_checkpoint_frequency == 0
        ):
            logger.info(f"Saving periodic checkpoint at epoch {epoch}")
            opt = self.optimizer
            sched = self.scheduler
            es = self.early_stopping
            self.save_checkpoint(
                epoch=epoch,
                dataset=train_loader,
                optimizer_state_dict=(
                    {"optimizer": opt.state_dict()} if opt is not None else None
                ),
                scheduler_state_dict=(
                    sched.state_dict() if sched is not None else None
                ),
                early_stopping_state=(es.state_dict() if es is not None else None),
            )

    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute waveform reconstruction training via BaseTrainer infrastructure."""
        self._setup_training_metadata()
        current_epoch = self.get_epoch()
        for epoch in range(current_epoch, self.num_epochs):
            train_metrics, val_metrics, test_metrics, loss_improved = (
                self._run_training_epoch_cycle(
                    epoch, train_loader, val_loader, test_loader
                )
            )
            self.log_epoch_metrics_unified(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                best_loss=self.get_best_loss(),
                loss_improved=loss_improved,
            )
            if self.check_early_stopping_common():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            val_metric = val_metrics.get(METRIC_KEY_LOSS) if val_metrics else None
            self._handle_epoch_checkpointing(epoch, val_metric, train_loader)
        if self.is_rank0:
            logger.info("Waveform reconstruction training completed")
            self.close_all()


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_waveform_reconstruction_trainer",
    node=WaveformReconstructionTrainerConfig,
    group="trainer",
)
