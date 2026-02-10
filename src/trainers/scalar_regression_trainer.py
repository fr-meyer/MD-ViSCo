"""Scalar regression trainer for BP prediction and other scalar regression
tasks.

This module provides ScalarRegressionTrainer, a trainer specialized for
models that output scalar predictions (e.g., blood pressure values, heart
rate estimates). It handles the complete training, validation, and testing
workflow for scalar regression models.

**Supported Models:**
- PatchTST: Direct scalar prediction (num_targets: 2 for SBP/DBP)
- ShallowUNetBP: Direct scalar prediction
- MDViSCo BPModel: Waveform output → BP extractor computes scalars
- NABNet BP Cascade: Two-stage model with scalar dict output

**Key Features:**
- Unified loss computation using scalar predictions vs scalar targets
- Processor-based output parsing and metric computation
- Support for both single-direction and multi-directional training
- Distributed training support via PyTorch DDP

**Exports:**
- ScalarRegressionTrainer: Main trainer class
- ScalarRegressionTrainerConfig: Hydra-compatible configuration dataclass
"""

# Standard library imports
import logging
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
from src.utils.bp import get_dbp
from src.utils.bp import get_sbp

logger = logging.getLogger(__name__)


@dataclass
class ScalarRegressionTrainerConfig(TrainerBaseConfig):
    """Configuration for ScalarRegressionTrainer with Hydra-compatible defaults.

    This trainer works with any model that outputs scalar predictions.
    Compatible models: PatchTST, ShallowUNetBP, MDViSCo BPModel
    """

    _target_: str = "src.trainers.scalar_regression_trainer.ScalarRegressionTrainer"
    trainer_name: str = "ScalarReg"

    # Scalar regression-specific configuration
    input_preprocessing: dict[str, Any] = MISSING


class ScalarRegressionTrainer(BaseTrainer):
    """Trainer for scalar regression tasks (BP prediction, HR estimation, etc.).

    **STRICT REQUIREMENTS:**
    - Model outputs must include 'predictions' tensor (scalar predictions)
    - Batch must contain normalized target 'y' [B, 3] when BP is target
      (from collate function)
    - Criteria must accept: input (scalar predictions), target (scalar y),
      **kwargs (optional context)

    **LOSS COMPUTATION CONTRACT:**
    - Loss is computed using scalar predictions vs scalar target 'y'
    - Waveform targets (y_waveform) may be passed in ground_truth kwargs as
      optional context
    - Criteria should NOT rely on y_pred_waveform/y_waveform-only loss paths
    - For waveform-supervised loss, use WaveformReconstructionTrainer instead

    **COMPATIBLE MODELS:**
    - PatchTST: Direct scalar prediction (num_targets: 2 for SBP/DBP)
    - ShallowUNetBP: Direct scalar prediction
    - MDViSCo BPModel: Waveform output -> BP extractor computes scalars
    - NABNet BP Cascade: Two-stage model with scalar dict output

    **TRAINING MODES:**
    - Single-direction (direction_mode="single"): Required for NABNet, PatchTST
    - Multi-direction (direction_mode="multi"): Supported by MDViSCo refinement models

    **PROCESSOR:**
    - Uses ScalarOutputProcessor (or variant) for scalar prediction tasks
    - Processor extracts scalars from model outputs and provides metrics
    - All stages call processor.process(...) exactly once for unified metrics

    **CRITERION REQUIREMENTS:**
    Criteria used with this trainer must:
    1. Accept 'input' parameter (scalar predictions tensor)
    2. Accept 'target' parameter (scalar ground truth from ground_truth['y'])
    3. Accept **kwargs for optional context (embeddings, waveform targets, demographics)
    4. NOT rely on removed y_pred_waveform/y_waveform-only loss paths
    5. If waveform loss is needed, compute it from waveform tensors in kwargs
       (e.g., from model_outputs or ground_truth['y_waveform'])

    For waveform-supervised experiments, migrate to WaveformReconstructionTrainer
    or explicitly update the criterion to compute waveform loss from kwargs.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ScalarRegressionTrainer with config-driven processor and model.

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
                "Processor must be passed via config (trainer.processor) or as "
                "a keyword argument. Positional processor arguments are not "
                "supported. Please update your code to use: "
                "trainer.processor=<processor_instance> in config, or pass "
                "processor=<processor_instance> as a keyword argument."
            )

        if self.processor is None:
            raise ValueError(
                "ScalarRegressionTrainer requires a processor. Specify via "
                "config: trainer.processor"
            )
        logger.info(
            "ScalarRegressionTrainer initialized with processor: %s",
            type(self.processor).__name__,
        )

        logger.info("Scalar regression trainer initialized")

    def _validate_configuration(self):
        """Validate trainer configuration.

        No-op for scalar regression; no extra checks required.
        """
        pass

    def _extract_target_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract target from unified batch structure for loss computation.

        ScalarRegressionTrainer strictly requires a scalar target 'y' for
        loss computation. When BP is the target, batch["y"] contains
        normalized BP [B, 3] (SBP, DBP, MAP). This method extracts scalar
        BP targets from batch["y"].

        Args:
            batch: Batch dictionary with unified structure containing:
                - 'y': Normalized BP targets [B, 3] when BP is target
                  (REQUIRED)
                - 'bp_raw': Raw BP values [B, 3] (optional, for
                  metrics/denormalization only)

        Returns:
            Dict[str, torch.Tensor]: Target dictionary for loss computation
                with required 'y' key.
                - 'y': Scalar target tensor [B, 2] for SBP/DBP (REQUIRED)
                - 'y_sbp': SBP values [B] (from normalized targets)
                - 'y_dbp': DBP values [B] (from normalized targets)
                - 'y_sbp_raw', 'y_dbp_raw': Raw BP values [B] (optional,
                  from bp_raw if available)
                - 'y_waveform': Waveform target [B, 1, T] (optional, passed
                  as context only)

        Raises:
            ValueError: If batch["y"] is not present or has incorrect shape
                for BP targets
        """
        target = {}

        # When BP is the target, batch["y"] contains normalized BP [B, 3]
        # (SBP, DBP, MAP)
        if "y" not in batch:
            raise ValueError(
                "ScalarRegressionTrainer requires batch['y'] for BP targets. "
                "Ensure the collate function provides normalized targets in batch['y']."
            )

        y_bp = batch["y"]

        # Remove only extra leading batch dimensions while preserving target dimension
        while y_bp.dim() > 2 and y_bp.size(0) == 1:
            y_bp = y_bp.squeeze(0)
        if y_bp.dim() == 1:
            y_bp = y_bp.unsqueeze(0)  # Handle single sample case

        if y_bp.dim() != 2 or y_bp.size(1) != 3:
            raise ValueError(
                f"Expected batch['y'] to have shape [B, 3] for BP targets, "
                f"but got shape {tuple(y_bp.shape)}. "
                f"For waveform targets, use WaveformReconstructionTrainer instead."
            )

        # Extract SBP and DBP from normalized BP target
        target["y_sbp"] = y_bp[:, 0]  # Column 0: SBP
        target["y_dbp"] = y_bp[:, 1]  # Column 1: DBP
        target["y"] = y_bp[:, 0:2]  # [B, 2] - SBP and DBP only

        # Extract raw BP values if available (for metrics/denormalization, not for loss)
        if "bp_raw" in batch:
            bp_raw = batch["bp_raw"]
            # Handle potential extra dimensions
            if bp_raw.dim() > 2:
                bp_raw = bp_raw.squeeze()
            if bp_raw.dim() == 1:
                bp_raw = bp_raw.unsqueeze(0)
            target["y_sbp_raw"] = get_sbp(bp_raw)
            target["y_dbp_raw"] = get_dbp(bp_raw)

        # Extract waveform target if available (optional context only,
        # not used for loss)
        # Criteria should not rely on y_waveform for loss computation;
        # if waveform loss is needed, use WaveformReconstructionTrainer or
        # update criterion to compute waveform loss from waveform tensors
        # passed in kwargs (e.g., from model_outputs or ground_truth)
        if "y_abp" in batch:
            target["y_waveform"] = batch["y_abp"]

        return target

    def _get_main_output(self, outputs) -> torch.Tensor:
        """Extract main output from v3 canonical format.

        Args:
            outputs: Model outputs (tensor or dict with 'predictions' key)

        Returns:
            torch.Tensor: Main output tensor for loss computation
        """
        if isinstance(outputs, dict):
            if "predictions" in outputs:
                return outputs["predictions"]
            else:
                keys_str = str(list(outputs.keys()))
                raise ValueError(
                    f"Dict outputs must include 'predictions' key. Got keys: {keys_str}"
                )
        elif isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, (list, tuple)):
            return outputs[0]
        else:
            output_type = type(outputs)
            raise TypeError(
                f"Unexpected output format: {output_type}. Expected dict with "
                "'predictions', tensor, list, or tuple."
            )

    def _compute_loss(self, model_outputs, batch, model, ground_truth):
        """Unified loss computation for all stages (train/val/test).

        ScalarRegressionTrainer strictly requires:
        - 'predictions' tensor in model_outputs (scalar predictions)
        - 'y' scalar target in ground_truth (from _extract_target_from_batch)

        Criteria should accept:
        - input: scalar predictions tensor
        - target: scalar ground truth tensor (ground_truth['y'])
        - **kwargs: optional context including waveform targets, embeddings,
          demographics

        This method replaces the old compute_training_loss and works for
        both training and validation stages.

        Args:
            model_outputs: Model predictions and embeddings (must include 'predictions')
            batch: Input batch containing targets and demographic fields
            model: Model instance
            ground_truth: Target dictionary with required 'y' key (scalar target)

        Returns:
            torch.Tensor: Computed loss

        Raises:
            KeyError: If 'predictions' not in model_outputs or 'y' not in ground_truth
            ValueError: If 'y' target is not scalar (e.g., waveform-shaped)
        """
        predictions = model_outputs.get("predictions")
        if predictions is None:
            raise KeyError(
                "Model outputs must include 'predictions' for scalar loss computation."
            )

        if "y" not in ground_truth:
            raise KeyError(
                "ScalarRegressionTrainer requires scalar target 'y' in "
                "ground_truth. Ensure _extract_target_from_batch() provides "
                "'y' from batch['y'] (normalized BP targets). "
                "For waveform-supervised loss, use "
                "WaveformReconstructionTrainer instead."
            )

        y_target = ground_truth["y"]

        # Scalar targets should be 1D [B] or 2D [B, num_targets],
        # not 3D [B, C, T]
        if y_target.dim() >= 3:
            raise ValueError(
                f"ScalarRegressionTrainer requires scalar target 'y', but "
                f"received waveform-shaped tensor with shape "
                f"{tuple(y_target.shape)}. Expected shape [B] or "
                f"[B, num_targets]. For waveform-supervised loss, use "
                f"WaveformReconstructionTrainer instead."
            )

        # Extract demographic fields from batch for WCL loss computation
        # These are treated as "inputs" (context) rather than "targets" (supervision)
        demo_keys = ["age_raw", "gender_raw", "height_raw", "weight_raw", "bmi_raw"]
        demo_dict = {
            k: batch[k] for k in demo_keys if k in batch and torch.is_tensor(batch[k])
        }

        # Merge all kwargs: model_outputs (embeddings + predictions),
        # ground_truth (BP targets), demographics (context)
        # Criteria should accept input (scalar predictions), target (scalar y),
        # and any waveform targets passed in **ground_truth as optional
        # context (e.g., y_waveform)
        if self.criterion is None:
            raise RuntimeError("criterion must be set before _compute_loss")
        raw = self.criterion(
            input=predictions,
            target=y_target,
            **model_outputs,
            **ground_truth,
            **demo_dict,
        )
        if isinstance(raw, dict):
            return raw["total_loss"]
        if isinstance(raw, tuple):
            return raw[0]
        return raw

    # ============================================================================
    # UNIFIED LOSS AND METRICS COMPUTATION METHODS
    # ============================================================================

    def _step_core(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, float], Any]:
        """Pure compute: parse, forward, target, loss, metrics. No optimizer/AMP.

        Compute both loss and metrics for all stages. Follows the same pattern
        as WaveformReconstructionTrainer for consistency.

        Uses OutputProcessor for output parsing and metrics via the unified
        ``processor.process()`` method.

        Padding Handling:
            Loss is computed on PADDED tensors (both prediction and target)
            without trimming. This follows industry best practices:
            - HuggingFace Transformers: Pad inputs/targets, use attention masks
            - PyTorch NLP: Pad sequences, mask loss for padding tokens
            - Fairseq: Pad to max sequence length in batch

            Benefits of this approach:
            1. Consistent tensor shapes -> better GPU utilization
            2. Small padding ratio (e.g., 30/1280 = 2.3%) -> negligible
               gradient impact
            3. Simpler training code -> no conditional logic
            4. Clean separation -> trimming handled by
               `processor.process(..., stage="val"/"test")`

            The processor now uses a unified `process()` interface:
            - `stage="train"`: minimal parsing, no trimming unless explicitly requested
            - `stage="val"/"test"`: full trimming and metric computation

            See processor class docstrings for complete pipeline documentation.

        Basic MAE/MSE values originate from processors (metrics.basic) so the trainer
        only surfaces whatever metrics the processor returns, ensuring a single source
        of truth for logging.

        Args:
            model: The model (DDP-wrapped or not)
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")

        Returns:
            loss: Computed loss
            metrics: Dictionary of metrics
            outputs: Model outputs
        """
        if stage == "train" and self.compute_full_metrics_during_train:
            cast("dict[str, Any]", batch)["_compute_full_metrics_during_train"] = True

        outputs = model(batch)
        processed_outputs = (
            self.processor.process(outputs, batch, stage=stage)
            if self.processor is not None
            else {"predictions": outputs}
        )
        parsed_output = processed_outputs["predictions"]

        # Extract target using centralized method
        y_target = self._extract_target_from_batch(batch)

        # Build model_outputs dict for loss computation
        # If model returns dict (with embeddings), use it; otherwise wrap parsed tensor
        if isinstance(outputs, dict):
            model_outputs_for_loss = dict(outputs)
            primary_predictions = self._get_main_output(outputs)
            model_outputs_for_loss.setdefault("predictions", primary_predictions)
        else:
            model_outputs_for_loss = {"predictions": parsed_output}

        loss = self._compute_loss(model_outputs_for_loss, batch, model, y_target)

        metrics = {"loss": float(loss.detach())}
        processor_metrics = processed_outputs.get("metrics") or {}
        if processor_metrics:
            metrics.update(self._summarize_processor_metrics(processor_metrics))

        return loss, metrics, outputs

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prediction step with full post-processing for inference/evaluation.

        This method is called by evaluators (Phase 8) to get fully processed outputs.
        including trimmed waveforms and extracted scalar values (SBP/DBP).

        Args:
            batch: Input batch dictionary with unified structure

        Returns:
            Dict[str, Tensor]: Processed outputs with keys:
                - 'waveform': Trimmed waveform predictions [B, C, T]
                - 'sbp': Systolic blood pressure [B, 1] (if extract_scalars=True)
                - 'dbp': Diastolic blood pressure [B, 1] (if extract_scalars=True)

        Examples:
            >>> # Called by evaluator
            >>> batch = test_loader.next()
            >>> processed = trainer.predict_step(batch)
            >>> print(processed.keys())  # ['waveform', 'sbp', 'dbp']

        Note:
            This method is used during evaluation/testing. Training uses
            `_step_core()` which calls `processor.process(..., stage=stage)`
            once per batch.
        """
        return super().predict_step(batch)

    def _summarize_processor_metrics(
        self, processor_metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Convert processor metric tensors into scalar summaries for logging."""
        summarized: dict[str, float] = {}

        def _flatten(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for name, nested in value.items():
                    next_prefix = f"{prefix}_{name}" if prefix else name
                    _flatten(next_prefix, nested)
                return
            if torch.is_tensor(value):
                summarized[prefix] = float(value.mean().detach())
            elif isinstance(value, (float, int)):
                summarized[prefix] = float(value)
            elif value is None:
                return

        for group, values in processor_metrics.items():
            _flatten(group, values)

        return summarized

    # ============================================================================
    # EPOCH METHODS - Using shared implementation from BaseTrainer
    # ============================================================================

    def _on_epoch_end(
        self, stage: str, stage_metrics: dict[str, float], master_process: bool
    ):
        """Override to add validation-specific logging for scalar regression."""
        # Print validation results for user feedback (matching legacy behavior)
        if stage == "val" and master_process:
            logger.info("\nValidation Results:")
            scalar_mse = stage_metrics.get("loss", 0.0)
            scalar_mae = stage_metrics.get("basic_mae", 0.0)
            logger.info(f"Scalar: MSE={scalar_mse:.4f}, MAE={scalar_mae:.4f}")
            if "bp_sbp_mae" in stage_metrics:
                bp_sbp_mae = stage_metrics["bp_sbp_mae"]
                bp_dbp_mae = stage_metrics["bp_dbp_mae"]
                logger.info(f"BP: SBP MAE={bp_sbp_mae:.2f}, DBP MAE={bp_dbp_mae:.2f}")
                if "bp_sbp_mse" in stage_metrics and "bp_dbp_mse" in stage_metrics:
                    bp_sbp_mse = stage_metrics["bp_sbp_mse"]
                    bp_dbp_mse = stage_metrics["bp_dbp_mse"]
                    logger.info(
                        f"BP: SBP MSE={bp_sbp_mse:.2f}, DBP MSE={bp_dbp_mse:.2f}"
                    )

    def _train_epoch(self, epoch, model, train_loader, optim, device, master_process):
        """Training epoch using the shared runner from BaseTrainer."""
        return self._run_epoch(
            epoch, model, train_loader, device, master_process, "train", optim
        )

    def _validate_epoch(self, epoch, model, val_loader, device, master_process):
        """Run validation epoch using the shared runner from BaseTrainer."""
        return self._run_epoch(epoch, model, val_loader, device, master_process, "val")

    def _test_epoch(self, epoch, model, test_loader, device, master_process):
        """Test epoch using the shared runner from BaseTrainer."""
        return self._run_epoch(
            epoch, model, test_loader, device, master_process, "test"
        )

    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute scalar regression training using provided infrastructure
        from BaseTrainer."""
        # Cache vital channel mapping once (optimization: same for all splits)
        self._vital_channel_mapping = self._build_vital_channel_mapping()
        mapping_display = {
            v.name: idx for v, idx in self._vital_channel_mapping.items()
        }
        logger.info(f"Using preprocessing-defined channel mapping: {mapping_display}")

        # Determine training direction
        if self.direction_mode == DirectionMode.MULTI:
            if self.is_rank0:
                dir_keys = [d.key() for d in self.directions]
                logger.info(
                    f"Training multi-directional model with directions: {dir_keys}"
                )
        else:
            # Single direction training: use first available direction
            if self.is_rank0:
                dir_keys = [d.key() for d in self.directions]
                logger.info(f"Training single-direction {dir_keys} model...")

        # -------- Training Loop --------
        current_epoch = self.get_epoch()
        if self.get_best_loss() is not None:
            best_loss_val = self.get_best_loss()
            logger.info(
                f"Resuming training from epoch {current_epoch} with "
                f"best_loss {best_loss_val}"
            )

        for epoch in range(current_epoch, self.num_epochs):
            # Use base trainer's method to set sampler epoch (training only)
            self.set_sampler_epoch(epoch)

            # -------- Train --------
            # Layout setting is handled by _run_epoch in BaseTrainer
            train_metrics = self._train_epoch(
                epoch,
                self.model,
                train_loader,
                self.optimizer,
                self.device,
                self.is_rank0,
            )

            # -------- Validate --------
            val_metric = None
            val_metrics = {}
            if val_loader is not None:
                # Layout setting is handled by _run_epoch in BaseTrainer
                val_metrics = self._validate_epoch(
                    epoch, self.model, val_loader, self.device, self.is_rank0
                )
                val_metric = val_metrics.get(
                    "loss", 0.0
                )  # Use loss as the validation metric for scheduler

            # -------- Scheduler / logging / checkpointing --------
            loss_improved = None
            if val_metric is not None:
                self.step_scheduler(val_metric)

                if self.early_stopping:
                    self.early_stopping(val_metric)
                loss_improved = self.set_best_loss(val_metric)
                if loss_improved:
                    logger.info(f"New best loss: {val_metric:.6f}")
                self.update_training_state(epoch)

                if self.is_rank0:
                    final_loss = val_metrics.get("loss", 0.0)
                    final_mse = val_metrics.get("basic_mse", 0.0)
                    final_mae = val_metrics.get("basic_mae", 0.0)
                    logger.info(
                        f"\nValidation Loss: {final_loss:.4f}, "
                        f"Basic MSE: {final_mse:.4f}, Basic MAE: {final_mae:.4f}"
                    )

            # -------- Test Phase (if test_loader provided) --------
            test_metrics = {}
            if test_loader is not None:
                # Layout setting is handled by _run_epoch in BaseTrainer
                test_metrics = self._test_epoch(
                    epoch, self.model, test_loader, self.device, self.is_rank0
                )

                if self.is_rank0:
                    final_test_loss = test_metrics.get("loss", 0.0)
                    final_test_mse = test_metrics.get("basic_mse", 0.0)
                    final_test_mae = test_metrics.get("basic_mae", 0.0)
                    logger.info(
                        f"\nTest Loss: {final_test_loss:.4f}, "
                        f"Basic MSE: {final_test_mse:.4f}, "
                        f"Basic MAE: {final_test_mae:.4f}"
                    )

            self.log_epoch_metrics_unified(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                best_loss=self.get_best_loss(),
                loss_improved=loss_improved,  # Pass the improvement status
            )

            if self.check_early_stopping_common():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if (
                val_metric is not None
                and self.get_best_loss() is not None
                and val_metric == self.get_best_loss()
            ):
                best_loss_val = self.get_best_loss()
                logger.info(
                    f"Saving best model checkpoint with loss: {best_loss_val:.6f}"
                )
                self.save_checkpoint(
                    epoch=None,  # Best-only slot
                    dataset=train_loader,
                    optimizer_state_dict=(
                        {"optimizer": self.optimizer.state_dict()}
                        if self.optimizer is not None
                        else None
                    ),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler else None
                    ),
                    early_stopping_state=(
                        self.early_stopping.state_dict()
                        if self.early_stopping
                        else None
                    ),
                )

            # Periodic checkpointing (rank-0 only, barrier handled in base)
            if (
                self.save_checkpoint_frequency is not None
                and epoch % self.save_checkpoint_frequency == 0
            ):
                logger.info(f"Saving periodic checkpoint at epoch {epoch}")
                self.save_checkpoint(
                    epoch=epoch,  # Periodic slot
                    dataset=train_loader,
                    optimizer_state_dict=(
                        {"optimizer": self.optimizer.state_dict()}
                        if self.optimizer is not None
                        else None
                    ),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler else None
                    ),
                    early_stopping_state=(
                        self.early_stopping.state_dict()
                        if self.early_stopping
                        else None
                    ),
                )

        if self.is_rank0:
            logger.info("Scalar regression training completed")
            self.close_all()


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_scalar_regression_trainer",
    node=ScalarRegressionTrainerConfig,
    group="trainer",
)
