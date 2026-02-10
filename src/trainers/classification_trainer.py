"""Classification Trainer.

This trainer provides a generic classification trainer for any classification task
while preserving the existing data pipeline and evaluation logic.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

# Local imports
from src.loggings.metrics import metrics
from src.processors import OutputProcessor
from src.trainers.trainer import BaseTrainer
from src.trainers.trainer import TrainerBaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ClassificationTrainerConfig(TrainerBaseConfig):
    """Configuration for ClassificationTrainer with Hydra-compatible defaults.

    This trainer works with any v3 classification model.
    Compatible models: AFClassifier (preferred) and custom classification models
    registered via Hydra.
    """

    _target_: str = "src.trainers.classification_trainer.ClassificationTrainer"
    trainer_name: str = "Classify"

    # Data preprocessing configuration
    input_preprocessing: dict[str, Any] = field(
        default_factory=lambda: {
            "source": {"vital": "ppg", "norm": "minmax_zc"},
            "target": {"vital": "ecg", "norm": "minmax_zc"},
            "class_labels": "af_labels",
        }
    )

    # No additional parameters needed - everything comes from BaseTrainer
    # - num_epochs: from BaseTrainer
    # - save_checkpoint_frequency: from BaseTrainer
    # - learning_rate: from BaseTrainer
    # - scheduler_patience: from BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """Generic classification trainer that follows the standard BaseTrainer pattern.

    This trainer provides a generic classification trainer for any classification task
    while preserving the existing data pipeline and evaluation logic.

    **IMPORTANT: Currently supports binary classification only.**
    Multi-class classification is not supported due to AUC computation limitations.

    Architecture:
    - self.model: Classification model (loaded via Hydra)
    - self.criterion: Classification loss (loaded via Hydra)
    - self.early_stopping: EarlyStopping (loaded via Hydra)
    - self.optimizer: Adam optimizer (created by BaseTrainer after DDP wrapping)
    - self.scheduler: ReduceLROnPlateau scheduler (created by BaseTrainer after DDP
        wrapping)
    - self.processor: OutputProcessor (inherited from BaseTrainer, loaded via Hydra)

    The trainer uses the standard BaseTrainer infrastructure for optimizer/scheduler
    creation, ensuring DDP compatibility and consistency with other trainers.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ClassificationTrainer with config-driven processor and model.

        Raises:
            TypeError: If processor is passed positionally instead of via config
                or keyword.
            ValueError: If processor is None (processor is required).
        """
        super().__init__(*args, **kwargs)

        # Defensive check for legacy positional processor argument
        if not isinstance(self.trainer_name, str) or (
            len(args) > 0 and isinstance(args[0], OutputProcessor)
        ):
            raise TypeError(
                "Processor must be passed via config (trainer.processor) or as a "
                "keyword argument. "
                "Positional processor arguments are not supported. "
                "Please update your code to use: "
                "trainer.processor=<processor_instance> in config, "
                "or pass processor=<processor_instance> as a keyword argument."
            )

        if self.processor is None:
            raise ValueError(
                "ClassificationTrainer requires a processor. Specify via config: "
                "trainer.processor"
            )
        logger.info(
            "ClassificationTrainer initialized with processor %s",
            type(self.processor).__name__,
        )

        logger.info(
            "Classification trainer initialized with model: %s",
            type(self.model).__name__,
        )

    def _extract_target_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract target from unified batch structure for loss computation.

        For classification, we extract the input signal which contains
        classification information.

        Args:
            batch: Batch dictionary with unified structure

        Returns:
            torch.Tensor: Target signal tensor [B, 1, T] containing classification
                information
        """
        # Extract target using the same pattern as other trainers
        waveform = batch["x"]  # All channels [B, C, T]
        tgt_idxs = batch["tgt_idxs"]  # Target channel indices [B]
        batch_arange = torch.arange(waveform.size(0), device=waveform.device)
        y_target = waveform[batch_arange, tgt_idxs].unsqueeze(1)  # [B, 1, T]

        return y_target

    def _get_main_output(self, outputs) -> torch.Tensor:
        """Extract main output from v3 canonical format.

        Args:
            outputs: Model outputs (tensor or dict with 'predictions' key)

        Returns:
            torch.Tensor: Main output tensor
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, dict):
            if "predictions" in outputs:
                return outputs["predictions"]
            else:
                keys_str = str(list(outputs.keys()))
                raise ValueError(
                    f"Dict outputs must include 'predictions' key. Got keys: {keys_str}"
                )
        else:
            output_type = type(outputs)
            raise ValueError(
                f"Unexpected output format: {output_type}. Expected tensor or "
                "dict with 'predictions' key."
            )

    def _step_core(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, float], Any]:
        """Core step following standard PyTorch pattern.

        Uses OutputProcessor for output parsing and post-processing.

        Args:
            model: The classification model (self.model)
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")

        Returns:
            loss: Computed loss
            metrics: Dictionary of metrics
            outputs: Model outputs

        Raises:
            ValueError: If processor outputs lack required keys or output format
                is invalid.
            KeyError: If processor contract is violated (e.g. 'class_predictions'
                missing).
        """
        # Extract class labels
        class_labels = batch["class_labels"].long()

        # Forward pass through model
        model_output = model(batch)

        processed_outputs = (
            self.processor.process(model_output, batch, stage=stage)
            if self.processor is not None
            else {"predictions": model_output}
        )
        logits = processed_outputs["predictions"]

        if self.criterion is None:
            raise RuntimeError("criterion must be set before _step_core")
        raw_loss = self.criterion(logits, class_labels)
        if isinstance(raw_loss, dict):
            loss = raw_loss["total_loss"]
        elif isinstance(raw_loss, tuple):
            loss = raw_loss[0]
        else:
            loss = raw_loss

        # Extract predictions and probabilities from processor
        with torch.no_grad():
            try:
                predictions = processed_outputs["class_predictions"]
            except KeyError as exc:
                raise ValueError(
                    "ClassificationOutputProcessor contract violation: "
                    "'class_predictions' key missing from outputs. "
                    "Ensure processor.process() returns standardized outputs."
                ) from exc

            try:
                probabilities = processed_outputs["probabilities"]
            except KeyError as exc:
                raise ValueError(
                    "ClassificationOutputProcessor contract violation: "
                    "'probabilities' key missing from outputs. "
                    "Ensure processor.process() returns standardized outputs."
                ) from exc

            batch_size = class_labels.size(0)

            if predictions.dim() != 1 or predictions.size(0) != batch_size:
                pred_shape_str = str(tuple(predictions.shape))
                raise ValueError(
                    f"Expected class_predictions shape [B], got {pred_shape_str}. "
                    "Processor contract violation."
                )

            if (
                probabilities.dim() != 2
                or probabilities.size(0) != batch_size
                or probabilities.size(1) != 2
            ):
                prob_shape = tuple(probabilities.shape)
                raise ValueError(
                    f"Expected probabilities shape [B, 2], got {prob_shape}. "
                    "Processor contract violation."
                )

            if predictions.dtype not in (torch.int32, torch.int64, torch.long):
                pred_dtype = predictions.dtype
                raise ValueError(
                    f"Expected class_predictions dtype integer/long, got "
                    f"{pred_dtype}. Processor contract violation."
                )

            if not torch.is_floating_point(probabilities):
                prob_dtype = probabilities.dtype
                raise ValueError(
                    f"Expected probabilities dtype floating point, got "
                    f"{prob_dtype}. Processor contract violation."
                )

            accuracy = (predictions == class_labels).float().mean()
            f1_score = self._compute_f1_score(predictions, class_labels)

        metrics = {
            "loss": float(loss.detach()),
            "accuracy": float(accuracy.detach()),
            "f1_score": f1_score,
        }

        outputs = {
            "y_pred": logits,  # [B, 1] - for loss computation
            "predictions": predictions,  # [B] - for accuracy/F1
            "probabilities": probabilities,  # [B, 2] - for AUC computation
        }

        return loss, metrics, outputs

    def _run_epoch(
        self, epoch, model, data_loader, device, master_process, stage: str, optim=None
    ):
        """Run generic epoch for both training and validation.

        Args:
            epoch: Current epoch number
            model: The model to run
            data_loader: DataLoader for the stage
            device: Device to run on
            master_process: Whether this is the master process
            stage: Stage name ("train", "val", or "test")
            optim: Optimizer (required for training, None for validation/test)

        Raises:
            ValueError: If stage is not one of "train", "val", or "test".
        """
        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Stage must be 'train', 'val', or 'test', got {stage}")

        self.metrics.reset_meters(stage)

        if self.is_rank0:
            if stage == "train":
                self.create_training_progress_bar(data_loader, epoch, master_process)
            elif stage == "val":
                self.create_validation_progress_bar(data_loader, epoch, master_process)
            else:  # test
                self.create_test_progress_bar(data_loader, epoch, master_process)

        if stage == "train":
            model.train()
            context_manager = torch.enable_grad()
        else:  # val or test
            model.eval()
            context_manager = torch.no_grad()

        epoch_metrics: dict[str, float] = {}
        try:
            with context_manager:
                if stage in ["val", "test"]:
                    # For validation/test, collect all data for epoch-level metrics
                    all_predictions = []
                    all_labels = []
                    all_probabilities = []
                    total_loss = 0.0
                    num_batches = 0

                    for step, batch in enumerate(data_loader):
                        prepared_batch = {
                            k: (
                                v.to(device, non_blocking=True)
                                if torch.is_tensor(v)
                                else v
                            )
                            for k, v in batch.items()
                        }

                        # Use prepared_batch with unified structure
                        loss, metrics, outputs = self._step_core(
                            model, prepared_batch, stage=stage
                        )

                        class_labels = prepared_batch["class_labels"].long()

                        all_predictions.append(outputs["predictions"].cpu())
                        all_labels.append(class_labels.cpu())
                        # Binary classification: use positive class probability
                        # (column 1) for ROC-AUC computation.
                        all_probabilities.append(outputs["probabilities"][:, 1].cpu())

                        total_loss += float(loss.detach())
                        num_batches += 1

                        # Use unified metrics logging (every rank)
                        with self.metrics.aggregate(stage):
                            # Use metrics directly from _step_core (loss is
                            # already included)
                            self.log_step_metrics_unified(metrics, step)

                        if self.is_rank0:
                            current_metrics = self.metrics.get_smoothed_values(stage)
                            to_log = stage == "train"
                            self.update_progress_bar(
                                current_metrics,
                                step,
                                is_rank0=self.is_rank0,
                                to_log=to_log,
                            )

                        # Clear memory (avoid per-batch empty_cache to prevent
                        # fragmentation)
                        del batch, prepared_batch, outputs, loss, metrics

                    all_predictions = torch.cat(all_predictions, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                    all_probabilities = torch.cat(all_probabilities, dim=0)

                    # Binary AF assumption: num_classes == 2 and probabilities[:, 1]
                    # stores the positive class.
                    roc_auc, pr_auc = self._compute_epoch_level_aucs(
                        all_probabilities, all_labels
                    )

                    epoch_f1 = self._compute_f1_score(all_predictions, all_labels)
                    epoch_accuracy = (all_predictions == all_labels).float().mean()
                    epoch_loss = total_loss / num_batches

                    # Override the aggregated metrics with epoch-level calculations
                    loss_val = (
                        float(epoch_loss.item())
                        if isinstance(epoch_loss, torch.Tensor)
                        else float(epoch_loss)
                    )
                    epoch_metrics: dict[str, float] = {
                        "loss": loss_val,
                        "accuracy": float(epoch_accuracy),
                        "f1_score": float(epoch_f1),
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                    }

                    if self.is_rank0:
                        self.update_progress_bar(
                            epoch_metrics,
                            len(data_loader) - 1,
                            is_rank0=self.is_rank0,
                            to_log=False,
                        )

                else:  # Training stage
                    if optim is None:
                        raise RuntimeError("optimizer is required for training stage")
                    for step, batch in enumerate(data_loader):
                        prepared_batch = {
                            k: (
                                v.to(device, non_blocking=True)
                                if torch.is_tensor(v)
                                else v
                            )
                            for k, v in batch.items()
                        }

                        # Use prepared_batch with unified structure
                        loss, metrics, outputs = self._step_core(
                            model, prepared_batch, stage=stage
                        )

                        # Training side effects
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                        optim.step()

                        # Use unified metrics logging (every rank)
                        with self.metrics.aggregate(stage):
                            # Use metrics directly from _step_core (loss is
                            # already included)
                            self.log_step_metrics_unified(metrics, step)

                        if self.is_rank0:
                            current_metrics = self.metrics.get_smoothed_values(stage)
                            to_log = stage == "train"
                            self.update_progress_bar(
                                current_metrics,
                                step,
                                is_rank0=self.is_rank0,
                                to_log=to_log,
                            )

                        # Clear memory (avoid per-batch empty_cache to prevent
                        # fragmentation)
                        del batch, prepared_batch, outputs, loss, metrics

        finally:
            if self.is_rank0:
                self.close_progress_bar()

        # Memory cleanup after epoch (not per-batch to avoid fragmentation)
        torch.cuda.empty_cache()

        # Synchronize metrics across all ranks for correct global averages
        _, world_size, _ = self._get_distributed_config()
        self.metrics.sync_distributed(device, world_size=world_size)

        # epoch_metrics set in val/test branch above; train uses smoothed metrics
        if stage in ["val", "test"]:
            return epoch_metrics
        return self.metrics.get_smoothed_values(stage)

    def _train_epoch(self, epoch, model, train_loader, optim, device, master_process):
        """Train epoch using the generic runner."""
        return self._run_epoch(
            epoch, model, train_loader, device, master_process, "train", optim
        )

    def _validate_epoch(self, epoch, model, val_loader, device, master_process):
        """Validate epoch using the generic runner."""
        return self._run_epoch(epoch, model, val_loader, device, master_process, "val")

    def _test_epoch(self, epoch, model, test_loader, device, master_process):
        """Test epoch using the generic runner."""
        return self._run_epoch(
            epoch, model, test_loader, device, master_process, "test"
        )

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prediction step with full post-processing for inference/evaluation.

        This method is called by evaluators (Phase 8) to get fully processed outputs
        including predictions and class probabilities.

        Args:
            batch: Input batch dictionary with unified structure

        Returns:
            Dict[str, Tensor]: Processed outputs with keys:
                - 'predictions': Binary predictions [B] (0 or 1)
                - 'probabilities': Class probabilities [B, 2] (sklearn-compatible
                    format)
                - 'logits': Raw logits [B, 1] (optional, for debugging)

        Examples:
            >>> # Called by evaluator
            >>> batch = test_loader.next()
            >>> processed = trainer.predict_step(batch)
            >>> print(processed.keys())  # ['predictions', 'probabilities', 'logits']

        Note:
            This method is used during evaluation, not training. Training uses
            ``_step_core()`` which routes every batch through ``processor.process(...)``
            exactly once to obtain logits, probabilities, and extras.
        """
        return super().predict_step(batch)

    def _compute_f1_score(self, predictions, targets):
        """Compute F1 score."""
        # Simple implementation - you might want to use sklearn.metrics
        tp = ((predictions == 1) & (targets == 1)).sum().float()
        fp = ((predictions == 1) & (targets == 0)).sum().float()
        fn = ((predictions == 0) & (targets == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return f1.item()

    def _compute_epoch_level_aucs(self, all_probabilities, all_labels):
        """Compute epoch-level ROC-AUC and PR-AUC.

        Args:
            all_probabilities: All probability predictions for the epoch [N]
            all_labels: All ground truth labels for the epoch [N]

        Returns:
            tuple: (roc_auc, pr_auc) scores

        Raises:
            ValueError: If multi-class classification is detected (only binary
                supported)
        """
        try:
            unique_classes = torch.unique(all_labels)
            if len(unique_classes) > 2:
                raise ValueError(
                    "ClassificationTrainer currently only supports binary "
                    "classification. "
                    "For multi-class, use separate evaluation metrics."
                )

            if len(unique_classes) < 2:
                class_ratio = torch.mean(all_labels.float())
                return 0.5, class_ratio

            # NumPy for sklearn/metric APIs
            labels_np = all_labels.cpu().numpy()
            probabilities_np = all_probabilities.cpu().numpy()

            roc_auc = roc_auc_score(labels_np, probabilities_np)
            pr_auc = average_precision_score(labels_np, probabilities_np)

            return roc_auc, pr_auc

        except (ValueError, RuntimeError) as e:
            logger.warning("Epoch-level AUC calculation failed: %s", e)
            return 0.5, 0.5

    def _execute_training_logic(self, train_loader, val_loader, test_loader):
        """Execute training using standard BaseTrainer training pattern."""
        logger.info("Starting classification training...")

        current_epoch = self.get_epoch()
        if self.get_best_loss() is not None:
            best_loss = self.get_best_loss()
            logger.info(
                f"Resuming training from epoch {current_epoch} with "
                f"best_loss {best_loss}"
            )

        for epoch in range(current_epoch, self.num_epochs):
            # Use base trainer's method to set sampler epoch (training only)
            self.set_sampler_epoch(epoch)

            # -------- Train --------
            train_metrics = self._train_epoch(
                epoch,
                self.model,
                train_loader,
                self.optimizer,
                self.device,
                self.is_rank0,
            )

            # -------- Validate --------
            loss_improved = None
            val_metric = None
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(
                    epoch, self.model, val_loader, self.device, self.is_rank0
                )
                # Validation metric: pr_auc (alternatives: loss, f1_score).
                val_metric = -(val_metrics.get("pr_auc", 0.0))

                self.step_scheduler(val_metric)

                if self.early_stopping:
                    self.early_stopping(val_metric)

                loss_improved = self.set_best_loss(val_metric)
                if loss_improved:
                    logger.info(f"New best loss: {val_metric:.6f}")

                self.update_training_state(epoch)

                if self.is_rank0:
                    final_loss = val_metrics.get("loss", 0.0)
                    final_accuracy = val_metrics.get("accuracy", 0.0)
                    final_f1 = val_metrics.get("f1_score", 0.0)
                    final_roc_auc = val_metrics.get("roc_auc", 0.0)
                    final_pr_auc = val_metrics.get("pr_auc", 0.0)
                    val_msg = (
                        f"\nValidation Loss: {final_loss:.4f}, "
                        f"Accuracy: {final_accuracy:.4f}, "
                        f"F1: {final_f1:.4f}, ROC-AUC: {final_roc_auc:.4f}, "
                        f"PR-AUC: {final_pr_auc:.4f}"
                    )
                    logger.info(val_msg)

            # -------- Test Phase (if test_loader provided) --------
            test_metrics = {}
            if test_loader is not None:
                test_metrics = self._test_epoch(
                    epoch,
                    self.model,
                    test_loader,
                    self.device,
                    self.is_rank0,
                )

                if self.is_rank0:
                    final_test_loss = test_metrics.get("loss", 0.0)
                    final_test_accuracy = test_metrics.get("accuracy", 0.0)
                    final_test_f1 = test_metrics.get("f1_score", 0.0)
                    final_test_roc_auc = test_metrics.get("roc_auc", 0.0)
                    final_test_pr_auc = test_metrics.get("pr_auc", 0.0)
                    test_msg = (
                        f"\nTest Loss: {final_test_loss:.4f}, "
                        f"Accuracy: {final_test_accuracy:.4f}, "
                        f"F1: {final_test_f1:.4f}, ROC-AUC: {final_test_roc_auc:.4f}, "
                        f"PR-AUC: {final_test_pr_auc:.4f}"
                    )
                    logger.info(test_msg)

            # Log epoch metrics
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
            logger.info("Classification training completed")
            self.close_all()

    def _log_detailed_results(self, metrics_dict, split_name):
        """Log detailed classification results."""
        logger.info(f"\nDetailed Results for {split_name} set:")
        logger.info("=" * 50)

        logger.info(f"Accuracy: {metrics_dict['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics_dict['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {metrics_dict['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {metrics_dict['pr_auc']:.4f}")

        # Log metrics to metrics system
        metrics.log_scalar(f"{split_name}_accuracy", metrics_dict["accuracy"])
        metrics.log_scalar(f"{split_name}_f1_score", metrics_dict["f1_score"])
        metrics.log_scalar(f"{split_name}_roc_auc", metrics_dict["roc_auc"])
        metrics.log_scalar(f"{split_name}_pr_auc", metrics_dict["pr_auc"])


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_classification_trainer",
    node=ClassificationTrainerConfig,
    group="trainer",
)
