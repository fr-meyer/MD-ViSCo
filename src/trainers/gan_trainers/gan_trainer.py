"""GAN trainer for adversarial waveform generation using WGAN-GP.

This module provides GANTrainer, a specialized trainer for Wasserstein GAN with
Gradient Penalty (WGAN-GP) based waveform generation models like P2E-WGAN.
"""

# Standard library imports
import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.nn.parallel import (
    DistributedDataParallel as DDP,  # noqa: N817  # conventional alias DDP
)

# Local imports
from src.processors.output_processor import OutputProcessor
from src.trainers.trainer import BaseTrainer
from src.trainers.trainer import TrainerBaseConfig
from src.trainers.trainer import recursively_set_layout
from src.trainers.trainer import unwrap_model

logger = logging.getLogger(__name__)


@dataclass
class GANTrainerConfig(TrainerBaseConfig):
    """Configuration for GANTrainer with Hydra-compatible defaults.

    This trainer uses a unified optimizer configuration where a single optimizer
    partial function is called twice with different parameters (generator and
    discriminator). This simplifies configuration while maintaining dual optimizer
    functionality internally.
    """

    _target_: str = "src.trainers.gan_trainers.gan_trainer.GANTrainer"

    # GAN-specific parameters
    trainer_name: str = "gan"  # "approximation" or "refinement"

    # Processor configuration
    processor: Any | None = (
        None  # Instantiated via Hydra defaults - OutputProcessor for waveform
        # processing with extractor
    )

    # GAN training parameters
    n_critic: int = 5  # Number of discriminator updates per generator update
    # Note: lambda_gp and lambda_sample are handled by the criterion, not the trainer

    # Unified optimizer configuration
    optimizer: Any = (
        # Hydra will create partial function (used for both generator and discriminator)
        MISSING
    )
    scheduler: Any | None = (
        None  # Hydra will create partial function, None if not provided (for generator)
    )

    # Legacy GAN optimizer parameters (for backward compatibility)
    g_betas: tuple[float, float] = (0.5, 0.999)  # Generator Adam betas
    d_betas: tuple[float, float] = (0.5, 0.999)  # Discriminator Adam betas


class GANTrainer(BaseTrainer):
    """Unified WGAN-GP trainer for approximation and refinement stages.

    Uses CheckpointIO for checkpoint loading (SafeTensors, weights_only). Model
    weights must be loaded on rank 0 before DDP wrapping; optimizers and
    scheduler are created after DDP wrap. Dual optimizers: generator and
    discriminator (n_critic discriminator steps per generator step).

    Note:
        Checkpoint layout: flat multi-model (model_G_state_dict, model_D_state_dict,
        optimizer_G_state_dict, optimizer_D_state_dict, scheduler_state_dict,
        early_stopping_state, additional_info). DDP flow: rank 0 loads checkpoint
        and applies weights to G/D; then all ranks wrap with DDP, create
        optimizers/scheduler, then load trainer state from broadcast.

    See Also:
        src.utils.checkpoint_io.CheckpointIO: Checkpoint schema and
            extract_multi_model_state() for GAN checkpoints.
    """

    def __init__(
        self,
        n_critic: int = 5,
        optimizer: Any = None,
        scheduler: Any = None,
        processor: OutputProcessor | None = None,
        g_betas: tuple[float, float] = (0.5, 0.999),
        d_betas: tuple[float, float] = (0.5, 0.999),
        *args,
        **kwargs,
    ):
        """Initialize GANTrainer with GAN-specific and inherited parameters.

        Raises:
            ValueError: If n_critic is less than 1.
            TypeError: If processor is passed in an unsupported way (from base).
        """
        super().__init__(*args, processor=processor, **kwargs)
        self.n_critic = n_critic
        self.g_betas = g_betas
        self.d_betas = d_betas
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        if self.n_critic < 1:
            raise ValueError(f"n_critic must be >= 1, got {self.n_critic}")

    # Checkpoint Loading (inherited from BaseTrainer):
    # GANTrainer does not override _load_checkpoint_from_disk(), so it uses
    # BaseTrainer's refactored method which already uses CheckpointIO.load().
    # The checkpoint loading flow is:
    # 1. BaseTrainer._load_checkpoint_from_disk() uses CheckpointIO.load()
    # 2. BaseTrainer.prepare_model_weights() uses CheckpointIO.extract_model_state()
    #    and BaseModel.load_checkpoint() for both generator and discriminator
    # 3. BaseTrainer.load_trainer_states() uses CheckpointIO.extract_optimizer(),
    #    extract_scheduler(), and extract_early_stopping() for dual optimizers
    # For GAN checkpoints, CheckpointIO.extract_multi_model_state() can be used
    # with keys={'generator': 'model_G_state_dict', 'discriminator':
    # 'model_D_state_dict'}

    def prepare_model_weights(self, models=None):
        """Override to handle dual model (G and D) weight loading.

        CRITICAL: Called BEFORE DDP wrapping on rank 0 only.
        Uses BaseTrainer._apply_model_weights which supports multi-model dicts.
        """
        if not hasattr(self, "_stored_checkpoint") or self._stored_checkpoint is None:
            logger.warning("No stored checkpoint available for model weight loading")
            return

        if not self.is_rank0:
            return

        # Pass generator and discriminator to base class _apply_model_weights
        # It will extract 'model_G_state_dict' and 'model_D_state_dict' automatically
        self._apply_model_weights(
            self._stored_checkpoint,
            {
                "model_G": self.model.generator,
                "model_D": self.model.discriminator,
            },
        )
        logger.info(
            "Rank 0: GAN model weights (G and D) loaded (will be broadcast by DDP)"
        )

    def _pack_trainer_payload(self, checkpoint):
        """Override to pack dual optimizer states (G and D)."""
        if self.checkpoint_io is None:
            raise RuntimeError("checkpoint_io must be set before _pack_trainer_payload")
        io = self.checkpoint_io
        payload: dict[str, Any] = {}

        # Pack dual optimizer states
        if self.optimizer and self.load_optimizer:
            optimizer_g_state = io.extract_optimizer(
                checkpoint, key="optimizer_G_state_dict"
            )
            optimizer_d_state = io.extract_optimizer(
                checkpoint, key="optimizer_D_state_dict"
            )

            if optimizer_g_state or optimizer_d_state:
                payload["optimizers"] = {}
                if optimizer_g_state:
                    payload["optimizers"]["generator"] = optimizer_g_state
                if optimizer_d_state:
                    payload["optimizers"]["discriminator"] = optimizer_d_state

        # Pack scheduler state (single scheduler for generator)
        if self.scheduler and self.load_scheduler:
            scheduler_state = io.extract_scheduler(
                checkpoint, key="scheduler_state_dict"
            )
            if scheduler_state:
                payload["scheduler"] = scheduler_state

        # Pack early stopping state
        if self.early_stopping:
            early_stopping_state = io.extract_early_stopping(
                checkpoint, key="early_stopping_state"
            )
            if early_stopping_state:
                payload["early_stopping"] = early_stopping_state

        # Pack flags
        payload["flags"] = {
            "epoch": checkpoint.get("epoch", 0),
            "best_loss": checkpoint.get("best_loss", None),
            "load_model_weights": self.load_model_weights,
            "load_optimizer": self.load_optimizer,
            "load_scheduler": self.load_scheduler,
        }

        return payload

    def _load_trainer_states_from_payload(self, payload):
        """Override to load dual optimizer states (G and D)."""
        if self.optimizer and "optimizers" in payload:
            if (
                "generator" in payload["optimizers"]
                and payload["optimizers"]["generator"] is not None
            ):
                self.optimizer["generator"].load_state_dict(
                    payload["optimizers"]["generator"]
                )
                logger.info("Loaded generator optimizer state")
            if (
                "discriminator" in payload["optimizers"]
                and payload["optimizers"]["discriminator"] is not None
            ):
                self.optimizer["discriminator"].load_state_dict(
                    payload["optimizers"]["discriminator"]
                )
                logger.info("Loaded discriminator optimizer state")

        if self.scheduler and "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
            logger.info("Loaded scheduler state")

        if self.early_stopping and "early_stopping" in payload:
            self.early_stopping.load_state_dict(payload["early_stopping"])
            logger.info("Loaded early stopping state")

        if "flags" in payload:
            self.training_metadata.update(payload["flags"])
            logger.info(f"Loaded training metadata: {self.training_metadata}")

    def _extract_target_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract and narrow target channel from unified batch for GAN training.

        This narrows multi-channel targets in `batch['y']` using per-sample indices
        in `batch['tgt_idxs']` so the discriminator receives exactly one target
        channel per sample. The returned `result['y']` has shape [B, 1, T]. If
        present, SBP/DBP scalars are also extracted from `batch['bp_raw']`.

        Args:
            batch: Batch dictionary with unified structure containing at minimum
                'y' [B, C_targets, T] and 'tgt_idxs' [B].

        Returns:
            Dict[str, torch.Tensor]: { 'y': [B, 1, T], optional 'y_sbp', 'y_dbp' }
        """
        if "y" not in batch:
            raise ValueError(
                "GAN trainer requires unified batch with key 'y' for targets"
            )
        if "tgt_idxs" not in batch:
            raise ValueError(
                "GAN trainer requires unified batch format with 'tgt_idxs' for "
                "per-sample target selection. "
                "Ensure the updated collate function provides 'tgt_idxs'."
            )

        y_targets = batch["y"]  # [B, C_targets, T] or [B, T]
        # Handle single-channel targets without channel dim
        if y_targets.ndim == 2:
            y_target = y_targets.unsqueeze(1)  # [B, 1, T]
        else:
            tgt_idxs = batch["tgt_idxs"]  # [B]
            # Ensure correct dtype/device before advanced indexing
            tgt_idxs = tgt_idxs.to(dtype=torch.long, device=y_targets.device)
            batch_arange = torch.arange(y_targets.size(0), device=y_targets.device)
            y_target = y_targets[batch_arange, tgt_idxs].unsqueeze(1)  # [B, 1, T]

        result = {"y": y_target}

        # Optional BP scalars from bp_raw if present: [B, 3] -> SBP, DBP, MAP
        if (
            "bp_raw" in batch
            and torch.is_tensor(batch["bp_raw"])
            and batch["bp_raw"].ndim >= 2
        ):
            result["y_sbp"] = batch["bp_raw"][:, 0]  # Column 0: SBP
            result["y_dbp"] = batch["bp_raw"][:, 1]  # Column 1: DBP

        return result

    def _extract_source_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract source signal from batch for conditional GAN training.

        This method delegates to the P2EWGAN model's extract_input() method which
        provides a DDP-safe interface for input extraction. The top-level model
        method handles DDP unwrapping internally, ensuring compatibility with
        distributed training setups. This ensures per-sample channel selection is
        correctly applied via src_idxs and src_mask.

        Args:
            batch: Batch dictionary with unified structure containing:
                - "x": Source signal [B, C, T]
                - "src_idxs": Channel indices [B, S_max] (optional)
                - "src_mask": Channel mask [B, S_max] (optional)

        Returns:
            torch.Tensor: Properly extracted source signal tensor with shape
                [B, S_max, T] where channels are selected according to src_idxs
                and src_mask
        """
        required_keys = ["x", "src_idxs", "src_mask"]
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise ValueError(
                f"GAN trainer requires unified batch format. Missing keys: {
                    missing_keys
                }. "
                f"Expected unified batch with: {required_keys}. "
                f"Use updated collate function that provides src_idxs and src_mask."
            )

        # Delegate to model's DDP-safe extraction logic for proper channel selection
        if self.model is None:
            raise RuntimeError("model must be set before _extract_source_from_batch")
        result = self.model.extract_input(batch)
        return result["x"]

    def _get_main_output(self, outputs) -> torch.Tensor:
        """Handle different GAN model output formats for consistent processing.

        For GAN training, outputs can be generator outputs, discriminator outputs,
        or composite outputs depending on the model structure.

        Args:
            outputs: Raw model outputs (can be tensor, tuple, dict, etc.)

        Returns:
            torch.Tensor: Main output tensor for loss computation
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, (list, tuple)):
            # For GANs, typically return the first output (generator output)
            return outputs[0]
        elif isinstance(outputs, dict):
            # For GANs with structured outputs, return generator output
            if "generator_output" in outputs:
                return outputs["generator_output"]
            elif "fake" in outputs:
                return outputs["fake"]
            elif "output" in outputs:
                return outputs["output"]
            else:
                for _key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        return value
                raise ValueError("No tensor found in outputs dict")
        else:
            raise ValueError(f"Unexpected output format for GAN: {type(outputs)}")

    def _step_core(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        stage: str,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, float], Any]:
        """GAN-specific step core that returns one loss based on mode.

        This method computes a single loss based on the requested mode,
        handling all GAN loss computation in a unified way.

        Args:
            model: The GAN model (DDP-wrapped or not) containing generator and
                discriminator
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")
            mode: Loss type ("discriminator", "generator", "validation")

        Returns:
            loss: Single computed loss
            metrics: Dictionary of metrics
        """
        if self.criterion is None:
            raise RuntimeError("criterion must be set before _step_core")
        generator = model.generator
        discriminator = model.discriminator

        y_target = self._extract_target_from_batch(batch)  # Target signal

        real_b = y_target["y"]

        if mode == "discriminator":
            # Generate fake data WITHOUT gradients (matching vanilla implementation)
            with torch.no_grad():
                outputs = generator(batch)
            fake_b = self._get_main_output(outputs)
            real_a = self._extract_source_from_batch(batch)

            real_pred = self._get_main_output(discriminator(batch, real_b))
            fake_pred = self._get_main_output(discriminator(batch, fake_b.detach()))

            # interpolates needed for gradient penalty
            gp_loss = getattr(self.criterion, "gp_loss", None)
            if gp_loss is None:
                raise RuntimeError("criterion must have gp_loss for discriminator mode")
            interpolates, d_interpolates = gp_loss.get_interpolates(
                real_b, fake_b, discriminator, real_a
            )
            d_interpolates = self._get_main_output(d_interpolates)
            raw_loss = self.criterion(
                real_data=real_b,
                fake_data=fake_b,
                real_pred=real_pred,
                fake_pred=fake_pred,
                interpolates=interpolates,
                d_interpolates=d_interpolates,
            )
            loss = (
                raw_loss["total_loss"]
                if isinstance(raw_loss, dict)
                else raw_loss[0]
                if isinstance(raw_loss, tuple)
                else raw_loss
            )

            mse_loss = F.mse_loss(fake_b, real_b)
            l1_loss = F.l1_loss(fake_b, real_b)
            metrics = {
                "d_loss": float(loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "l1_loss": float(l1_loss.detach()),
            }

            # BP metrics extraction using processor
            if self.processor is not None:
                try:
                    processor_payload = self._process_fake_waveform(
                        fake_b, batch, stage
                    )
                    self._merge_processor_bp_metrics(metrics, processor_payload)
                except Exception as e:
                    logger.debug(
                        f"BP extraction failed in discriminator mode ({stage} stage): {
                            e
                        }"
                    )

            return loss, metrics, None

        elif mode == "generator":
            # Generate fake data with gradients
            outputs = generator(batch)
            fake_b = self._get_main_output(outputs)
            real_a = self._extract_source_from_batch(batch)
            fake_pred = self._get_main_output(discriminator(batch, fake_b))

            raw_g = self.criterion(
                real_data=real_b,
                fake_data=fake_b,
                real_pred=fake_pred,  # Dummy value for generator mode
                fake_pred=fake_pred,
                # interpolates and d_interpolates default to None for generator mode
            )
            loss = (
                raw_g["total_loss"]
                if isinstance(raw_g, dict)
                else raw_g[0]
                if isinstance(raw_g, tuple)
                else raw_g
            )

            mse_loss = F.mse_loss(fake_b, real_b)
            l1_loss = F.l1_loss(fake_b, real_b)
            metrics = {
                "g_loss": float(loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "l1_loss": float(l1_loss.detach()),
            }

            # BP metrics extraction using processor
            if self.processor is not None:
                try:
                    processor_payload = self._process_fake_waveform(
                        fake_b, batch, stage
                    )
                    self._merge_processor_bp_metrics(metrics, processor_payload)
                except Exception as e:
                    logger.debug(
                        f"BP extraction failed in generator mode ({stage} stage): {e}"
                    )

            return loss, metrics, None

        elif mode == "validation":
            # Generate fake data without gradients
            with torch.no_grad():
                outputs = generator(batch)
                fake_b = self._get_main_output(outputs)
                real_a = self._extract_source_from_batch(batch)
                real_pred = self._get_main_output(discriminator(batch, real_b))
                fake_pred = self._get_main_output(discriminator(batch, fake_b))

            # interpolates needed for gradient penalty
            gp_loss_val = getattr(self.criterion, "gp_loss", None)
            if gp_loss_val is None:
                raise RuntimeError("criterion must have gp_loss for validation mode")
            with torch.enable_grad():
                interpolates, d_interpolates = gp_loss_val.get_interpolates(
                    real_b, fake_b, discriminator, real_a
                )
                d_interpolates = self._get_main_output(d_interpolates)
                raw_d = self.criterion(
                    real_data=real_b,
                    fake_data=fake_b,
                    real_pred=real_pred,
                    fake_pred=fake_pred,
                    interpolates=interpolates,
                    d_interpolates=d_interpolates,
                )
                d_loss = (
                    raw_d["total_loss"]
                    if isinstance(raw_d, dict)
                    else raw_d[0]
                    if isinstance(raw_d, tuple)
                    else raw_d
                )

            raw_g_val = self.criterion(
                real_data=real_b,
                fake_data=fake_b,
                real_pred=fake_pred,  # Dummy value for generator mode
                fake_pred=fake_pred,
                # interpolates and d_interpolates default to None for generator mode
            )
            g_loss = (
                raw_g_val["total_loss"]
                if isinstance(raw_g_val, dict)
                else raw_g_val[0]
                if isinstance(raw_g_val, tuple)
                else raw_g_val
            )

            mse_loss = F.mse_loss(fake_b, real_b)
            l1_loss = F.l1_loss(fake_b, real_b)
            metrics = {
                "d_loss": float(d_loss.detach()),
                "g_loss": float(g_loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "l1_loss": float(l1_loss.detach()),
            }

            # BP metrics extraction using processor
            if self.processor is not None:
                try:
                    processor_payload = self._process_fake_waveform(
                        fake_b, batch, stage
                    )
                    self._merge_processor_bp_metrics(metrics, processor_payload)
                except Exception as e:
                    logger.debug(
                        f"BP extraction failed in validation mode ({stage} stage): {e}"
                    )

            # for compatibility with base signature
            combined_loss = d_loss + g_loss
            return combined_loss, metrics, None

        else:
            raise ValueError(
                f"Invalid mode: {
                    mode
                }. Must be 'discriminator', 'generator', or 'validation'"
            )

    def _run_epoch(
        self, epoch, model, data_loader, device, master_process, stage: str, optim=None
    ):
        """Run generic epoch for GAN training with train/val/test support.

        This method handles the unique GAN training requirements including:
        - Dual optimizer training (generator and discriminator)
        - n_critic discriminator updates per generator update
        - Proper progress bar management and metrics synchronization

        Note: Sampler epoch setting is handled by _execute_training_logic for
        training only. Validation and test stages use deterministic sampling
        without epoch setting.

        Args:
            epoch: Current epoch number
            model: The GAN model to run
            data_loader: DataLoader for the stage
            device: Device to run on
            master_process: Whether this is the master process
            stage: Stage name ("train", "val", or "test")
            optim: Optimizer dict (required for training, None for validation/test)
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
            train_mapping = getattr(self, "_vital_channel_mapping", None)
            if train_mapping is not None:
                recursively_set_layout(model, train_mapping)
                logger.debug(
                    "Training phase: Model layout set using input_preprocessing mapping"
                )
            model.train()
            context_manager = torch.enable_grad()
        elif stage == "val":
            val_mapping = getattr(self, "_vital_channel_mapping", None)
            if val_mapping is not None:
                recursively_set_layout(model, val_mapping)
                logger.debug(
                    "Validation phase: Model layout set using input_preprocessing "
                    "mapping"
                )
            model.eval()
            context_manager = torch.no_grad()
        else:  # test
            test_mapping = getattr(self, "_vital_channel_mapping", None)
            if test_mapping is not None:
                recursively_set_layout(model, test_mapping)
                logger.debug(
                    "Test phase: Model layout set using input_preprocessing mapping"
                )
            model.eval()
            context_manager = torch.no_grad()

        try:
            with context_manager:
                for step, batch in enumerate(data_loader):
                    prepared_batch = {
                        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }

                    # One-time validation of S_max vs generator.in_channels to
                    # prevent mismatches
                    if stage == "train" and step == 0:
                        sample_x = self._extract_source_from_batch(
                            prepared_batch
                        )  # [B, S_max, T]
                        s_max = sample_x.shape[1]
                        gen = getattr(model.generator, "module", model.generator)
                        if s_max != gen.in_channels:
                            msg = (
                                f"Input channel mismatch: S_max={
                                    s_max
                                } from batch but generator.in_channels={
                                    gen.in_channels
                                }. "
                                f"Update model config (in_channels) to match "
                                f"S_max or ensure single-source inputs for "
                                f"P2EWGAN."
                            )
                            logger.error(msg)
                            raise ValueError(msg)

                    if stage == "train":
                        # ✅ GAN-specific dual optimization logic using mode-based
                        # _step_core
                        if self.optimizer is None:
                            raise RuntimeError("optimizer required for training")
                        d_metrics: dict[str, float] = {}
                        g_metrics: dict[str, float] = {}

                        # Train discriminator n_critic times
                        for _ in range(self.n_critic):
                            self.optimizer["discriminator"].zero_grad(set_to_none=True)
                            d_loss, d_metrics, _ = self._step_core(
                                model, prepared_batch, stage=stage, mode="discriminator"
                            )
                            d_loss.backward()
                            self.optimizer["discriminator"].step()

                        # Train generator once
                        self.optimizer["generator"].zero_grad(set_to_none=True)
                        g_loss, g_metrics, _ = self._step_core(
                            model, prepared_batch, stage=stage, mode="generator"
                        )
                        g_loss.backward()
                        self.optimizer["generator"].step()

                        # Combine metrics from both discriminator and generator
                        metrics = {**d_metrics, **g_metrics}

                    else:
                        # ✅ Validation/test: use _step_core with validation mode
                        _combined_loss, metrics, _ = self._step_core(
                            model, prepared_batch, stage=stage, mode="validation"
                        )

                    # ✅ UNIFIED METRICS LOGGING (every rank)
                    with self.metrics.aggregate(stage):
                        self.log_step_metrics_unified(metrics, step)

                    # ✅ UNIFIED PROGRESS BAR UPDATE (rank-0 only)
                    if self.is_rank0:
                        current_metrics = self.metrics.get_smoothed_values(stage)
                        to_log = stage == "train"
                        self.update_progress_bar(
                            current_metrics, step, is_rank0=self.is_rank0, to_log=to_log
                        )

                    # ✅ UNIFIED MEMORY CLEANUP
                    del batch, prepared_batch, metrics

        finally:
            if self.is_rank0:
                self.close_progress_bar()

        # Memory cleanup after epoch (not per-batch to avoid fragmentation)
        torch.cuda.empty_cache()

        # Synchronize metrics across all ranks for correct global averages
        _, world_size, _ = self._get_distributed_config()
        self.metrics.sync_distributed(device, world_size=world_size)
        stage_metrics = self.metrics.get_smoothed_values(stage)

        # Print validation results for user feedback (matching legacy behavior)
        if stage == "val" and master_process:
            logger.info("\nValidation Results:")
            logger.info(f"D Loss: {stage_metrics.get('d_loss', 0.0):.4f}")
            logger.info(f"G Loss: {stage_metrics.get('g_loss', 0.0):.4f}")
            if "real_pred" in stage_metrics:
                logger.info(
                    f"Real Pred: {stage_metrics['real_pred']:.4f}, Fake Pred: {
                        stage_metrics['fake_pred']:.4f}"
                )

        return stage_metrics

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

    def _setup_ddp_wrapping(self, model, local_rank: int):
        """Override DDP wrapping for GAN models with proper parameter handling."""
        # TIMING: DDP wrapping broadcasts model weights from rank 0 to all ranks.
        # GAN may not use all params every step; DDP needs find_unused_parameters=True.
        self.model.generator = DDP(
            model.generator,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=True,
        )
        self.model.discriminator = DDP(
            model.discriminator,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=True,
        )

    def create_optimizer(self):
        """Create dual optimizers for generator and discriminator using unified
        configuration.

        Uses a single optimizer partial function called twice with different
        parameters. This maintains dual optimizer functionality while simplifying
        configuration.
        """
        if self.optimizer_partial is None:
            raise ValueError(
                "Optimizer config not provided. Specify via Hydra: trainer.optimizer"
            )
        params_g = getattr(
            self.model.generator, "module", self.model.generator
        ).parameters()
        params_d = getattr(
            self.model.discriminator, "module", self.model.discriminator
        ).parameters()
        generator_optimizer = self.optimizer_partial(params=params_g)
        discriminator_optimizer = self.optimizer_partial(params=params_d)
        if (
            hasattr(generator_optimizer, "param_groups")
            and len(generator_optimizer.param_groups) > 0
            and "betas" in generator_optimizer.param_groups[0]
        ):
            generator_optimizer.param_groups[0]["betas"] = self.g_betas
            discriminator_optimizer.param_groups[0]["betas"] = self.d_betas

        self.optimizer = {
            "generator": generator_optimizer,
            "discriminator": discriminator_optimizer,
        }
        logger.info(
            f"GAN optimizers created: {self.optimizer['generator'].name}, {
                self.optimizer['discriminator'].name
            }"
        )

    def create_scheduler(self):
        """Create scheduler for generator optimizer."""
        if self.scheduler_partial is None:
            self.scheduler = None
            logger.info("No scheduler configured")
            return

        assert self.optimizer is not None
        self.scheduler = self.scheduler_partial(optimizer=self.optimizer["generator"])
        assert self.scheduler is not None
        logger.info(f"GAN scheduler created: {self.scheduler.name}")

    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute GAN training using provided infrastructure from BaseTrainer."""
        # Cache vital channel mapping once (optimization: same for all splits)
        self._vital_channel_mapping = self._build_vital_channel_mapping()
        mapping = self._vital_channel_mapping
        mapping_display = {v.name: idx for v, idx in mapping.items()} if mapping else {}
        logger.info(f"Using preprocessing-defined channel mapping: {mapping_display}")

        assert self.optimizer is not None, (
            "GAN optimizer must be created before training"
        )

        # -------- Training Loop --------
        current_epoch = self.get_epoch()
        if self.get_best_loss() is not None:
            logger.info(
                f"Resuming training from epoch {current_epoch} with best_loss {
                    self.get_best_loss()
                }"
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
            val_metric = None
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(
                    epoch, self.model, val_loader, self.device, self.is_rank0
                )
                # Scheduler uses mse_loss (legacy: previously d_loss+g_loss).
                val_metric = val_metrics.get("mse_loss", 0.0)

            # -------- Test Phase (if test_loader provided) --------
            test_metrics = {}
            if test_loader is not None:
                test_metrics = self._test_epoch(
                    epoch, self.model, test_loader, self.device, self.is_rank0
                )

                if self.is_rank0:
                    final_test_loss = test_metrics.get("mse_loss", 0.0)
                    logger.info(f"\nTest Loss: {final_test_loss:.4f}")

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
                    final_loss = val_metrics.get("d_loss", 0.0) + val_metrics.get(
                        "g_loss", 0.0
                    )
                    logger.info(f"\nValidation Loss: {final_loss:.4f}")

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
                logger.info(
                    f"Saving best model checkpoint with loss: {
                        self.get_best_loss():.6f}"
                )
                # GAN-SPECIFIC: Save best model checkpoint with BOTH generator
                # and discriminator
                # Checkpoint format follows CheckpointIO canonical GAN schema:
                # - model_G_state_dict: Generator weights
                # - model_D_state_dict: Discriminator weights
                # - optimizer_G_state_dict: Generator optimizer state
                # - optimizer_D_state_dict: Discriminator optimizer state
                # - scheduler_state_dict: Single scheduler for generator only
                self.save_checkpoint(
                    epoch=None,
                    dataset=train_loader,
                    model_state_dict={
                        "model_G_state_dict": unwrap_model(
                            self.model.generator
                        ).state_dict(),
                        "model_D_state_dict": unwrap_model(
                            self.model.discriminator
                        ).state_dict(),
                    },
                    optimizer_state_dict={
                        "optimizer_G_state_dict": self.optimizer[
                            "generator"
                        ].state_dict(),
                        "optimizer_D_state_dict": self.optimizer[
                            "discriminator"
                        ].state_dict(),
                    },
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler else None
                    ),  # Single scheduler
                    early_stopping_state=(
                        self.early_stopping.state_dict()
                        if self.early_stopping
                        else None
                    ),
                    additional_info={
                        "stage": self.trainer_name,
                        "n_critic": self.n_critic,
                    },
                )

            # Periodic checkpointing (rank-0 only, barrier handled in base)
            if (
                self.save_checkpoint_frequency is not None
                and epoch % self.save_checkpoint_frequency == 0
            ):
                logger.info(f"Saving periodic checkpoint at epoch {epoch}")
                # GAN-SPECIFIC: Periodic checkpointing with BOTH generator and
                # discriminator
                # Same canonical format as best checkpoint for consistency
                self.save_checkpoint(
                    epoch=epoch,
                    dataset=train_loader,
                    model_state_dict={
                        "model_G_state_dict": unwrap_model(
                            self.model.generator
                        ).state_dict(),
                        "model_D_state_dict": unwrap_model(
                            self.model.discriminator
                        ).state_dict(),
                    },
                    optimizer_state_dict={
                        "optimizer_G_state_dict": self.optimizer[
                            "generator"
                        ].state_dict(),
                        "optimizer_D_state_dict": self.optimizer[
                            "discriminator"
                        ].state_dict(),
                    },
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler else None
                    ),  # Single scheduler
                    early_stopping_state=(
                        self.early_stopping.state_dict()
                        if self.early_stopping
                        else None
                    ),
                    additional_info={
                        "stage": self.trainer_name,
                        "n_critic": self.n_critic,
                    },
                )

        if self.is_rank0:
            logger.info("GAN training completed")
            self.close_all()

    def _process_fake_waveform(
        self, fake_waveform: torch.Tensor, batch: dict[str, Any], stage: str
    ) -> dict[str, Any] | None:
        """Build processor payload from fake waveform for BP metrics extraction.

        Runs the configured OutputProcessor on the generated fake waveform and
        batch to compute blood pressure and other scalar metrics (e.g. for
        logging during GAN training).

        Args:
            fake_waveform: Generated waveform tensor from the generator.
            batch: Batch dict (must contain ground-truth keys expected by the
                processor, e.g. bp_raw).
            stage: Stage name ("train", "val", or "test") for processor behavior.

        Returns:
            Processor payload dict (e.g. with "metrics" key), or None if no
            processor is configured.
        """
        if self.processor is None:
            return None
        batch["extract_scalars"] = True
        if stage == "train":
            batch["_compute_full_metrics_during_train"] = True
        model_output = {"predictions": fake_waveform, "extras": {}}
        return self.processor.process(model_output, batch, stage=stage)

    def _merge_processor_bp_metrics(
        self, metrics: dict[str, float], processor_payload: dict[str, Any] | None
    ) -> None:
        """Merge BP metrics from processor payload into the metrics dict in place.

        Extracts the "bp" sub-dict from processor_payload["metrics"] and adds
        each key with a "bp_" prefix to the given metrics dict (e.g. bp_mae,
        bp_sbp). Tensor values are reduced with mean().detach(); scalars are
        stored as float.

        Args:
            metrics: Dict to update with BP metrics (modified in place).
            processor_payload: Result from _process_fake_waveform or None; no-op
                if None or if it has no "metrics"/"bp" data.
        """
        if not processor_payload:
            return
        bp_metrics = (
            ((processor_payload.get("metrics") or {}).get("bp"))
            if isinstance(processor_payload, dict)
            else None
        )
        if not bp_metrics:
            return
        for key, value in bp_metrics.items():
            metric_key = f"bp_{key}"
            if torch.is_tensor(value):
                metrics[metric_key] = float(value.mean().detach())
            elif isinstance(value, (float, int)):
                metrics[metric_key] = float(value)

    def _move_model_to_device(self, train_loader=None):
        """Override to handle GAN models with generator and discriminator.

        GAN models always have both generator and discriminator components
        that need to be moved to device. This method ensures both are properly
        placed on the target device.

        Args:
            model: The GAN model containing generator and discriminator

        Returns:
            The model with both components moved to device
        """
        if self.model is not None:
            if self.is_rank0:
                logger.info(
                    f"Moving GAN generator and discriminator to device: {
                        self._get_device()
                    }"
                )

            # TIMING: Move both generator and discriminator to device BEFORE DDP
            # wrapping. This ensures weights are on the correct device before DDP
            # broadcasts them
            unwrap_model(self.model.generator).to(self._get_device())
            unwrap_model(self.model.discriminator).to(self._get_device())

    def log_epoch_metrics_unified(
        self, epoch, train_metrics, val_metrics, test_metrics, best_loss, loss_improved
    ):
        """Override base trainer logging to handle dual optimizers (GAN-specific).

        This method handles the dual optimizer structure used in GAN training,
        logging separate learning rates for generator and discriminator.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            test_metrics: Test metrics dictionary
            best_loss: Best loss achieved so far
            loss_improved: Whether loss improved this epoch
        """
        if not self.is_rank0:
            return  # Only rank-0 logs to W&B/TensorBoard

        wandb_obj = getattr(self.progress_bar, "wandb", None)
        if wandb_obj is not None and wandb_obj._is_ready():
            try:
                log_dict = {}

                # Log ALL train metrics with simple prefixing
                for metric_name, metric_value in train_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(
                        metric_value
                    ):
                        log_dict[f"train/{metric_name}"] = float(metric_value)

                # Log ALL val metrics with simple prefixing
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(
                        metric_value
                    ):
                        log_dict[f"val/{metric_name}"] = float(metric_value)

                # Log ALL test metrics with simple prefixing
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(
                        metric_value
                    ):
                        log_dict[f"test/{metric_name}"] = float(metric_value)

                # Add standard metrics with dual optimizer support
                opt = self.optimizer
                log_dict.update(
                    {
                        "epoch": epoch,
                        "learning_rate_G": float(opt["generator"].param_groups[0]["lr"])
                        if opt is not None
                        else 0.0,
                        "learning_rate_D": float(
                            opt["discriminator"].param_groups[0]["lr"]
                        )
                        if opt is not None
                        else 0.0,
                        "best_loss": float(best_loss or 0.0),
                        "improved": (
                            loss_improved if loss_improved is not None else False
                        ),
                    }
                )

                if hasattr(self, "early_stopping") and self.early_stopping:
                    log_dict.update(
                        {
                            "early_stopping/counter": int(
                                getattr(self.early_stopping, "counter", 0)
                            ),
                            "early_stopping/best_loss": float(
                                getattr(self.early_stopping, "best_loss", 0.0)
                            ),
                            "early_stopping/patience": int(
                                getattr(self.early_stopping, "patience", 0)
                            ),
                        }
                    )

                # Log to WandB
                if wandb_obj is not None:
                    wandb_obj.log(log_dict, is_rank0=self.is_rank0)

            except (RuntimeError, KeyError, AttributeError) as e:
                logger.error("Failed to log metrics to WandB: %s", e, exc_info=True)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(group="trainer", name="base_gan_trainer", node=GANTrainerConfig)
