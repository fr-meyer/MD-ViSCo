# Standard library imports
import logging
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any, Dict, List, Optional, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
import torch.nn.functional as F
# Local imports
from src.dataset.base_dataset import Sample

from src.utils.train_utils import EarlyStopping
from src.utils.utils_preprocessing import print_model_parameters, isExist_dir
from src.criterions.wgan_gp_loss import WGANGPLoss
from src.utils.checkpoint_utils import load_checkpoint_components
from ..trainer import BaseTrainer, TrainerBaseConfig


logger = logging.getLogger(__name__)

@dataclass
class GANTrainerConfig(TrainerBaseConfig):
    """Configuration for GANTrainer with Hydra-compatible defaults"""
    _target_: str = "src.trainers.gan_trainers.gan_trainer.GANTrainer"
    
    # GAN-specific parameters
    trainer_name: str = "gan"  # "approximation" or "refinement"
    
    # GAN training parameters
    n_critic: int = 5  # Number of discriminator updates per generator update
    # Note: lambda_gp and lambda_sample are handled by the criterion, not the trainer
    
    # GAN optimizer parameters
    g_betas: Tuple[float, float] = (0.5, 0.999)  # Generator Adam betas
    d_betas: Tuple[float, float] = (0.5, 0.999)  # Discriminator Adam betas


class GANTrainer(BaseTrainer):
    """
    Unified WGAN-GP trainer that works with both approximation and refinement stages.
    
    Features:
    - Hydra-compatible instantiation with individual parameters
    - Unified setup with type-filtered config flattening
    - Automatic project name generation based on trainer_name
    - Complete config traceability in wandb
    - No subclass overrides required for wandb setup
    """
    
    def __init__(self, n_critic: int = 5, 
                 g_betas: Tuple[float, float] = (0.5, 0.999),
                 d_betas: Tuple[float, float] = (0.5, 0.999),
                 *args, **kwargs):
        
        # Call parent constructor with all other parameters
        super().__init__(*args, **kwargs)
        
        # Store only GAN-specific parameters
        self.n_critic = n_critic
        self.g_betas = g_betas
        self.d_betas = d_betas
        
        # Validate GAN parameters
        if self.n_critic < 1:
            raise ValueError(f"n_critic must be >= 1, got {self.n_critic}")

    # def _extract_target_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     """Extract target from unified batch structure for GAN training.
        
    #     For GAN training, we need both real and fake data. The target is typically
    #     the real data that the generator should learn to produce.
        
    #     Args:
    #         batch: Batch dictionary with unified structure
            
    #     Returns:
    #         torch.Tensor: Real data tensor for GAN training
    #     """
    #     # For GAN training, extract real data from batch
    #     # The batch structure depends on the specific GAN implementation
    #     if "real" in batch:
    #         return batch["real"]
    #     elif "x" in batch:
    #         return batch["x"]  # Input data as real data
    #     elif "input" in batch:
    #         return batch["input"]
    #     else:
    #         # Fallback: use the first tensor in the batch
    #         for key, value in batch.items():
    #             if isinstance(value, torch.Tensor):
    #                 return value
    #         raise ValueError("No suitable target found in batch for GAN training")

    def _extract_source_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract source signal from batch for conditional GAN training.
        
        For conditional GAN training, we need the source signal that conditions
        the generation process.
        
        Args:
            batch: Batch dictionary with unified structure
            
        Returns:
            torch.Tensor: Source signal tensor for conditional GAN training
        """
        if "x" in batch:
            return batch["x"]  # Source signal (condition)
        elif "input" in batch:
            return batch["input"]
        elif "source" in batch:
            return batch["source"]
        else:
            raise ValueError("No source signal found in batch for conditional GAN training")

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
                # Return first tensor value
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        return value
                raise ValueError("No tensor found in outputs dict")
        else:
            raise ValueError(f"Unexpected output format for GAN: {type(outputs)}")

    def _step_core(self, model, batch, *, stage: str, mode: str = None):
        """GAN-specific step core that returns one loss based on mode.
        
        This method computes a single loss based on the requested mode,
        handling all GAN loss computation in a unified way.
        
        Args:
            model: The GAN model (DDP-wrapped or not) containing generator and discriminator
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")
            mode: Loss type ("discriminator", "generator", "validation")
            
        Returns:
            loss: Single computed loss
            metrics: Dictionary of metrics
        """
        # Extract generator and discriminator
        generator = model.generator
        discriminator = model.discriminator
        
        # Extract source (condition) and target signals
        # real_A = batch.get("x")  # Source signal (condition)
        # real_B = batch.get("y")  # Target signal

        y_target = self._extract_target_from_batch(batch) # Target signal

        real_B = y_target["y"]
        
        if mode == "discriminator":
            # Generate fake data WITHOUT gradients (matching vanilla implementation)
            with torch.no_grad():
                outputs = generator(batch)
            fake_B = outputs["fake_b"]
            real_A = outputs["real_a"]

            real_pred = discriminator(batch, real_B)
            fake_pred = discriminator(batch, fake_B.detach())
            
            # Compute discriminator loss - interpolates are needed for discriminator mode
            interpolates, d_interpolates = self.criterion.gp_loss.get_interpolates(real_B, fake_B, discriminator, real_A)
            loss = self.criterion(
                real_data=real_B,
                fake_data=fake_B,
                real_pred=real_pred,
                fake_pred=fake_pred,
                interpolates=interpolates,
                d_interpolates=d_interpolates
            )

            mse_loss = F.mse_loss(fake_B, real_B)
            l1_loss = F.l1_loss(fake_B, real_B)
            metrics = {
                "d_loss": float(loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "l1_loss": float(l1_loss.detach()),
                # "real_pred": float(real_pred.mean().detach()),
                # "fake_pred": float(fake_pred.mean().detach())
            }

            if "y_pred_sbp" in outputs and  "y_dbp" in y_target:
                metrics["sbp_mae"] = F.l1_loss(outputs["y_pred_sbp"], y_target["y_sbp"]).item()
                metrics["sbp_mse"] = F.mse_loss(outputs["y_pred_sbp"], y_target["y_sbp"]).item()
            
            if "y_pred_dbp" in outputs and "y_dbp" in y_target:
                metrics["dbp_mae"] = F.l1_loss(outputs["y_pred_dbp"], y_target["y_dbp"]).item()
                metrics["dbp_mse"] = F.mse_loss(outputs["y_pred_dbp"], y_target["y_dbp"]).item()
            
            return loss, metrics
            
        elif mode == "generator":
            # Generate fake data with gradients
            outputs = generator(batch)
            fake_B = outputs["fake_b"]
            real_A = outputs["real_a"]
            fake_pred = discriminator(batch, fake_B)
            
            # Compute generator loss - no interpolates needed for generator mode
            loss = self.criterion(
                real_data=real_B,
                fake_data=fake_B,
                real_pred=fake_pred,  # Dummy value for generator mode
                fake_pred=fake_pred
                # interpolates and d_interpolates default to None for generator mode
            )
            
            mse_loss = F.mse_loss(fake_B, real_B)
            l1_loss = F.l1_loss(fake_B, real_B)
            metrics = {
                "g_loss": float(loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "l1_loss": float(l1_loss.detach()),
                # "fake_pred": float(fake_pred.mean().detach())
            }

            if "y_pred_sbp" in outputs and  "y_dbp" in y_target:
                metrics["sbp_mae"] = F.l1_loss(outputs["y_pred_sbp"], y_target["y_sbp"]).item()
                metrics["sbp_mse"] = F.mse_loss(outputs["y_pred_sbp"], y_target["y_sbp"]).item()
            
            if "y_pred_dbp" in outputs and "y_dbp" in y_target:
                metrics["dbp_mae"] = F.l1_loss(outputs["y_pred_dbp"], y_target["y_dbp"]).item()
                metrics["dbp_mse"] = F.mse_loss(outputs["y_pred_dbp"], y_target["y_dbp"]).item()
            
            return loss, metrics
            
        elif mode == "validation":
            # Generate fake data without gradients
            with torch.no_grad():
                outputs = generator(batch)
                fake_B = outputs["fake_b"]
                real_A = outputs["real_a"]
                real_pred = discriminator(batch, real_B)
                fake_pred = discriminator(batch, fake_B)
            
            # Compute validation loss (discriminator loss) - interpolates are needed
            with torch.enable_grad():
                interpolates, d_interpolates = self.criterion.gp_loss.get_interpolates(real_B, fake_B, discriminator, real_A)
                d_loss = self.criterion(
                    real_data=real_B,
                    fake_data=fake_B,
                    real_pred=real_pred,
                    fake_pred=fake_pred,
                    interpolates=interpolates,
                    d_interpolates=d_interpolates
                )

            # Compute generator loss - no interpolates needed for generator mode
            g_loss = self.criterion(
                real_data=real_B,
                fake_data=fake_B,
                real_pred=fake_pred,  # Dummy value for generator mode
                fake_pred=fake_pred
                # interpolates and d_interpolates default to None for generator mode
            )

            mse_loss = F.mse_loss(fake_B, real_B)
            l1_loss = F.l1_loss(fake_B, real_B)
            metrics = {
                "d_loss": float(d_loss.detach()),
                "g_loss": float(g_loss.detach()),
                "mse_loss": float(mse_loss.detach()),
                "l1_loss": float(l1_loss.detach()),
                # "real_pred": float(real_pred.mean().detach()),
                # "fake_pred": float(fake_pred.mean().detach())
            }

            if "y_pred_sbp" in outputs and  "y_dbp" in y_target:
                metrics["sbp_mae"] = F.l1_loss(outputs["y_pred_sbp"], y_target["y_sbp"]).item()
                metrics["sbp_mse"] = F.mse_loss(outputs["y_pred_sbp"], y_target["y_sbp"]).item()
            
            if "y_pred_dbp" in outputs and "y_dbp" in y_target:
                metrics["dbp_mae"] = F.l1_loss(outputs["y_pred_dbp"], y_target["y_dbp"]).item()
                metrics["dbp_mse"] = F.mse_loss(outputs["y_pred_dbp"], y_target["y_dbp"]).item()
            
            return d_loss, g_loss, metrics
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'discriminator', 'generator', or 'validation'")

    def _run_epoch(self, epoch, model, data_loader, device, master_process, stage: str, optim=None):
        """Generic epoch runner for GAN training with train/val/test support.
        
        This method handles the unique GAN training requirements including:
        - Dual optimizer training (generator and discriminator)
        - n_critic discriminator updates per generator update
        - Proper progress bar management and metrics synchronization
        
        Note: Sampler epoch setting is handled by _execute_training_logic for training only.
        Validation and test stages use deterministic sampling without epoch setting.
        
        Args:
            epoch: Current epoch number
            model: The GAN model to run
            data_loader: DataLoader for the stage
            device: Device to run on
            master_process: Whether this is the master process
            stage: Stage name ("train", "val", or "test")
            optim: Optimizer dict (required for training, None for validation/test)
        """
        # Validate stage parameter
        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Stage must be 'train', 'val', or 'test', got {stage}")
        
        # Reset metrics for new epoch
        self.metrics.reset_meters(stage)
        
        # Create appropriate progress bar
        if self.is_rank0:
            if stage == "train":
                self.create_training_progress_bar(data_loader, epoch, master_process)
            elif stage == "val":
                self.create_validation_progress_bar(data_loader, epoch, master_process)
            else:  # test
                self.create_test_progress_bar(data_loader, epoch, master_process)
        
        # Set model layout for this phase (if vitals dataset available)
        if stage == "train":
            if hasattr(self, 'train_vitals_dataset'):
                self.unwrap(model).set_layout(self.train_vitals_dataset)
                logger.debug(f"Training phase: Model layout set using training vitals_dataset")
            model.train()
            context_manager = torch.enable_grad()
        elif stage == "val":
            if hasattr(self, 'val_vitals_dataset'):
                self.unwrap(model).set_layout(self.val_vitals_dataset)
                logger.debug(f"Validation phase: Model layout set using validation vitals_dataset")
            model.eval()
            context_manager = torch.no_grad()
        else:  # test
            if hasattr(self, 'test_vitals_dataset'):
                self.unwrap(model).set_layout(self.test_vitals_dataset)
                logger.debug(f"Test phase: Model layout set using test vitals_dataset")
            model.eval()
            context_manager = torch.no_grad()
        
        try:
            with context_manager:
                for step, batch in enumerate(data_loader):
                    # Move tensors to device (filtering non-tensor values)
                    prepared_batch = {
                        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
                        for k, v in batch.items()
                    }
                    
                    if stage == "train":
                        # ✅ GAN-specific dual optimization logic using mode-based _step_core
                        
                        # Train discriminator n_critic times
                        for _ in range(self.n_critic):
                            self.optimizer["D"].zero_grad(set_to_none=True)
                            d_loss, d_metrics = self._step_core(
                                model, prepared_batch, stage=stage, mode="discriminator"
                            )
                            d_loss.backward()
                            self.optimizer["D"].step()
                        
                        # Train generator once
                        self.optimizer["G"].zero_grad(set_to_none=True)
                        g_loss, g_metrics = self._step_core(
                            model, prepared_batch, stage=stage, mode="generator"
                        )
                        g_loss.backward()
                        self.optimizer["G"].step()
                        
                        # Combine metrics from both discriminator and generator
                        metrics = {**d_metrics, **g_metrics}
                        
                    else:
                        # ✅ Validation/test: use _step_core with validation mode
                        d_loss, g_loss, metrics = self._step_core(
                            model, prepared_batch, stage=stage, mode="validation"
                        )
                    
                    # ✅ UNIFIED METRICS LOGGING (every rank)
                    with self.metrics.aggregate(stage):
                        self.log_step_metrics_unified(metrics, step)
                    
                    # ✅ UNIFIED PROGRESS BAR UPDATE (rank-0 only)
                    if self.is_rank0:
                        current_metrics = self.metrics.get_smoothed_values(stage)
                        to_log = True if stage == "train" else False
                        self.update_progress_bar(
                            current_metrics, 
                            step, 
                            is_rank0=self.is_rank0, 
                            to_log=to_log
                        )
                    
                    # ✅ UNIFIED MEMORY CLEANUP
                    del batch, prepared_batch, d_loss, g_loss, metrics
        
        finally:
            if self.is_rank0:
                self.close_progress_bar()
        
        # Memory cleanup after epoch (not per-batch to avoid fragmentation)
        torch.cuda.empty_cache()
        
        # Synchronize metrics across all ranks for correct global averages
        _, world_size, _ = self._get_distributed_config()
        self.metrics.sync_distributed(device, world_size=world_size)
        
        # Get aggregated metrics (now properly synchronized across all ranks)
        stage_metrics = self.metrics.get_smoothed_values(stage)
        
        # Print validation results for user feedback (matching legacy behavior)
        if stage == "val" and master_process:
            logger.info(f"\nValidation Results:")
            logger.info(f"D Loss: {stage_metrics.get('d_loss', 0.0):.4f}")
            logger.info(f"G Loss: {stage_metrics.get('g_loss', 0.0):.4f}")
            if "real_pred" in stage_metrics:
                logger.info(f"Real Pred: {stage_metrics['real_pred']:.4f}, Fake Pred: {stage_metrics['fake_pred']:.4f}")
        
        return stage_metrics

    def _train_epoch(self, epoch, model, train_loader, optim, device, master_process):
        """Training epoch using the generic runner."""
        return self._run_epoch(epoch, model, train_loader, device, master_process, "train", optim)

    def _validate_epoch(self, epoch, model, val_loader, device, master_process):
        """Validation epoch using the generic runner."""
        return self._run_epoch(epoch, model, val_loader, device, master_process, "val")

    def _test_epoch(self, epoch, model, test_loader, device, master_process):
        """Test epoch using the generic runner."""
        return self._run_epoch(epoch, model, test_loader, device, master_process, "test")
    
    def _setup_ddp_wrapping(self, model, local_rank: int):
        """Override DDP wrapping for GAN models with proper parameter handling"""
        self.model.generator = DDP(model.generator, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=True)
        self.model.discriminator = DDP(model.discriminator, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=True)
    
    def create_optimizer(self):
        """Create GAN optimizers - called AFTER DDP wrapping"""  
        self.optimizer = {
            "G": torch.optim.Adam(self.model.generator.module.parameters(), lr=self.learning_rate, betas=self.g_betas),
            "D": torch.optim.Adam(self.model.discriminator.module.parameters(), lr=self.learning_rate, betas=self.d_betas),
        }

    def create_scheduler(self):
        """Create GAN scheduler - only for generator (matching previous implementation)"""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer["G"], 'min', patience=self.scheduler_patience
        )

    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute GAN training using provided infrastructure from BaseTrainer"""
        # Create separate vitals_dataset variables for each split (similar to other trainers)
        from src.dataset.core import get_vitals_dataset
        train_vitals_dataset = get_vitals_dataset(train_loader)
        val_vitals_dataset = get_vitals_dataset(val_loader)
        test_vitals_dataset = get_vitals_dataset(test_loader)
        
        # Store vitals datasets as instance variables for each phase
        self.train_vitals_dataset = train_vitals_dataset
        self.val_vitals_dataset = val_vitals_dataset
        self.test_vitals_dataset = test_vitals_dataset
        
        logger.info(f"Using training vitals_dataset: {[v.name for v in train_vitals_dataset.vitals()]}")
        
        # -------- Training Loop --------
        # Get training state from metadata - no need to recreate
        current_epoch = self.get_epoch()
        if self.get_best_loss() is not None:
            logger.info(f"Resuming training from epoch {current_epoch} with best_loss {self.get_best_loss()}")

        for epoch in range(current_epoch, self.num_epochs):
            # Use base trainer's method to set sampler epoch (training only)
            self.set_sampler_epoch(epoch)
            
            # -------- Train --------
            # Call the refactored _train_epoch method
            train_metrics = self._train_epoch(epoch, self.model, train_loader, self.optimizer, self.device, self.is_rank0)
            
            # -------- Validate --------
            val_metric = None
            val_metrics = {}
            if val_loader is not None:
                # Call the refactored _validate_epoch method
                val_metrics = self._validate_epoch(epoch, self.model, val_loader, self.device, self.is_rank0)
                val_metric = val_metrics.get("loss", 0.0)  # Use loss as the validation metric for scheduler

            # -------- Test Phase (if test_loader provided) --------
            test_metrics = {}
            if test_loader is not None:
                # Call the test epoch method
                test_metrics = self._test_epoch(epoch, self.model, test_loader, self.device, self.is_rank0)
                
                if self.is_rank0:
                    final_test_loss = test_metrics.get("loss", 0.0)
                    logger.info(f"\nTest Loss: {final_test_loss:.4f}")
            
            # -------- Scheduler / logging / checkpointing --------
            loss_improved = None
            if val_metric is not None:
                self.step_scheduler(val_metric)
            
                # Update early stopping
                if self.early_stopping:
                    self.early_stopping(val_metric)
                
                # Check if loss improved before updating state
                loss_improved = self.set_best_loss(val_metric)  # Check improvement first
                if loss_improved:
                    logger.info(f"New best loss: {val_metric:.6f}")
                
                # Update base trainer state (epoch only, best_loss already handled)
                self.update_training_state(epoch)
                
                if self.is_rank0:
                    final_loss = val_metrics.get("loss", 0.0)
                    logger.info(f"\nValidation Loss: {final_loss:.4f}")
            
            self.log_epoch_metrics_unified(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                best_loss=self.get_best_loss(),
                loss_improved=loss_improved  # Pass the improvement status
            )

            # Check early stopping
            if self.check_early_stopping_common():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save best model checkpoint
            if val_metric is not None and self.get_best_loss() is not None and val_metric == self.get_best_loss():
                logger.info(f"Saving best model checkpoint with loss: {self.get_best_loss():.6f}")
                self.save_checkpoint(
                    epoch=None, # Save best model checkpoint
                    dataset=train_loader,
                    model_state_dict={
                        "G": self.unwrap(self.model.generator).state_dict(),
                        "D": self.unwrap(self.model.discriminator).state_dict()
                    },
                    optimizer_state_dict={
                        "G": self.optimizer["G"].state_dict(),
                        "D": self.optimizer["D"].state_dict()
                    },
                    scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,  # Single scheduler
                    early_stopping_state=self.early_stopping.state_dict() if self.early_stopping else None,
                    additional_info={
                        "stage": self.trainer_name, 
                        "n_critic": self.n_critic
                    }
                )
            
            # Periodic checkpointing (rank-0 only, barrier handled in base)
            if epoch % self.save_checkpoint_frequency == 0:
                logger.info(f"Saving periodic checkpoint at epoch {epoch}")
                self.save_checkpoint(
                    epoch=epoch, # Save periodic checkpoint
                    dataset=train_loader,
                    model_state_dict={
                        "G": self.unwrap(self.model.generator).state_dict(),
                        "D": self.unwrap(self.model.discriminator).state_dict()
                    },
                    optimizer_state_dict={
                        "G": self.optimizer["G"].state_dict(),
                        "D": self.optimizer["D"].state_dict()
                    },
                    scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,  # Single scheduler
                    early_stopping_state=self.early_stopping.state_dict() if self.early_stopping else None,
                    additional_info={
                        "stage": self.trainer_name, 
                        "n_critic": self.n_critic
                    }
                )
        
        if self.is_rank0:
            logger.info("GAN training completed")
            self.close_all()


    def _move_model_to_device(self):
        """Override to handle GAN models with generator and discriminator
        
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
                logger.info(f"Moving GAN generator and discriminator to device: {self._get_device()}")
            
            # Move both components to device - handle DDP wrapping
            self.unwrap(self.model.generator).to(self._get_device())
            self.unwrap(self.model.discriminator).to(self._get_device())



# cs = ConfigStore.instance()
# cs.store(name="gan_trainer", node=GANTrainerConfig, group="trainer")