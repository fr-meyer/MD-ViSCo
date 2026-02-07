# Standard library imports
import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, List, Optional, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore

# Local imports
# from src.dataload.base_dataset import Sample


# from src.utils.train_utils import EarlyStopping
from src.utils.bp_utils import extract_bp_values
# from src.utils.utils_preprocessing import (
#     print_model_parameters,
#     isExist_dir,
#     Min_Max_Norm_Torch,
#     check_file_exists
# )
# from src.utils.checkpoint_utils import load_checkpoint_components
# Import criterion classes with factory methods
# from src.criterions.multi_regression_loss import MultiRegressionLoss
# from src.criterions.multi_regression_multi_wcl_loss import MultiRegressionMultiWCLLoss
# from src.dataload import create_training_datasets
from .trainer import BaseTrainer, TrainerBaseConfig, DirectionMode


logger = logging.getLogger(__name__)

@dataclass
class RefinementTrainerConfig(TrainerBaseConfig):
    """Configuration for RefinementTrainer with Hydra-compatible defaults"""
    _target_: str = "src.trainers.refinement_trainer.RefinementTrainer"
    trainer_name: str = "Ref"
    
    # Refinement-specific configuration
    
    # Data preprocessing configuration (inherits from TrainerBaseConfig)
    input_preprocessing: Dict[str, str] = field(default_factory=lambda: {
        "x": "waveform_minmax_zc",  # Default for refinement
        "waveform": "waveform_minmax_zc",
        "y": "waveform_minmax_zc",  # Default for refinement
        "y_abp": "abp_global_minmax",  # Default for refinement
        "y_bp": "bp_global_minmax",  # Default for refinement
        "bp_raw": "bp_raw"
    })


class RefinementTrainer(BaseTrainer):
    """Trainer for the refinement stage with unified support for multiple model types
    
    Supports both single-direction and multi-directional training modes.
    - MDViSCo can be trained uni-directionally (direction_mode="single") or multi-directionally (direction_mode="multi")
    - NABNet, PatchTST, PPG2ABP can only be trained uni-directionally (direction_mode="single")
    - Users must explicitly specify direction_mode="single" or "multi" - no automatic detection
    
    Features unified loss and metrics computation:
    - Single _step_core method computes both loss and comprehensive metrics for all stages
    - Training now gets the same rich metrics as validation (MSE, MAE, BP metrics)
    - Consistent with ApproximationTrainer architecture
    
    Channel configuration is handled by the dataset layer, not the trainer.
    """
    
    def __init__(self, *args, **kwargs):
        # Call parent constructor with all parameters via kwargs
        super().__init__(*args, **kwargs)

        # Store only refinement-specific parameters (already handled by config)
        logger.info("Refinement trainer initialized")
    
    def _validate_configuration(self):
        """Validate trainer configuration"""
        pass  # No specific validation needed for refinement trainer

    def _extract_target_from_batch(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract target from unified batch structure for loss computation.
        
        Args:
            batch: Batch dictionary with unified structure
            
        Returns:
            torch.Tensor: Target tensor for loss computation
        """
        target = {}

        if "y_bp" in batch:
            target["y_sbp"] = batch["y_bp"][:, 0]
            target["y_dbp"] = batch["y_bp"][:, 1]

        if "y_pred_waveform" in outputs:
            target["y_waveform"] = batch["y_abp"]
            return target
        
        if "y_bp" in batch:
            target["y"] = batch["y_bp"][:, 0:2].squeeze(-1)
            return target
        
        if "bp_raw" in batch:
            target["y_sbp_raw"] = batch["bp_raw"][:, 0]
            target["y_dbp_raw"] = batch["bp_raw"][:, 1]
        
        return target
        
    def _get_main_output(self, outputs) -> torch.Tensor:
        """Handle different model output formats for consistent processing.
        
        Args:
            outputs: Raw model outputs (can be tensor, tuple, dict, etc.)
            
        Returns:
            torch.Tensor: Main output tensor for loss computation
        """
        if "y" in outputs:
            # For waveform models, outputs should be a tensor
            if isinstance(outputs, torch.Tensor):
                return outputs
            elif isinstance(outputs, (list, tuple)):
                return outputs[0]  # Return first element if tuple/list
            else:
                raise ValueError(f"Unexpected output format for waveform: {type(outputs)}")
        elif "y_bp" in outputs:
            # For BP models, outputs might be a dict or tensor
            if isinstance(outputs, dict):
                # Return the main prediction key (implementation depends on model)
                if "pred" in outputs:
                    return outputs["pred"]
                elif "output" in outputs:
                    return outputs["output"]
                else:
                    # Return first tensor value
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            return value
                    raise ValueError("No tensor found in outputs dict")
            elif isinstance(outputs, torch.Tensor):
                return outputs
            else:
                raise ValueError(f"Unexpected output format for scalar_bp: {type(outputs)}")
        else:
            raise ValueError(f"Unexpected output format: {type(outputs)}")
    
    def _compute_loss(self, model_outputs, batch, model, ground_truth):
        """Unified loss computation for all stages (train/val/test).
        
        This method replaces the old compute_training_loss and works for
        both training and validation stages.
        
        Args:
            model_outputs: Model predictions
            batch: Input batch containing targets
            model: Model instance
            
        Returns:
            torch.Tensor: Computed loss
        """
        if "y_pred_waveform" in model_outputs:
            return self.criterion(model_outputs['y_pred_waveform'], ground_truth['y_waveform'])
        

        return self.criterion(input=model_outputs['y_pred'], target=ground_truth['y'], **model_outputs, **ground_truth)

    # ============================================================================
    # UNIFIED LOSS AND METRICS COMPUTATION METHODS
    # ============================================================================
    
    def _compute_metrics(self, model_outputs, batch, model, ground_truth):
        """Unified metrics computation for all stages (train/val/test).
        
        This method replaces the old compute_validation_metrics and works for
        both training and validation stages, providing comprehensive metrics.
        
        Unified method that handles all model types and batch formats:
        - MDViSCo models with ECG/PPG predictions
        - General waveform models  
        - BP prediction models
        
        Args:
            model_outputs: Model outputs (dict or tensor)
            batch: Input batch with various BP key patterns
            model: Model instance
            
        Returns:
            dict: Comprehensive metrics (mse, mae, bp_metrics, etc.)
        """
        metrics = {}

        sbp_mae = torch.nn.functional.l1_loss(model_outputs["y_pred_sbp"], ground_truth["y_sbp"]).item()
        dbp_mae = torch.nn.functional.l1_loss(model_outputs["y_pred_dbp"], ground_truth["y_dbp"]).item()
        sbp_mse = torch.nn.functional.mse_loss(model_outputs["y_pred_sbp"], ground_truth["y_sbp"]).item()
        dbp_mse = torch.nn.functional.mse_loss(model_outputs["y_pred_dbp"], ground_truth["y_dbp"]).item()

        metrics.update({
                    'sbp_mae': sbp_mae,
                    'dbp_mae': dbp_mae,
                    'sbp_mse': sbp_mse,
                    'dbp_mse': dbp_mse
                })

        return metrics

    def _step_core(self, model, batch, *, stage: str):
        """Pure compute: parse, forward, target, loss, metrics. No optimizer/AMP.
        
        Unified method that computes both loss and metrics for all stages.
        Follows the same pattern as ApproximationTrainer for consistency.
        
        Args:
            model: The model (DDP-wrapped or not)
            batch: Batch dictionary with unified structure
            stage: Training stage ("train", "val", "test")
            
        Returns:
            loss: Computed loss
            metrics: Dictionary of metrics
            outputs: Model outputs
        """
        # Pass raw batch to model, let it handle input processing
        outputs = model(batch)  # Call forward on wrapper (DDP intercepts automatically)

        # Extract target using centralized method
        y_target = self._extract_target_from_batch(batch, outputs)
        
        # Compute loss using the unified loss computation
        loss = self._compute_loss(outputs, batch, model, y_target)
        
        # Compute comprehensive metrics using the unified metrics computation
        metrics = self._compute_metrics(outputs, batch, model, y_target)
        
        # Add loss to metrics for consistency with approximation trainer
        metrics["loss"] = float(loss.detach())
        
        return loss, metrics, outputs
    
    # ============================================================================
    # UNIFIED EPOCH METHODS - Following ApproximationTrainer pattern
    # ============================================================================
    
    def _run_epoch(self, epoch, model, data_loader, device, master_process, stage: str, optim=None):
        """Generic epoch runner for both training and validation.
        
        Args:
            epoch: Current epoch number
            model: The model to run
            data_loader: DataLoader for the stage
            device: Device to run on
            master_process: Whether this is the master process
            stage: Stage name ("train", "val", or "test")
            optim: Optimizer (required for training, None for validation/test)
        """
        # Validate stage parameter
        if stage not in ["train", "val", "test"]:
            raise ValueError(f"Stage must be 'train', 'val', or 'test', got {stage}")
        
        # CRITICAL: Set sampler epoch for distributed training
        if hasattr(data_loader, 'sampler') and data_loader.sampler is not None:
            data_loader.sampler.set_epoch(epoch)
        
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
        
        # Set model layout for this phase
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
                    # Batch already contains direction metadata from collate
                    # Just move tensors to device (filtering non-tensor values)
                    prepared_batch = {
                        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v 
                        for k, v in batch.items()
                    }
                    
                    # Use prepared_batch with unified structure
                    loss, metrics, outputs = self._step_core(model, prepared_batch, stage=stage)
                    
                    # Stage-specific operations
                    if stage == "train":
                        # Training side effects
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                        optim.step()
                    
                    # Use unified metrics logging (every rank)
                    with self.metrics.aggregate(stage):
                        # Use metrics directly from _step_core (loss is already included)
                        self.log_step_metrics_unified(metrics, step)
                    
                    # Update progress bar (rank-0 only) with stage awareness
                    if self.is_rank0:
                        current_metrics = self.metrics.get_smoothed_values(stage)
                        to_log = True if stage == "train" else False
                        self.update_progress_bar(
                            current_metrics, 
                            step, 
                            is_rank0=self.is_rank0, 
                            to_log=to_log
                        )
                    
                    # Clear memory (avoid per-batch empty_cache to prevent fragmentation)
                    del batch, prepared_batch, outputs, loss, metrics
        
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
            logger.info(f"Waveform: MSE={stage_metrics.get('loss', 0.0):.4f}, MAE={stage_metrics.get('mae', 0.0):.4f}")
            if "sbp_mae" in stage_metrics:
                logger.info(f"BP: SBP MAE={stage_metrics['sbp_mae']:.2f}, DBP MAE={stage_metrics['dbp_mae']:.2f}")
                logger.info(f"BP: SBP MSE={stage_metrics['sbp_mse']:.2f}, DBP MSE={stage_metrics['dbp_mse']:.2f}")
        
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
    
    # ============================================================================
    # LEGACY METHODS REMOVED
    # ============================================================================
    # train_single_direction() method removed - functionality fully replaced by:
    # - BaseTrainer.run_training() for infrastructure setup
    # - _execute_training_logic() for training implementation  
    # - Unified epoch methods (_train_epoch, _validate_epoch, _test_epoch)
    # ============================================================================

    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute refinement training using provided infrastructure from BaseTrainer"""
        
        # Create separate vitals_dataset variables for each split
        from src.dataset.core import get_vitals_dataset
        train_vitals_dataset = get_vitals_dataset(train_loader)
        val_vitals_dataset = get_vitals_dataset(val_loader)
        test_vitals_dataset = get_vitals_dataset(test_loader)
        
        # Store vitals datasets as instance variables for each phase
        self.train_vitals_dataset = train_vitals_dataset
        self.val_vitals_dataset = val_vitals_dataset
        self.test_vitals_dataset = test_vitals_dataset
        
        logger.info(f"Using training vitals_dataset: {[v.name for v in train_vitals_dataset.vitals()]}")
        
        # Determine training direction
        if self.direction_mode == DirectionMode.MULTI:
            if self.is_rank0:
                logger.info(f"Training multi-directional model with directions: {[d.key() for d in self.directions]}")
        else:
            # Single direction training: use first available direction
            if self.is_rank0:
                logger.info(f"Training single-direction {[d.key() for d in self.directions]} model...")
        
        # -------- Training Loop --------
        # Get training state from metadata - no need to recreate
        current_epoch = self.get_epoch()
        if self.get_best_loss() is not None:
            logger.info(f"Resuming training from epoch {current_epoch} with best_loss {self.get_best_loss()}")

        for epoch in range(current_epoch, self.num_epochs):
            # Use base trainer's method to set sampler epoch (training only)
            self.set_sampler_epoch(epoch)
            
            # -------- Train --------
            # Set layout for training phase
            self.unwrap(self.model).set_layout(self.train_vitals_dataset)
            logger.debug(f"Training phase: Model layout set using training vitals_dataset")
            
            # Call the refactored _train_epoch method
            train_metrics = self._train_epoch(epoch, self.model, train_loader, self.optimizer, self.device, self.is_rank0)
            
            # -------- Validate --------
            val_metric = None
            val_metrics = {}
            if val_loader is not None:
                # Set layout for validation phase
                self.unwrap(self.model).set_layout(self.val_vitals_dataset)
                logger.debug(f"Validation phase: Model layout set using validation vitals_dataset")
                
                # Call the refactored _validate_epoch method
                val_metrics = self._validate_epoch(epoch, self.model, val_loader, self.device, self.is_rank0)
                val_metric = val_metrics.get("loss", 0.0)  # Use loss as the validation metric for scheduler

            
            # -------- Scheduler / logging / checkpointing --------
            loss_improved = None
            if val_metric is not None:
                self.step_scheduler(val_metric)
                
                # NEW: Update early stopping
                if self.early_stopping:
                    self.early_stopping(val_metric)
                
                # NEW: Check if loss improved before updating state
                loss_improved = self.set_best_loss(val_metric)  # Check improvement first
                if loss_improved:
                    logger.info(f"New best loss: {val_metric:.6f}")
                
                # NEW: Update base trainer state (epoch only, best_loss already handled)
                self.update_training_state(epoch)
                
                if self.is_rank0:
                    final_loss = val_metrics.get("loss", 0.0)
                    final_mse = val_metrics.get("mse", 0.0)
                    final_mae = val_metrics.get("mae", 0.0)
                    logger.info(f"\nValidation Loss: {final_loss:.4f}, MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")

            # -------- Test Phase (if test_loader provided) --------
            test_metrics = {}
            if test_loader is not None:
                # Set layout for test phase
                self.unwrap(self.model).set_layout(self.test_vitals_dataset)
                logger.debug(f"Test phase: Model layout set using test vitals_dataset")
                
                # Call the test epoch method
                test_metrics = self._test_epoch(epoch, self.model, test_loader, self.device, self.is_rank0)
                
                if self.is_rank0:
                    final_test_loss = test_metrics.get("loss", 0.0)
                    final_test_mse = test_metrics.get("mse", 0.0)
                    final_test_mae = test_metrics.get("mae", 0.0)
                    logger.info(f"\nTest Loss: {final_test_loss:.4f}, MSE: {final_test_mse:.4f}, MAE: {final_test_mae:.4f}")

            self.log_epoch_metrics_unified(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                best_loss=self.get_best_loss(),
                loss_improved=loss_improved  # Pass the improvement status
            )

            # NEW: Check early stopping
            if self.check_early_stopping_common():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # NEW: Save best model checkpoint
            if val_metric is not None and self.get_best_loss() is not None and val_metric == self.get_best_loss():
                logger.info(f"Saving best model checkpoint with loss: {self.get_best_loss():.6f}")
                self.save_checkpoint(
                    epoch=None, # Save best model checkpoint
                    dataset=train_loader,
                    model_state_dict={'model': self.unwrap(self.model).state_dict()},
                    optimizer_state_dict={'optimizer': self.optimizer.state_dict()},
                    scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                    early_stopping_state=self.early_stopping.state_dict() if self.early_stopping else None,
                    additional_info={}
                )
                
            
            # Periodic checkpointing (rank-0 only, barrier handled in base)
            if epoch % self.save_checkpoint_frequency == 0:
                logger.info(f"Saving periodic checkpoint at epoch {epoch}")
                self.save_checkpoint(
                    epoch=epoch, # Save periodic checkpoint
                    dataset=train_loader,
                    model_state_dict={'model': self.unwrap(self.model).state_dict()},
                    optimizer_state_dict={'optimizer': self.optimizer.state_dict()},
                    scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                    early_stopping_state=self.early_stopping.state_dict() if self.early_stopping else None,
                    additional_info={}
                )
        
        if self.is_rank0:
            logger.info("Refinement training completed")
            self.close_all()


cs = ConfigStore.instance()
cs.store(name="base_refinement_trainer", node=RefinementTrainerConfig, group="trainer")