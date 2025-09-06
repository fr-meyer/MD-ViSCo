# Standard library imports
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.core.config_store import ConfigStore

# Local imports
from src.dataset.base_dataset import Sample
from .trainer import BaseTrainer, TrainerBaseConfig, DirectionMode
from src.core.domain import Direction, Vital
from src.dataset.core import get_vitals_dataset

logger = logging.getLogger(__name__)

@dataclass
class ApproximationTrainerConfig(TrainerBaseConfig):
    """Configuration for ApproximationTrainer with Hydra-compatible defaults"""
    _target_: str = "src.trainers.approximation_trainer.ApproximationTrainer"
    trainer_name: str = "App"

    
    # Data preprocessing configuration (inherits from TrainerBaseConfig)
    # Override default preprocessing if needed for approximation stage
    input_preprocessing: Dict[str, str] = field(default_factory=lambda: {
        "x": "waveform_minmax_zc",  # Default for approximation
        "y": "waveform_minmax_zc",  # Default for approximation
    })


class ApproximationTrainer(BaseTrainer):
    """Trainer for the approximation stage with unified support for multiple model types
    
    Supports both single-direction and multi-directional training modes.
    - MDViSCo can be trained uni-directionally (direction_mode="single") or multi-directionally (direction_mode="multi")
    - NABNet, PatchTST, PPG2ABP can only be trained uni-directionally (direction_mode="single")
    - Users must explicitly specify direction_mode="single" or "multi" - no automatic detection
    """
    
    def __init__(self,
                 *args, **kwargs):
        
        # Call parent constructor with all parameters via kwargs
        super().__init__(*args, **kwargs)
        
        # Store only approximation-specific parameters
        logger.info(f"Approximation trainer initialized with model_name: {self._get_model_name()}, direction_mode: {self.direction_mode.value}")

    def _get_main_output(self, outputs):
        """Handle different model output formats"""
        if not isinstance(outputs, (tuple, list)):
            return outputs
        return outputs[0]   

    def _extract_target_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract target from unified batch structure for loss computation.
        
        Args:
            batch: Batch dictionary with unified structure
            
        Returns:
            torch.Tensor: Target tensor [B, T] for loss computation
        """
        # Extract target from unified batch structure
        waveform = batch["x"]
        tgt_idxs = batch["tgt_idxs"]
        batch_arange = torch.arange(waveform.size(0), device=waveform.device)
        y_target = waveform[batch_arange, tgt_idxs].unsqueeze(1)  # [B, 1, T]
        
        return y_target

    def _step_core(self, model, batch, *, stage: str):
        """Pure compute: parse, forward, target, loss, metrics. No optimizer/AMP.
        
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
        y_target = self._extract_target_from_batch(batch)

        y_pred = outputs['y_pred']
        
        # Get main output for loss computation
        main_output = self._get_main_output(y_pred)
        
        # Compute loss using consistent criterion for all stages
        loss = self.criterion(y_pred, y_target)

        # Compute metrics
        mae = torch.nn.functional.l1_loss(main_output, y_target)
        mse = torch.nn.functional.mse_loss(main_output, y_target)
        
        metrics = {
            "loss": float(loss.detach()),
            "mae": float(mae.detach()),
            "mse": float(mse.detach())
        }
        
        return loss, metrics, outputs

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
            self.unwrap(model).set_layout(self.train_vitals_dataset)
            logger.debug(f"Training phase: Model layout set using training vitals_dataset")
            model.train()
            context_manager = torch.enable_grad()
        elif stage == "val":
            self.unwrap(model).set_layout(self.val_vitals_dataset)
            logger.debug(f"Validation phase: Model layout set using validation vitals_dataset")
            model.eval()
            context_manager = torch.no_grad()
        else:  # test
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
        
        # Return aggregated metrics (now properly synchronized across all ranks)
        return self.metrics.get_smoothed_values(stage)

    def _train_epoch(self, epoch, model, train_loader, optim, device, master_process):
        """Training epoch using the generic runner."""
        return self._run_epoch(epoch, model, train_loader, device, master_process, "train", optim)

    def _validate_epoch(self, epoch, model, val_loader, device, master_process):
        """Validation epoch using the generic runner."""
        return self._run_epoch(epoch, model, val_loader, device, master_process, "val")

    def _test_epoch(self, epoch, model, test_loader, device, master_process):
        """Test epoch using the generic runner."""
        return self._run_epoch(epoch, model, test_loader, device, master_process, "test")
            
    def _execute_training_logic(
        self,
        train_loader,
        val_loader,
        test_loader,
    ):
        """Execute approximation training using provided infrastructure from BaseTrainer"""
        
        # Create separate vitals_dataset variables for each split
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
                val_metric = val_metrics.get("loss", 0.0)  # Use MSE as the validation metric for scheduler
            
            # -------- Scheduler / logging / checkpointing --------
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
            logger.info("Approximation training completed")
            self.close_all()

    cs = ConfigStore.instance()
    cs.store(name="base_approximation_trainer", node=ApproximationTrainerConfig, group="trainer")