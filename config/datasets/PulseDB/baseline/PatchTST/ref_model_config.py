# Standard library imports
import os
import time
from dataclasses import dataclass

# Third-party imports
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Local imports
from config.datasets.dataset_configs import PulseDBBaseConfig
from baseline.PatchTST import RefinementPatchTST
from utils.ddp_utils import print_memory_stats, create_ddp_dataloaders
from utils.utils_preprocessing import (
    print_model_parameters,
    isExist_dir,
    Min_Max_Norm_Torch
)
from utils.test_utils import (
    calculate_bp_metrics,
    print_abp_evaluation_results
)
from utils.train_utils import EarlyStopping
from .app_model_config import PulseDBApproximationPatchTSTConfig

@dataclass
class PulseDBRefinementPatchTSTConfig(PulseDBBaseConfig):
    """PulseDB dataset configuration for PatchTST model for BP prediction.
    Predicts SBP and DBP from waveform (PPG, ECG) and demographic information."""
    
    args: dict = None
    # Direction configuration
    direction: str = None  # If None, train all combinations, otherwise format: "SOURCE2TARGET" (e.g. "PPG2ABP")
    
    # Base configuration
    path_folder: str = 'RefModel/PulseDB'

    is_finetuning: bool = False
    use_patient_split: bool = False
    
    # Model Architecture configuration
    context_length: int = 1285  # Length of input sequence
    patch_len: int = 16  # Length of each patch
    stride: int = 8  # Stride between patches
    d_model: int = 1024  # Dimension of model
    num_encoder_layers: int = 16  # Number of encoder layers
    num_heads: int = 16  # Number of attention heads
    dropout: float = 0.1  # Attention dropout rate
    fc_dropout: float = 0.1  # Dropout rate for fully connected layers
    head_dropout: float = 0.1  # Dropout rate for prediction head
    use_cls_token: bool = False  # Use CLS token for BP prediction
    
    # Demographics configuration
    pi: bool = False  # Whether to use patient information/demographics
    in_channels: int = 1  # 1 waveform + 5 demographics when pi=True, 1 when pi=False
    num_targets: int = 2  # SBP and DBP

    def __post_init__(self):
        # Call parent class's __post_init__
        super().__post_init__()
        if self.args is not None:
            self.batch_size = self.args.batch_size
            self.test_batch_size = self.args.test_batch_size
            self.num_epochs = self.args.num_epochs
            self.learning_rate = self.args.learning_rate
            self.scheduler_patience = self.args.scheduler_patience
            if hasattr(self.args, 'resume_training'):
                self.resume_training = self.args.resume_training
            self.early_stopping_patience = self.args.early_stopping_patience
            self.checkpoint_name = self.args.checkpoint_name
            self.checkpoint_epoch = self.args.checkpoint_epoch
            self.direction = self.args.direction
            self.model_type = self.args.model_type

            # Check if pi flag is in args
            if hasattr(self.args, 'pi'):
                self.pi = self.args.pi
            if hasattr(self.args, 'is_finetuning'):
                self.is_finetuning = self.args.is_finetuning
            if hasattr(self.args, 'use_patient_split'):
                self.use_patient_split = self.args.use_patient_split
            if hasattr(self.args, 'is_pretraining'):
                self.is_pretraining = self.args.is_pretraining
            
            if hasattr(self.args, 'seed'):
                self.seed = self.args.seed
        
        # Update in_channels based on pi flag
        self.in_channels = 1 if self.pi else 1
        
        # Parse direction if specified
        if self.direction is not None:
            source, target = self._parse_direction(self.direction)
            self.source_channel = getattr(self, f"{source.lower()}_label")
            self.target_channel = getattr(self, f"{target.lower()}_label")
        self.set_seed()

    def create_model(self):
        """Create RefinementPatchTST model instance"""
        return RefinementPatchTST(
            input_size=self.in_channels,  # Will be 1 if pi=False, 6 if pi=True
            output_size=self.num_targets,
            context_length=self.context_length,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            head_dropout=self.head_dropout,
            use_cls_token=self.use_cls_token
        )

    def _prepare_batch(self, batch_data, device, source_channel, target_channel):
        """
        Prepare batch data by appending demographic information sequence-wise instead of channel-wise.
        This reduces memory usage by not expanding demographics across all timesteps.
        """
        # Get waveform from source channel
        waveform = batch_data[8][:, source_channel:source_channel+1]  # [batch_size, 1, seq_len]
        waveform = waveform.transpose(1, 2)  # [batch_size, seq_len, 1]
        
        # Get target BP values
        sbp = batch_data[10][:, 0:1]
        dbp = batch_data[10][:, 1:2]
        target_values = torch.cat([sbp, dbp], dim=1)
        
        if self.pi:
            batch_size, seq_len, wav_channel = waveform.shape
            
            # Get demographic data - age, gender, height, weight, bmi
            demographics = batch_data[5]  # shape: [batch_size, 5]
            
            # Create mask for missing demographic values
            demo_mask = ~torch.isnan(demographics)  # [batch_size, 5]
            
            # Replace NaN with zeros
            demographics = torch.nan_to_num(demographics, nan=0.0)
            
            # Reshape demographics to append as sequence
            # Take each demographic value and make it a sequence point
            demo_seq = demographics.unsqueeze(-1)  # [batch_size, 5, 1]
            
            # Concatenate along sequence dimension
            combined_input = torch.cat([waveform, demo_seq], dim=1)  # [batch_size, 1285, 1]
            
            # Create observed mask
            waveform_mask = torch.ones((batch_size, seq_len, 1), dtype=torch.bool)
            demo_mask = demo_mask.unsqueeze(-1)  # [batch_size, 5, 1]
            past_observed_mask = torch.cat([waveform_mask, demo_mask], dim=1)  # [batch_size, 1285, 1]
            
            return {
                "x": combined_input.to(device, torch.float32),
                "target_values": target_values.to(device, torch.float32),
                "past_observed_mask": past_observed_mask.to(device)
            }
        else:
            # If not using demographic information, just use the waveform
            return {
                "x": waveform.to(device, torch.float32),
                "target_values": target_values.to(device, torch.float32),
                "past_observed_mask": None  # No mask needed when not using demographics
            }

    def get_source_target_pairs(self):
        """Get all valid source-target pairs for training"""
        # All possible signal types as sources (ECG, PPG)
        sources = [self.ecg_label, self.ppg_label]
        target = self.abp_label  # ABP is always the target
        
        # Generate all valid pairs (source, target)
        pairs = []
        for source in sources:
            pairs.append((source, target))
        return pairs

    def get_model_name(self, source, target):
        """Get model name for a specific source-target pair"""
        return f"{self.channel_names[source]}2{self.channel_names[target]}"

    def get_checkpoint_path(self, source, target, epoch: int = None, saving: bool = False):
        """Get checkpoint path for a specific source-target pair"""
        model_name = self.get_model_name(source, target)
        sub_folder = f'RefModel_PatchTST_BS_{self.batch_size}_E_{self.num_epochs}_LR_{self.learning_rate}_PI_{self.pi}_P_{self.scheduler_patience}_ES_{self.early_stopping_patience}'
        
        # Add finetuning suffix if in finetuning mode
        if self.is_finetuning:
            sub_folder += '_finetuning'
        
        # Add patient split suffix if using patient split
        if self.use_patient_split:
            sub_folder += '_Patient_Split'
        
        checkpoint_dir = os.path.join(self.abs_path, self.path_folder, sub_folder)

        if saving:
            if epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}_epoch_{epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}.pt')
        else:
            if self.checkpoint_epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}_epoch_{self.checkpoint_epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}.pt')
        
        return {
            'dir': checkpoint_dir,
            'file': checkpoint_file,
            'sub_folder': sub_folder,
            'is_finetuning': self.is_finetuning  # Add this field to match reference implementations
        }

    def _parse_direction(self, direction):
        """Parse direction string into source channel and ABP target"""
        try:
            source, target = direction.split('2')
            valid_sources = set(self.channel_names.values()) - {'ABP'}  # All channels except ABP
            if source not in valid_sources:
                raise ValueError(f"Invalid source channel '{source}'. Valid source channels are {valid_sources}")
            if target != 'ABP':
                raise ValueError(f"Invalid target channel '{target}'. Only 'ABP' is allowed as target for BP prediction")
            return source, target
        except ValueError as e:
            raise ValueError(f"Invalid direction format {direction}. Must be SOURCE2ABP (e.g. PPG2ABP). {str(e)}")

    def _train_epoch(self, epoch, ddp_model, train_loader, optim, device, master_process, 
                     source_channel, target_channel):
        """Run one training epoch for BP prediction"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        # Initialize epoch metrics
        epoch_mae_loss = 0.0  # Total MAE loss for training
        epoch_sbp_mae = 0.0
        epoch_dbp_mae = 0.0
        epoch_sbp_mse = 0.0
        epoch_dbp_mse = 0.0
        num_batches = 0
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            ddp_model.train()
            tepoch.set_description(f"Train - Epoch {epoch}")
            
            for step, train_file in enumerate(tepoch):
                # Prepare batch with source/target channels
                batch = self._prepare_batch(train_file, device, source_channel, target_channel)
                
                # Training step
                optim.zero_grad(set_to_none=True)
                
                # Forward pass
                outputs = ddp_model(
                    batch["x"],
                    past_observed_mask=batch.get("past_observed_mask", None)
                )
                
                # Calculate MAE loss for training (combined loss)
                mae_loss = F.l1_loss(outputs[:, :], batch["target_values"][:, :])
                with torch.no_grad():
                # Calculate separate losses for SBP and DBP for monitoring
                    sbp_mae = F.l1_loss(outputs[:, 0].detach(), batch["target_values"][:, 0])
                    dbp_mae = F.l1_loss(outputs[:, 1].detach(), batch["target_values"][:, 1])
                    sbp_mse = F.mse_loss(outputs[:, 0].detach(), batch["target_values"][:, 0])
                    dbp_mse = F.mse_loss(outputs[:, 1].detach(), batch["target_values"][:, 1])

                # Use MAE for training
                mae_loss.backward()
                optim.step()

                # Accumulate losses for epoch average
                epoch_mae_loss += mae_loss.item()
                epoch_sbp_mae += sbp_mae.item()
                epoch_dbp_mae += dbp_mae.item()
                epoch_sbp_mse += sbp_mse.item()
                epoch_dbp_mse += dbp_mse.item()
                num_batches += 1

                # Log training metrics per step (only for master process)
                if master_process and wandb.run is not None:
                    try:
                        wandb.log({
                            "train/step": step + epoch * len(train_loader),
                            "train/step_loss": mae_loss.item(),
                            "train/step_sbp_mae": sbp_mae.item(),
                            "train/step_dbp_mae": dbp_mae.item(),
                            "train/step_sbp_mse": sbp_mse.item(),
                            "train/step_dbp_mse": dbp_mse.item()
                        })
                    except Exception as e:
                        print(f"Failed to log to wandb: {e}")
                
                # Update progress bar
                tepoch.set_postfix(
                    mae=f"{mae_loss.item():.4f}",
                    sbp_mae=f"{sbp_mae.item():.4f}",
                    dbp_mae=f"{dbp_mae.item():.4f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}"
                )
                
                # Clear memory
                del batch, mae_loss, outputs, sbp_mae, dbp_mae, sbp_mse, dbp_mse
                torch.cuda.empty_cache()
        
        # Calculate epoch averages
        metrics = {
            "mae": epoch_mae_loss / num_batches,
            "sbp_mae": epoch_sbp_mae / num_batches,
            "dbp_mae": epoch_dbp_mae / num_batches,
            "sbp_mse": epoch_sbp_mse / num_batches,
            "dbp_mse": epoch_dbp_mse / num_batches
        }
        
        return metrics

    def _validate_epoch(self, epoch, ddp_model, val_loader, device, master_process,
                       source_channel, target_channel):
        """Run one validation epoch for BP prediction"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch_val:
                ddp_model.eval()
                total_mae_loss = 0.0
                total_sbp_mae = 0.0
                total_dbp_mae = 0.0
                total_sbp_mse = 0.0
                total_dbp_mse = 0.0
                num_batches = 0
                tepoch_val.set_description(f"Val - Epoch {epoch}")
                
                for val_file in tepoch_val:
                    # Prepare batch with source/target channels
                    batch = self._prepare_batch(val_file, device, source_channel, target_channel)
                    
                    # Forward pass
                    outputs = ddp_model(
                        batch["x"],
                        past_observed_mask=batch.get("past_observed_mask", None)
                    )
                    
                    # Calculate losses
                    mae_loss = F.l1_loss(outputs[:, :], batch["target_values"][:, :])
                    sbp_mae = F.l1_loss(outputs[:, 0], batch["target_values"][:, 0])
                    dbp_mae = F.l1_loss(outputs[:, 1], batch["target_values"][:, 1])
                    sbp_mse = F.mse_loss(outputs[:, 0], batch["target_values"][:, 0])
                    dbp_mse = F.mse_loss(outputs[:, 1], batch["target_values"][:, 1])
                    
                    # Accumulate losses
                    total_mae_loss += mae_loss.item()
                    total_sbp_mae += sbp_mae.item()
                    total_dbp_mae += dbp_mae.item()
                    total_sbp_mse += sbp_mse.item()
                    total_dbp_mse += dbp_mse.item()
                    num_batches += 1
                    
                    # Update progress bar
                    tepoch_val.set_postfix(
                        mae=f"{mae_loss.item():.4f}",
                        sbp_mae=f"{sbp_mae.item():.4f}",
                        dbp_mae=f"{dbp_mae.item():.4f}"
                    )
                    
                    # Clear memory
                    del batch, outputs, mae_loss, sbp_mae, dbp_mae, sbp_mse, dbp_mse
                    torch.cuda.empty_cache()
                
                # Calculate average losses
                metrics = {
                    "mae": total_mae_loss / num_batches,
                    "sbp_mae": total_sbp_mae / num_batches,
                    "dbp_mae": total_dbp_mae / num_batches,
                    "sbp_mse": total_sbp_mse / num_batches,
                    "dbp_mse": total_dbp_mse / num_batches
                }
                
                # Gather metrics from all processes
                metrics_tensor = torch.tensor(
                    [metrics[k] for k in metrics.keys()],
                    device=device
                )
                gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_metrics, metrics_tensor)
                
                final_metrics = torch.stack(gathered_metrics).mean(dim=0)
                
                # Convert back to dictionary
                metrics = {k: final_metrics[i].item() for i, k in enumerate(metrics.keys())}
                
                if master_process:
                    print(f"\nValidation MAE: {metrics['mae']:.4f}")
                    print(f"SBP MAE: {metrics['sbp_mae']:.4f}, DBP MAE: {metrics['dbp_mae']:.4f}")
                    print(f"SBP MSE: {metrics['sbp_mse']:.4f}, DBP MSE: {metrics['dbp_mse']:.4f}")
                
                return metrics

    def load_checkpoint_with_retry(self, rank, model, optim, scheduler, early_stopping, checkpoint_path,
                                 max_retries=3, wait_time=5):
        """Helper function to load checkpoint with retry logic"""
        training_state = {
            'epoch_checkpoint': 0,
            'best_recon_total_loss': None
        }
        
        # Initialize checkpoint_loaded for all ranks
        checkpoint_loaded = False        
        if rank == 0:  # Only rank 0 handles initial loading
            try:
                if self.is_finetuning:
                    if self.resume_training:
                        # Try loading finetuning checkpoint first
                        try:
                            # Check if finetuning checkpoint exists
                            is_finetuning_orig = self.is_finetuning
                            checkpoint_info = self.get_checkpoint_path(self.source_channel, self.target_channel)
                            
                            if os.path.exists(checkpoint_info['file']):
                                checkpoint = torch.load(checkpoint_info['file'])
                                
                                # Load model state dict
                                state_dict = checkpoint['model_state_dict']
                                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                                model.load_state_dict(new_state_dict, strict=False)
                                
                                # Load complete training state
                                training_state['epoch_checkpoint'] = checkpoint['epoch']
                                training_state['best_recon_total_loss'] = checkpoint.get('best_recon_total_loss', None)
                                
                                # Load early stopping state
                                if 'early_stopping_state' in checkpoint:
                                    early_stopping.load_state_dict(checkpoint['early_stopping_state'])
                                
                                # Load optimizer and scheduler states
                                optim.load_state_dict(checkpoint['optim_state_dict'])
                                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                                
                                checkpoint_loaded = True
                                print(f"Resumed finetuning from existing finetuning checkpoint {checkpoint_info['file']}")
                            else:
                                raise FileNotFoundError(f"No finetuning checkpoint found in {checkpoint_info['file']}")
                        except Exception as e:
                            print(f"Failed to load finetuning checkpoint: {e}")
                            print("Attempting to load regular checkpoint for finetuning...")
                            try:
                                # Switch to regular checkpoint
                                is_finetuning_orig = self.is_finetuning
                                self.is_finetuning = False
                                checkpoint_info = self.get_checkpoint_path(self.source_channel, self.target_channel)
                                self.is_finetuning = is_finetuning_orig
                                
                                if os.path.exists(checkpoint_info['file']):
                                    checkpoint = torch.load(checkpoint_info['file'])
                                    
                                    # Load model state dict only
                                    state_dict = checkpoint['model_state_dict']
                                    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                                    model.load_state_dict(new_state_dict, strict=False)
                                    
                                    # Reset training state for new finetuning
                                    training_state['epoch_checkpoint'] = 0
                                    training_state['best_recon_total_loss'] = None
                                    
                                    checkpoint_loaded = False
                                    print(f"Starting finetuning from regular checkpoint {checkpoint_info['file']}")
                                else:
                                    raise FileNotFoundError(f"No regular checkpoint found for finetuning in {checkpoint_info['file']}")
                            except Exception as e:
                                print(f"Failed to load regular checkpoint for finetuning: {e}")
                                print("Starting finetuning from scratch")
                    else:
                        # Not resuming, must load regular checkpoint for finetuning
                        try:
                            # Switch to regular checkpoint
                            is_finetuning_orig = self.is_finetuning
                            self.is_finetuning = False
                            checkpoint_info = self.get_checkpoint_path(self.source_channel, self.target_channel)
                            self.is_finetuning = is_finetuning_orig
                            
                            if os.path.exists(checkpoint_info['file']):
                                checkpoint = torch.load(checkpoint_info['file'])
                                
                                # Load model state dict only
                                state_dict = checkpoint['model_state_dict']
                                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                                model.load_state_dict(new_state_dict, strict=False)
                                
                                # Reset training state for new finetuning
                                training_state['epoch_checkpoint'] = 0
                                training_state['best_recon_total_loss'] = None
                                
                                checkpoint_loaded = False
                                print(f"Starting new finetuning from regular checkpoint {checkpoint_info['file']}")
                            else:
                                raise FileNotFoundError(f"No regular checkpoint found for finetuning in {checkpoint_info['file']}")
                        except Exception as e:
                            print(f"Failed to load regular checkpoint for finetuning: {e}")
                            print("Starting finetuning from scratch")
                else:  # Regular training
                    if self.resume_training:
                        # Check if checkpoint exists
                        if os.path.exists(checkpoint_path):
                            # Try to load complete checkpoint
                            for attempt in range(max_retries):
                                try:
                                    print(f"Loading checkpoint (attempt {attempt + 1}/{max_retries})")
                                    checkpoint = torch.load(checkpoint_path)

                                    # Load model state dict
                                    state_dict = checkpoint['model_state_dict']
                                    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                                    
                                    missing_keys, unexpected_keys = model.load_state_dict(
                                        new_state_dict,
                                        strict=False
                                    )

                                    # Print warnings about keys if any
                                    if missing_keys:
                                        print(f"Warning: Missing keys: {len(missing_keys)} keys")
                                        print(f"First few missing keys: {missing_keys[:5]}")
                                    if unexpected_keys:
                                        print(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
                                        print(f"First few unexpected keys: {unexpected_keys[:5]}")

                                    # Load complete training state
                                    training_state['epoch_checkpoint'] = checkpoint['epoch']
                                    training_state['best_recon_total_loss'] = checkpoint.get('best_recon_total_loss', None)

                                    # Load early stopping state
                                    if 'early_stopping_state' in checkpoint:
                                        early_stopping.load_state_dict(checkpoint['early_stopping_state'])

                                    # Load optimizer and scheduler states
                                    optim.load_state_dict(checkpoint['optim_state_dict'])
                                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                                    checkpoint_loaded = True
                                    print(f'Successfully loaded checkpoint from {checkpoint_path}')
                                    break

                                except Exception as e:
                                    print(f"Failed to load checkpoint (attempt {attempt + 1}): {str(e)}")
                                    if attempt < max_retries - 1:
                                        print(f"Waiting {wait_time} seconds before retrying...")
                                        time.sleep(wait_time)
                                    else:
                                        print("Failed all attempts to load checkpoint")
                                        print("Starting training from scratch...")
                        else:
                            print(f"Checkpoint not found at {checkpoint_path}. Starting training from scratch...")
                    else:
                        print(f"resume_training=False. Starting training from scratch...")
            except Exception as e:
                if self.is_finetuning:
                    raise  # Re-raise exception for finetuning
                print(f"Error during checkpoint loading: {e}")
                print("Starting training from scratch")

        # First barrier to ensure checkpoint loading decision is synchronized
        dist.barrier()
        
        # Broadcast checkpoint_loaded flag from rank 0 to all processes
        checkpoint_loaded_tensor = torch.tensor([checkpoint_loaded], device=f'cuda:{rank}')
        dist.broadcast(checkpoint_loaded_tensor, src=0)
        checkpoint_loaded = checkpoint_loaded_tensor.item()
        
        # Broadcast training state from rank 0 to all processes
        epoch_tensor = torch.tensor([training_state['epoch_checkpoint']], device=f'cuda:{rank}')
        best_loss_tensor = torch.tensor([training_state['best_recon_total_loss'] if training_state['best_recon_total_loss'] is not None else -1], device=f'cuda:{rank}')
        dist.broadcast(epoch_tensor, src=0)
        dist.broadcast(best_loss_tensor, src=0)
        
        training_state['epoch_checkpoint'] = epoch_tensor.item()
        training_state['best_recon_total_loss'] = best_loss_tensor.item() if best_loss_tensor.item() != -1 else None
        
        # Load optimizer/scheduler/early_stopping state on all non-zero ranks if checkpoint was successfully loaded
        if checkpoint_loaded and rank != 0:
            # Non-zero ranks load optimizer, scheduler, and early stopping states
            try:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
                
                # Load optimizer and scheduler states
                optim.load_state_dict(checkpoint['optim_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Load early stopping state if available
                if 'early_stopping_state' in checkpoint:
                    early_stopping.load_state_dict(checkpoint['early_stopping_state'])
                
                print(f"Rank {rank} - Successfully loaded optimizer/scheduler/early stopping states from {checkpoint_path}")
            except Exception as e:
                print(f"Rank {rank} - Warning: Could not load optimizer/scheduler/early stopping states from {checkpoint_path}: {e}")
        
        # Final barrier to ensure all processes are synchronized
        dist.barrier()
        return training_state

    def _update_training_state(self, epoch, val_loss, ddp_model, optim, scheduler,
                               early_stopping, checkpoint_path, best_loss, prev_lr):
        """Update training state including scheduler, early stopping, and checkpoints"""
        # Step scheduler and display learning rate
        scheduler.step(val_loss)
        current_lr = optim.param_groups[0]['lr']
        
        if current_lr != prev_lr:
            print(f"Learning rate changed: {prev_lr:.2e} -> {current_lr:.2e}")
        
        # Update early stopping
        early_stopping(val_loss)
        
        # Ensure checkpoint directory exists before saving
        checkpoint_dir = os.path.dirname(checkpoint_path)
        isExist_dir(checkpoint_dir)
        
        # Save checkpoint if better than previous best
        if best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss:
            torch.save({
                'model_state_dict': ddp_model.module.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),  # Save epoch+1 here
                'best_recon_total_loss': val_loss,
                'is_finetuning': self.is_finetuning  # Add this line
            }, checkpoint_path)
            best_loss['best_recon_total_loss'] = val_loss
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if epoch % 5 == 0:
            # Get correct checkpoint path for this epoch
            checkpoint_info = self.get_checkpoint_path(self.source_channel, self.target_channel, epoch, saving=True)
            checkpoint_path_epoch = checkpoint_info['file']
            torch.save({
                'model_state_dict': ddp_model.module.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),  # Save epoch+1 here
                'best_recon_total_loss': val_loss,
                'is_finetuning': self.is_finetuning  # Add this line
            }, checkpoint_path_epoch)
            print(f"Saved checkpoint to {checkpoint_path_epoch}")
        
        return current_lr

    def _check_early_stopping(self, early_stopping, rank, device):
        """Check if early stopping should be triggered"""
        if rank == 0 and early_stopping.early_stop:
            print("Early stopping triggered")
        
        # Broadcast early stopping status to all processes
        early_stop_tensor = torch.tensor([early_stopping.early_stop], device=device)
        dist.broadcast(early_stop_tensor, src=0)
        
        return early_stop_tensor.item()

    def _cleanup(self, train_loader, val_loader, test_loader=None):
        """Clean up resources"""
        try:
            # Clean up dataloader workers
            if train_loader is not None and hasattr(train_loader, '_iterator'):
                train_loader._iterator = None
            if val_loader is not None and hasattr(val_loader, '_iterator'):
                val_loader._iterator = None
            if test_loader is not None and hasattr(test_loader, '_iterator'):
                test_loader._iterator = None

            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def trainer(self, dataset: tuple, rank: int, world_size: int, ddp_config=None):
        """Train model(s) based on configuration"""
        # Check wandb initialization for master process
        if rank == 0:
            if wandb.run is None:
                print("Warning: wandb is not properly initialized!")
            else:
                print(f"wandb run: {wandb.run.name}")
            
        if self.direction is not None:
            # Train single direction
            if rank == 0:
                print(f"\nTraining {self.direction} model...")
            self._train_single_direction(dataset, rank, world_size, ddp_config,
                                      self.source_channel, self.target_channel)
        else:
            # Train all combinations
            if rank == 0:
                print("\nTraining all source-target combinations...")
            source_target_pairs = self.get_source_target_pairs()
            for source, target in source_target_pairs:
                if rank == 0:
                    print(f"\nTraining {self.channel_names[source]}2{self.channel_names[target]} model...")
                self._train_single_direction(dataset, rank, world_size, ddp_config, source, target)

    def _train_single_direction(self, dataset: tuple, rank: int, world_size: int, ddp_config, source_channel, target_channel):
        """Train for a specific source-target direction"""
        try:
            # Setup device and data
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{rank}')
            master_process = rank == 0

            # Get datasets
            train_dataset, val_dataset, test_dataset = dataset

            # Create dataloaders
            dataloader_settings = ddp_config.get_dataloader_settings() if ddp_config else None
            train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_ddp_dataloaders(
                dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                rank=rank,
                world_size=world_size,
                batch_size=self.batch_size,
                dataloader_settings=dataloader_settings
            )

            # Create and setup model
            model = self.create_model().to(device)
            if master_process:
                print_model_parameters(model)
            
            # Setup training components
            optim = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, 'min', verbose=master_process,
                patience=self.scheduler_patience,
                threshold=1e-4
            )

            early_stopping = EarlyStopping(
                patience=self.early_stopping_patience,
                threshold=1e-4,
                threshold_mode='rel',
                verbose=master_process,
                delta=0
            )

            # Get checkpoint info for this direction
            checkpoint_info = self.get_checkpoint_path(source_channel, target_channel)
            
            # Load checkpoint if exists - pass unwrapped model
            training_state = self.load_checkpoint_with_retry(
                rank=rank,
                model=model,  # Pass unwrapped model
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_path=checkpoint_info['file']
            )
            
            # Wrap model with DDP AFTER loading checkpoint
            ddp_model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=ddp_config.find_unused_parameters if ddp_config else False,
                static_graph=ddp_config.static_graph if ddp_config else True
            )

            epoch_checkpoint = training_state['epoch_checkpoint']
            best_loss = {'best_recon_total_loss': training_state['best_recon_total_loss']}
            prev_lr = optim.param_groups[0]['lr']

            # Training loop
            print_memory_stats(rank, "Before training loop")
            for e in range(epoch_checkpoint, self.num_epochs):
                # Training and validation with metrics
                train_metrics = self._train_epoch(e, ddp_model, train_loader, optim, device, master_process, 
                                source_channel=source_channel, target_channel=target_channel)
                val_metrics = self._validate_epoch(e, ddp_model, val_loader, device, master_process,
                                              source_channel=source_channel, target_channel=target_channel)
                test_metrics = self._validate_epoch(e, ddp_model, test_loader, device, master_process,
                                              source_channel=source_channel, target_channel=target_channel)

                if master_process:
                    # Get checkpoint info for this direction
                    checkpoint_info = self.get_checkpoint_path(source_channel, target_channel, saving=True)
                    prev_lr = self._update_training_state(
                        e, val_metrics["mae"], ddp_model, optim, scheduler,
                        early_stopping, checkpoint_info['file'], best_loss, prev_lr
                    )

                    # Log epoch metrics to wandb
                    if wandb.run is not None:
                        try:
                            wandb.log({
                                "epoch": e,
                                "train/epoch_loss": float(train_metrics["mae"]),
                                "train/epoch_sbp_mae": float(train_metrics["sbp_mae"]),
                                "train/epoch_dbp_mae": float(train_metrics["dbp_mae"]),
                                "train/epoch_sbp_mse": float(train_metrics["sbp_mse"]),
                                "train/epoch_dbp_mse": float(train_metrics["dbp_mse"]),
                                "val/epoch_loss": float(val_metrics["mae"]),
                                "val/epoch_sbp_mae": float(val_metrics["sbp_mae"]),
                                "val/epoch_dbp_mae": float(val_metrics["dbp_mae"]),
                                "val/epoch_sbp_mse": float(val_metrics["sbp_mse"]),
                                "val/epoch_dbp_mse": float(val_metrics["dbp_mse"]),
                                "test/epoch_loss": float(test_metrics["mae"]),
                                "test/epoch_sbp_mae": float(test_metrics["sbp_mae"]),
                                "test/epoch_dbp_mae": float(test_metrics["dbp_mae"]),
                                "test/epoch_sbp_mse": float(test_metrics["sbp_mse"]),
                                "test/epoch_dbp_mse": float(test_metrics["dbp_mse"]),
                                "learning_rate": float(optim.param_groups[0]['lr']),
                                "best_loss": float(best_loss['best_recon_total_loss']) if best_loss['best_recon_total_loss'] is not None else None,
                                "improved": bool(best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_metrics["mae"]),
                                "early_stopping_counter": int(early_stopping.counter)
                            })
                        except Exception as e:
                            print(f"Failed to log to wandb: {e}")

                # Check for early stopping
                if self._check_early_stopping(early_stopping, rank, device):
                    break

                # Clear cache periodically
                if ddp_config and e % ddp_config.empty_cache_frequency == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Rank {rank} encountered error training {self.channel_names[source_channel]}2{self.channel_names[target_channel]}: {e}")
            raise
        finally:
            # Cleanup
            if 'train_loader' in locals():
                self._cleanup(train_loader, val_loader, test_loader)

        if rank == 0:
            print(f"Finished training {self.channel_names[source_channel]}2{self.channel_names[target_channel]} model")

        print(f"Finished training on rank {rank}")
    
    def test(self, num_workers:int):
        """Test the model performance on test set"""
        # Model configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create and load the approximation model for inference
        app_config = PulseDBApproximationPatchTSTConfig(args=self.args)
        app_model = app_config.create_model().to(device)
        
        # Get source and target channels from the configured direction
        source, target = self._parse_direction(self.direction)
        source_channel_app = getattr(app_config, f"{source.lower()}_label")
        target_channel_app = getattr(app_config, f"{target.lower()}_label")
        
        # Get checkpoint info
        checkpoint_info = app_config.get_checkpoint_path(source_channel_app, target_channel_app)
        print(f"Loading approximation model from {checkpoint_info['file']} for {self.direction} cascade")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_info['file'])
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        app_model.load_state_dict(state_dict)
        app_model.eval()

        # Initialize dataset and dataloader
        test_dataset = self.create_test_dataset()
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.test_batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=num_workers
        )

        # Determine which directions to test
        if self.direction is None:
            # Get all possible source-target pairs
            source_target_pairs = self.get_source_target_pairs()
        else:
            # Use the specified direction
            source, target = self._parse_direction(self.direction)
            source_target_pairs = [(getattr(self, f"{source.lower()}_label"), 
                                getattr(self, f"{target.lower()}_label"))]

        # Initialize metrics dictionary
        best_loss = {
            'total_Sample': len(test_dataset),
        }

        # Add metrics for each direction
        for source_channel, target_channel in source_target_pairs:
            direction = f"{self.channel_names[source_channel]}2{self.channel_names[target_channel]}"
            # Initialize all required keys for calculate_bp_metrics
            best_loss.update({
                f'{direction}_loss': [],
                f'{direction}_ME_loss': [],
                f'{direction}_SBP_loss': [], f'{direction}_DBP_loss': [], f'{direction}_MAP_loss': [],
                f'{direction}_SBP_ME_loss': [], f'{direction}_DBP_ME_loss': [], f'{direction}_MAP_ME_loss': [],
                f'{direction}_SBP_BHS_5': 0, f'{direction}_SBP_BHS_10': 0, f'{direction}_SBP_BHS_15': 0,
                f'{direction}_DBP_BHS_5': 0, f'{direction}_DBP_BHS_10': 0, f'{direction}_DBP_BHS_15': 0,
                f'{direction}_MAP_BHS_5': 0, f'{direction}_MAP_BHS_10': 0, f'{direction}_MAP_BHS_15': 0,
            })

        # Test each direction
        for source_channel, target_channel in source_target_pairs:
            direction = f"{self.channel_names[source_channel]}2{self.channel_names[target_channel]}"
            print(f"\nTesting {direction} model...")

            # Get checkpoint path and load model
            checkpoint_info = self.get_checkpoint_path(source_channel, target_channel)
            model = self.create_model()
            
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_info['file'])
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
                model = model.to(device)
                model.eval()

                # Print model parameters
                print(f"\nModel Parameters for {direction}:")
                print_model_parameters(model)

                with torch.no_grad():
                    with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                        tepoch_test.set_description(f"Testing {direction}")
                        
                        for batch_idx, test_file in enumerate(tepoch_test):
                            # Get raw waveform for approximation model
                            x = test_file[8][:, source_channel:source_channel + 1].transpose(1, 2).to(device, torch.float32)
                            
                            # Use _prepare_batch for refinement model input (includes demographics if pi=True)
                            batch = self._prepare_batch(test_file, device, source_channel, target_channel)
                            
                            # Get target values in consistent format
                            y_target = test_file[2][:, self.abp_label:self.abp_label+1, 15:-15].to(device, torch.float32)
                            sbp = test_file[3][:, 0:1].to(device, torch.float32)
                            dbp = test_file[3][:, 1:2].to(device, torch.float32)
                            
                            # Get approximation model prediction
                            x_abp = app_model(x).unsqueeze(1)  # PatchTST expects (batch, seq_len, channels)
                            x_abp = Min_Max_Norm_Torch(x_abp[:, :, 15:-15])  # Normalize the output
                            
                            # Forward pass to get BP predictions using prepared batch
                            bp_predictions = model(
                                batch["x"],
                                past_observed_mask=batch.get("past_observed_mask", None)
                            )
                            
                            # Extract SBP and DBP predictions
                            sbp_pred = bp_predictions[:, 0:1]
                            dbp_pred = bp_predictions[:, 1:2]
                            
                            # Calculate BP metrics
                            metrics = calculate_bp_metrics(
                                predictions=(sbp_pred, dbp_pred),
                                waveform=x_abp,
                                sbp_true=sbp,
                                dbp_true=dbp,
                                abp_gt=y_target,  # Pass the trimmed ground truth ABP waveform
                                prefix=direction,
                                best_loss=best_loss,
                                global_min=self.dbp_min,
                                global_max=self.sbp_max,
                                normalize=True
                            )
                            
                            # Update progress bar
                            tepoch_test.set_postfix({
                                "SBP_MAE": f"{metrics['sbp_mae']:.2f}",
                                "DBP_MAE": f"{metrics['dbp_mae']:.2f}",
                                "Batch": f"{tepoch_test.n}/{len(test_loader)}"
                            })
                            
                            # Clear memory
                            del batch, x, y_target, sbp, dbp, bp_predictions, sbp_pred, dbp_pred, x_abp
                            torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error testing {direction} model: {str(e)}")
                continue

            finally:
                # Clean up model for this direction
                if 'model' in locals():
                    del model
                    torch.cuda.empty_cache()

        # Clean up approximation model
        del app_model
        torch.cuda.empty_cache()

        # Print final results
        print_abp_evaluation_results(best_loss, self.direction, args=self.args)