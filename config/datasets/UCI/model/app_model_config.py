# Standard library imports
import os
import time
import collections
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Local imports
from config.datasets.dataset_configs import UCIBaseConfig
from model.ApproximationModel import UNet_SwinUnet
from utils.ddp_utils import print_memory_stats, create_ddp_dataloaders
from utils.train_utils import EarlyStopping
from utils.test_utils import (
    calculate_reconstruction_losses,
    print_reconstruction_results,
    calculate_pearson_correlations,
    print_correlation_results,
    calculate_bpm_metrics,
    print_bpm_results,
    extract_waveform_features,
    save_waveform_features
)
from utils.utils_preprocessing import (
    print_model_parameters,
    isExist_dir,
    check_file_exists,
    Min_Max_Norm_Torch
)

# PulseDB Model Configurations
@dataclass
class UCIApproximationConfig(UCIBaseConfig):
    """UCI dataset configuration for Approximation model"""
    args: dict = None

    in_channels: int = 1
    out_channels: int = 1

    num_domains: int = 3
    init_features: int = 64
    kernel_size: int = 3
    patch_size: int = 4
    depth: int = 1
    upsample_scale: list[int] = field(default_factory=lambda: [4]) 

    # Training configuration
    training_mix_waveform_types: bool = True
    training_allow_identity: bool = False

    # Direction configuration
    direction: str = None  # If None, train all combinations, otherwise format: "SOURCE2TARGET" (e.g. "PPG2ABP")

    def __post_init__(self):
        # Instead of calling super().__post_init__(), directly initialize attributes
        super().__post_init__()
        # Initialize base attributes first
        self.path_folder: str = 'AppModel/UCI'
        if self.args is not None:
            self.batch_size = self.args.batch_size
            self.test_batch_size = self.args.test_batch_size
            self.num_epochs = self.args.num_epochs
            self.learning_rate = self.args.learning_rate
            self.scheduler_patience = self.args.scheduler_patience
            self.model_type = self.args.model_type
            if hasattr(self.args, 'resume_training'):
                self.resume_training = self.args.resume_training
            self.early_stopping_patience = self.args.early_stopping_patience
            self.checkpoint_name = self.args.checkpoint_name
            if hasattr(self.args, 'checkpoint_epoch'):
                self.checkpoint_epoch = self.args.checkpoint_epoch
            self.seed = self.args.seed
            if hasattr(self.args, 'calculate_bpm'):
                self.calculate_bpm = self.args.calculate_bpm
            if hasattr(self.args, 'batch_size_approximation_model'):
                if self.args.batch_size_approximation_model != self.batch_size:
                    self.batch_size = self.args.batch_size_approximation_model
            if hasattr(self.args, 'num_epochs_approximation_model'):
                self.num_epochs = self.args.num_epochs_approximation_model
            if hasattr(self.args, 'direction'):
                self.direction = self.args.direction
            if hasattr(self.args, 'extract_waveform_features'):
                self.extract_waveform_features = self.args.extract_waveform_features
        # Parse direction if specified
        if self.direction is not None:
            source, target = self._parse_direction(self.direction)
            self.source_channel = getattr(self, f"{source.lower()}_label")
            self.target_channel = getattr(self, f"{target.lower()}_label")
        self.set_seed()

    def _parse_direction(self, direction):
        """Parse direction string into source and target channels"""
        try:
            source, target = direction.split('2')
            valid_channels = set(self.channel_names.values())
            if source not in valid_channels:
                raise ValueError(f"Invalid source channel '{source}'. Valid channels are {valid_channels}")
            if target not in valid_channels:
                raise ValueError(f"Invalid target channel '{target}'. Valid channels are {valid_channels}")
            if source == target:
                raise ValueError(f"Source and target channels cannot be the same: {direction}")
            print(f"Parsed direction: {source}2{target}")
            return source, target
        except ValueError as e:
            raise ValueError(f"Invalid direction format {direction}. Must be SOURCE2TARGET (e.g. PPG2ABP). {str(e)}")

    def create_model(self):
        """Create approximation model instance"""
        return UNet_SwinUnet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            init_features=self.init_features,
            k=self.kernel_size,
            style_dim=self.num_domains,
            upsample_scale=self.upsample_scale,
            input_size=self.input_size,
            patch_size=self.patch_size,
            depth=self.depth
        )

    def _prepare_batch(self, batch_data, device):
        """Prepare batch data for training with mixed waveform support"""
        if self.direction is not None:
            # Uni-directional training using predefined source and target channels
            x_real = batch_data[0][:, self.source_channel:self.source_channel + 1].to(device, torch.float32)
            x_real_target = batch_data[0][:, self.target_channel:self.target_channel + 1].to(device, torch.float32)

            domain_shift_target = torch.nn.functional.one_hot(
                torch.tensor(self.target_channel),
                self.num_domains
            ).expand((x_real.size(0), self.num_domains)).to(device, torch.float32)

            return {
                "x_real": x_real,
                "x_real_target": x_real_target,
                "domain_shift_target": domain_shift_target,
                "mixed_batch": False,
                "rand_origin": self.source_channel,
                "rand_target": self.target_channel
            }
        elif self.training_mix_waveform_types:
            # Generate random targets and origins for each sample in batch
            batch_size = batch_data[0].size(0)
            rand_targets = torch.randint(0, 3, (batch_size,)).to(device)
            rand_origins = torch.randint(0, 3, (batch_size,)).to(device)
            
            if not self.training_allow_identity:
                # Ensure origin and target are different for each sample
                same_indices = (rand_targets == rand_origins)
                while same_indices.any():
                    rand_origins[same_indices] = torch.randint(0, 3, (same_indices.sum(),)).to(device)
                    same_indices = (rand_targets == rand_origins)

            # Prepare inputs and targets
            x_real = torch.stack([
                batch_data[0][i, origin:origin + 1]
                for i, origin in enumerate(rand_origins)
            ]).to(device, torch.float32)

            x_real_target = torch.stack([
                batch_data[0][i, target:target + 1]
                for i, target in enumerate(rand_targets)
            ]).to(device, torch.float32)

            domain_shift_target = torch.nn.functional.one_hot(
                rand_targets,
                self.num_domains
            ).to(device, torch.float32)

            return {
                "x_real": x_real,
                "x_real_target": x_real_target,
                "domain_shift_target": domain_shift_target,
                "mixed_batch": True,
                "rand_targets": rand_targets,
                "rand_origins": rand_origins
            }
        else:
            # Original behavior - same source/target for whole batch
            rand_target = torch.randint(0, 3, (1,)).numpy()[0]
            rand_origin = torch.randint(0, 3, (1,)).numpy()[0]
            
            if not self.training_allow_identity:
                while(rand_target == rand_origin):
                    rand_origin = torch.randint(0, 3, (1,)).numpy()[0]

            x_real = batch_data[0][:, rand_origin:rand_origin + 1].to(device, torch.float32)
            x_real_target = batch_data[0][:, rand_target:rand_target + 1].to(device, torch.float32)

            domain_shift_target = torch.nn.functional.one_hot(
                torch.tensor(rand_target),
                self.num_domains
            ).expand((x_real.size(0), self.num_domains)).to(device, torch.float32)

            return {
                "x_real": x_real,
                "x_real_target": x_real_target,
                "domain_shift_target": domain_shift_target,
                "mixed_batch": False,
                "rand_origin": rand_origin,
                "rand_target": rand_target
            }

    def _train_epoch(self, epoch, ddp_model, train_loader, optim, device, master_process):
        """Run one training epoch with mixed waveform support"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        domains_shift_code = {
            str(self.ecg_label): "ECG",
            str(self.ppg_label): "PPG", 
            str(self.abp_label): "ABP"
        }

        # Initialize epoch metrics
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        num_batches = 0
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            ddp_model.train()
            tepoch.set_description(f"Train - Epoch {epoch}")

            for step, train_file in enumerate(tepoch):
                # Prepare batch
                batch = self._prepare_batch(train_file, device)

                # Training step
                optim.zero_grad(set_to_none=True)
                y = ddp_model(batch["x_real"], batch["domain_shift_target"])
                # Calculate losses
                mse_loss = torch.nn.functional.mse_loss(batch["x_real_target"], y)
                with torch.no_grad():
                    mae_loss = torch.nn.functional.l1_loss(batch["x_real_target"], y.detach())
                loss = mse_loss  # Use MSE as training loss
                
                loss.backward()
                optim.step()

                # Add synchronization barrier after each step to ensure all processes are in sync
                dist.barrier()

                # Accumulate losses for epoch average
                epoch_mse_loss += mse_loss.item()
                epoch_mae_loss += mae_loss.item()
                num_batches += 1

                # Log training metrics per step (only for master process)
                if master_process and wandb.run is not None:
                    wandb.log({
                        "train/step": step + epoch * len(train_loader),
                        "train/step_mse": mse_loss.item(),
                        "train/step_mae": mae_loss.item()
                        # ,
                        # "train/learning_rate": optim.param_groups[0]['lr']
                    })

                # Update progress bar with appropriate information
                if batch["mixed_batch"]:
                    tepoch.set_postfix(
                        Mixed_Batch="True",
                        Identity=str(self.training_allow_identity),
                        R_loss=f"{loss.item():.4f}",
                        LR=f"{optim.param_groups[0]['lr']:.2e}"
                    )
                else:
                    tepoch.set_postfix(
                        Origin=domains_shift_code[str(batch["rand_origin"])],
                        Target=domains_shift_code[str(batch["rand_target"])],
                        Identity=str(self.training_allow_identity),
                        R_loss=f"{loss.item():.4f}",
                        LR=f"{optim.param_groups[0]['lr']:.2e}"
                    )

                # Clear memory
                del batch, loss, mse_loss, mae_loss, y
                torch.cuda.empty_cache()
            
            return epoch_mse_loss / num_batches, epoch_mae_loss / num_batches

    def _validate_epoch(self, epoch, ddp_model, val_loader, device, master_process):
        """Run one validation epoch"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)

        # Extract rank from device
        rank = device.index
    
        if self.direction is not None:
            # Uni-directional validation
            mse_metrics = {f'{self.direction}_loss': []}
            mae_metrics = {f'{self.direction}_loss': []}
        else:
            # Multi-directional validation
            mse_metrics = {
                'PPG2ECG_loss': [],
                'ABP2ECG_loss': [],
                'ABP2PPG_loss': [],
                'ECG2PPG_loss': [],
                'PPG2ABP_loss': [],
                'ECG2ABP_loss': []
            }

            mae_metrics = {
                'PPG2ECG_loss': [],
                'ABP2ECG_loss': [],
                'ABP2PPG_loss': [],
                'ECG2PPG_loss': [],
                'PPG2ABP_loss': [],
                'ECG2ABP_loss': []
            }
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch_val:
                ddp_model.eval()
                tepoch_val.set_description(f"Val - Epoch {epoch}")

                for val_file in tepoch_val:
                    if self.direction is not None:
                        # Uni-directional validation
                        input_signal = val_file[0][:, self.source_channel:self.source_channel + 1].to(device, torch.float32)
                        target_signal = val_file[0][:, self.target_channel:self.target_channel + 1].to(device, torch.float32)
                        
                        # Create domain target
                        domain_shift_target = torch.nn.functional.one_hot(
                            torch.tensor(self.target_channel),
                            self.num_domains
                        ).expand((input_signal.size(0), self.num_domains)).to(device, torch.float32)
                        
                        # Get reconstruction
                        recon = ddp_model(input_signal, domain_shift_target)
                        
                        # Calculate losses
                        mae_loss = torch.nn.functional.l1_loss(recon, target_signal)
                        mse_loss = torch.nn.functional.mse_loss(recon, target_signal)
                        
                        mae_metrics[f'{self.direction}_loss'].append(mae_loss.item())
                        mse_metrics[f'{self.direction}_loss'].append(mse_loss.item())
                        
                        # Update progress bar
                        tepoch_val.set_postfix(
                            mae=f"{mae_loss.item():.4f}",
                            mse=f"{mse_loss.item():.4f}"
                        )
                    else:
                        # Get inputs for each domain
                        input_ii = val_file[0][:, self.ecg_label:self.ecg_label + 1].to(device, torch.float32)
                        input_ppg = val_file[0][:, self.ppg_label:self.ppg_label + 1].to(device, torch.float32)
                        input_abp = val_file[0][:, self.abp_label:self.abp_label + 1].to(device, torch.float32)

                        # Create domain targets
                        s_ii = torch.nn.functional.one_hot(torch.tensor(self.ecg_label), self.num_domains).expand(
                            (input_ii.size(0), self.num_domains)).to(device, torch.float32)
                        s_ppg = torch.nn.functional.one_hot(torch.tensor(self.ppg_label), self.num_domains).expand(
                            (input_ii.size(0), self.num_domains)).to(device, torch.float32)
                        s_abp = torch.nn.functional.one_hot(torch.tensor(self.abp_label), self.num_domains).expand(
                            (input_ii.size(0), self.num_domains)).to(device, torch.float32)

                        # Get ground truth outputs
                        output_ii = val_file[0][:, self.ecg_label:self.ecg_label + 1].to(device, torch.float32)
                        output_ppg = val_file[0][:, self.ppg_label:self.ppg_label + 1].to(device, torch.float32)
                        output_abp = val_file[0][:, self.abp_label:self.abp_label + 1].to(device, torch.float32)

                        # Calculate losses for each transformation
                        # II reconstructions
                        recon_ppg2ii = ddp_model(input_ppg, s_ii)
                        recon_abp2ii = ddp_model(input_abp, s_ii)
                        # Calculate both MAE and MSE
                        mae_ppg2ii = torch.nn.functional.l1_loss(recon_ppg2ii, output_ii)
                        mae_abp2ii = torch.nn.functional.l1_loss(recon_abp2ii, output_ii)
                        mse_ppg2ii = torch.nn.functional.mse_loss(recon_ppg2ii, output_ii)
                        mse_abp2ii = torch.nn.functional.mse_loss(recon_abp2ii, output_ii)
                        
                        mae_metrics['PPG2ECG_loss'].append(mae_ppg2ii.item())
                        mae_metrics['ABP2ECG_loss'].append(mae_abp2ii.item())
                        mse_metrics['PPG2ECG_loss'].append(mse_ppg2ii.item())
                        mse_metrics['ABP2ECG_loss'].append(mse_abp2ii.item())

                        # PPG reconstructions
                        recon_abp2ppg = ddp_model(input_abp, s_ppg)
                        recon_ii2ppg = ddp_model(input_ii, s_ppg)

                        mae_abp2ppg = torch.nn.functional.l1_loss(recon_abp2ppg, output_ppg)
                        mae_ii2ppg = torch.nn.functional.l1_loss(recon_ii2ppg, output_ppg)
                        mse_abp2ppg = torch.nn.functional.mse_loss(recon_abp2ppg, output_ppg)
                        mse_ii2ppg = torch.nn.functional.mse_loss(recon_ii2ppg, output_ppg)
                        
                        mae_metrics['ABP2PPG_loss'].append(mae_abp2ppg.item())
                        mae_metrics['ECG2PPG_loss'].append(mae_ii2ppg.item())
                        mse_metrics['ABP2PPG_loss'].append(mse_abp2ppg.item())
                        mse_metrics['ECG2PPG_loss'].append(mse_ii2ppg.item())

                        # ABP reconstructions
                        recon_ppg2abp = ddp_model(input_ppg, s_abp)
                        recon_ii2abp = ddp_model(input_ii, s_abp)
                        
                        mae_ppg2abp = torch.nn.functional.l1_loss(recon_ppg2abp, output_abp)
                        mae_ii2abp = torch.nn.functional.l1_loss(recon_ii2abp, output_abp)
                        mse_ppg2abp = torch.nn.functional.mse_loss(recon_ppg2abp, output_abp)
                        mse_ii2abp = torch.nn.functional.mse_loss(recon_ii2abp, output_abp)
                        
                        mae_metrics['PPG2ABP_loss'].append(mae_ppg2abp.item())
                        mae_metrics['ECG2ABP_loss'].append(mae_ii2abp.item())
                        mse_metrics['PPG2ABP_loss'].append(mse_ppg2abp.item())
                        mse_metrics['ECG2ABP_loss'].append(mse_ii2abp.item())

                        # Calculate mean of all losses for this batch
                        current_mae_losses = [
                            mae_ppg2abp.item(), mae_ii2abp.item(), 
                            mae_ppg2ii.item(), mae_abp2ii.item(),
                            mae_abp2ppg.item(), mae_ii2ppg.item()
                        ]
                        current_mse_losses = [
                            mse_ppg2abp.item(), mse_ii2abp.item(), 
                            mse_ppg2ii.item(), mse_abp2ii.item(),
                            mse_abp2ppg.item(), mse_ii2ppg.item()
                        ]
                    
                        mean_mae = sum(current_mae_losses) / len(current_mae_losses)
                        mean_mse = sum(current_mse_losses) / len(current_mse_losses)

                        tepoch_val.set_postfix(
                            mean_mae=f"{mean_mae:.4f}",
                            mean_mse=f"{mean_mse:.4f}"
                        )

                    # Add synchronization barrier after each validation batch
                    dist.barrier(device_ids=[rank])

                # Calculate mean losses for each metric
                mean_mae_losses = {k: torch.tensor(np.mean(v), device=device) for k, v in mae_metrics.items()}
                mean_mse_losses = {k: torch.tensor(np.mean(v), device=device) for k, v in mse_metrics.items()}
                
                # Gather losses from all processes
                gathered_mae_losses = {}
                gathered_mse_losses = {}
                
                for k, v in mean_mae_losses.items():
                    gathered_tensor = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensor, v)
                    gathered_mae_losses[k] = torch.stack(gathered_tensor).mean().item()
            
                for k, v in mean_mse_losses.items():
                    gathered_tensor = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_tensor, v)
                    gathered_mse_losses[k] = torch.stack(gathered_tensor).mean().item()

                # Calculate mean of all transformation losses
                mean_mae = sum(gathered_mae_losses.values()) / len(gathered_mae_losses)
                mean_mse = sum(gathered_mse_losses.values()) / len(gathered_mse_losses)

                if master_process:
                    print(f"\nValidation Results - Epoch {epoch}:")
                    print(f"Mean MAE: {mean_mae:.4f}, Mean MSE: {mean_mse:.4f}")

                # Add final synchronization barrier at the end of validation
                dist.barrier(device_ids=[rank])
                
                return mean_mse, mean_mae

    def load_checkpoint_with_retry(self, rank, model, optim, scheduler, early_stopping, checkpoint_path, max_retries=3, wait_time=5):
        """Helper function to load checkpoint with retry logic
        
        Args:
            rank (int): Process rank in distributed training
            model (nn.Module): The unwrapped model (only loaded on rank 0)
            optim (Optimizer): Model optimizer (loaded on all ranks)
            scheduler (LRScheduler): Learning rate scheduler (loaded on all ranks)
            early_stopping (EarlyStopping): Early stopping handler (loaded on all ranks)
            checkpoint_path (str): Path to checkpoint file
            max_retries (int): Maximum number of retry attempts
            wait_time (int): Wait time between retries in seconds
        """
        training_state = {
            'epoch_checkpoint': 0,
            'best_recon_total_loss': None
        }
        
        # Initialize checkpoint_loaded for all ranks
        checkpoint_loaded = False
        checkpoint_exists = False
        
        # Check if we should attempt to resume training
        if self.resume_training:
            # Check if checkpoint exists (only on rank 0)
            if rank == 0:
                checkpoint_exists = os.path.exists(checkpoint_path)
                print(f"Checking for checkpoint at {checkpoint_path}: {'Found' if checkpoint_exists else 'Not found'}")
            
            # Broadcast checkpoint existence to all processes
            checkpoint_exists_tensor = torch.tensor([checkpoint_exists], device=f'cuda:{rank}')
            dist.broadcast(checkpoint_exists_tensor, src=0)
            checkpoint_exists = checkpoint_exists_tensor.item()
            
            # If checkpoint exists, try to load it
            if checkpoint_exists:
                # Only rank 0 loads the model checkpoint
                if rank == 0:
                    # Try to load complete checkpoint
                    for attempt in range(max_retries):
                        try:
                            print(f"Loading checkpoint (attempt {attempt + 1}/{max_retries})")
                            checkpoint = torch.load(checkpoint_path)

                            # Load model state dict (only on rank 0)
                            state_dict = checkpoint['model_state_dict']
                            # Remove 'module.' prefix if it exists in the checkpoint
                            new_state_dict = {k[7:] if k.startswith('module.') else k: v
                                            for k, v in state_dict.items()}

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
                    if rank == 0:
                        print(f"Checkpoint not found at {checkpoint_path}. Starting training from scratch...")
        else:
            if rank == 0:
                print(f"resume_training=False. Starting training from scratch...")

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
        if checkpoint_loaded and checkpoint_exists and rank != 0:
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
        
        # Get correct checkpoint path
        checkpoint_info = self.get_checkpoint_info(saving=True)
        checkpoint_path = checkpoint_info['file']
        
        # Ensure checkpoint directory exists before saving
        checkpoint_dir = os.path.dirname(checkpoint_path)
        isExist_dir(checkpoint_dir)
        
        # Save checkpoint if better than previous best
        if best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss:
            if best_loss['best_recon_total_loss'] is not None:
                print(f"Validation loss improved from {best_loss['best_recon_total_loss']:.6f} to {val_loss:.6f}")
            torch.save({
                'model_state_dict': ddp_model.module.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),
                'best_recon_total_loss': val_loss
            }, checkpoint_path)
            best_loss['best_recon_total_loss'] = val_loss
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            # Get correct checkpoint path
            checkpoint_info_epoch = self.get_checkpoint_info(epoch=epoch, saving=True)
            checkpoint_path_epoch = checkpoint_info_epoch['file']
            torch.save({
                'model_state_dict': ddp_model.module.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),
                'best_recon_total_loss': val_loss
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
            for loader in [train_loader, val_loader, test_loader]:
                if loader is not None and hasattr(loader, '_iterator'):
                    loader._iterator = None
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def get_checkpoint_info(self, epoch: int=None, saving: bool=False):
        """Get checkpoint directory and file information
            
        Returns:
            dict: Contains checkpoint directory and file paths
        """
        # Base folder name
        base_folder = f'AppModel_UNetSwinT_BS_{self.batch_size}_E_{self.num_epochs}_LR_{self.learning_rate}_P_{self.scheduler_patience}_ES_{self.early_stopping_patience}'
            
        checkpoint_dir = os.path.join(self.abs_path, self.path_folder, base_folder)

        # Get checkpoint file name
        if self.direction is not None:
            # For uni-directional training, include direction in checkpoint name
            checkpoint_prefix = f'{self.direction}_{self.checkpoint_name}'
        else:
            # For multi-directional training, keep original behavior
            checkpoint_prefix = self.checkpoint_name

        # Saving checkpoint use current epoch
        if saving:
            if epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{checkpoint_prefix}_epoch_{epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{checkpoint_prefix}.pt')
        else:
            # Loading checkpoint use checkpoint_epoch
            if self.checkpoint_epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{checkpoint_prefix}_epoch_{self.checkpoint_epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{checkpoint_prefix}.pt')
        
        return {
            'dir': checkpoint_dir,
            'file': checkpoint_file,
            'sub_folder': base_folder
        }

    def trainer(self, dataset: tuple, rank: int, world_size: int, ddp_config=None):
        """DDP trainer for the approximation model
        
        Args:
            rank (int): Process rank
            world_size (int): Total number of processes
            ddp_config (DDPConfig, optional): DDP configuration. Defaults to None.
        """
        # Initialize variables that might need cleanup
        train_loader = None
        val_loader = None

        try:
            # Setup device
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{rank}')
            master_process = rank == 0

            # Get datasets
            train_dataset, val_dataset, test_dataset = dataset

            # Create dataloaders using DDP utility
            dataloader_settings = ddp_config.get_dataloader_settings()
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

            # Load checkpoint before DDP wrapping
            checkpoint_info = self.get_checkpoint_info()
            training_state = self.load_checkpoint_with_retry(
                rank=rank,
                model=model,  # Pass unwrapped model
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_path=checkpoint_info['file']
            )

            # Wrap model with DDP after loading checkpoint
            ddp_model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=ddp_config.find_unused_parameters if ddp_config else False,
                static_graph=ddp_config.static_graph if ddp_config else True
            )

            epoch_checkpoint = training_state['epoch_checkpoint']
            best_loss = {'best_recon_total_loss': training_state['best_recon_total_loss']}
            prev_lr = optim.param_groups[0]['lr']

            # Check wandb initialization for master process
            if master_process:
                if wandb.run is None:
                    print("Warning: wandb is not properly initialized!")
                else:
                    print(f"wandb run: {wandb.run.name}")

            # Training loop
            print_memory_stats(rank, "Before training loop")
            for e in range(epoch_checkpoint, self.num_epochs):
                train_mse, train_mae = self._train_epoch(e, ddp_model, train_loader, optim, device, master_process)
                val_mse, val_mae = self._validate_epoch(e, ddp_model, val_loader, device, master_process)
                test_mse, test_mae = self._validate_epoch(e, ddp_model, test_loader, device, master_process)
                if master_process:
                    prev_lr = self._update_training_state(
                        e, val_mae, ddp_model, optim, scheduler,
                        early_stopping, checkpoint_info['file'], best_loss, prev_lr
                    )
                    if wandb.run is not None:
                        try:
                            log_dict = {
                                "epoch": e,
                                "train/epoch_mse": float(train_mse),
                                "train/epoch_mae": float(train_mae),
                                "val/epoch_mse": float(val_mse),
                                "val/epoch_mae": float(val_mae),
                                "test/epoch_mse": float(test_mse),
                                "test/epoch_mae": float(test_mae),
                                "learning_rate": float(optim.param_groups[0]['lr']),
                                "best_loss": float(best_loss['best_recon_total_loss']) if best_loss['best_recon_total_loss'] is not None else None,
                                "improved": bool(best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_mae),
                                "early_stopping_counter": int(early_stopping.counter)
                                }
                            wandb.log(log_dict)
                        except Exception as e:
                            print(f"Failed to log to wandb: {e}")

                # Check for early stopping
                if self._check_early_stopping(early_stopping, rank, device):
                    break

                # Clear cache periodically based on ddp_config
                if ddp_config and e % ddp_config.empty_cache_frequency == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Rank {rank} encountered error: {e}")
            raise e
        finally:
            # Only cleanup if loaders were created
            if train_loader is not None or val_loader is not None:
                self._cleanup(train_loader, val_loader, test_loader)

        print(f"Finished training on rank {rank}.")
    
    def test(self, num_workers:int):
        """Test the model performance on test set
        
        Args:
            num_workers (int): Number of dataloader workers
        """
        # Model configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dataset and dataloader
        test_dataset = self.create_test_dataset()
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.test_batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=num_workers
        )

        # Get checkpoint path and load model
        checkpoint_info = self.get_checkpoint_info()
        model = self.create_model()
        
        # Load checkpoint
        if os.path.exists(checkpoint_info['file']):
            checkpoint = torch.load(checkpoint_info['file'])
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
            print(f"Loaded checkpoint from {checkpoint_info['file']}")
        else:
            # For unidirectional case, try loading from direction-specific checkpoint
            if self.direction is not None:
                # Get checkpoint info with direction
                direction_checkpoint_info = self.get_checkpoint_info()
                if os.path.exists(direction_checkpoint_info['file']):
                    checkpoint = torch.load(direction_checkpoint_info['file'])
                    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
                    print(f"Loaded direction-specific checkpoint from {direction_checkpoint_info['file']}")
                else:
                    raise FileNotFoundError(f"No checkpoint found for testing direction {self.direction}")
            else:
                raise FileNotFoundError("No checkpoint found for testing")

        model = model.to(device)
        model.eval()

        # Print model parameters
        print_model_parameters(model)

        # If only extracting waveform features, skip the rest of test evaluation
        if hasattr(self, 'extract_waveform_features') and self.extract_waveform_features:
            print("Extracting waveform features from reconstructed signals...")
            
            # Define directions and their corresponding source-target pairs
            directions = {
                'ppg2ecg': {'source': 'ppg', 'target': 'ecg'},
                'abp2ecg': {'source': 'abp', 'target': 'ecg'},
                'abp2ppg': {'source': 'abp', 'target': 'ppg'},
                'ecg2ppg': {'source': 'ecg', 'target': 'ppg'}
            }
            
            # Process each direction separately
            for direction, signal_info in directions.items():
                print(f"\nProcessing {direction.upper()} direction...")
                direction_features = {}
                
                with torch.no_grad():
                    # Check if features file already exists
                    features_dir = os.path.join('./results/features_extraction', 'UCI', 'MD-ViSCo', f'seed_{self.seed}')
                    filename = f'features_{direction.upper()}.h5'
                    filepath = os.path.join(features_dir, filename)
                    
                    try:
                        check_file_exists(filepath)
                    except FileExistsError as e:
                        print(f"\nSkipping {direction.upper()} feature extraction: {e}")
                        continue
                    
                    with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                        tepoch_test.set_description(f"Extracting {direction.upper()} Features")
                        
                        for batch_idx, test_file in enumerate(tepoch_test):
                            # Get source and target signals
                            source_signal = test_file[0][:, getattr(self, f"{signal_info['source']}_label"):getattr(self, f"{signal_info['source']}_label") + 1].to(device, torch.float32)
                            target_signal = test_file[0][:, getattr(self, f"{signal_info['target']}_label"):getattr(self, f"{signal_info['target']}_label") + 1].to(device, torch.float32)
                            
                            # Create domain encoding for target
                            target_domain = torch.nn.functional.one_hot(
                                torch.tensor(getattr(self, f"{signal_info['target']}_label")),
                                self.num_domains
                            ).expand((source_signal.size(0), self.num_domains)).to(device, torch.float32)
                            
                            # Get reconstruction
                            recon = model(source_signal, target_domain)
                            
                            # Extract features from reconstructed waveforms
                            batch_size = recon.size(0)
                            with tqdm(range(batch_size), unit="sample", ncols=100, leave=False) as pbar:
                                pbar.set_description(f"Batch {batch_idx}")
                                for i in pbar:
                                    # Calculate global index from batch_idx and sample index
                                    global_idx = batch_idx * self.test_batch_size + i
                                    
                                    # Extract features for this sample
                                    features = extract_waveform_features(
                                        ecg_signal=Min_Max_Norm_Torch(recon[i:i+1, :, :])[0, 0, :].cpu().numpy() if signal_info['target'] == 'ecg' else None,
                                        ppg_signal=Min_Max_Norm_Torch(recon[i:i+1, :, :])[0, 0, :].cpu().numpy() if signal_info['target'] == 'ppg' else None
                                    )
                                    
                                    # Store features using global index as key
                                    direction_features[global_idx] = features
                                    
                                    # Update inner progress bar
                                    pbar.set_postfix({
                                        'sample': f"{i+1}/{batch_size}",
                                        'idx': global_idx,
                                        'features': '✓' if features[f'{signal_info["target"]}_features'] is not None else '✗'
                                    })
                            
                            # Clear memory
                            torch.cuda.empty_cache()
                            
                            
                    # Save features for this direction
                    print(f"\nSaving {direction.upper()} features...")
                    save_waveform_features(
                        features_dict=direction_features,
                        dataset='UCI',
                        model_name='MD-ViSCo',
                        seed=self.seed,
                        direction=direction.upper(),
                        output_dir=self.abs_path
                    )
                
            print("Feature extraction and saving completed.")
            return
        else:
            # Regular test evaluation continues here
            # Loss function
            l1 = nn.L1Loss(reduction='none')

            # Initialize metrics dictionaries
            if self.direction is not None:
                # Uni-directional case
                mae_metrics = {f'{self.direction}_loss': []}
                correlation_metrics = collections.defaultdict(list)
            else:
                # Multi-directional case
                mae_metrics = {
                    'ppg2ecg_loss': [], 'abp2ecg_loss': [],
                    'abp2ppg_loss': [], 'ecg2ppg_loss': [],
                    'ppg2abp_loss': [], 'ecg2abp_loss': []
                }
                correlation_metrics = collections.defaultdict(list)
            
            # Initialize BPM metrics if calculate_bpm is True
            bpm_metrics = {} if hasattr(self, 'calculate_bpm') and self.calculate_bpm else None

            with torch.no_grad():
                with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                    tepoch_test.set_description("Testing")
                    
                    for batch_idx, test_file in enumerate(tepoch_test):
                        # Domain labels
                        ii_label = self.ecg_label
                        ppg_label = self.ppg_label
                        abp_label = self.abp_label
                        
                        # Get ground truth rates from dataset (UCI specific positions)
                        ground_truth_rates = None
                        if hasattr(self, 'calculate_bpm') and self.calculate_bpm:
                            try:
                                # UCI dataset has heart rate at position 6 and pulse rate at position 7
                                heart_rate = test_file[6].to(device) if test_file[6] is not None else None
                                pulse_rate = test_file[7].to(device) if test_file[7] is not None else None
                                if heart_rate is not None and pulse_rate is not None:
                                    ground_truth_rates = torch.stack([heart_rate, pulse_rate], dim=1)
                            except Exception as e:
                                print(f"Warning: Could not load rate data: {e}")

                        if self.direction is not None:
                            # Uni-directional case
                            source, target = self.direction.split('2')
                            source_label = getattr(self, f"{source.lower()}_label")
                            target_label = getattr(self, f"{target.lower()}_label")
                            
                            # Prepare input and target
                            input_signal = test_file[0][:, source_label:source_label + 1].to(device, torch.float32)
                            target_signal = test_file[0][:, target_label:target_label + 1].to(device, torch.float32)
                            
                            # Create domain encoding for target
                            target_domain = torch.nn.functional.one_hot(
                                torch.tensor(target_label),
                                self.num_domains
                            ).expand((input_signal.size(0), self.num_domains)).to(device, torch.float32)
                            
                            # Get reconstruction
                            recon = model(input_signal, target_domain)
                            
                            # Calculate losses
                            losses, batch_losses = calculate_reconstruction_losses(
                                {self.direction: recon},
                                {target: target_signal},
                                l1
                            )
                            
                            # Calculate Pearson correlations
                            correlations = calculate_pearson_correlations(
                                {self.direction: recon},
                                {target: target_signal},
                                correlation_metrics
                            )
                            
                            # Calculate BPM metrics if enabled
                            if bpm_metrics is not None and ground_truth_rates is not None:
                                # Create target BPM dictionary using ground truth rates
                                target_bpm = {
                                    'ecg': ground_truth_rates[:, 0],  # Heart rate from ECG
                                    'ppg': ground_truth_rates[:, 1]   # Pulse rate from PPG
                                }
                                # Calculate BPM metrics
                                batch_bpm_means = calculate_bpm_metrics(
                                    recons={self.direction: recon},
                                    target_bpm=target_bpm,
                                    bpm_metrics=bpm_metrics,
                                    sampling_rate=125  # Assuming 125Hz sampling rate
                                )
                                # Add BPM metrics to progress bar
                                progress_metrics = {
                                    **{f"{k}_mae": torch.mean(v.cpu()) for k, v in losses.items()},
                                    **{f"{k}_corr": v for k, v in correlations.items()},
                                    **{f"BPM_{k}": f"{v:.2f}" for k, v in batch_bpm_means.items()}
                                }
                            else:
                                # Original progress metrics without BPM
                                progress_metrics = {
                                    **{f"{k}_mae": torch.mean(v.cpu()) for k, v in losses.items()},
                                    **{f"{k}_corr": v for k, v in correlations.items()}
                                }
                            
                            # Update mae_metrics dictionary
                            for k, v in batch_losses.items():
                                mae_metrics[k].extend(v)
                            
                            tepoch_test.set_postfix(**progress_metrics)
                            
                        else:
                            # Multi-directional case - existing code
                            # Prepare inputs
                            inputs = {
                                'ecg': test_file[0][:, ii_label:ii_label + 1].to(device, torch.float32),
                                'ppg': test_file[0][:, ppg_label:ppg_label + 1].to(device, torch.float32),
                                'abp': test_file[0][:, abp_label:abp_label + 1].to(device, torch.float32)
                            }
                            
                            # Prepare targets
                            targets = {
                                'ecg': test_file[0][:, ii_label:ii_label + 1].to(device, torch.float32),
                                'ppg': test_file[0][:, ppg_label:ppg_label + 1].to(device, torch.float32),
                                'abp': test_file[0][:, abp_label:abp_label + 1].to(device, torch.float32)
                            }
                            
                            # Prepare domain encodings
                            domains = {
                                'ecg': torch.nn.functional.one_hot(torch.tensor(ii_label), self.num_domains),
                                'ppg': torch.nn.functional.one_hot(torch.tensor(ppg_label), self.num_domains),
                                'abp': torch.nn.functional.one_hot(torch.tensor(abp_label), self.num_domains)
                            }
                            
                            for d in domains:
                                domains[d] = domains[d].expand((inputs['ecg'].size(0), self.num_domains)).to(device, torch.float32)
                            
                            # Get reconstructions
                            recons = {
                                'ppg2ecg': model(inputs['ppg'], domains['ecg']),
                                'abp2ecg': model(inputs['abp'], domains['ecg']),
                                'abp2ppg': model(inputs['abp'], domains['ppg']),
                                'ecg2ppg': model(inputs['ecg'], domains['ppg']),
                                'ppg2abp': model(inputs['ppg'], domains['abp']),
                                'ecg2abp': model(inputs['ecg'], domains['abp'])
                            }
                            
                            # Calculate losses
                            losses, batch_losses = calculate_reconstruction_losses(recons, targets, l1)
                            
                            # Calculate Pearson correlations
                            correlations = calculate_pearson_correlations(recons, targets, correlation_metrics)
                            
                            # Calculate BPM metrics if enabled and ground truth rates are available
                            if bpm_metrics is not None and ground_truth_rates is not None:
                                # Create target BPM dictionary using ground truth rates
                                target_bpm = {
                                    'ecg': ground_truth_rates[:, 0],  # Heart rate from ECG
                                    'ppg': ground_truth_rates[:, 1]   # Pulse rate from PPG
                                }
                                # Calculate BPM metrics
                                batch_bpm_means = calculate_bpm_metrics(
                                    recons=recons,
                                    target_bpm=target_bpm,
                                    bpm_metrics=bpm_metrics,
                                    sampling_rate=125  # Assuming 125Hz sampling rate
                                )
                                # Add BPM metrics to progress bar
                                progress_metrics = {
                                    **{f"{k}_mae": torch.mean(v.cpu()) for k, v in losses.items()},
                                    **{f"{k}_corr": v for k, v in correlations.items()},
                                    **{f"BPM_{k}": f"{v:.2f}" for k, v in batch_bpm_means.items()}
                                }
                            else:
                                # Original progress metrics without BPM
                                progress_metrics = {
                                    **{f"{k}_mae": torch.mean(v.cpu()) for k, v in losses.items()},
                                    **{f"{k}_corr": v for k, v in correlations.items()}
                                }
                            
                            # Update mae_metrics dictionary
                            for k, v in batch_losses.items():
                                mae_metrics[k].extend(v)
                            
                            tepoch_test.set_postfix(**progress_metrics)

                        # Clear memory
                        torch.cuda.empty_cache()

            # Print final results
            print("\nMAE Results:")
            print_reconstruction_results(mae_metrics, args=self.args)
            print("\nCorrelation Results:")
            print_correlation_results(correlation_metrics, args=self.args)
            
            # Print BPM results if enabled
            if bpm_metrics:
                print("\nBPM Results:")
                print_bpm_results(bpm_metrics, args=self.args)
