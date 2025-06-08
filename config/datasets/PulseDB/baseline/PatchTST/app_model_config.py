# Standard library imports
import os
import time
from dataclasses import dataclass

# Third-party imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Local imports
from config.datasets.dataset_configs import PulseDBBaseConfig
from baseline.PatchTST import ApproximationPatchTST
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

@dataclass
class PulseDBApproximationPatchTSTConfig(PulseDBBaseConfig):
    """PulseDB dataset configuration for PatchTST model.
    Supports approximation between any combination of signals (PPG, ECG, ABP)."""
    
    args: dict = None
    # Direction configuration
    direction: str = None  # If None, train all combinations, otherwise format: "SOURCE2TARGET" (e.g. "PPG2ABP")
    
    # Base configuration
    path_folder: str = 'AppModel/PulseDB'
    
    # Model Architecture configuration
    context_length: int = 1280  # Length of input sequence
    patch_len: int = 16  # Length of each patch
    stride: int = 8  # Stride between patches
    d_model: int = 640  # Dimension of model
    num_encoder_layers: int = 10  # Number of encoder layers
    num_heads: int = 10  # Number of attention heads
    dropout: float = 0.1  # Attention dropout rate
    fc_dropout: float = 0.1  # Dropout rate for fully connected layers
    head_dropout: float = 0.1  # Dropout rate for prediction head
    in_channels: int = 1
    out_channels: int = 1

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
            if hasattr(self.args, 'seed'):
                self.seed = self.args.seed
            self.model_type = self.args.model_type
            if hasattr(self.args, 'calculate_bpm'):
                self.calculate_bpm = self.args.calculate_bpm
            if hasattr(self.args, 'batch_size_approximation_model'):
                if self.args.batch_size_approximation_model != self.batch_size:
                    self.batch_size = self.args.batch_size_approximation_model
            if hasattr(self.args, 'num_epochs_approximation_model'):
                self.num_epochs = self.args.num_epochs_approximation_model
            if hasattr(self.args, 'extract_waveform_features'):
                self.extract_waveform_features = self.args.extract_waveform_features

        # Parse direction if specified
        if self.direction is not None:
            source, target = self._parse_direction(self.direction)
            self.source_channel = getattr(self, f"{source.lower()}_label")
            self.target_channel = getattr(self, f"{target.lower()}_label")
        self.set_seed()

    def create_model(self):
        """Create PatchTST model instance"""
        return ApproximationPatchTST(
            input_size=self.in_channels,
            output_size=self.context_length,
            context_length=self.context_length,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            head_dropout=self.head_dropout
        )

    def _prepare_batch(self, batch_data, device, source_channel, target_channel):
        """Prepare batch data for training"""
        # Get input and target signals
        x = batch_data[8][:, source_channel:source_channel + 1].to(device, torch.float32)
        y_target = batch_data[8][:, target_channel:target_channel + 1].to(device, torch.float32)

        return {
            "x": x,
            "y_target": y_target
        }

    def _train_epoch(self, epoch, ddp_model, train_loader, optim, device, master_process, 
                     source_channel, target_channel):
        """Run one training epoch"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        # Initialize epoch metrics
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        num_batches = 0
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            ddp_model.train()
            tepoch.set_description(f"Train - Epoch {epoch}")
            
            for step, train_file in enumerate(tepoch):
                # Prepare batch with source/target channels
                batch = self._prepare_batch(train_file, device, source_channel, target_channel)
                
                # Training step
                optim.zero_grad(set_to_none=True)
                outputs = ddp_model(batch["x"].permute(0, 2, 1)).unsqueeze(1)
                
                # Calculate MSE loss
                mse_loss = torch.nn.functional.mse_loss(batch["y_target"], outputs)
                with torch.no_grad():
                    mae_loss = torch.nn.functional.l1_loss(batch["y_target"], outputs.detach())
                
                loss = mse_loss  # Use MSE for training
                loss.backward()
                optim.step()

                # Accumulate losses for epoch average
                epoch_mse_loss += mse_loss.item()
                epoch_mae_loss += mae_loss.item()
                num_batches += 1

                # Log training metrics per step (only for master process)
                if master_process and wandb.run is not None:
                    try:
                        wandb.log({
                            "train/step": step + epoch * len(train_loader),
                            "train/step_mse": mse_loss.item(),
                            "train/step_mae": mae_loss.item()
                        })
                    except Exception as e:
                        print(f"Failed to log to wandb: {e}")
                
                # Update progress bar
                tepoch.set_postfix(
                    mse=f"{mse_loss.item():.4f}",
                    mae=f"{mae_loss.item():.4f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}"
                )
                
                # Clear memory
                del batch, loss, outputs, mse_loss, mae_loss
                torch.cuda.empty_cache()
        
        return epoch_mse_loss / num_batches, epoch_mae_loss / num_batches

    def _validate_epoch(self, epoch, ddp_model, val_loader, device, master_process,
                       source_channel, target_channel):
        """Run one validation epoch"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch_val:
                ddp_model.eval()
                total_mse_loss = 0
                total_mae_loss = 0
                num_batches = 0
                tepoch_val.set_description(f"Val - Epoch {epoch}")
                
                for val_file in tepoch_val:
                    # Prepare batch with source/target channels
                    batch = self._prepare_batch(val_file, device, source_channel, target_channel)
                    outputs = ddp_model(batch["x"].permute(0, 2, 1)).unsqueeze(1)
                    
                    # Calculate losses
                    batch_mse_loss = torch.nn.functional.mse_loss(batch["y_target"], outputs)
                    batch_mae_loss = torch.nn.functional.l1_loss(batch["y_target"], outputs)
                    
                    total_mse_loss += batch_mse_loss.item()
                    total_mae_loss += batch_mae_loss.item()
                    num_batches += 1
                    
                    tepoch_val.set_postfix(
                        mse=f"{batch_mse_loss.item():.4f}",
                        mae=f"{batch_mae_loss.item():.4f}"
                    )
                    
                    del batch, outputs, batch_mse_loss, batch_mae_loss
                    torch.cuda.empty_cache()
                
                avg_mse = total_mse_loss / num_batches
                avg_mae = total_mae_loss / num_batches
                
                # Gather losses from all processes
                mse_tensor = torch.tensor([avg_mse], device=device)
                mae_tensor = torch.tensor([avg_mae], device=device)
                
                gathered_mse = [torch.zeros_like(mse_tensor) for _ in range(dist.get_world_size())]
                gathered_mae = [torch.zeros_like(mae_tensor) for _ in range(dist.get_world_size())]
                
                dist.all_gather(gathered_mse, mse_tensor)
                dist.all_gather(gathered_mae, mae_tensor)
                
                final_mse = torch.stack(gathered_mse).mean().item()
                final_mae = torch.stack(gathered_mae).mean().item()
                
                if master_process:
                    print(f"\nValidation MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")
                
                return final_mse, final_mae

    def get_source_target_pairs(self):
        """Get all valid source-target pairs for training"""
        signals = [self.ecg_label, self.ppg_label, self.abp_label]
        pairs = []
        for source in signals:
            for target in signals:
                if source != target:
                    pairs.append((source, target))
        return pairs

    def get_model_name(self, source, target):
        """Get model name for a specific source-target pair"""
        return f"{self.channel_names[source]}2{self.channel_names[target]}"

    def get_checkpoint_path(self, source, target, epoch: int = None, saving: bool = False):
        """Get checkpoint path for a specific source-target pair"""
        model_name = self.get_model_name(source, target)
        sub_folder = f'AppModel_PatchTST_BS_{self.batch_size}_E_{self.num_epochs}_LR_{self.learning_rate}_P_{self.scheduler_patience}_ES_{self.early_stopping_patience}'
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
            'sub_folder': sub_folder
        }

    # Include remaining helper methods from other configs
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
            return source, target
        except ValueError as e:
            raise ValueError(f"Invalid direction format {direction}. Must be SOURCE2TARGET (e.g. PPG2ABP). {str(e)}")

    def load_checkpoint_with_retry(self, rank, model, optim, scheduler, early_stopping, checkpoint_path,
                           max_retries=3, wait_time=5):
        """Helper function to load checkpoint with retry logic"""
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

                            # Load early stopping state on rank 0
                            if 'early_stopping_state' in checkpoint:
                                early_stopping.load_state_dict(checkpoint['early_stopping_state'])

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
                'best_recon_total_loss': val_loss
            }, checkpoint_path)
            best_loss['best_recon_total_loss'] = val_loss
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if epoch % 5 == 0:
            # Get correct checkpoint path
            checkpoint_info_epoch = self.get_checkpoint_path(self.source_channel, self.target_channel, epoch, saving=True)
            checkpoint_path_epoch = checkpoint_info_epoch['file']
            torch.save({
                'model_state_dict': ddp_model.module.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),  # Save epoch+1 here
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
                train_mse, train_mae = self._train_epoch(e, ddp_model, train_loader, optim, device, master_process, 
                                source_channel=source_channel, target_channel=target_channel)
                val_mse, val_mae = self._validate_epoch(e, ddp_model, val_loader, device, master_process,
                                              source_channel=source_channel, target_channel=target_channel)
                test_mse, test_mae = self._validate_epoch(e, ddp_model, test_loader, device, master_process,
                                              source_channel=source_channel, target_channel=target_channel)

                if master_process:
                    # Get checkpoint info for this direction
                    checkpoint_info = self.get_checkpoint_path(source_channel, target_channel, saving=True)
                    prev_lr = self._update_training_state(
                        e, val_mae, ddp_model, optim, scheduler,
                        early_stopping, checkpoint_info['file'], best_loss, prev_lr
                    )

                    # Log epoch metrics to wandb
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

        # Determine which direction to test
        if self.direction is None:
            raise ValueError("Direction must be specified for testing (e.g. 'PPG2ABP', 'ECG2ABP', etc.)")
        
        # Parse direction to get source and target
        source, target = self._parse_direction(self.direction)
        source_channel = getattr(self, f"{source.lower()}_label")
        target_channel = getattr(self, f"{target.lower()}_label")

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
            print(f"\nModel Parameters for {self.direction}:")
            print_model_parameters(model)

            # If only extracting waveform features, skip the rest of test evaluation
            if hasattr(self, 'extract_waveform_features') and self.extract_waveform_features:
                print(f"Extracting waveform features from {self.direction} reconstructed signals...")
                
                # Check if target is ECG or PPG
                if target.lower() not in ['ecg', 'ppg']:
                    print(f"\nSkipping {self.direction} feature extraction: Target signal {target} is not ECG or PPG")
                    return
                
                # Initialize features dictionary
                direction_features = {}
                
                with torch.no_grad():
                    # Check if features file already exists
                    features_dir = os.path.join('./results/features_extraction', 'PulseDB', 'PatchTST', f'seed_{self.seed}')
                    filename = f'features_{self.direction}.h5'
                    filepath = os.path.join(features_dir, filename)
                    
                    try:
                        check_file_exists(filepath)
                    except FileExistsError as e:
                        print(f"\nSkipping {self.direction} feature extraction: {e}")
                        return
                    
                    with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                        tepoch_test.set_description(f"Extracting {self.direction} Features")
                        
                        for batch_idx, test_file in enumerate(tepoch_test):
                            # Get source and target signals
                            source_signal = test_file[8][:, source_channel:source_channel + 1].to(device, torch.float32)
                            target_signal = test_file[8][:, target_channel:target_channel + 1].to(device, torch.float32)
                            
                            # Get global indices from the dataset
                            global_indices = test_file[1]  # Global indices
                            
                            # Get reconstruction
                            recon = model(source_signal.permute(0, 2, 1)).unsqueeze(1)
                            
                            # Extract features from reconstructed waveforms
                            batch_size = recon.size(0)
                            with tqdm(range(batch_size), unit="sample", ncols=100, leave=False) as pbar:
                                pbar.set_description(f"Batch {batch_idx}")
                                for i in pbar:
                                    # Get global index
                                    global_idx = global_indices[i].item()
                                    
                                    # Extract features for this sample
                                    features = extract_waveform_features(
                                        ecg_signal=Min_Max_Norm_Torch(recon[i:i+1, :, 15:-15])[0, 0, :].cpu().numpy() if target.lower() == 'ecg' else None,
                                        ppg_signal=Min_Max_Norm_Torch(recon[i:i+1, :, 15:-15])[0, 0, :].cpu().numpy() if target.lower() == 'ppg' else None
                                    )
                                    
                                    # Store features using global index as key
                                    direction_features[global_idx] = features
                                    
                                    # Update inner progress bar
                                    pbar.set_postfix({
                                        'sample': f"{i+1}/{batch_size}",
                                        'idx': global_idx,
                                        'features': '✓' if features[f'{target.lower()}_features'] is not None else '✗'
                                    })
                            
                            # Clear memory
                            torch.cuda.empty_cache()
                    
                    # Save features for this direction
                    print(f"\nSaving {self.direction} features...")
                    save_waveform_features(
                        features_dict=direction_features,
                        dataset='PulseDB',
                        model_name='PatchTST',
                        seed=self.seed,
                        direction=self.direction,
                        output_dir=self.abs_path
                    )
                
                print("Feature extraction and saving completed.")
                return
            else:
                # Regular test evaluation continues here
                # Loss function
                l1 = nn.L1Loss(reduction='none')

                # Initialize metrics dictionaries
                mae_metrics = {f'{self.direction.lower()}_loss': []}
                correlation_metrics = {f'pC_{self.direction.lower()}_loss': []}
                
                # Initialize BPM metrics if calculate_bpm is True
                bpm_metrics = {} if hasattr(self, 'calculate_bpm') and self.calculate_bpm else None

                with torch.no_grad():
                    with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                        tepoch_test.set_description(f"Testing {self.direction}")
                        
                        for batch_idx, test_file in enumerate(tepoch_test):
                            # Get ground truth rates from dataset if BPM calculation is enabled
                            ground_truth_rates = None
                            if bpm_metrics is not None:
                                ground_truth_rates = test_file[9].to(device)  # Index 9 contains [heart_rate, pulse_rate]
                            
                            # Prepare batch data
                            x = test_file[8][:, source_channel:source_channel + 1].to(device, torch.float32)
                            y_target = test_file[8][:, target_channel:target_channel + 1].to(device, torch.float32)
                            
                            # Get reconstruction
                            output = model(x.permute(0, 2, 1)).unsqueeze(1)
                            
                            # Format data for helper functions
                            recons = {self.direction.lower(): output}  # Use lowercase direction
                            targets = {target.lower(): y_target}
                            
                            # Calculate losses
                            losses, batch_losses = calculate_reconstruction_losses(recons, targets, l1)
                            
                            # Calculate Pearson correlations
                            correlations = calculate_pearson_correlations(recons, targets, correlation_metrics)
                            
                            # Update mae_metrics dictionary
                            mae_metrics[f'{self.direction.lower()}_loss'].extend(batch_losses[f'{self.direction.lower()}_loss'])

                            # Calculate BPM metrics if enabled
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
                                # Update progress bar with all metrics
                                tepoch_test.set_postfix({
                                    "mae": f"{torch.mean(losses[self.direction.lower()].cpu()):.4f}",
                                    "corr": f"{correlations[self.direction.lower()]:.4f}" if self.direction.lower() in correlations else "0.0000",
                                    **{f"BPM_{k}": f"{v:.2f}" for k, v in batch_bpm_means.items()}
                                })
                            else:
                                # Update progress bar with basic metrics
                                tepoch_test.set_postfix({
                                    "mae": f"{torch.mean(losses[self.direction.lower()].cpu()):.4f}",
                                    "corr": f"{correlations[self.direction.lower()]:.4f}" if self.direction.lower() in correlations else "0.0000"
                                })
                            
                            # Clear memory
                            del x, y_target, output, recons, targets, losses, batch_losses, correlations
                            if ground_truth_rates is not None:
                                del ground_truth_rates
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

        except Exception as e:
            print(f"Error testing {self.direction} model: {str(e)}")
            raise
        finally:
            # Clean up model
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()
