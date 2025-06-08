# Standard library imports
from dataclasses import dataclass
import os

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
from baseline.ShallowUNet import ShallowUNet
from utils.ddp_utils import print_memory_stats, create_ddp_dataloaders
from utils.train_utils import EarlyStopping
from utils.utils_preprocessing import (
    print_model_parameters,
    isExist_dir,
    Min_Max_Norm_Torch
)
from utils.test_utils import (
    calculate_bp_metrics,
    print_abp_evaluation_results
)
from .app_model_config import PulseDBApproximationNabNetConfig

@dataclass
class PulseDBRefinementNabNetConfig(PulseDBBaseConfig):
    """PulseDB dataset configuration for Refinement model"""
    args: dict = None
    # Direction configuration
    direction: str = None  # If None, train all combinations, otherwise format: "SOURCE2ABP" (e.g. "PPG2ABP")
    path_folder: str = 'RefModel/PulseDB'
    is_finetuning: bool = False  # Add finetuning flag
    use_patient_split: bool = False

    # ShallowUNet specific parameters
    model_depth: int = 1
    model_width: int = 84
    feature_number: int = 1024
    hidden_size: int = 100
    num_channel: int = 1
    output_nums: int = 1
    kernel_size: int = 3
    problem_type: str = 'Regression'
    deep_supervision: int = 0
    autoencoder: int = 1
    guided_attention: int = 0
    use_transconv: bool = True
    use_lstm: int = 0

    # DDP-specific settings for refinement model
    find_unused_parameters: bool = True  # Override base setting if needed
    empty_cache_frequency: int = 10  # More frequent cache clearing for refinement

    def __post_init__(self):
        super().__post_init__()
        if self.args is not None:
            # Basic training parameters
            self.batch_size = self.args.batch_size
            self.test_batch_size = self.args.test_batch_size
            self.num_epochs = self.args.num_epochs
            self.learning_rate = self.args.learning_rate
            self.scheduler_patience = self.args.scheduler_patience
            self.early_stopping_patience = self.args.early_stopping_patience
            self.model_type = self.args.model_type
            # Checkpoint settings
            self.checkpoint_name = self.args.checkpoint_name
            self.checkpoint_epoch = self.args.checkpoint_epoch
            
            # Direction configuration
            if hasattr(self.args, 'direction'):
                self.direction = self.args.direction
            
            # Boolean flags
            if hasattr(self.args, 'is_finetuning'):
                self.is_finetuning = self.args.is_finetuning
            if hasattr(self.args, 'use_patient_split'):
                self.use_patient_split = self.args.use_patient_split
            if hasattr(self.args, 'resume_training'):
                self.resume_training = self.args.resume_training
            if hasattr(self.args, 'is_pretraining'):
                self.is_pretraining = self.args.is_pretraining
            
            # Additional settings
            if hasattr(self.args, 'seed'):
                self.seed = self.args.seed

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
            valid_sources = set(self.channel_names.values()) - {'ABP'}  # All channels except ABP can be source
            if source not in valid_sources:
                raise ValueError(f"Invalid source channel '{source}'. Valid source channels are {valid_sources}")
            if target != 'ABP':
                raise ValueError(f"Invalid target channel '{target}'. Only 'ABP' is allowed as target for refinement")
            return source, target
        except ValueError as e:
            raise ValueError(f"Invalid direction format {direction}. Must be SOURCE2ABP (e.g. PPG2ABP). {str(e)}")

    def get_source_target_pairs(self):
        """Get all valid source-target pairs for training"""
        # All possible signal types except ABP
        sources = [self.ecg_label, self.ppg_label]  # All signals except ABP
        target = self.abp_label  # ABP is always the target
        
        # Generate all valid pairs (source, target)
        pairs = []
        for source in sources:
            pairs.append((source, target))
        return pairs

    def get_model_name(self, source, target):
        """Get model name for a specific source-target pair"""
        return f"{self.channel_names[source]}2{self.channel_names[target]}"

    def get_checkpoint_info(self, is_finetuning=False, epoch: int=None, saving: bool=False):
        """Get checkpoint directory and file information"""
        source = self.source_channel if self.direction else None
        target = self.target_channel if self.direction else None
        model_name = self.get_model_name(source, target)
        sub_folder = f'RefModel_ShallowUnet_BS_{self.batch_size}_E_{self.num_epochs}_LR_{self.learning_rate}_P_{self.scheduler_patience}_ES_{self.early_stopping_patience}'
        
        # Add finetuning suffix if in finetuning mode
        if is_finetuning:
            sub_folder += '_finetuning'
        
        # Add patient split suffix if using patient split
        if self.use_patient_split:
            sub_folder += '_Patient_Split'
        
        checkpoint_dir = os.path.join(self.abs_path, self.path_folder, sub_folder)
        
        # Saving checkpoint use current epoch
        if saving:
            if epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}_epoch_{epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}.pt')
        else:
            # Loading checkpoint use checkpoint_epoch
            if self.checkpoint_epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}_epoch_{self.checkpoint_epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{model_name}_{self.checkpoint_name}.pt')
        
        return {
            'dir': checkpoint_dir,
            'file': checkpoint_file,
            'sub_folder': sub_folder,
            'is_finetuning': is_finetuning
        }

    def create_model(self):
        """Create ShallowUNet model instance"""
        return ShallowUNet(
            input_size=self.input_size,
            feature_number=self.feature_number,
            hidden_size=self.hidden_size,
            model_depth=self.model_depth,
            num_channel=self.num_channel,
            model_width=self.model_width,
            kernel_size=self.kernel_size,
            problem_type=self.problem_type,
            output_nums=self.output_nums,
            ds=self.deep_supervision,
            ae=self.autoencoder,
            ag=self.guided_attention,
            is_transconv=self.use_transconv,
            lstm=self.use_lstm
        )

    def trainer(self, dataset:tuple, rank: int, world_size: int, ddp_config=None):
        """Train model(s) based on configuration"""
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

    def _train_single_direction(self, dataset:tuple, rank: int, world_size: int, ddp_config, source_channel, target_channel):
        """Train for a specific source-target direction in two steps"""
        try:
            # Step 1: Train waveform reconstruction
            if rank == 0:
                print("\nStep 1: Training waveform reconstruction...")
            self._train_waveform_step(dataset, rank, world_size, ddp_config, source_channel, target_channel)

            # Step 2: Train BP prediction
            if rank == 0:
                print("\nStep 2: Training BP prediction...")
            self._train_bp_step(dataset, rank, world_size, ddp_config, source_channel, target_channel)

        except Exception as e:
            print(f"Rank {rank} encountered error training {self.channel_names[source_channel]}2{self.channel_names[target_channel]}: {e}")
            raise

    def _train_waveform_step(self, dataset:tuple, rank: int, world_size: int, ddp_config, source_channel, target_channel):
        """First training step: Waveform reconstruction"""
        try:
            # Setup device and data
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{rank}')
            master_process = rank == 0

            # Get datasets and create dataloaders
            train_dataset, val_dataset, test_dataset = dataset
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

            # Load checkpoint if exists
            training_state = self.load_checkpoint_with_retry(
                rank=rank,
                model=model,
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                device=device,
                current_prefix='wave_'
            )

            # Now wrap with DDP
            ddp_model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=self.find_unused_parameters,
                static_graph=ddp_config.static_graph if ddp_config else True
            )

            epoch_checkpoint = training_state['epoch_checkpoint']
            best_loss = {'best_recon_total_loss': training_state['best_recon_total_loss']}
            prev_lr = optim.param_groups[0]['lr']

            # Training loop
            print_memory_stats(rank, "Before waveform training loop")
            for e in range(epoch_checkpoint, self.num_epochs):
                # Train for one epoch
                train_loss = self._train_waveform_epoch(e, ddp_model, train_loader, optim, device, master_process, 
                                                      source_channel, target_channel)
                
                # Validate on validation set
                val_loss = self._validate_waveform_epoch(e, ddp_model, val_loader, device, master_process,
                                                       source_channel, target_channel)
                
                # Test on test set
                test_loss = self._validate_waveform_epoch(e, ddp_model, test_loader, device, master_process,
                                                        source_channel, target_channel)

                if master_process:
                    # Update training state (scheduler, early stopping, checkpoints)
                    prev_lr = self._update_training_state(
                        e, val_loss, ddp_model, optim, scheduler,
                        early_stopping, best_loss, prev_lr, bp_pred=False
                    )
                    
                    # Log comprehensive metrics to wandb (only if run exists)
                    if wandb.run is not None:
                        try:
                            log_dict = {
                                "waveform/epoch": e,
                                "train/waveform/epoch_loss": float(train_loss),
                                "val/waveform/epoch_loss": float(val_loss),
                                "test/waveform/epoch_loss": float(test_loss),
                                "waveform/learning_rate": float(optim.param_groups[0]['lr']),
                                "waveform/best_loss": float(best_loss['best_recon_total_loss']) if best_loss['best_recon_total_loss'] is not None else None,
                                "waveform/improved": bool(best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss),
                                "waveform/early_stopping_counter": int(early_stopping.counter)
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

        finally:
            # Cleanup wandb
            if rank == 0 and wandb.run is not None:
                wandb.finish()
            if 'train_loader' in locals():
                self._cleanup(train_loader, val_loader, test_loader)

    def _train_bp_step(self, dataset:tuple, rank: int, world_size: int, ddp_config, source_channel, target_channel):
        """Second training step: BP prediction"""        
        try:
            # Setup device and data
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{rank}')
            master_process = rank == 0

            # Get datasets and create dataloaders
            train_dataset, val_dataset, test_dataset = dataset
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

            # Load checkpoint if exists
            training_state = self.load_checkpoint_with_retry(
                rank=rank,
                model=model,
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                device=device,
                current_prefix='bp_'
            )

            # Freeze UNet parameters after loading checkpoint
            model.unet.eval()  # Set UNet to eval mode
            for param in model.unet.parameters():
                param.requires_grad = False

            # Now wrap with DDP
            ddp_model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=self.find_unused_parameters,
                static_graph=ddp_config.static_graph if ddp_config else True
            )

            epoch_checkpoint = training_state['epoch_checkpoint']
            best_loss = {'best_recon_total_loss': training_state['best_recon_total_loss']}
            prev_lr = optim.param_groups[0]['lr']

            # Wandb initialization for second phase BP prediction.
            # Only handle wandb in rank 0
            if rank == 0 and self.args.project_name is not None:
                try:
                    wandb.init(
                        entity="single_waveform",
                        project=self.args.project_name,
                        config=vars(self.args)
                    )
                except Exception as e:
                    print(f"Failed to start wandb run: {str(e)}")

                if wandb.run is None:
                    print("wandb run not found in rank 0 process - logs will not be sent to wandb")
                else:
                    print(f"Using wandb run: {wandb.run.name}")

            # Training loop
            print_memory_stats(rank, "Before BP training loop")
            for e in range(epoch_checkpoint, self.num_epochs):
                # Train for one epoch
                train_losses = self._train_bp_epoch(e, ddp_model, train_loader, optim, device, master_process, 
                                                  source_channel, target_channel)
                
                # Validate on validation set
                val_losses = self._validate_bp_epoch(e, ddp_model, val_loader, device, master_process,
                                                   source_channel, target_channel)
                
                # Test on test set 
                test_losses = self._validate_bp_epoch(e, ddp_model, test_loader, device, master_process,
                                                        source_channel, target_channel)

                if master_process:
                    # Update training state (scheduler, early stopping, checkpoints)
                    prev_lr = self._update_training_state(
                        e, val_losses['total_loss'], ddp_model, optim, scheduler,
                        early_stopping, best_loss, prev_lr, bp_pred=True
                    )
                    
                    # Log comprehensive metrics to wandb (only if run exists)
                    if wandb.run is not None:
                        try:
                            log_dict = {
                                "epoch": e,
                                "train/epoch_loss": float(train_losses['total_loss']),
                                "train/epoch_sbp_mae": float(train_losses['sbp_loss']),
                                "train/epoch_dbp_mae": float(train_losses['dbp_loss']),
                                "train/epoch_sbp_mse": float(train_losses['sbp_mse']),
                                "train/epoch_dbp_mse": float(train_losses['dbp_mse']),
                                "val/epoch_loss": float(val_losses['total_loss']),
                                "val/epoch_sbp_mae": float(val_losses['sbp_loss']),
                                "val/epoch_dbp_mae": float(val_losses['dbp_loss']),
                                "val/epoch_sbp_mse": float(val_losses['sbp_mse']),
                                "val/epoch_dbp_mse": float(val_losses['dbp_mse']),
                                "learning_rate": float(optim.param_groups[0]['lr']),
                                "best_loss": float(best_loss['best_recon_total_loss']) if best_loss['best_recon_total_loss'] is not None else None,
                                "improved": bool(best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_losses['total_loss']),
                                "early_stopping_counter": int(early_stopping.counter)
                            }
                            
                            # Add test metrics
                            log_dict.update({
                                "test/epoch_loss": float(test_losses['total_loss']),
                                "test/epoch_sbp_mae": float(test_losses['sbp_loss']),
                                "test/epoch_dbp_mae": float(test_losses['dbp_loss']),
                                "test/epoch_sbp_mse": float(test_losses['sbp_mse']),
                                "test/epoch_dbp_mse": float(test_losses['dbp_mse'])
                            })
                            
                            wandb.log(log_dict)
                        except Exception as e:
                            print(f"Failed to log to wandb: {e}")

                # Check for early stopping
                if self._check_early_stopping(early_stopping, rank, device):
                    break

                # Clear cache periodically
                if ddp_config and e % ddp_config.empty_cache_frequency == 0:
                    torch.cuda.empty_cache()

        finally:
            if 'train_loader' in locals():
                self._cleanup(train_loader, val_loader, test_loader)

    def _prepare_batch(self, batch_data, device, source_channel, target_channel):
        """Prepare batch data for training"""
        return {
            "x": batch_data[8][:, source_channel:source_channel+1].to(device, torch.float32),
            "y_target": batch_data[11][:, :].to(device, torch.float32),
            "sbp": batch_data[10][:, 0:1].to(device, torch.float32),
            "dbp": batch_data[10][:, 1:2].to(device, torch.float32)
        }

    def _train_waveform_epoch(self, epoch, ddp_model, train_loader, optim, device, master_process, 
                             source_channel, target_channel):
        """Train one epoch for waveform reconstruction"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        # Initialize epoch loss tracking
        epoch_total_loss = 0.0
        num_batches = 0
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            ddp_model.train()
            tepoch.set_description(f"Train Waveform - Epoch {epoch}")

            for step, train_file in enumerate(tepoch):
                # Prepare batch
                batch = self._prepare_batch(train_file, device, source_channel, target_channel)
                
                # Training step
                optim.zero_grad(set_to_none=True)
                waveform_pred = ddp_model(batch["x"], return_abp=True)  # Only get waveform prediction
                loss = nn.MSELoss()(waveform_pred, batch["y_target"])
                loss.backward()
                optim.step()
                
                # Update epoch average
                epoch_total_loss += loss.item()
                num_batches += 1
                
                # Log every step to wandb (only master process) with correct key format
                if master_process and wandb.run is not None:
                    try:
                        wandb.log({
                            "train/waveform/step": step + epoch * len(train_loader),
                            "train/waveform/step_loss": loss.item(),
                        })
                    except Exception as e:
                        print(f"Failed to log step metrics to wandb: {e}")

                # Update progress bar
                tepoch.set_postfix(loss=f"{loss.item():.4f}")

                # Clear memory
                del batch, loss, waveform_pred
                torch.cuda.empty_cache()
                
            # Calculate epoch average loss
            epoch_avg_loss = epoch_total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Gather losses from all processes
            loss_tensor = torch.tensor([epoch_avg_loss], device=device)
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss_tensor)
            final_loss = torch.stack(gathered_losses).mean().item()
            
            return final_loss

    def _validate_waveform_epoch(self, epoch, ddp_model, val_loader, device, master_process,
                                source_channel, target_channel):
        """Validate one epoch for waveform reconstruction"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
                ddp_model.eval()
                total_loss = 0
                num_batches = 0
                tepoch.set_description(f"Val Waveform - Epoch {epoch}")

                for val_file in tepoch:
                    # Prepare batch
                    batch = self._prepare_batch(val_file, device, source_channel, target_channel)
                    
                    # Forward pass
                    waveform_pred = ddp_model(batch["x"], return_abp=True)  # Only get waveform prediction
                    loss = nn.MSELoss()(waveform_pred, batch["y_target"])
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")
                    
                    del batch, loss, waveform_pred

                # Calculate average loss
                avg_loss = total_loss / num_batches
                
                # Gather losses from all processes
                loss_tensor = torch.tensor([avg_loss], device=device)
                gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_losses, loss_tensor)
                final_loss = torch.stack(gathered_losses).mean().item()
                
                if master_process:
                    print(f"\nValidation Waveform Loss: {final_loss:.4f}")
                
                return final_loss

    def _train_bp_epoch(self, epoch, ddp_model, train_loader, optim, device, master_process,
                        source_channel, target_channel):
        """Train one epoch for BP prediction"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        # Initialize epoch loss tracking
        epoch_total_loss = 0.0
        epoch_sbp_loss = 0.0  # MAE
        epoch_dbp_loss = 0.0  # MAE
        epoch_sbp_mse = 0.0   # MSE
        epoch_dbp_mse = 0.0   # MSE
        num_batches = 0
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            # Access the underlying model through .module
            ddp_model.eval()  # Set entire model to eval mode
            ddp_model.module.unet.eval()  # Set UNet specifically to eval mode
            ddp_model.module.mlp_sbp.train()  # Set SBP MLP to train mode
            ddp_model.module.mlp_dbp.train()  # Set DBP MLP to train mode
            tepoch.set_description(f"Train BP - Epoch {epoch}")

            for step, train_file in enumerate(tepoch):
                # Prepare batch
                batch = self._prepare_batch(train_file, device, source_channel, target_channel)
                
                # Training step
                optim.zero_grad(set_to_none=True)
                sbp_pred, dbp_pred = ddp_model(batch["x"], return_abp=False)  # Get BP predictions
                
                # Calculate BP losses (both MAE and MSE)
                sbp_loss = nn.L1Loss()(sbp_pred, batch["sbp"])  # MAE
                dbp_loss = nn.L1Loss()(dbp_pred, batch["dbp"])  # MAE
                with torch.no_grad():
                    sbp_mse = nn.MSELoss()(sbp_pred.detach(), batch["sbp"])  # MSE
                    dbp_mse = nn.MSELoss()(dbp_pred.detach(), batch["dbp"])  # MSE
                
                # Total loss remains same (using MAE for training)
                total_loss = sbp_loss + dbp_loss
                
                total_loss.backward()
                optim.step()

                # Update epoch averages
                epoch_total_loss += total_loss.item()
                epoch_sbp_loss += sbp_loss.item()
                epoch_dbp_loss += dbp_loss.item()
                epoch_sbp_mse += sbp_mse.item()
                epoch_dbp_mse += dbp_mse.item()
                num_batches += 1
                
                # Log every step to wandb (only master process) with correct key format
                if master_process and wandb.run is not None:
                    try:
                        wandb.log({
                            "train/step": step + epoch * len(train_loader),
                            "train/step_loss": total_loss.item(),
                            "train/step_sbp_mae": sbp_loss.item(),
                            "train/step_dbp_mae": dbp_loss.item(),
                            "train/step_sbp_mse": sbp_mse.item(),
                            "train/step_dbp_mse": dbp_mse.item()
                        })
                    except Exception as e:
                        print(f"Failed to log step metrics to wandb: {e}")

                # Update progress bar
                tepoch.set_postfix(
                    sbp_mae=f"{sbp_loss.item():.4f}",
                    dbp_mae=f"{dbp_loss.item():.4f}",
                    sbp_mse=f"{sbp_mse.item():.4f}",
                    dbp_mse=f"{dbp_mse.item():.4f}"
                )

                # Clear memory
                del batch, sbp_loss, dbp_loss, total_loss, sbp_pred, dbp_pred, sbp_mse, dbp_mse
                torch.cuda.empty_cache()
                
            # Calculate epoch averages
            if num_batches > 0:
                epoch_avg_total_loss = epoch_total_loss / num_batches
                epoch_avg_sbp_loss = epoch_sbp_loss / num_batches
                epoch_avg_dbp_loss = epoch_dbp_loss / num_batches
                epoch_avg_sbp_mse = epoch_sbp_mse / num_batches
                epoch_avg_dbp_mse = epoch_dbp_mse / num_batches
            else:
                epoch_avg_total_loss = float('inf')
                epoch_avg_sbp_loss = float('inf')
                epoch_avg_dbp_loss = float('inf')
                epoch_avg_sbp_mse = float('inf')
                epoch_avg_dbp_mse = float('inf')
            
            # Gather losses from all processes (including MSE)
            loss_tensor = torch.tensor([
                epoch_avg_total_loss, 
                epoch_avg_sbp_loss, 
                epoch_avg_dbp_loss,
                epoch_avg_sbp_mse,
                epoch_avg_dbp_mse
            ], device=device)
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss_tensor)
            
            final_total_loss = torch.stack([t[0] for t in gathered_losses]).mean().item()
            final_sbp_loss = torch.stack([t[1] for t in gathered_losses]).mean().item()
            final_dbp_loss = torch.stack([t[2] for t in gathered_losses]).mean().item()
            final_sbp_mse = torch.stack([t[3] for t in gathered_losses]).mean().item()
            final_dbp_mse = torch.stack([t[4] for t in gathered_losses]).mean().item()
            
            return {
                "total_loss": final_total_loss,
                "sbp_loss": final_sbp_loss,    # MAE
                "dbp_loss": final_dbp_loss,    # MAE
                "sbp_mse": final_sbp_mse,      # MSE
                "dbp_mse": final_dbp_mse       # MSE
            }

    def _validate_bp_epoch(self, epoch, ddp_model, val_loader, device, master_process,
                          source_channel, target_channel):
        """Validate one epoch for BP prediction"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
                ddp_model.eval()
                total_sbp_loss = 0
                total_dbp_loss = 0
                total_sbp_mse = 0  # Track MSE for SBP
                total_dbp_mse = 0  # Track MSE for DBP
                num_batches = 0
                tepoch.set_description(f"Val BP - Epoch {epoch}")

                for val_file in tepoch:
                    # Prepare batch
                    batch = self._prepare_batch(val_file, device, source_channel, target_channel)
                    
                    # Forward pass
                    sbp_pred, dbp_pred = ddp_model(batch["x"], return_abp=False)  # Get BP predictions
                    
                    # Calculate BP losses (L1 and MSE)
                    sbp_loss = nn.L1Loss()(sbp_pred, batch["sbp"])
                    dbp_loss = nn.L1Loss()(dbp_pred, batch["dbp"])
                    sbp_mse = nn.MSELoss()(sbp_pred, batch["sbp"])
                    dbp_mse = nn.MSELoss()(dbp_pred, batch["dbp"])
                    
                    # Accumulate losses
                    total_sbp_loss += sbp_loss.item()
                    total_dbp_loss += dbp_loss.item()
                    total_sbp_mse += sbp_mse.item()
                    total_dbp_mse += dbp_mse.item()
                    num_batches += 1
                    
                    tepoch.set_postfix(
                        sbp_loss=f"{sbp_loss.item():.4f}",
                        dbp_loss=f"{dbp_loss.item():.4f}"
                    )
                    
                    del batch, sbp_loss, dbp_loss, sbp_pred, dbp_pred, sbp_mse, dbp_mse

                # Calculate average losses
                if num_batches > 0:
                    avg_sbp_loss = total_sbp_loss / num_batches
                    avg_dbp_loss = total_dbp_loss / num_batches
                    avg_sbp_mse = total_sbp_mse / num_batches
                    avg_dbp_mse = total_dbp_mse / num_batches
                    avg_total_loss = (avg_sbp_loss + avg_dbp_loss) / 2
                else:
                    avg_sbp_loss = float('inf')
                    avg_dbp_loss = float('inf')
                    avg_sbp_mse = float('inf')
                    avg_dbp_mse = float('inf')
                    avg_total_loss = float('inf')
                
                # Gather losses from all processes
                loss_tensor = torch.tensor([avg_total_loss, avg_sbp_loss, avg_dbp_loss, avg_sbp_mse, avg_dbp_mse], device=device)
                gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_losses, loss_tensor)
                
                final_total_loss = torch.stack([t[0] for t in gathered_losses]).mean().item()
                final_sbp_loss = torch.stack([t[1] for t in gathered_losses]).mean().item()
                final_dbp_loss = torch.stack([t[2] for t in gathered_losses]).mean().item()
                final_sbp_mse = torch.stack([t[3] for t in gathered_losses]).mean().item()
                final_dbp_mse = torch.stack([t[4] for t in gathered_losses]).mean().item()
                
                if master_process:
                    print(f"\nValidation BP Results:")
                    print(f"Total Loss: {final_total_loss:.4f}")
                    print(f"SBP Loss (MAE): {final_sbp_loss:.4f}")
                    print(f"DBP Loss (MAE): {final_dbp_loss:.4f}")
                    print(f"SBP MSE: {final_sbp_mse:.4f}")
                    print(f"DBP MSE: {final_dbp_mse:.4f}")
                
                return {
                    "total_loss": final_total_loss,
                    "sbp_loss": final_sbp_loss,
                    "dbp_loss": final_dbp_loss,
                    "sbp_mse": final_sbp_mse,
                    "dbp_mse": final_dbp_mse
                }

    def _load_checkpoint(self, model, optim, scheduler, early_stopping, 
                        is_finetuning: bool, device: str, load_optimizer: bool, rank: int, current_prefix: str):
        """Helper function to load checkpoint"""
        checkpoint_info = self.get_checkpoint_info(is_finetuning=is_finetuning)
        print(f"Rank {rank} - Loading checkpoint from {checkpoint_info['file']}")
        
        if not os.path.exists(checkpoint_info['file']):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_info['file']}")
        
        checkpoint = torch.load(checkpoint_info['file'], map_location=device)
        
        # Only rank 0 loads model weights (DDP will sync)
        if rank == 0:
            # print(f"Loading checkpoint from {checkpoint_info['file']}")
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {k[7:] if k.startswith('module.') else k: v 
                             for k, v in state_dict.items()}
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"Rank {rank} - Successfully loaded model state from {checkpoint_info['file']}")
            
            if missing_keys:
                print(f"Warning: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys}")
        
        # Load optimizer and scheduler states if requested (all processes)
        if load_optimizer:
            if f'{current_prefix}optim_state_dict' in checkpoint:
                optim.load_state_dict(checkpoint[f'{current_prefix}optim_state_dict'])
                print(f"Rank {rank} - Successfully loaded optimizer state from {checkpoint_info['file']}")
            if f'{current_prefix}scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint[f'{current_prefix}scheduler_state_dict'])
                print(f"Rank {rank} - Successfully loaded scheduler state from {checkpoint_info['file']}")
            if f'{current_prefix}early_stopping_state' in checkpoint:
                early_stopping.load_state_dict(checkpoint[f'{current_prefix}early_stopping_state'])
                print(f"Rank {rank} - Successfully loaded early stopping state from {checkpoint_info['file']}")
        return {
            'epoch_checkpoint': checkpoint.get(f'{current_prefix}epoch', 0),
            'best_recon_total_loss': checkpoint.get(f'{current_prefix}best_recon_total_loss', None)
        }

    def _update_training_state(self, epoch, val_loss, ddp_model, optim, scheduler,
                               early_stopping, best_loss, prev_lr, bp_pred=False):
        """Update training state including scheduler, early stopping, and checkpoints"""
        # Set prefix based on training phase
        current_prefix = 'bp_' if bp_pred else 'wave_'
        
        # Step scheduler and display learning rate
        scheduler.step(val_loss)
        current_lr = optim.param_groups[0]['lr']
        
        if current_lr != prev_lr:
            print(f"Learning rate changed: {prev_lr:.2e} -> {current_lr:.2e}")
        
        # Update early stopping
        early_stopping(val_loss)
        
        # Save checkpoint if better than previous best
        if best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss:
            if best_loss['best_recon_total_loss'] is not None:
                print(f"Validation loss improved from {best_loss['best_recon_total_loss']:.6f} to {val_loss:.6f}")
            
            if self.save_checkpoint(
                ddp_model=ddp_model,
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                epoch=epoch,
                val_loss=val_loss,
                is_best=True,
                current_prefix=current_prefix
            ):
                best_loss['best_recon_total_loss'] = val_loss
        
        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            self.save_checkpoint(
                ddp_model=ddp_model,
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                epoch=epoch,
                val_loss=val_loss,
                is_best=False,
                current_prefix=current_prefix
            )
        
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
        """Clean up resources
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
        """
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
    
    def test(self, num_workers: int):
        """Test the model performance on test set

        Args:
            num_workers (int): Number of dataloader workers
        """
        app_model_config = PulseDBApproximationNabNetConfig(args=self.args)

        # Model configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dataset with patient-level split for testing
        test_dataset = self.create_test_dataset()
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.test_batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=num_workers
        )

        # Initialize metrics dictionary
        best_loss = {
            'total_Sample': len(test_dataset),
        }

        # Determine which directions to test
        if self.direction is None:
            directions = ['PPG2ABP', 'ECG2ABP']
        else:
            directions = [self.direction]
        
        if self.direction is None:
            # Get all possible source-target pairs
            source_target_pairs = self.get_source_target_pairs()
        else:
            # Use the specified direction
            source, target = self._parse_direction(self.direction)
            source_target_pairs = [(getattr(self, f"{source.lower()}_label"), 
                                  getattr(self, f"{target.lower()}_label"))]

        # Add metrics for each direction
        for direction in directions:
            source = direction.split('2')[0]
            best_loss.update({
                f'{direction}_loss': [], f'{direction}_ME_loss': [],
                f'{direction}_SBP_loss': [], f'{direction}_DBP_loss': [], f'{direction}_MAP_loss': [],
                f'{direction}_SBP_ME_loss': [], f'{direction}_DBP_ME_loss': [], f'{direction}_MAP_ME_loss': [],
                f'{direction}_SBP_BHS_5': 0, f'{direction}_SBP_BHS_10': 0, f'{direction}_SBP_BHS_15': 0,
                f'{direction}_DBP_BHS_5': 0, f'{direction}_DBP_BHS_10': 0, f'{direction}_DBP_BHS_15': 0,
                f'{direction}_MAP_BHS_5': 0, f'{direction}_MAP_BHS_10': 0, f'{direction}_MAP_BHS_15': 0,
            })

        # Process each direction
        for direction in directions:
            source = direction.split('2')[0]
            source_channel = getattr(self, f"{source.lower()}_label")

            # Create and setup models for this direction
            g_model_app = app_model_config.create_model().to(device)
            model = self.create_model().to(device)

            # Get checkpoint paths and load models
            app_checkpoint_info = app_model_config.get_checkpoint_path(source_channel, self.abp_label)
            checkpoint_info = self.get_checkpoint_info(is_finetuning=self.is_finetuning)

            try:
                # Load approximation model checkpoint
                app_checkpoint = torch.load(app_checkpoint_info['file'])
                g_model_app.load_state_dict({k.replace('module.', ''): v for k, v in app_checkpoint['model_state_dict'].items()})
                g_model_app.eval()
                print(f'\nLoaded approximation model checkpoint for {direction}')

                # Load refinement model checkpoint
                checkpoint = torch.load(checkpoint_info['file'])
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
                model.eval()
                print(f'Loaded refinement model checkpoint for {direction}')

                # Print model parameters
                print(f"\nApproximation Model Parameters for {direction}:")
                print_model_parameters(g_model_app)
                print(f"\nRefinement Model Parameters for {direction}:")
                print_model_parameters(model)

            except Exception as e:
                print(f"Failed to load models for {direction}: {e}")
                continue

            # Testing loop for this direction
            with torch.no_grad():
                with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                    tepoch_test.set_description(f"Testing {direction}")
                    
                    for test_file in tepoch_test:
                        # Prepare batch data
                        batch = {
                            "x": test_file[8].to(device, torch.float32),
                            "y_target": test_file[2][:, self.abp_label:self.abp_label+1, 15:-15].to(device, torch.float32),
                            "sbp": test_file[3][:, 0:1].to(device, torch.float32),
                            "dbp": test_file[3][:, 1:2].to(device, torch.float32)
                        }

                        # Get approximation model prediction
                        x = batch["x"][:, source_channel:source_channel + 1]
                        x_abp = g_model_app(x)

                        if isinstance(x_abp, list):
                            x_abp = Min_Max_Norm_Torch(x_abp[0][:, :, 15:-15])
                        else:
                            x_abp = Min_Max_Norm_Torch(x_abp[:, :, 15:-15])
                        
                        # Get refinement model prediction
                        outputs = model(x)
                        waveform_pred = x_abp
                        sbp_pred, dbp_pred = outputs[0], outputs[1]

                        # Calculate metrics
                        metrics = calculate_bp_metrics(
                            predictions=(sbp_pred, dbp_pred) if sbp_pred is not None else None,
                            waveform=waveform_pred,
                            sbp_true=batch["sbp"],
                            dbp_true=batch["dbp"],
                            abp_gt=batch["y_target"],
                            prefix=direction,
                            best_loss=best_loss,
                            global_min=self.dbp_min,
                            global_max=self.sbp_max,
                            normalize=True
                        )

                        # Update progress bar
                        tepoch_test.set_postfix(**{
                            f"{direction}_MAE": f"{metrics['waveform_mae']:.2f}",
                            "Batch": f"{tepoch_test.n}/{len(test_loader)}"
                        })

                        # Clear cache periodically
                        if tepoch_test.n % 25 == 0:
                            torch.cuda.empty_cache()

            # Clean up models for this direction
            del g_model_app, model
            torch.cuda.empty_cache()

        # Print final evaluation results
        print_abp_evaluation_results(best_loss, self.direction, args=self.args)

    def load_checkpoint_with_retry(self, rank: int, model, optim, scheduler, early_stopping, device: str, current_prefix: str):
        """Load checkpoint based on training mode and resume settings"""
        training_state = {
            'epoch_checkpoint': 0,
            'best_recon_total_loss': None
        }
        
        # if rank == 0:  # Only rank 0 handles initial loading
        try:
            if self.is_finetuning:
                if self.resume_training:
                    # Try loading finetuning checkpoint first
                    try:
                        training_state = self._load_checkpoint(
                            model, optim, scheduler, early_stopping,
                            is_finetuning=True, device=device,
                            load_optimizer=True, rank=rank,
                            current_prefix=current_prefix
                        )
                        print("Resumed finetuning from existing finetuning checkpoint")
                    except Exception as e:
                        print(f"Failed to load finetuning checkpoint: {e}")
                        print("Attempting to load regular checkpoint for finetuning...")
                        try:
                            training_state = self._load_checkpoint(
                                model, optim, scheduler, early_stopping,
                                is_finetuning=False, device=device,
                                load_optimizer=False, rank=rank,
                                current_prefix=current_prefix
                            )
                            # Reset epoch count for new finetuning
                            training_state['epoch_checkpoint'] = 0
                            training_state['best_recon_total_loss'] = None
                            print("Starting finetuning from regular checkpoint")
                        except Exception as e:
                            raise Exception("Cannot resume finetuning: No valid checkpoint found") from e
                elif current_prefix == 'bp_':
                    try:
                        training_state = self._load_checkpoint(
                            model, optim, scheduler, early_stopping,
                            is_finetuning=True, device=device,
                            load_optimizer=False, rank=rank,
                            current_prefix=current_prefix
                        )
                        # Reset epoch count for BP prediction finetuning
                        training_state['epoch_checkpoint'] = 0
                        training_state['best_recon_total_loss'] = None
                        print("BP Phase: Loaded existing checkpoint from wave finetuning phase")
                    except Exception as e:
                        raise Exception("Cannot load pretrained wave finetuning phase: No valid checkpoint found") from e
                else:
                    # Not resuming, must load regular checkpoint for finetuning
                    try:
                        training_state = self._load_checkpoint(
                            model, optim, scheduler, early_stopping,
                            is_finetuning=False, device=device,
                            load_optimizer=False, rank=rank,
                            current_prefix=current_prefix
                        )
                        # Reset epoch count for new finetuning
                        training_state['epoch_checkpoint'] = 0
                        training_state['best_recon_total_loss'] = None
                        print("Starting new finetuning from regular checkpoint")
                    except Exception as e:
                        raise Exception("Cannot start finetuning: No regular checkpoint found") from e
            else:  # Regular training
                if self.resume_training:
                    try:
                        training_state = self._load_checkpoint(
                            model, optim, scheduler, early_stopping,
                            is_finetuning=False, device=device,
                            load_optimizer=True, rank=rank,
                            current_prefix=current_prefix
                        )
                        print("Resumed training from regular checkpoint")
                    except Exception as e:
                        print(f"Failed to load regular checkpoint: {e}")
                        print("Starting training from scratch")
                elif current_prefix == 'bp_':
                    try:
                        training_state = self._load_checkpoint(
                            model, optim, scheduler, early_stopping,
                            is_finetuning=False, device=device,
                            load_optimizer=False, rank=rank,
                            current_prefix=current_prefix
                        )
                        # Reset epoch count for BP prediction finetuning
                        training_state['epoch_checkpoint'] = 0
                        training_state['best_recon_total_loss'] = None
                        print("BP Phase: Loaded existing checkpoint from wave phase")
                    except Exception as e:
                        raise Exception("Cannot load pretrained wave phase: No valid checkpoint found") from e
                else:
                    print("Starting new training from scratch")
        except Exception as e:
            if self.is_finetuning:
                raise  # Re-raise exception for finetuning
            print(f"Error during checkpoint loading: {e}")
            print("Starting training from scratch")
        
        # Synchronize processes
        dist.barrier()
        
        # Broadcast training state from rank 0 to all processes
        state_tensor = torch.tensor(
            [training_state['epoch_checkpoint'],
             training_state['best_recon_total_loss'] if training_state['best_recon_total_loss'] is not None else -1],
            device=device
        )
        dist.broadcast(state_tensor, src=0)
        
        if rank != 0:
            training_state = {
                'epoch_checkpoint': int(state_tensor[0].item()),
                'best_recon_total_loss': state_tensor[1].item() if state_tensor[1].item() != -1 else None
            }
        
        return training_state

    def save_checkpoint(self, ddp_model, optim, scheduler, early_stopping, epoch: int, 
                       val_loss: float, is_best: bool = False, current_prefix: str = ''):
        """Save checkpoint with consistent format"""
        try:
            # Get checkpoint info based on whether this is best model or periodic save
            checkpoint_info = self.get_checkpoint_info(
                is_finetuning=self.is_finetuning,
                epoch=None if is_best else epoch,
                saving=True
            )
            
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(checkpoint_info['file'])
            isExist_dir(checkpoint_dir)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': ddp_model.module.state_dict(),
                f'{current_prefix}optim_state_dict': optim.state_dict(),
                f'{current_prefix}scheduler_state_dict': scheduler.state_dict(),
                f'{current_prefix}early_stopping_state': early_stopping.state_dict(),
                f'{current_prefix}epoch': int(epoch+1),
                f'{current_prefix}best_recon_total_loss': val_loss,
                'is_finetuning': self.is_finetuning
            }
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_info['file'])
            
            save_type = "best model" if is_best else f"epoch {epoch}"
            mode_type = "finetuning" if self.is_finetuning else "regular"
            print(f"Saved {save_type} {mode_type} checkpoint to {checkpoint_info['file']}")
            
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            return False