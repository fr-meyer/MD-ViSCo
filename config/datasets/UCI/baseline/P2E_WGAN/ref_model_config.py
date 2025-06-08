# Standard library imports
import os
from dataclasses import dataclass
import time

# Third-party imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Local imports
from config.datasets.dataset_configs import UCIBaseConfig
from baseline.P2E_WGAN import (
    GeneratorUNet,
    Discriminator,
    weights_init_normal,
    compute_gradient_penalty
)
from utils.ddp_utils import print_memory_stats, create_ddp_dataloaders
from utils.train_utils import EarlyStopping
from utils.utils_preprocessing import (
    print_model_parameters,
    isExist_dir
)
from utils.test_utils import calculate_bp_metrics, print_abp_evaluation_results, extract_bp_values

@dataclass
class UCIRefinementP2EWGANConfig(UCIBaseConfig):
    """UCI dataset configuration for P2E-WGAN refinement model.
    Supports refinement between any combination of signals (PPG, ECG, ABP)."""
    
    args: dict = None
    # Direction configuration
    direction: str = None  # If None, train all combinations, otherwise format: "SOURCE2TARGET" (e.g. "PPG2ABP")
    
    # Base configuration
    path_folder: str = 'RefModel/UCI'  # Change to UCI path
    is_finetuning: bool = False
    use_patient_split: bool = False
    
    # Model Architecture configuration
    in_channels: int = 1
    out_channels: int = 1
    
    # Network capacity configuration
    generator_init_filters: int = 640  # Controls generator network capacity
    discriminator_init_filters: int = 320  # Controls discriminator network capacity
    
    # WGAN specific parameters
    lambda_gp: float = 10.0  # Gradient penalty lambda
    n_critic: int = 5  # Number of critic iterations per generator iteration
    
    def __post_init__(self):
        # Call parent class's __post_init__
        super().__post_init__()
        if self.args is not None:
            # Basic training parameters
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
            self.model_type = self.args.model_type

            # Direction configuration
            if hasattr(self.args, 'direction'):
                self.direction = self.args.direction
            
            # Boolean flags
            if hasattr(self.args, 'is_finetuning'):
                self.is_finetuning = self.args.is_finetuning
            if hasattr(self.args, 'use_patient_split'):
                self.use_patient_split = self.args.use_patient_split
            if hasattr(self.args, 'is_pretraining'):
                self.is_pretraining = self.args.is_pretraining
            if hasattr(self.args, 'seed'):
                self.seed = self.args.seed
        

        # Parse direction if specified
        if self.direction is not None:
            source, target = self._parse_direction(self.direction)
            self.source_channel = getattr(self, f"{source.lower()}_label")
            self.target_channel = getattr(self, f"{target.lower()}_label")
        self.set_seed()

    def create_model(self):
        """Create Generator and Discriminator model instances with configured capacities"""
        generator = GeneratorUNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            init_filters=self.generator_init_filters
        )
        discriminator = Discriminator(
            in_channels=self.in_channels,
            init_filters=self.discriminator_init_filters
        )
        
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        
        return generator, discriminator

    def _prepare_batch(self, batch_data, device, source_channel, target_channel):
        """Prepare batch data for training"""
        # Update indexing for UCI dataset structure
        x = batch_data[0][:, source_channel:source_channel + 1].to(device, torch.float32)
        y_target = batch_data[1][:, :].to(device, torch.float32)

        # If target is ABP, also extract SBP and DBP values for BP metrics calculation
        if self.channel_names[target_channel] == 'ABP':
            sbp = batch_data[5][:, 0:1].to(device, torch.float32)  # Update index for UCI dataset
            dbp = batch_data[5][:, 1:2].to(device, torch.float32)  # Update index for UCI dataset
            return {
                "x": x,
                "y_target": y_target,
                "sbp": sbp,
                "dbp": dbp
            }
        else:
            return {
                "x": x,
                "y_target": y_target
            }

    def _parse_direction(self, direction):
        """Parse direction string into source and target channels.
        For refinement, only SOURCE2ABP formats are supported."""
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
        """Get all valid source-target pairs for training.
        For refinement, only SOURCE2ABP pairs are supported."""
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

    def get_checkpoint_path(self, source, target, epoch:int=None, saving:bool=False):
        """Get checkpoint path for a specific source-target pair"""
        model_name = self.get_model_name(source, target)
        sub_folder = f'RefModel_UnetWGAN_BS_{self.batch_size}_E_{self.num_epochs}_LR_{self.learning_rate}_P_{self.scheduler_patience}_ES_{self.early_stopping_patience}'
        
        # Add finetuning suffix if in finetuning mode
        if hasattr(self, 'is_finetuning') and self.is_finetuning:
            sub_folder += '_finetuning'
        
        # Add patient split suffix if using patient split
        if hasattr(self, 'use_patient_split') and self.use_patient_split:
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
            'is_finetuning': self.is_finetuning if hasattr(self, 'is_finetuning') else False
        }

    def _train_epoch(self, epoch, ddp_generator, ddp_discriminator, train_loader, 
                     gen_optim, disc_optim, device, master_process, 
                     source_channel, target_channel):
        # Extract rank from device
        rank = device.index

        """Run one training epoch"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        # Initialize epoch metrics
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_g_mse_loss = 0.0
        epoch_g_mae_loss = 0.0
        epoch_sbp_mae = 0.0
        epoch_dbp_mae = 0.0
        epoch_sbp_mse = 0.0
        epoch_dbp_mse = 0.0
        num_batches = 0
        g_updates = 0
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            tepoch.set_description(f"Train - Epoch {epoch}")
            
            for step, train_file in enumerate(tepoch):
                # Prepare batch
                batch = self._prepare_batch(train_file, device, source_channel, target_channel)
                real_A = batch["x"]
                real_B = batch["y_target"]
                
                if step % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    gen_optim.zero_grad(set_to_none=True)
                    
                    # Generate fake signals
                    fake_B = ddp_generator(real_A)
                    
                    # Loss measures generator's ability to fool the discriminator
                    fake_validity = ddp_discriminator(fake_B, real_A)
                    adv_loss = -torch.mean(fake_validity)
                    
                    # Add MSE reconstruction loss
                    mse_loss = torch.nn.functional.mse_loss(fake_B, real_B)
                    # Also calculate MAE loss for logging (not used in optimization)
                    with torch.no_grad():
                        mae_loss = torch.nn.functional.l1_loss(fake_B.detach(), real_B)
                    lambda_sample = 50  # Same value as in original script
                    
                    # Combine losses
                    g_loss = adv_loss + lambda_sample * mse_loss
                    
                    g_loss.backward()
                    gen_optim.step()
                    
                    # Track losses for epoch averages
                    epoch_g_loss += g_loss.item()
                    epoch_g_mse_loss += mse_loss.item()
                    epoch_g_mae_loss += mae_loss.item()
                    g_updates += 1
                    
                    # Calculate BP metrics
                    with torch.no_grad():
                        # Extract SBP and DBP from predicted and ground truth waveforms
                        pred_sbp, pred_dbp = extract_bp_values(fake_B.detach())
                        
                        # Calculate BP metrics
                        sbp_mae = torch.nn.functional.l1_loss(pred_sbp, batch["sbp"])
                        dbp_mae = torch.nn.functional.l1_loss(pred_dbp, batch["dbp"])
                        sbp_mse = torch.nn.functional.mse_loss(pred_sbp, batch["sbp"])
                        dbp_mse = torch.nn.functional.mse_loss(pred_dbp, batch["dbp"])
                    
                    # Track BP metrics
                    epoch_sbp_mae += sbp_mae.item()
                    epoch_dbp_mae += dbp_mae.item()
                    epoch_sbp_mse += sbp_mse.item()
                    epoch_dbp_mse += dbp_mse.item()
                else:
                    # Generate fake signals without gradient tracking
                    with torch.no_grad():
                        fake_B = ddp_generator(real_A)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                disc_optim.zero_grad(set_to_none=True)
                
                # Real signals
                real_validity = ddp_discriminator(real_B, real_A)
                # Fake signals
                fake_validity = ddp_discriminator(fake_B.detach(), real_A)
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    ddp_discriminator.module, real_B, fake_B.detach(), real_A,
                    real_validity.shape[1:3], device
                )
                
                # Discriminator loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
                d_loss.backward()
                disc_optim.step()
                
                epoch_d_loss += d_loss.item()
                num_batches += 1
                
                # Log training metrics
                if master_process and wandb.run is not None:
                    try:
                        log_dict = {
                            "train/step": step + epoch * len(train_loader),
                            "train/d_loss": d_loss.item(),
                            "train/g_loss": g_loss.item() if step % self.n_critic == 0 else None,
                            "train/g_adv_loss": adv_loss.item() if step % self.n_critic == 0 else None,
                            "train/step_loss": mse_loss.item() if step % self.n_critic == 0 else None,
                            "train/gp": gradient_penalty.item()
                        }
                        
                        # Add BP metrics if target is ABP and we just updated the generator
                        if step % self.n_critic == 0:
                            log_dict.update({
                                "train/step_sbp_mae": sbp_mae.item(),
                                "train/step_dbp_mae": dbp_mae.item(),
                                "train/step_sbp_mse": sbp_mse.item(),
                                "train/step_dbp_mse": dbp_mse.item()
                            })
                        
                        wandb.log(log_dict)
                    except Exception as e:
                        print(f"Failed to log to wandb: {e}")
                
                # Update progress bar
                postfix_dict = {
                    "d_loss": f"{d_loss.item():.4f}",
                }
                
                if step % self.n_critic == 0:
                    postfix_dict.update({
                        "g_loss": f"{g_loss.item():.4f}",
                        "g_mse": f"{mse_loss.item():.4f}",
                        "g_mae": f"{mae_loss.item():.4f}"
                    })
                    
                    postfix_dict.update({
                        "sbp_mae": f"{sbp_mae.item():.2f}",
                        "dbp_mae": f"{dbp_mae.item():.2f}"
                    })
                else:
                    postfix_dict.update({
                        "g_loss": "N/A",
                        "g_mse": "N/A",
                        "g_mae": "N/A"
                    })
                    
                    postfix_dict.update({
                        "sbp_mae": "N/A",
                        "dbp_mae": "N/A"
                    })
                
                tepoch.set_postfix(**postfix_dict)
                
                # Clear memory
                del batch, real_A, real_B, fake_B, d_loss, gradient_penalty
                if step % self.n_critic == 0:
                    del g_loss, mse_loss, mae_loss, adv_loss
                    del pred_sbp, pred_dbp, sbp_mae, dbp_mae, sbp_mse, dbp_mse
                torch.cuda.empty_cache()
                dist.barrier(device_ids=[rank])
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / g_updates if g_updates > 0 else 0
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_mse = epoch_g_mse_loss / g_updates if g_updates > 0 else 0
        avg_g_mae = epoch_g_mae_loss / g_updates if g_updates > 0 else 0
        
        avg_sbp_mae = epoch_sbp_mae / g_updates if g_updates > 0 else 0
        avg_dbp_mae = epoch_dbp_mae / g_updates if g_updates > 0 else 0
        avg_sbp_mse = epoch_sbp_mse / g_updates if g_updates > 0 else 0
        avg_dbp_mse = epoch_dbp_mse / g_updates if g_updates > 0 else 0
        
        return {
            "g_loss": avg_g_loss,
            "d_loss": avg_d_loss,
            "g_mse": avg_g_mse,
            "g_mae": avg_g_mae,
            "sbp_mae": avg_sbp_mae,
            "dbp_mae": avg_dbp_mae,
            "sbp_mse": avg_sbp_mse,
            "dbp_mse": avg_dbp_mse
        }

    def _validate_epoch(self, epoch, ddp_generator, val_loader, device, master_process,
                       source_channel, target_channel):
        # Extract rank from device
        rank = device.index
        """Run one validation epoch"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch_val:
                ddp_generator.eval()
                total_mse_loss = 0
                total_mae_loss = 0
                total_sbp_mae = 0
                total_dbp_mae = 0
                total_sbp_mse = 0
                total_dbp_mse = 0
                num_batches = 0
                tepoch_val.set_description(f"Val - Epoch {epoch}")
                
                for val_file in tepoch_val:
                    # Prepare batch
                    batch = self._prepare_batch(val_file, device, source_channel, target_channel)
                    
                    # Generate fake signals
                    fake_B = ddp_generator(batch["x"])
                    
                    # Calculate MSE and MAE losses
                    mse_loss = torch.nn.functional.mse_loss(batch["y_target"], fake_B)
                    mae_loss = torch.nn.functional.l1_loss(batch["y_target"], fake_B)
                    
                    total_mse_loss += mse_loss.item()
                    total_mae_loss += mae_loss.item()
                    
                    # Calculate BP metrics if target is ABP
                    postfix_dict = {
                        "mse": f"{mse_loss.item():.4f}",
                        "mae": f"{mae_loss.item():.4f}"
                    }
                    
                    # Extract SBP and DBP from predicted waveform
                    pred_sbp, pred_dbp = extract_bp_values(fake_B)
                    
                    # Calculate BP metrics
                    sbp_mae = torch.nn.functional.l1_loss(pred_sbp, batch["sbp"])
                    dbp_mae = torch.nn.functional.l1_loss(pred_dbp, batch["dbp"])
                    sbp_mse = torch.nn.functional.mse_loss(pred_sbp, batch["sbp"])
                    dbp_mse = torch.nn.functional.mse_loss(pred_dbp, batch["dbp"])
                    
                    total_sbp_mae += sbp_mae.item()
                    total_dbp_mae += dbp_mae.item()
                    total_sbp_mse += sbp_mse.item()
                    total_dbp_mse += dbp_mse.item()
                    
                    postfix_dict.update({
                        "sbp_mae": f"{sbp_mae.item():.2f}",
                        "dbp_mae": f"{dbp_mae.item():.2f}"
                    })
                    
                    tepoch_val.set_postfix(**postfix_dict)
                    
                    num_batches += 1
                    
                    del batch, fake_B, mse_loss, mae_loss
                    del pred_sbp, pred_dbp, sbp_mae, dbp_mae, sbp_mse, dbp_mse
                    torch.cuda.empty_cache()
                    dist.barrier(device_ids=[rank])
                
                avg_mse = total_mse_loss / num_batches
                avg_mae = total_mae_loss / num_batches
                
                # Gather losses from all processes
                avg_sbp_mae = total_sbp_mae / num_batches
                avg_dbp_mae = total_dbp_mae / num_batches
                avg_sbp_mse = total_sbp_mse / num_batches
                avg_dbp_mse = total_dbp_mse / num_batches
                
                metrics_tensor = torch.tensor(
                    [avg_mse, avg_mae, avg_sbp_mae, avg_dbp_mae, avg_sbp_mse, avg_dbp_mse], 
                    device=device
                )
                gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_metrics, metrics_tensor)
                
                # Average metrics across all processes
                final_metrics = torch.stack(gathered_metrics).mean(dim=0)
                
                final_mse = final_metrics[0].item()
                final_mae = final_metrics[1].item()
                final_sbp_mae = final_metrics[2].item()
                final_dbp_mae = final_metrics[3].item()
                final_sbp_mse = final_metrics[4].item()
                final_dbp_mse = final_metrics[5].item()
                
                if master_process:
                    print(f"\nValidation Results:")
                    print(f"Waveform: MSE={final_mse:.4f}, MAE={final_mae:.4f}")
                    print(f"BP: SBP MAE={final_sbp_mae:.2f} mmHg, DBP MAE={final_dbp_mae:.2f} mmHg")
                    print(f"BP: SBP MSE={final_sbp_mse:.2f}, DBP MSE={final_dbp_mse:.2f}")
                
                # Return metrics dictionary
                return {
                    "mse": final_mse,
                    "mae": final_mae,
                    "sbp_mae": final_sbp_mae,
                    "dbp_mae": final_dbp_mae,
                    "sbp_mse": final_sbp_mse,
                    "dbp_mse": final_dbp_mse
                }

    def load_checkpoint_with_retry(self, rank, generator, discriminator, gen_optim, disc_optim, scheduler, early_stopping, 
                                   checkpoint_path, max_retries=3, wait_time=5):
        """Helper function to load checkpoint with retry logic and finetuning support.
        
        Args:
            rank (int): Process rank in distributed training
            generator (nn.Module): The unwrapped generator model (only loaded on rank 0)
            discriminator (nn.Module): The unwrapped discriminator model (only loaded on rank 0)
            gen_optim (Optimizer): Generator optimizer (loaded on all ranks)
            disc_optim (Optimizer): Discriminator optimizer (loaded on all ranks)
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
                                
                                # Load generator state dict
                                gen_state_dict = checkpoint['generator_state_dict']
                                new_gen_state_dict = {k.replace('module.', ''): v for k, v in gen_state_dict.items()}
                                generator.load_state_dict(new_gen_state_dict, strict=False)
                                
                                # Load discriminator state dict
                                disc_state_dict = checkpoint['discriminator_state_dict']
                                new_disc_state_dict = {k.replace('module.', ''): v for k, v in disc_state_dict.items()}
                                discriminator.load_state_dict(new_disc_state_dict, strict=False)
                                
                                # Load complete training state
                                training_state['epoch_checkpoint'] = checkpoint['epoch']
                                training_state['best_recon_total_loss'] = checkpoint.get('best_recon_total_loss', None)
                                
                                # Load early stopping state
                                if 'early_stopping_state' in checkpoint:
                                    early_stopping.load_state_dict(checkpoint['early_stopping_state'])
                                
                                # Load optimizer and scheduler states
                                gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])
                                disc_optim.load_state_dict(checkpoint['disc_optim_state_dict'])
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
                                    
                                    # Load generator state dict only
                                    gen_state_dict = checkpoint['generator_state_dict']
                                    new_gen_state_dict = {k.replace('module.', ''): v for k, v in gen_state_dict.items()}
                                    generator.load_state_dict(new_gen_state_dict, strict=False)
                                    
                                    # Load discriminator state dict only
                                    disc_state_dict = checkpoint['discriminator_state_dict']
                                    new_disc_state_dict = {k.replace('module.', ''): v for k, v in disc_state_dict.items()}
                                    discriminator.load_state_dict(new_disc_state_dict, strict=False)
                                    
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
                                
                                # Load generator state dict only
                                gen_state_dict = checkpoint['generator_state_dict']
                                new_gen_state_dict = {k.replace('module.', ''): v for k, v in gen_state_dict.items()}
                                generator.load_state_dict(new_gen_state_dict, strict=False)
                                
                                # Load discriminator state dict only
                                disc_state_dict = checkpoint['discriminator_state_dict']
                                new_disc_state_dict = {k.replace('module.', ''): v for k, v in disc_state_dict.items()}
                                discriminator.load_state_dict(new_disc_state_dict, strict=False)
                                
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

                                    # Load generator state dict
                                    gen_state_dict = checkpoint['generator_state_dict']
                                    new_gen_state_dict = {k.replace('module.', ''): v for k, v in gen_state_dict.items()}
                                    
                                    gen_missing_keys, gen_unexpected_keys = generator.load_state_dict(
                                        new_gen_state_dict,
                                        strict=False
                                    )

                                    # Print warnings about generator keys if any
                                    if gen_missing_keys:
                                        print(f"Warning: Missing keys in generator: {len(gen_missing_keys)} keys")
                                        print(f"First few missing keys: {gen_missing_keys[:5]}")
                                    if gen_unexpected_keys:
                                        print(f"Warning: Unexpected keys in generator: {len(gen_unexpected_keys)} keys")
                                        print(f"First few unexpected keys: {gen_unexpected_keys[:5]}")
                                    
                                    # Load discriminator state dict
                                    disc_state_dict = checkpoint['discriminator_state_dict']
                                    new_disc_state_dict = {k.replace('module.', ''): v for k, v in disc_state_dict.items()}
                                    
                                    disc_missing_keys, disc_unexpected_keys = discriminator.load_state_dict(
                                        new_disc_state_dict,
                                        strict=False
                                    )
                                    
                                    # Print warnings about discriminator keys if any
                                    if disc_missing_keys:
                                        print(f"Warning: Missing keys in discriminator: {len(disc_missing_keys)} keys")
                                        print(f"First few missing keys: {disc_missing_keys[:5]}")
                                    if disc_unexpected_keys:
                                        print(f"Warning: Unexpected keys in discriminator: {len(disc_unexpected_keys)} keys")
                                        print(f"First few unexpected keys: {disc_unexpected_keys[:5]}")

                                    # Load complete training state
                                    training_state['epoch_checkpoint'] = checkpoint['epoch']
                                    training_state['best_recon_total_loss'] = checkpoint.get('best_recon_total_loss', None)

                                    # Load early stopping state
                                    if 'early_stopping_state' in checkpoint:
                                        early_stopping.load_state_dict(checkpoint['early_stopping_state'])

                                    # Load optimizer and scheduler states
                                    gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])
                                    disc_optim.load_state_dict(checkpoint['disc_optim_state_dict'])
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
        best_loss_tensor = torch.tensor(
            [training_state['best_recon_total_loss'] if training_state['best_recon_total_loss'] is not None else -1],
            device=f'cuda:{rank}')
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
                gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])
                disc_optim.load_state_dict(checkpoint['disc_optim_state_dict'])
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

    def _update_training_state(self, epoch, val_loss, ddp_generator, ddp_discriminator,
                             gen_optim, disc_optim, scheduler, early_stopping, 
                             checkpoint_path, best_loss, prev_lr):
        """Update training state including scheduler, early stopping, and checkpoints"""
        # Step scheduler and display learning rate
        scheduler.step(val_loss)
        current_lr = gen_optim.param_groups[0]['lr']
        
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
                'generator_state_dict': ddp_generator.module.state_dict(),
                'discriminator_state_dict': ddp_discriminator.module.state_dict(),
                'gen_optim_state_dict': gen_optim.state_dict(),
                'disc_optim_state_dict': disc_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),  # Save as completed epoch (epoch+1)
                'best_recon_total_loss': val_loss,
                'is_finetuning': self.is_finetuning
            }, checkpoint_path)
            best_loss['best_recon_total_loss'] = val_loss
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if epoch % 5 == 0:
            # Get correct checkpoint path
            checkpoint_info_epoch = self.get_checkpoint_path(self.source_channel, self.target_channel, epoch, saving=True)
            checkpoint_path_epoch = checkpoint_info_epoch['file']
            torch.save({
                'generator_state_dict': ddp_generator.module.state_dict(),
                'discriminator_state_dict': ddp_discriminator.module.state_dict(),
                'gen_optim_state_dict': gen_optim.state_dict(),
                'disc_optim_state_dict': disc_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),  # Save as completed epoch (epoch+1)
                'best_recon_total_loss': val_loss,
                'is_finetuning': self.is_finetuning
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

            # Create models first (unwrapped) 
            generator, discriminator = self.create_model()
            generator = generator.to(device)
            discriminator = discriminator.to(device)
            
            if master_process:
                print_model_parameters(generator)
                print_model_parameters(discriminator)
            
            # Setup optimizers
            gen_optim = torch.optim.Adam(generator.parameters(), lr=self.learning_rate)
            disc_optim = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate)
            
            # Setup scheduler (only for generator)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                gen_optim, 'min', verbose=master_process,
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
            # Load checkpoint if exists - now loading on unwrapped models
            training_state = self.load_checkpoint_with_retry(
                rank=rank,
                generator=generator,  # Pass unwrapped generator model
                discriminator=discriminator,  # Pass unwrapped discriminator model
                gen_optim=gen_optim,
                disc_optim=disc_optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_path=checkpoint_info['file']
            )

            # Now wrap models with DDP AFTER loading checkpoint
            ddp_generator = DDP(
                generator,
                device_ids=[rank],
                find_unused_parameters=ddp_config.find_unused_parameters if ddp_config else False,
                static_graph=ddp_config.static_graph if ddp_config else True
            )
            
            ddp_discriminator = DDP(
                discriminator,
                device_ids=[rank],
                find_unused_parameters=ddp_config.find_unused_parameters if ddp_config else False,
                static_graph=ddp_config.static_graph if ddp_config else True
            )

            epoch_checkpoint = training_state['epoch_checkpoint']
            best_loss = {'best_recon_total_loss': training_state['best_recon_total_loss']}
            prev_lr = gen_optim.param_groups[0]['lr']

            # Training loop
            print_memory_stats(rank, "Before training loop")
            for e in range(epoch_checkpoint, self.num_epochs):
                # Training and validation with metrics
                train_metrics = self._train_epoch(
                    e, ddp_generator, ddp_discriminator, train_loader,
                    gen_optim, disc_optim, device, master_process,
                    source_channel=source_channel, target_channel=target_channel
                )
                
                val_metrics = self._validate_epoch(
                    e, ddp_generator, val_loader, device, master_process,
                    source_channel=source_channel, target_channel=target_channel
                )
                
                test_metrics = self._validate_epoch(
                    e, ddp_generator, test_loader, device, master_process,
                    source_channel=source_channel, target_channel=target_channel
                )

                if master_process:
                    # Get checkpoint info for this direction
                    checkpoint_info = self.get_checkpoint_path(source_channel, target_channel, saving=True)
                    
                    # Use MAE loss for early stopping and checkpointing
                    val_loss_for_checkpoint = val_metrics["mae"]
                    
                    prev_lr = self._update_training_state(
                        e, val_loss_for_checkpoint, ddp_generator, ddp_discriminator,
                        gen_optim, disc_optim, scheduler,
                        early_stopping, checkpoint_info['file'],
                        best_loss, prev_lr
                    )

                    # Log epoch metrics to wandb
                    if wandb.run is not None:
                        try:
                            log_dict = {
                                "epoch": e,
                                "train/epoch_g_loss": float(train_metrics["g_loss"]),
                                "train/epoch_d_loss": float(train_metrics["d_loss"]),
                                "train/epoch_loss": float(train_metrics["g_mse"]),
                                "val/epoch_loss": float(val_metrics["mse"]),
                                "test/epoch_loss": float(test_metrics["mse"]),
                                "learning_rate": float(gen_optim.param_groups[0]['lr']),
                                "best_loss": float(best_loss['best_recon_total_loss']) if best_loss['best_recon_total_loss'] is not None else None,
                                "improved": bool(best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss_for_checkpoint),
                                "early_stopping_counter": int(early_stopping.counter)
                            }
                            
                            # Add BP metrics if available (target is ABP)
                            if "sbp_mae" in train_metrics:
                                log_dict.update({
                                    "train/epoch_sbp_mae": float(train_metrics["sbp_mae"]),
                                    "train/epoch_dbp_mae": float(train_metrics["dbp_mae"]),
                                    "train/epoch_sbp_mse": float(train_metrics["sbp_mse"]),
                                    "train/epoch_dbp_mse": float(train_metrics["dbp_mse"]),
                                    "val/epoch_sbp_mae": float(val_metrics["sbp_mae"]),
                                    "val/epoch_dbp_mae": float(val_metrics["dbp_mae"]),
                                    "val/epoch_sbp_mse": float(val_metrics["sbp_mse"]),
                                    "val/epoch_dbp_mse": float(val_metrics["dbp_mse"]),
                                    "test/epoch_sbp_mae": float(test_metrics["sbp_mae"]),
                                    "test/epoch_dbp_mae": float(test_metrics["dbp_mae"]),
                                    "test/epoch_sbp_mse": float(test_metrics["sbp_mse"]),
                                    "test/epoch_dbp_mse": float(test_metrics["dbp_mse"])
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

        # Initialize BP metrics dictionary for ABP targets
        bp_metrics = {
            'total_Sample': len(test_dataset),
        }

        # Test each direction
        for source_channel, target_channel in source_target_pairs:
            # Get model name for this direction
            direction = f"{self.channel_names[source_channel]}2{self.channel_names[target_channel]}"
            
            # Add BP metrics if target is ABP
            is_abp_target = self.channel_names[target_channel] == 'ABP'
            if is_abp_target:
                bp_metrics.update({
                    f'{direction}_loss': [], f'{direction}_ME_loss': [],
                    f'{direction}_SBP_loss': [], f'{direction}_DBP_loss': [], f'{direction}_MAP_loss': [],
                    f'{direction}_SBP_ME_loss': [], f'{direction}_DBP_ME_loss': [], f'{direction}_MAP_ME_loss': [],
                    f'{direction}_SBP_BHS_5': 0, f'{direction}_SBP_BHS_10': 0, f'{direction}_SBP_BHS_15': 0,
                    f'{direction}_DBP_BHS_5': 0, f'{direction}_DBP_BHS_10': 0, f'{direction}_DBP_BHS_15': 0,
                    f'{direction}_MAP_BHS_5': 0, f'{direction}_MAP_BHS_10': 0, f'{direction}_MAP_BHS_15': 0,
                })

            print(f"\nTesting {direction} model...")

            # Get checkpoint path and load model
            checkpoint_info = self.get_checkpoint_path(source_channel, target_channel)
            generator, _ = self.create_model()  # Only need generator for testing
            
            try:
                print(f"Loaded refinement model from {checkpoint_info['file']} for {self.direction} cascade")
                # Load checkpoint
                checkpoint = torch.load(checkpoint_info['file'])
                # Handle module prefix in state dict keys
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint['generator_state_dict'].items()}
                generator.load_state_dict(state_dict)
                generator = generator.to(device)
                generator.eval()

                # Print model parameters
                print(f"\nModel Parameters for {direction}:")
                print_model_parameters(generator)

                with torch.no_grad():
                    with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                        tepoch_test.set_description(f"Testing {direction}")
                        
                        for batch_idx, test_file in enumerate(tepoch_test):
                            # Prepare batch data
                            batch = self._prepare_batch(test_file, device, source_channel, target_channel)
                            
                            # Generate fake signals
                            fake_B = generator(batch["x"])
                            
                            # If target is ABP, calculate BP metrics
                            if is_abp_target:
                                # Get ground truth values
                                sbp_true = test_file[2][:, 0:1].to(device, torch.float32)
                                dbp_true = test_file[2][:, 1:2].to(device, torch.float32)
                                # sbp_true = batch["sbp"]
                                # dbp_true = batch["dbp"]
                                
                                # Unnormalized ABP ground truth
                                x_abp_u = test_file[10].to(device, torch.float32)
                                
                                # Calculate BP metrics
                                batch_bp_metrics = calculate_bp_metrics(
                                    predictions=None,
                                    waveform=fake_B,
                                    sbp_true=sbp_true,
                                    dbp_true=dbp_true,
                                    abp_gt=x_abp_u,
                                    prefix=direction,
                                    best_loss=bp_metrics,
                                    global_min=self.dbp_min,
                                    global_max=self.sbp_max,
                                    normalize=True
                                )
                                
                                # Update progress bar with BP metrics
                                tepoch_test.set_postfix(
                                    sbp_mae=f"{batch_bp_metrics['sbp_mae']:.2f}",
                                    dbp_mae=f"{batch_bp_metrics['dbp_mae']:.2f}"
                                )

                            # Clear memory
                            del batch, fake_B
                            if is_abp_target:
                                del batch_bp_metrics, x_abp_u, sbp_true, dbp_true
                            torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error testing {direction} model: {str(e)}")
                continue

            finally:
                # Clean up model for this direction
                if 'generator' in locals():
                    del generator
                    torch.cuda.empty_cache()

        # Print BP evaluation results if any ABP targets were tested
        if any(self.channel_names[target] == 'ABP' for _, target in source_target_pairs):
            print("\nBlood Pressure Evaluation Results:")
            print_abp_evaluation_results(bp_metrics, self.direction, args=self.args)
