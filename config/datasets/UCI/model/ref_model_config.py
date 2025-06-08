# Standard library imports
import os
from dataclasses import dataclass

# Third-party imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Local imports
from config.datasets.dataset_configs import UCIBaseConfig
from model.RefinementModel import BPModel
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ddp_utils import print_memory_stats, create_ddp_dataloaders
from utils.utils_preprocessing import (
    print_model_parameters,
    isExist_dir,
    Min_Max_Norm_Torch
)
from utils.train_utils import EarlyStopping
from utils.test_utils import calculate_bp_metrics, print_abp_evaluation_results
from .app_model_config import UCIApproximationConfig

@dataclass
class UCIRefinementConfig(UCIBaseConfig):
    """UCI dataset configuration for Refinement model"""
    args: dict = None
    is_finetuning: bool = False
    use_patient_split: bool = False
    
    # DDP-specific settings for refinement model
    find_unused_parameters: bool = True
    empty_cache_frequency: int = 10

    bp_norm: bool = True  # UCI uses normalized BP by default
    calculate_bpm: bool = False

    # Model architecture settings
    image_embedding: int = 10240  # For PulseDB version
    text_embedding: int = 768
    projection_dim: int = 512
    dropout: float = 0.1

    # Training settings
    temperature: float = 4
        
    # BP Model Settings 
    wcl: bool = False
    pi: bool = False  # PI is False for UCI
    wcl_age_threshold: float = 0.0235
    d_model: int = 64

    def __post_init__(self):
        super().__post_init__()
        self.path_folder: str = 'RefModel/UCI'  # Changed to UCI path
        
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
            
            # Boolean flags
            if hasattr(self.args, 'is_finetuning'):
                self.is_finetuning = self.args.is_finetuning
            if hasattr(self.args, 'bp_norm'):
                self.bp_norm = self.args.bp_norm
            if hasattr(self.args, 'use_patient_split'):
                self.use_patient_split = self.args.use_patient_split
            if hasattr(self.args, 'resume_training'):
                self.resume_training = self.args.resume_training
            if hasattr(self.args, 'is_pretraining'):
                self.is_pretraining = self.args.is_pretraining
            
            # Additional settings
            if hasattr(self.args, 'seed'):
                self.seed = self.args.seed
            
            # Add WCL and PI handling
            if hasattr(self.args, 'wcl'):
                self.wcl = self.args.wcl
            if hasattr(self.args, 'pi'):
                self.pi = self.args.pi
            self.seed = self.args.seed
            
        self.d_model = 64 if self.pi else 104
        self.set_seed()

    def create_model(self):
        """Create refinement model instance"""
        return BPModel(
            temperature=self.temperature,
            image_embedding=self.image_embedding,
            text_embedding=self.text_embedding,
            wcl_age_threshold=self.wcl_age_threshold if self.wcl else None,
            wcl=self.wcl,
            w_length=self.input_size,
            normalized_bp=self.bp_norm,
            projection_dim=self.projection_dim,
            dropout=self.dropout,
            pi=self.pi,
            d_model=self.d_model
        )

    def get_checkpoint_info(self, is_finetuning=False, epoch: int=None, saving: bool=False):
        """Get checkpoint directory and file information"""
        sub_folder = f'RefModel_BPModel_BS_{self.batch_size}_E_{self.num_epochs}_LR_{self.learning_rate}_WCL_{self.wcl}_PI_{self.pi}_P_{self.scheduler_patience}_ES_{self.early_stopping_patience}'
        # Add finetuning suffix if in finetuning mode
        if is_finetuning:
            sub_folder += '_finetuning'

        if self.bp_norm:
            sub_folder += '_BP_NORM'
        else:
            sub_folder += '_BP_RAW'

        if self.use_patient_split:
            sub_folder += '_Patient_Split'

        checkpoint_dir = os.path.join(self.abs_path, self.path_folder, sub_folder)
        # Saving checkpoint use current epoch
        if saving:
            if epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{self.checkpoint_name}_epoch_{epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{self.checkpoint_name}.pt')
        else:
            # Loading checkpoint use checkpoint_epoch
            if self.checkpoint_epoch is not None:
                checkpoint_file = os.path.join(checkpoint_dir, f'{self.checkpoint_name}_epoch_{self.checkpoint_epoch}.pt')
            else:
                checkpoint_file = os.path.join(checkpoint_dir, f'{self.checkpoint_name}.pt')
        
        return {
            'dir': checkpoint_dir,
            'file': checkpoint_file,
            'sub_folder': sub_folder,
            'is_finetuning': is_finetuning
        }

    def load_checkpoint_with_retry(self, rank: int, model, optim, scheduler, early_stopping, device: str):
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
                            load_optimizer=True, rank=rank
                        )
                        print("Resumed finetuning from existing finetuning checkpoint")
                    except Exception as e:
                        print(f"Failed to load finetuning checkpoint: {e}")
                        print("Attempting to load regular checkpoint for finetuning...")
                        try:
                            training_state = self._load_checkpoint(
                                model, optim, scheduler, early_stopping,
                                is_finetuning=False, device=device,
                                load_optimizer=False, rank=rank  # Don't load optimizer for new finetuning
                            )
                            # Reset epoch count for new finetuning
                            training_state['epoch_checkpoint'] = 0
                            training_state['best_recon_total_loss'] = None
                            print("Starting finetuning from regular checkpoint")
                        except Exception as e:
                            raise Exception("Cannot resume finetuning: No valid checkpoint found") from e
                else:
                    # Not resuming, must load regular checkpoint for finetuning
                    try:
                        training_state = self._load_checkpoint(
                            model, optim, scheduler, early_stopping,
                            is_finetuning=False, device=device,
                            load_optimizer=False, rank=rank  # Don't load optimizer for new finetuning
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
                            load_optimizer=True, rank=rank
                        )
                        print("Resumed training from regular checkpoint")
                    except Exception as e:
                        print(f"Failed to load regular checkpoint: {e}")
                        print("Starting training from scratch")
                else:
                    print("Starting new training from scratch")
        except Exception as e:
            if self.is_finetuning:
                raise  # Re-raise exception for finetuning
            print(f"Error during checkpoint loading: {e}")
            print("Starting training from scratch")
        
        # Synchronize processes
        dist.barrier()
        
        # Broadcast training state
        if rank == 0:
            state_tensor = torch.tensor(
                [training_state['epoch_checkpoint'],
                 training_state['best_recon_total_loss'] if training_state['best_recon_total_loss'] is not None else -1],
                device=device
            )
        else:
            state_tensor = torch.zeros(2, device=device)
        
        dist.broadcast(state_tensor, src=0)
        
        if rank != 0:
            training_state = {
                'epoch_checkpoint': int(state_tensor[0].item()),
                'best_recon_total_loss': state_tensor[1].item() if state_tensor[1].item() != -1 else None
            }
        
        return training_state

    def _load_checkpoint(self, model, optim, scheduler, early_stopping, 
                        is_finetuning: bool, device: str, load_optimizer: bool, rank: int):
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
            optim.load_state_dict(checkpoint['optim_state_dict'])
            print(f"Rank {rank} - Successfully loaded optimizer state from {checkpoint_info['file']}")
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Rank {rank} - Successfully loaded scheduler state from {checkpoint_info['file']}")
            if 'early_stopping_state' in checkpoint:
                early_stopping.load_state_dict(checkpoint['early_stopping_state'])
                print(f"Rank {rank} - Successfully loaded early stopping state from {checkpoint_info['file']}")
        
        return {
            'epoch_checkpoint': checkpoint['epoch'],
            'best_recon_total_loss': checkpoint.get('best_recon_total_loss', None)
        }

    def trainer(self, dataset: tuple, rank: int, world_size: int, ddp_config=None):
        """DDP trainer for the model"""
        train_loader = None
        val_loader = None
        
        try:
            # Setup device
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{rank}')
            master_process = rank == 0

            # Get datasets and create dataloaders
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
            
            # Create base model and move to device
            model = self.create_model().to(device)
            if master_process:  # Only print parameters on master process
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
            training_state = self.load_checkpoint_with_retry(
                rank=rank,
                model=model,
                optim=optim,
                scheduler=scheduler,
                early_stopping=early_stopping,
                device=device
            )
            
            # Now wrap model in DDP
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
                train_loss, bp_metrics = self._train_epoch(e, ddp_model, train_loader, optim, device, master_process)
                val_loss = self._validate_epoch(e, ddp_model, val_loader, device, master_process)
                test_loss = self._validate_epoch(e, ddp_model, test_loader, device, master_process)
                
                if master_process:
                    prev_lr = self._update_training_state(
                        e, val_loss, ddp_model, optim, scheduler,
                        early_stopping, best_loss, prev_lr
                    )
                    
                    # Add wandb logging
                    if wandb.run is not None:
                        try:
                            log_dict = {
                                "epoch": e,
                                "train/epoch_loss": float(train_loss),
                                "val/epoch_loss": float(val_loss['total_loss']),
                                "test/epoch_loss": float(test_loss['total_loss']),
                                "learning_rate": float(optim.param_groups[0]['lr']),
                                "best_loss": float(best_loss['best_recon_total_loss']) if best_loss['best_recon_total_loss'] is not None else None,
                                "improved": bool(best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss['total_loss']),
                                "early_stopping_counter": int(early_stopping.counter),
                                # Training BP metrics (already averaged across ECG and PPG)
                                "train/epoch_sbp_mse": float(bp_metrics['sbp_mse']),
                                "train/epoch_sbp_mae": float(bp_metrics['sbp_mae']),
                                "train/epoch_dbp_mse": float(bp_metrics['dbp_mse']),
                                "train/epoch_dbp_mae": float(bp_metrics['dbp_mae']),
                                # Validation BP metrics (already averaged)
                                "val/epoch_sbp_mse": float(val_loss['sbp_mse']),
                                "val/epoch_sbp_mae": float(val_loss['sbp_mae']),
                                "val/epoch_dbp_mse": float(val_loss['dbp_mse']),
                                "val/epoch_dbp_mae": float(val_loss['dbp_mae']),
                                # Test BP metrics (already averaged)
                                "test/epoch_sbp_mse": float(test_loss['sbp_mse']),
                                "test/epoch_sbp_mae": float(test_loss['sbp_mae']),
                                "test/epoch_dbp_mse": float(test_loss['dbp_mse']),
                                "test/epoch_dbp_mae": float(test_loss['dbp_mae'])
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
            print(f"Rank {rank} encountered error: {e}")
            raise e
        finally:
            # Only cleanup if loaders were created
            if train_loader is not None or val_loader is not None or test_loader is not None:
                self._cleanup(train_loader, val_loader, test_loader)
            
        print(f"Finished training on rank {rank}.")

    def _prepare_batch(self, batch_data, device):
        """Prepare batch data by moving it to device - UCI specific format"""
        return {
            "ecg": batch_data[0][:, self.ecg_label:self.ecg_label+1].permute(0, 2, 1).to(device, torch.float32),
            "ppg": batch_data[0][:, self.ppg_label:self.ppg_label+1].permute(0, 2, 1).to(device, torch.float32),
            "SBP": batch_data[2][:, 0:1].to(device, torch.float32),
            "DBP": batch_data[2][:, 1:2].to(device, torch.float32),
            "SBP_n": batch_data[5][:, 0:1].to(device, torch.float32),
            "DBP_n": batch_data[5][:, 1:2].to(device, torch.float32)
        }

    def _train_epoch(self, epoch, ddp_model, train_loader, optim, device, master_process):
        """Run one training epoch"""
        train_sampler = train_loader.sampler
        train_sampler.set_epoch(epoch)
        
        # Initialize epoch loss tracking
        epoch_total_loss = 0.0
        num_batches = 0
        
        # Initialize BP metric tracking
        bp_metrics = {
            'sbp_mse': 0.0, 'sbp_mae': 0.0,
            'dbp_mse': 0.0, 'dbp_mae': 0.0
        }
        
        with tqdm(train_loader, unit="batch", ncols=125, disable=not master_process) as tepoch:
            ddp_model.train()
            tepoch.set_description(f"Train - Epoch {epoch}")

            for step, train_file in enumerate(tepoch):
                try:
                    # Clear memory before processing batch
                    torch.cuda.empty_cache()
                    
                    # Prepare batch
                    batch = self._prepare_batch(train_file, device)

                    # Training step
                    optim.zero_grad(set_to_none=True)
                    ecg_loss, ppg_loss = ddp_model(batch)
                    
                    # Get BP predictions
                    with torch.no_grad():
                        y_ecg_sbp_dbp = ddp_model.module.get_SBP_DBP_fromECG(batch)
                        y_ppg_sbp_dbp = ddp_model.module.get_SBP_DBP_fromPPG(batch)
                    
                        # Calculate BP metrics (averaging ECG and PPG metrics)
                        ecg_sbp_mse = torch.nn.functional.mse_loss(y_ecg_sbp_dbp[0].detach(), batch['SBP_n']).item()
                        ppg_sbp_mse = torch.nn.functional.mse_loss(y_ppg_sbp_dbp[0].detach(), batch['SBP_n']).item()
                        bp_metrics['sbp_mse'] += (ecg_sbp_mse + ppg_sbp_mse) / 2

                        ecg_sbp_mae = torch.nn.functional.l1_loss(y_ecg_sbp_dbp[0].detach(), batch['SBP_n']).item()
                        ppg_sbp_mae = torch.nn.functional.l1_loss(y_ppg_sbp_dbp[0].detach(), batch['SBP_n']).item()
                        bp_metrics['sbp_mae'] += (ecg_sbp_mae + ppg_sbp_mae) / 2

                        ecg_dbp_mse = torch.nn.functional.mse_loss(y_ecg_sbp_dbp[1].detach(), batch['DBP_n']).item()
                        ppg_dbp_mse = torch.nn.functional.mse_loss(y_ppg_sbp_dbp[1].detach(), batch['DBP_n']).item()
                        bp_metrics['dbp_mse'] += (ecg_dbp_mse + ppg_dbp_mse) / 2

                        ecg_dbp_mae = torch.nn.functional.l1_loss(y_ecg_sbp_dbp[1].detach(), batch['DBP_n']).item()
                        ppg_dbp_mae = torch.nn.functional.l1_loss(y_ppg_sbp_dbp[1].detach(), batch['DBP_n']).item()
                        bp_metrics['dbp_mae'] += (ecg_dbp_mae + ppg_dbp_mae) / 2

                    loss = ecg_loss + ppg_loss
                    loss.backward()
                    optim.step()

                    # Update epoch average
                    epoch_total_loss += loss.item()
                    num_batches += 1

                    # Step-wise wandb logging (only master process)
                    if master_process and wandb.run is not None:
                        try:
                            wandb.log({
                                "train/step": step + epoch * len(train_loader),
                                "train/step_loss": loss.item(),
                                # Add simplified BP metrics (averaged across ECG and PPG)
                                "train/step_sbp_mse": bp_metrics['sbp_mse'] / (step + 1),
                                "train/step_sbp_mae": bp_metrics['sbp_mae'] / (step + 1),
                                "train/step_dbp_mse": bp_metrics['dbp_mse'] / (step + 1),
                                "train/step_dbp_mae": bp_metrics['dbp_mae'] / (step + 1),
                            })
                        except Exception as e:
                            print(f"Failed to log step metrics to wandb: {e}")

                    # Update progress bar
                    tepoch.set_postfix(
                        loss=loss.item()
                    )

                    # Clear memory
                    del batch, loss, ecg_loss, ppg_loss, y_ecg_sbp_dbp, y_ppg_sbp_dbp
                    torch.cuda.empty_cache()

                    # Clear cache periodically
                    if step % 25 == 0:
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Get rank from device
                        current_rank = dist.get_rank()
                        print(f"Rank {current_rank} - WARNING: out of memory in batch {step}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            # Calculate final epoch averages
            epoch_avg_loss = epoch_total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Average BP metrics over all batches
            for key in bp_metrics:
                bp_metrics[key] /= num_batches

            return epoch_avg_loss, bp_metrics  # Return both loss and BP metrics

    def _validate_epoch(self, epoch, ddp_model, val_loader, device, master_process):
        """Run one validation epoch"""
        val_sampler = val_loader.sampler
        val_sampler.set_epoch(epoch)
        
        with torch.no_grad():
            with tqdm(val_loader, unit="batch", ncols=125, disable=not master_process) as tepoch_val:
                ddp_model.eval()
                val_losses = {
                    'total_loss': [],
                    'sbp_mse': [], 'sbp_mae': [],
                    'dbp_mse': [], 'dbp_mae': []
                }
                tepoch_val.set_description(f"Val - Epoch {epoch}")

                for val_file in tepoch_val:
                    batch = self._prepare_batch(val_file, device)
                    
                    # Get main model losses
                    ecg_loss, ppg_loss = ddp_model(batch)
                    total_loss = ecg_loss + ppg_loss
                    
                    # Get BP predictions
                    y_sbp_dbp_ecg = ddp_model.module.get_SBP_DBP_fromECG(batch)
                    y_sbp_dbp_ppg = ddp_model.module.get_SBP_DBP_fromPPG(batch)
                    
                    # Calculate averaged BP prediction losses
                    val_losses['total_loss'].append(total_loss.item())
                    
                    # Average SBP metrics from ECG and PPG
                    ecg_sbp_mse = torch.nn.functional.mse_loss(y_sbp_dbp_ecg[0], batch['SBP_n']).item()
                    ppg_sbp_mse = torch.nn.functional.mse_loss(y_sbp_dbp_ppg[0], batch['SBP_n']).item()
                    val_losses['sbp_mse'].append((ecg_sbp_mse + ppg_sbp_mse) / 2)
                    
                    ecg_sbp_mae = torch.nn.functional.l1_loss(y_sbp_dbp_ecg[0], batch['SBP_n']).item()
                    ppg_sbp_mae = torch.nn.functional.l1_loss(y_sbp_dbp_ppg[0], batch['SBP_n']).item()
                    val_losses['sbp_mae'].append((ecg_sbp_mae + ppg_sbp_mae) / 2)
                    
                    # Average DBP metrics from ECG and PPG
                    ecg_dbp_mse = torch.nn.functional.mse_loss(y_sbp_dbp_ecg[1], batch['DBP_n']).item()
                    ppg_dbp_mse = torch.nn.functional.mse_loss(y_sbp_dbp_ppg[1], batch['DBP_n']).item()
                    val_losses['dbp_mse'].append((ecg_dbp_mse + ppg_dbp_mse) / 2)
                    
                    ecg_dbp_mae = torch.nn.functional.l1_loss(y_sbp_dbp_ecg[1], batch['DBP_n']).item()
                    ppg_dbp_mae = torch.nn.functional.l1_loss(y_sbp_dbp_ppg[1], batch['DBP_n']).item()
                    val_losses['dbp_mae'].append((ecg_dbp_mae + ppg_dbp_mae) / 2)

                    # Update progress bar
                    tepoch_val.set_postfix(
                        loss=total_loss.item()
                    )

                # Gather validation losses from all processes
                gathered_losses = {}
                for loss_name, loss_values in val_losses.items():
                    loss_tensor = torch.tensor(loss_values, device=device).mean()
                    if master_process:
                        gathered_list = [torch.zeros_like(loss_tensor, device=device) 
                                       for _ in range(dist.get_world_size())]
                        dist.gather(tensor=loss_tensor, gather_list=gathered_list, dst=0)
                        gathered_losses[loss_name] = torch.stack(gathered_list).mean().item()
                    else:
                        dist.gather(tensor=loss_tensor, gather_list=None, dst=0)
                        gathered_losses[loss_name] = None

                return gathered_losses

    def save_checkpoint(self, ddp_model, optim, scheduler, early_stopping, epoch: int, 
                       val_loss: float, is_best: bool = False):
        """Save checkpoint with consistent format
        
        Args:
            ddp_model: DDP wrapped model
            optim: Optimizer
            scheduler: Learning rate scheduler
            early_stopping: Early stopping handler
            epoch: Current epoch
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
        """
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
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state': early_stopping.state_dict(),
                'epoch': int(epoch+1),
                'best_recon_total_loss': val_loss,
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

    def _update_training_state(self, epoch, val_loss, ddp_model, optim, scheduler, 
                              early_stopping, best_loss, prev_lr):
        """Update training state including scheduler, early stopping, and checkpoints"""
        # Step scheduler with total_loss value and display learning rate
        scheduler.step(val_loss['total_loss'])  # Use the total_loss value from the dictionary
        current_lr = optim.param_groups[0]['lr']
        
        if current_lr != prev_lr:
            print(f"Learning rate changed: {prev_lr:.2e} -> {current_lr:.2e}")
        
        # Update early stopping with total_loss
        early_stopping(val_loss['total_loss'])  # Use the total_loss value here too
        
        # Save best model if improved (compare total_loss)
        if best_loss['best_recon_total_loss'] is None or best_loss['best_recon_total_loss'] > val_loss['total_loss']:
            if best_loss['best_recon_total_loss'] is not None:
                print(f"Validation loss improved from {best_loss['best_recon_total_loss']:.6f} to {val_loss['total_loss']:.6f}")
            
            if self.save_checkpoint(
                ddp_model, optim, scheduler, early_stopping,
                epoch, val_loss['total_loss'], is_best=True  # Pass total_loss here
            ):
                best_loss['best_recon_total_loss'] = val_loss['total_loss']
        
        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            self.save_checkpoint(
                ddp_model, optim, scheduler, early_stopping,
                epoch, val_loss['total_loss'], is_best=False  # Pass total_loss here too
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
        # Clean up dataloader workers
        for loader in [train_loader, val_loader, test_loader]:
            if loader is not None and hasattr(loader, '_iterator'):
                loader._iterator = None

    def test(self, num_workers: int):
        """Test the model performance on test set
        
        Args:
            num_workers (int): Number of dataloader workers
        """
        app_model_config = UCIApproximationConfig(args=self.args)

        # Model configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dataset with patient-level split for testing
        test_dataset = self.create_test_dataset()
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )

        # Create and setup models
        g_model_app = app_model_config.create_model().to(device)
        model = self.create_model().to(device)

        # Load checkpoints
        checkpoint_info = self.get_checkpoint_info(is_finetuning=self.is_finetuning)
        app_checkpoint_path = app_model_config.get_checkpoint_info()

        # Load approximation model checkpoint
        app_checkpoint = torch.load(app_checkpoint_path['file'], map_location=device)
        g_model_app.load_state_dict({k.replace('module.', ''): v for k, v in app_checkpoint['model_state_dict'].items()})
        print('Loaded approximation model checkpoint from {}'.format(app_checkpoint_path['file']))

        # Load refinement model checkpoint
        ref_checkpoint = torch.load(checkpoint_info['file'], map_location=device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in ref_checkpoint['model_state_dict'].items()})
        print('Loaded refinement model checkpoint from {}'.format(checkpoint_info['file']))

        # Set models to eval mode
        g_model_app.eval()
        model.eval()

        # Print model parameters
        print_model_parameters(g_model_app)
        print_model_parameters(model)

        # Initialize metrics dictionary
        best_loss = {
            'total_Sample': len(test_dataset),
            # PPG metrics
            'PPG2ABP_loss': [], 'PPG2ABP_ME_loss': [],
            'PPG2ABP_SBP_loss': [], 'PPG2ABP_DBP_loss': [], 'PPG2ABP_MAP_loss': [],
            'PPG2ABP_SBP_ME_loss': [], 'PPG2ABP_DBP_ME_loss': [], 'PPG2ABP_MAP_ME_loss': [],
            'PPG2ABP_SBP_BHS_5': 0, 'PPG2ABP_SBP_BHS_10': 0, 'PPG2ABP_SBP_BHS_15': 0,
            'PPG2ABP_DBP_BHS_5': 0, 'PPG2ABP_DBP_BHS_10': 0, 'PPG2ABP_DBP_BHS_15': 0,
            'PPG2ABP_MAP_BHS_5': 0, 'PPG2ABP_MAP_BHS_10': 0, 'PPG2ABP_MAP_BHS_15': 0,
            # ECG metrics
            'ECG2ABP_loss': [], 'ECG2ABP_ME_loss': [],
            'ECG2ABP_SBP_loss': [], 'ECG2ABP_DBP_loss': [], 'ECG2ABP_MAP_loss': [],
            'ECG2ABP_SBP_ME_loss': [], 'ECG2ABP_DBP_ME_loss': [], 'ECG2ABP_MAP_ME_loss': [],
            'ECG2ABP_SBP_BHS_5': 0, 'ECG2ABP_SBP_BHS_10': 0, 'ECG2ABP_SBP_BHS_15': 0,
            'ECG2ABP_DBP_BHS_5': 0, 'ECG2ABP_DBP_BHS_10': 0, 'ECG2ABP_DBP_BHS_15': 0,
            'ECG2ABP_MAP_BHS_5': 0, 'ECG2ABP_MAP_BHS_10': 0, 'ECG2ABP_MAP_BHS_15': 0,
        }

        # Testing loop
        with torch.no_grad():
            with tqdm(test_loader, unit="batch", ncols=125) as tepoch_test:
                tepoch_test.set_description("Testing")

                for test_file in tepoch_test:
                    # Prepare indices
                    ii_label, ppg_label, abp_label = self.ecg_label, self.ppg_label, self.abp_label

                    # Create one-hot encodings
                    s_ii = nn.functional.one_hot(torch.tensor(ii_label), app_model_config.num_domains).expand(
                        (test_file[0].size(0), app_model_config.num_domains)).to(device, torch.float32)

                    s_ppg = nn.functional.one_hot(torch.tensor(ppg_label), app_model_config.num_domains).expand(
                        (test_file[0].size(0), app_model_config.num_domains)).to(device, torch.float32)

                    s_abp = nn.functional.one_hot(torch.tensor(abp_label), app_model_config.num_domains).expand(
                        (test_file[0].size(0), app_model_config.num_domains)).to(device, torch.float32)


                    # Prepare batch data
                    x_ppg_zc = test_file[0][:, ppg_label:ppg_label + 1].to(device, torch.float32)
                    x_ii_zc = test_file[0][:, ii_label:ii_label + 1].to(device, torch.float32)
                    
                    x_abp_u = test_file[10].to(device, torch.float32)

                    # Prepare batch using the existing method
                    batch = self._prepare_batch(test_file, device)

                    # Get BP predictions
                    y_ppg_SBP_DBP = model.get_SBP_DBP_fromPPG(batch)
                    y_ecg_SBP_DBP = model.get_SBP_DBP_fromECG(batch)

                    # Get normalized waveforms from approximation model
                    x_ppg_abp = g_model_app(x_ppg_zc, s_abp)
                    x_ppg_abp = Min_Max_Norm_Torch(x_ppg_abp[:, :, :])

                    x_ii_abp = g_model_app(x_ii_zc, s_abp)
                    x_ii_abp = Min_Max_Norm_Torch(x_ii_abp[:, :, :])
                    # Calculate metrics for PPG
                    ppg_metrics = calculate_bp_metrics(
                        predictions=y_ppg_SBP_DBP,
                        waveform=x_ppg_abp,
                        sbp_true=batch['SBP'],
                        dbp_true=batch['DBP'],
                        abp_gt=x_abp_u,
                        prefix='PPG2ABP',
                        best_loss=best_loss,
                        global_min=self.dbp_min,
                        global_max=self.sbp_max,
                        normalize=self.bp_norm
                    )

                    # Calculate metrics for ECG
                    ecg_metrics = calculate_bp_metrics(
                        predictions=y_ecg_SBP_DBP,
                        waveform=x_ii_abp,
                        sbp_true=batch['SBP'],
                        dbp_true=batch['DBP'],
                        abp_gt=x_abp_u,
                        prefix='ECG2ABP',
                        best_loss=best_loss,
                        global_min=self.dbp_min,
                        global_max=self.sbp_max,
                        normalize=self.bp_norm
                    )

                    # Update progress bar
                    tepoch_test.set_postfix(
                        PPG_MAE=f"{ppg_metrics['waveform_mae']:.2f}",
                        ECG_MAE=f"{ecg_metrics['waveform_mae']:.2f}",
                        Batch=f"{tepoch_test.n}/{len(test_loader)}"
                    )

                    # Clear cache periodically
                    if tepoch_test.n % 25 == 0:
                        torch.cuda.empty_cache()

        # Print final evaluation results
        print_abp_evaluation_results(best_loss, args=self.args)
