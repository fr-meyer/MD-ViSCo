# Standard library imports
import os
import socket
import argparse
import logging
from dataclasses import dataclass

# Third-party imports
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

# Local imports
from config.config_factory import ConfigFactory
from config.base_config import BaseModelConfig
from utils.ddp_utils import cleanup, setup_ddp
from dataload.DataLoadPulseDB import PulseDBDataset
from dataload.DataLoadUCI import RecordSamplesDatasetUC

@dataclass
class DDPConfig:
    """DDP-specific configuration"""
    # Hardware Configuration
    gpu_ids: str  # Required parameter
    
    # DDP settings
    master_addr: str = 'localhost'
    master_port: str = '12355'  # Default port
    ddp_backend: str = 'nccl'
    empty_cache_frequency: int = 25
    find_unused_parameters: bool = False
    static_graph: bool = True
    
    # Hardware Configuration (continued)
    num_threads: int = None
    num_workers: int = None
    
    # DataLoader settings
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    timeout: int = 3600
    
    def __post_init__(self):
        """Initialize hardware-specific settings"""
        # Check and set available port
        self.master_port = str(self._find_free_port())
        
        # Get system resources
        cpu_count = mp.cpu_count()
        gpu_count = len(self.gpu_ids.split(','))
        
        # Calculate optimal thread and worker counts
        if self.num_threads is None:
            workers_needed = gpu_count * 2  # 4 workers per GPU
            self.num_threads = max(1, cpu_count - workers_needed - 2)
        
        if self.num_workers is None:
            max_workers = cpu_count - self.num_threads - 2
            self.num_workers = min(gpu_count * 2, max_workers)
        
        # Ensure integer types
        self.num_threads = int(self.num_threads // 8)
        # self.num_workers = int(self.num_workers)
        self.num_workers = int(2)
        
        logging.info(f"DDPConfig initialized with:")
        logging.info(f"- Total CPU cores: {cpu_count}")
        logging.info(f"- GPUs: {gpu_count}")
        logging.info(f"- OpenMP threads: {self.num_threads}")
        logging.info(f"- DataLoader workers: {self.num_workers}")
    
    def _find_free_port(self) -> int:
        """Find a free port to use for DDP communication"""
        try:
            # Create a socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Bind to port 0 lets OS assign a free port
            sock.bind(('', 0))
            # Get the assigned port number
            port = sock.getsockname()[1]
            sock.close()
            logging.info(f"Using port {port} for DDP communication")
            return port
        except Exception as e:
            logging.warning(f"Failed to find free port: {e}. Using default port {self.master_port}")
            return int(self.master_port)

    def setup_hardware(self):
        """Setup hardware environment"""
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        
        # Set OpenMP threads
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.num_threads)
        
        # Ensure PyTorch uses the same number of threads
        torch.set_num_threads(self.num_threads)

    def get_dataloader_settings(self):
        """Get DataLoader settings"""
        settings = {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers if self.num_workers > 0 else False,
            'prefetch_factor': self.prefetch_factor if self.num_workers > 0 else None,
            'timeout': 0 if self.num_workers == 0 else self.timeout
        }
        return {k: v for k, v in settings.items() if v is not None}

def setup_ddp_env(rank: int, world_size: int, ddp_config: DDPConfig):
    """Setup DDP environment"""
    setup_ddp(
        rank=rank,
        world_size=world_size,
        master_addr=ddp_config.master_addr,
        master_port=ddp_config.master_port,
        ddp_backend=ddp_config.ddp_backend
    )

def train_ddp(rank: int, config: BaseModelConfig, dataset_ddp: tuple, world_size: int, ddp_config: DDPConfig, args=None):
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Process {rank}] %(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Starting process for rank {rank}")
        
        # Only handle wandb in rank 0
        if rank == 0 and args.project_name is not None:
            try:
                wandb.init(
                    entity="single_waveform",
                    project=args.project_name,
                    config=vars(args)
                )
            except Exception as e:
                logging.error(f"Failed to start wandb run: {str(e)}")

            if wandb.run is None:
                logging.warning("wandb run not found in rank 0 process - logs will not be sent to wandb")
            else:
                logging.info(f"Using wandb run: {wandb.run.name}")
        
        # Setup DDP
        logging.info("Setting up DDP environment")
        setup_ddp_env(rank, world_size, ddp_config)
        
        # Run the trainer with finetuning parameter
        logging.info(f"Starting trainer (finetuning: {args.is_finetuning}, resume: {args.resume_training})")
        config.trainer(dataset_ddp, rank, world_size, ddp_config)
        
    except Exception as e:
        logging.error(f"Error in process {rank}: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup wandb
        if rank == 0 and wandb.run is not None:
            wandb.finish()
        
        logging.info(f"Cleaning up process {rank}")
        if dist.is_initialized():
            dist.destroy_process_group()
        cleanup()
        torch.cuda.empty_cache()

        # Define cleanup methods for different dataset types
        dataset_cleanup_methods = {
            PulseDBDataset: lambda x: x.cleanup_shared_memory(),
            RecordSamplesDatasetUC: lambda x: x.cleanup_shared_memory()
            # Add more dataset types and their cleanup methods here
        }

        # Clean up based on dataset type
        if dataset_ddp:
            train_dataset, val_dataset, test_dataset = dataset_ddp
            for dataset_type, cleanup_method in dataset_cleanup_methods.items():
                if isinstance(train_dataset, dataset_type):
                    cleanup_method(train_dataset)
                if isinstance(test_dataset, dataset_type):
                    cleanup_method(test_dataset)

def run_ddp_training(args):
    """Main entry point for DDP training"""
    try:
        # Setup logging for main process
        logging.basicConfig(
            level=logging.INFO,
            format='[Main Process] %(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Starting DDP training (finetuning: {args.is_finetuning}, resume: {args.resume_training})")

        # Get model config
        config = ConfigFactory.create_config(args)
        
        dataset_ddp = config.create_ddp_dataset()

        # Create and initialize DDPConfig
        ddp_config = DDPConfig(gpu_ids=args.gpu_ids)
        
        # Setup hardware environment
        ddp_config.setup_hardware()
        
        # Get number of GPUs
        n_gpus = len(ddp_config.gpu_ids.split(','))
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        
        # Clear CUDA cache before starting
        torch.cuda.empty_cache()
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        logging.info(f"Spawning {n_gpus} processes")
        
        # Spawn processes with error handling and pass is_finetuning
        ctx = mp.spawn(
            train_ddp,
            args=(config, dataset_ddp, n_gpus, ddp_config, args),
            nprocs=n_gpus,
            join=False
        )
        
        # Wait for processes with timeout and proper error handling
        try:
            ctx.join()
        except (TimeoutError, KeyboardInterrupt, Exception) as e:
            logging.error(f"Training interrupted: {str(e)}")
            # Ensure graceful shutdown of all processes
            for process in ctx.processes:
                if process and process.is_alive():
                    process.terminate()
                    process.join(timeout=10)  # Give processes time to cleanup
            raise
        
    except Exception as e:
        logging.error("Error in main process", exc_info=True)
        raise
    finally:
        # Ensure thorough cleanup
        if 'ctx' in locals():
            for process in ctx.processes:
                if process and process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
        torch.cuda.empty_cache()

        # Define cleanup methods for different dataset types
        dataset_cleanup_methods = {
            PulseDBDataset: lambda x: x.cleanup_shared_memory(),
            RecordSamplesDatasetUC: lambda x: x.cleanup_shared_memory()
            # Add more dataset types and their cleanup methods here
        }
        # Clean up based on dataset type
        if dataset_ddp:
            train_dataset, val_dataset, test_dataset = dataset_ddp
            for dataset_type, cleanup_method in dataset_cleanup_methods.items():
                if isinstance(train_dataset, dataset_type):
                    cleanup_method(train_dataset)
                if isinstance(test_dataset, dataset_type):
                    cleanup_method(test_dataset)
        logging.info("Training completed or terminated")

def parse_arguments():
    """Parse command line arguments for DDP training"""
    parser = argparse.ArgumentParser(description='Distributed Data Parallel Training')
    
    # Hardware settings
    parser.add_argument('--gpu_ids', type=str, required=True,
                      help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')
    
    # Dataset and model settings
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['pulsedb', 'uci'],
                      help='Dataset to use for training')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['approximation', 'refinement'],
                      help='Type of model to train')
    parser.add_argument('--model_name', type=str, default=None,
                      choices=['nabnet', 'ppg2abp', 'p2ewgan', 'patchtst', 'mdvisco'],
                      help='Name of the model architecture')
    parser.add_argument('--direction', type=str,
                      choices=['PPG2ABP', 'ECG2ABP', 'ABP2PPG', 'ABP2ECG', 'PPG2ECG', 'ECG2PPG'],
                      help='Direction of signal conversion')

    # Checkpoint settings
    parser.add_argument('--checkpoint_name', type=str, required=True,
                      help='Checkpoint name to use for training')
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                      help='Checkpoint epoch to use for training')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training (default: 256)')
    parser.add_argument('--batch_size_approximation_model', type=int, default=256,
                    help='Batch size for the approximation model (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='Batch size for testing (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                      help='Patience for learning rate scheduler (default: 3)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                      help='Patience for early stopping (default: 5)')
    parser.add_argument('--resume_training', action='store_true',
                      help='Whether to resume training from checkpoint')
    
    # Wandb settings
    parser.add_argument('--project_name', type=str, required=False, default=None,
                      help='Project name for wandb logging')

    # Add seed argument
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility of dataset split  (default: 42)')

    # Add calculate_bpm argument
    parser.add_argument('--calculate_bpm', action='store_true',
                      help='Whether to calculate beats per minute at test time')

    # Add num_epochs argument
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs for training (default: 100)')
    parser.add_argument('--num_epochs_approximation_model', type=int, default=100,
                      help='Number of epochs for training the approximation model (default: 100)')

    # # REFINEMENT MODEL ARGUMENTS
    # Add WCL, PI and patient split arguments for refinement model
    parser.add_argument('--wcl', action='store_true',
                      help='Refinement model - Whether to use Weighted Contrastive Loss')
    parser.add_argument('--pi', action='store_true',
                      help='Refinement model - Whether to use Patient Information only for PulseDB with our or PatchTST')
    parser.add_argument('--bp_norm', action='store_true',
                        help='Refinement model - Whether to use BP normalization')

    # Dataset split for refinement models
    parser.add_argument('--use_patient_split', action='store_true',
                        help='Whether to use patient-wise splitting')
    parser.add_argument('--is_finetuning', action='store_true',
                      help='Refinement model -Whether to perform finetuning')
    parser.add_argument('--is_pretraining', action='store_true',
                      help='Refinement model -Whether to perform pretraining')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Run DDP training with parsed arguments
    run_ddp_training(args)


