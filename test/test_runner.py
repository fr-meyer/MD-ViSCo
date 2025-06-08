# Standard library imports
import os
import argparse
import logging
from dataclasses import dataclass

# Third-party imports
import torch

# Local imports
from config.config_factory import ConfigFactory

@dataclass
class TestConfig:
    """Test-specific configuration"""
    # Hardware Configuration
    gpu_id: str = '7'
    num_threads: int = None
    num_workers: int = None
    
    # DataLoader settings
    pin_memory: bool = True
    timeout: int = 3600
    
    def __post_init__(self):
        """Initialize hardware-specific settings"""
        # Get system resources
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        
        # Calculate optimal thread and worker counts
        if self.num_threads is None:
            self.num_threads = min(2, cpu_count - 4)  # Reserve some cores for system
        
        if self.num_workers is None:
            self.num_workers = min(4, cpu_count - self.num_threads - 2)  # Use up to 4 workers
        
        # Ensure integer types
        self.num_threads = int(self.num_threads)
        self.num_workers = int(self.num_workers)
        
        logging.info(f"TestConfig initialized with:")
        logging.info(f"- Total CPU cores: {cpu_count}")
        logging.info(f"- GPU: {self.gpu_id}")
        logging.info(f"- OpenMP threads: {self.num_threads}")
        logging.info(f"- DataLoader workers: {self.num_workers}")
    
    def setup_hardware(self):
        """Setup hardware environment"""
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        
        # Set OpenMP threads
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.num_threads)
        
        # Ensure PyTorch uses the same number of threads
        torch.set_num_threads(self.num_threads)

def run_test(args):
    """Main entry point for model testing"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Starting model testing")

        # Create and initialize TestConfig
        test_config = TestConfig()
        test_config.gpu_id = args.gpu_ids
        
        # Setup hardware environment
        test_config.setup_hardware()
        
        # Clear CUDA cache before starting
        torch.cuda.empty_cache()
        
        # Get model config
        config = ConfigFactory.create_config(args=args)
        
        # Run the test
        logging.info("Starting testing")
        config.test(test_config.num_workers)
        
    except Exception as e:
        logging.error("Error during testing", exc_info=True)
        raise
    finally:
        # Ensure thorough cleanup
        torch.cuda.empty_cache()
        logging.info("Testing completed")

def parse_arguments():
    """Parse command line arguments for model testing"""
    parser = argparse.ArgumentParser(description='Model Testing')
    
    # Dataset and model settings
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['pulsedb', 'uci'],
                      help='Dataset to use for testing')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['approximation', 'refinement'],
                      help='Type of model to test')
    parser.add_argument('--model_name', type=str, default=None,
                      choices=['nabnet', 'ppg2abp', 'p2ewgan', 'patchtst', 'mdvisco'],
                      help='Name of the model architecture')
    parser.add_argument('--direction', type=str,
                      choices=['PPG2ABP', 'ECG2ABP', 'ABP2PPG', 'ABP2ECG', 'PPG2ECG', 'ECG2PPG'],
                      help='Direction of signal conversion')
    
    # Testing settings
    parser.add_argument('--gpu_ids', type=str, default='0',
                      help='GPU ID to use for testing (default: 0)')
    parser.add_argument('--is_finetuning', action='store_true',
                      help='Whether to test finetuned model')
    parser.add_argument('--bp_norm', action='store_true',
                      help='Whether to use BP normalization')
    parser.add_argument('--use_patient_split', action='store_true',
                      help='Whether to use patient-wise splitting')
    parser.add_argument('--calculate_bpm', action='store_true',
                      help='Whether to calculate beats per minute')

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

    # Add num_epochs argument
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training (default: 100)')
    parser.add_argument('--num_epochs_approximation_model', type=int, default=100,
                        help='Number of epochs for training the approximation model (default: 100)')

    # Add WCL and PI arguments for refinement model
    parser.add_argument('--wcl', action='store_true',
                        help='Whether to use Weighted Contrastive Loss')
    parser.add_argument('--pi', action='store_true',
                        help='Whether to use Patient Information')

    # Add seed argument
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility of dataset split  (default: 42)')

    # Add extract_waveform_features argument
    parser.add_argument('--extract_waveform_features', action='store_true',
                        help='Whether to extract waveform features during testing')

    # Wandb settings
    parser.add_argument('--project_name', type=str, required=False, default=None,
                        help='Project name for wandb logging')


    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Run test with parsed arguments
    run_test(args)