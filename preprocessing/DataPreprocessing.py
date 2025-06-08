"""
Preprocesses PulseDB data files by applying normalization and preparing them for model training.
The script takes .mat files as input and saves the preprocessed data in HDF5 format.
"""

# Standard library imports
import os

# Third-party imports
from config.config_factory import ConfigFactory

# Define the preprocessing configuration for each dataset
DATASET_CONFIG = {
    'Train_Subset': {
        'file_name': 'Train_Subset',
        'extension': 'mat',
        'dataset_name': 'pulsedb',
        'calculate_bpm': False
    },
    'CalFree_Test_Subset': {
        'file_name': 'CalFree_Test_Subset',
        'extension': 'mat',
        'dataset_name': 'pulsedb',
        'calculate_bpm': True
    },
    'UCI': {
        'file_name': 'UCI_Dataset_Preprocessed',
        'extension': 'h5',
        'dataset_name': 'uci',
        'calculate_bpm': False
    },
    # Add more datasets as needed
}

def process_dataset(config, dataset_params):
    """
    Process a single dataset according to the specified parameters.
    
    Args:
        config: Configuration object from ConfigFactory
        dataset_params (dict): Parameters for dataset processing
    """
    file_name = dataset_params['file_name']
    input_file = os.path.join(config.data_path, config.dataset_path, f'{file_name}.{dataset_params["extension"]}')
    output_file = os.path.join(config.data_path, config.dataset_path, config.preprocessed_path, f'{file_name}')

    # Print input and output paths
    print(f"Input file base path: {input_file}")
    print(f"Output file path: {output_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process the dataset with the specified parameters
    config.preprocess_dataset(
        input_file, 
        output_file,
        calculate_bpm=dataset_params['calculate_bpm']
    )
    print(f"Preprocessing complete for {file_name}. File saved to: {output_file}")

if __name__ == "__main__":
    # Process all datasets in the configuration
    for dataset_name, params in DATASET_CONFIG.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Create args object with dataset name
        class Args:
            def __init__(self, dataset_name):
                self.dataset = dataset_name
                self.model_type = None  # Set to None to get base config
        
        args = Args(params['dataset_name'])
        config = ConfigFactory.create_config(args)
        process_dataset(config, params)