r"""Preprocess PulseDB, UCI, MimicPERFormLarge, and MimicPERFormAF with Hydra.

This script can be run standalone to preprocess data files and save them in HDF5 format.

Features:
- Type-safe configuration using OmegaConf.structured()
- Automatic validation of configuration parameters
- Support for command-line overrides with type checking
- Input/output path validation
- Strict configuration validation to prevent typos
- Structured logging with Hydra integration
- Optional file saving control (save_files parameter)

Usage:
    python -m src.script.preprocess.preprocess preprocessor=pulsedb_preprocessing \\
        preprocessor.input_file=path/to/data.mat preprocessor.output_file=output/path
    python -m src.script.preprocess.preprocess preprocessor=uci_preprocessing \\
        preprocessor.input_file=path/to/data.h5 preprocessor.output_file=output/path \\
        preprocessor.sbp_max=200.0
    python -m src.script.preprocess.preprocess \\
        preprocessor=mimicperformlarge_preprocessing \\
        preprocessor.input_file=path/to/train.mat \\
        preprocessor.output_file=output/path
    python -m src.script.preprocess.preprocess \\
        preprocessor=mimicperformaf_preprocessing \\
        preprocessor.input_file=path/to/af_data.mat \\
        preprocessor.output_file=output/path

    # Skip saving files (processing only):
    python -m src.script.preprocess.preprocess preprocessor=pulsedb_preprocessing \\
        preprocessor.input_file=path/to/data.mat preprocessor.output_file=output/path \\
        preprocessor.save_files=false

    # UCI with custom parameters and no save:
    python -m src.script.preprocess.preprocess preprocessor=uci_preprocessing \\
        preprocessor.input_file=path/to/data.h5 preprocessor.output_file=output/path \\
        preprocessor.sbp_max=200.0 preprocessor.save_files=false

    # MimicPERFormAF with segment size and no save:
    python -m src.script.preprocess.preprocess \\
        preprocessor=mimicperformaf_preprocessing \\
        preprocessor.input_file=path/to/af_data.mat \\
        preprocessor.output_file=output/path \\
        preprocessor.segment_size=1024 preprocessor.save_files=false
"""

import logging

# Standard library imports
import os

# Third-party imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local imports
from . import import_preprocessors

# Trigger ConfigStore registration at module import time
preprocessors_imported = import_preprocessors()

logger = logging.getLogger(__name__)


def validate_paths(input_file: str, output_file: str):
    """Validate input and output paths before processing.

    Ensures the input file exists and creates the output directory
    if it doesn't exist.

    Args:
        input_file: Path to the input data file to be processed
        output_file: Path where the processed output will be saved

    Raises:
        FileNotFoundError: If the input file does not exist
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")


@hydra.main(
    version_base=None, config_path="../../conf", config_name="preprocessing_config"
)
def main(cfg: DictConfig):
    """Run preprocessing with Hydra configuration and enhanced validation.

    Args:
        cfg: Hydra configuration (DictConfig).
    """
    logger.info("=" * 60)
    logger.info("MD-ViSCo Preprocessing Configuration")
    logger.info("=" * 60)
    logger.info(f"Imported {preprocessors_imported} preprocessors")
    logger.info(f"Dataset: {cfg.preprocessor.dataset}")
    logger.info(f"Input file: {cfg.preprocessor.input_file}")
    logger.info(f"Output file: {cfg.preprocessor.output_file}")
    logger.info(f"Save files: {cfg.preprocessor.save_files}")
    if cfg.preprocessor.save_files:
        validate_paths(cfg.preprocessor.input_file, cfg.preprocessor.output_file)
    else:
        logger.info("Skipping path validation (save_files=False)")
        # Still validate input file exists
        if not os.path.exists(cfg.preprocessor.input_file):
            logger.error(f"Input file not found: {cfg.preprocessor.input_file}")
            raise FileNotFoundError(
                f"Input file not found: {cfg.preprocessor.input_file}"
            )

    # Instantiate preprocessor using Hydra (handles _target_ automatically)
    preprocessor = instantiate(cfg.preprocessor)
    logger.info(f"Instantiated preprocessor: {preprocessor.__class__.__name__}")

    output_file = preprocessor.preprocess_dataset(
        cfg.preprocessor.input_file, cfg.preprocessor.output_file
    )

    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)

    if cfg.preprocessor.save_files:
        logger.info(f"Output saved to: {output_file}")
    else:
        logger.info("Processing completed without saving files")
        logger.info(f"Would have saved to: {output_file}")


if __name__ == "__main__":
    main()
