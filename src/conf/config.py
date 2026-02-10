"""Configuration schema for MD-ViSCo training and evaluation pipeline.

Serves as the root schema for Hydra composition in train.py and test.py.
"""

# Standard library imports
from dataclasses import dataclass

# Local imports
from src.dataset.base_dataset import DatasetBaseConfig
from src.evaluators.base_evaluator import EvaluatorBaseConfig
from src.trainers.trainer import TrainerBaseConfig


@dataclass
class Config:
    """Main configuration class for MD-ViSCo training pipeline.

    Orchestrates core configuration components through composition. Serves as
    the root schema for Hydra's configuration composition system, allowing
    Hydra to validate and compose configuration groups defined in YAML files.

    The Config dataclass composes Hydra groups through the defaults system.
    Each attribute corresponds to a Hydra config group that can be selected
    via YAML defaults or CLI overrides. Hydra automatically instantiates the
    corresponding config classes and assigns them to the Config attributes.
    Additional configurations (model, criterion, optimizer, scheduler) are
    handled through Hydra's composition system via nested YAML configuration
    files. This class is registered with Hydra's ConfigStore in
    src/conf/__init__.py as "base_config", enabling Hydra to resolve it as
    the root config schema.

    Attributes:
        train_dataset: Training dataset configuration (split ratios, paths,
            strategies). Selected via defaults (e.g., train_dataset: train_uci)
            or CLI override.
        test_dataset: Test dataset configuration (split ratios, paths,
            strategies). Selected via defaults (e.g., test_dataset: test_uci)
            or CLI override.
        trainer: Trainer configuration (training parameters, hardware,
            logging, seed). Selected via defaults (e.g.,
            trainer: approximation_trainer_mdvisco) or CLI override.
        evaluator: Evaluator configuration (inference, checkpoint loading,
            metrics). Selected via defaults (e.g.,
            evaluator: blood_pressure_evaluator) or CLI override.

    Example:
        Config is instantiated automatically by Hydra in entry points:

        >>> @hydra.main(config_path="conf", config_name="config")
        >>> def main(cfg: Config):
        ...     print(cfg.train_dataset.dataset_path)
        ...     print(cfg.trainer.learning_rate)

        Override config groups via CLI: e.g. ``train_dataset=train_pulsedb
        trainer=refinement_trainer_nabnet``.
    """

    # Core configuration components
    train_dataset: DatasetBaseConfig | None = None
    test_dataset: DatasetBaseConfig | None = None
    trainer: TrainerBaseConfig | None = None
    evaluator: EvaluatorBaseConfig | None = None
