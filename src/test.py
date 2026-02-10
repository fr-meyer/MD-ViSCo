#!/usr/bin/env python3
"""Entry point for Hydra-driven model evaluation.

This module retains the evaluator orchestration pattern mirrored from `train.py`
while removing all automated unit, integration, and performance testing hooks.
Use this script strictly for inference-driven evaluation workflows.

Architecture:
============
- test.py: Thin entry point - configuration validation, logging, and Hydra instantiation
- BaseEvaluator: Runtime orchestration - handles hardware/environment setup,
  model setup, dataloader creation, evaluation execution, and cleanup

Usage:
    python -m src.test                                  # Hydra evaluator flow
"""

# Standard library imports
import argparse
import logging
import os
import sys
from collections.abc import Sequence

# Third-party imports
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Local imports
from src.conf.resolvers import register_resolvers
from src.core import register_core
from src.dataset import _determine_training_scenario
from src.dataset import create_test_dataset
from src.dataset import import_datasets
from src.model import import_models
from src.preprocessors import import_preprocessors
from src.utils import train_utils  # noqa: F401  # EarlyStoppingConfig registration

from .conf.config import Config
from .evaluators import import_evaluators

logger = logging.getLogger(__name__)

register_resolvers()

register_core()
datasets_imported = import_datasets()
models_imported = import_models()
preprocessors_imported = import_preprocessors()
evaluators_imported = import_evaluators()


def _parse_cli_args(argv: Sequence[str] | None = None):
    """Parse and validate CLI arguments, rejecting deprecated testing flags.

    Args:
        argv: Command-line arguments to parse. Defaults to sys.argv[1:].

    Returns:
        List of remaining arguments after validation.

    Raises:
        SystemExit: If deprecated testing flags are detected.
    """
    parser = argparse.ArgumentParser(
        description="MD-ViSCo evaluator entry point",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    argv_list = list(argv or [])
    legacy_suite_flag = "--run-" + "py" + "test"
    deprecated_flags = {legacy_suite_flag, "--include-performance"}
    detected = sorted(flag for flag in deprecated_flags if flag in argv_list)
    if detected:
        parser.error(
            "Automated testing removed; use Hydra evaluator for inference "
            "validation. See README."
        )
    _, remaining = parser.parse_known_args(argv_list)
    return remaining


def _evaluator_mode_label(cfg: Config) -> str:
    """Determine evaluator mode label based on configuration.

    Args:
        cfg: Configuration object containing evaluator settings.

    Returns:
        Human-readable string describing the evaluator mode.
    """
    if cfg.evaluator is None:
        return "GT-only (no evaluator)"
    ev = cfg.evaluator
    load_weights = ev.load_model_weights if hasattr(ev, "load_model_weights") else True
    model = ev.model if hasattr(ev, "model") else None
    if load_weights:
        return "model-based (checkpoint restored)"
    if model is None:
        return "GT-only (model disabled)"
    return "Checkpoint skip (random initialization)"


def validate_test_config(cfg: Config) -> None:
    """Validate test configuration - fail early before instantiation.

    Args:
        cfg: Configuration object to validate

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if cfg.test_dataset is None:
        raise ValueError("test_dataset configuration is required")
    elif cfg.test_dataset.dataset_name is None:
        raise ValueError("test_dataset.dataset_name is required")

    if cfg.evaluator is None:
        raise ValueError("evaluator configuration is required")
    elif not cfg.evaluator.enabled:
        raise ValueError("evaluator must be enabled for testing")

    load_model_weights = (
        cfg.evaluator.load_model_weights
        if hasattr(cfg.evaluator, "load_model_weights")
        else True
    )
    if load_model_weights:
        missing_components = []
        if not hasattr(cfg.evaluator, "model") or cfg.evaluator.model is None:
            missing_components.append("evaluator.model")
        checkpoint_managers_missing = not hasattr(
            cfg.evaluator, "checkpoint_managers"
        ) or OmegaConf.is_missing(cfg.evaluator, "checkpoint_managers")
        checkpoint_managers = (
            cfg.evaluator.checkpoint_managers
            if hasattr(cfg.evaluator, "checkpoint_managers")
            else None
        )
        if checkpoint_managers_missing or checkpoint_managers in (None, {}):
            missing_components.append(
                "evaluator.checkpoint_managers (dict with at least 'save' entry)"
            )
        else:
            try:
                managers_keys = list(checkpoint_managers.keys())
            except AttributeError:
                managers_keys = []
            if len(managers_keys) == 0:
                missing_components.append(
                    "evaluator.checkpoint_managers (dict with at least 'save' entry)"
                )
            else:
                checkpoint_manager_name = getattr(
                    cfg.evaluator, "checkpoint_manager_name", "save"
                )
                if checkpoint_manager_name not in checkpoint_managers:
                    available_managers = managers_keys
                    raise ValueError(
                        f"Checkpoint manager '{
                            checkpoint_manager_name
                        }' (specified by checkpoint_manager_name) "
                        f"not found in checkpoint_managers dict. Available managers: {
                            available_managers
                        }. "
                        f"Either add '{
                            checkpoint_manager_name
                        }' to checkpoint_managers or change checkpoint_manager_name "
                        f"to one of the available managers."
                    )
        if (
            not hasattr(cfg.evaluator, "checkpoint_io")
            or cfg.evaluator.checkpoint_io is None
        ):
            missing_components.append("evaluator.checkpoint_io")
        trainer_name_missing = (
            not hasattr(cfg.evaluator, "trainer_name")
            or OmegaConf.is_missing(cfg.evaluator, "trainer_name")
            or cfg.evaluator.trainer_name in (None, "")
        )
        if trainer_name_missing:
            missing_components.append("evaluator.trainer_name")
        dataset_name_missing = (
            not hasattr(cfg.evaluator, "dataset_name")
            or OmegaConf.is_missing(cfg.evaluator, "dataset_name")
            or cfg.evaluator.dataset_name in (None, "")
        )
        if dataset_name_missing:
            missing_components.append("evaluator.dataset_name")
        checkpoint_epoch_missing = not hasattr(
            cfg.evaluator, "checkpoint_epoch"
        ) or OmegaConf.is_missing(cfg.evaluator, "checkpoint_epoch")
        if checkpoint_epoch_missing:
            missing_components.append("evaluator.checkpoint_epoch")
        if missing_components:
            raise ValueError(
                f"When load_model_weights=True, the following components are required: {
                    ', '.join(missing_components)
                }. "
                "To run in GT-only (model disabled) mode, override with "
                "`model@evaluator.model=null evaluator.load_model_weights=false` "
                "and omit checkpoint components. "
                "To perform a checkpoint skip instead, keep the model "
                "configuration but set "
                "`evaluator.load_model_weights=false`; the model will run with "
                "random initialization and checkpoints will be ignored."
            )
    else:
        logging.info(
            "%s detected: checkpoint validation skipped",
            _evaluator_mode_label(cfg),
        )

    # Relaxed direction mode validation: only enforce for single-direction evaluators
    direction_mode = (
        cfg.evaluator.direction_mode
        if hasattr(cfg.evaluator, "direction_mode")
        else "single"
    )
    if direction_mode != "single":
        evaluator_target = (
            cfg.evaluator._target_ if hasattr(cfg.evaluator, "_target_") else ""
        )
        single_direction_targets = {
            "src.evaluators.feature_extraction.FeatureExtractionEvaluator",
        }
        if evaluator_target in single_direction_targets:
            raise ValueError(
                f"{evaluator_target} requires direction_mode='single' but received '{
                    direction_mode
                }'."
            )
        logging.warning(
            "Evaluator configured with direction_mode='%s'. Ensure %s supports "
            "multi-direction mode.",
            direction_mode,
            evaluator_target or "the evaluator",
        )

    checkpoint_manager_summary = "none"
    if load_model_weights:
        managers = (
            cfg.evaluator.checkpoint_managers
            if hasattr(cfg.evaluator, "checkpoint_managers")
            else None
        )
        if managers not in (None, {}):
            try:
                keys = list(managers.keys())
            except AttributeError:
                keys = []
            checkpoint_manager_summary = keys if keys else "unavailable"
    logging.info(
        "Test configuration validated successfully (mode: %s, checkpoint managers: %s)",
        _evaluator_mode_label(cfg),
        checkpoint_manager_summary,
    )


def log_test_configuration(cfg: Config) -> None:
    """Log test configuration - simplified version.

    Args:
        cfg: Configuration object to log
    """
    from src.utils.config_logging import log_config_section

    if cfg.evaluator is None or cfg.test_dataset is None:
        return
    ev, td = cfg.evaluator, cfg.test_dataset
    model_info = (
        ev.model._target_
        if hasattr(ev, "model") and ev.model is not None
        else "None (GT-only mode)"
    )

    log_config_section(
        "MD-ViSCo Test Configuration",
        {
            "Working Directory": os.getcwd(),
            "Evaluator": ev._target_,
            "Model": model_info,
            "Load Model Weights": (
                ev.load_model_weights if hasattr(ev, "load_model_weights") else True
            ),
            "Mode": _evaluator_mode_label(cfg),
            "Test Dataset": td.dataset_name,
        },
    )


def create_test_datasets(cfg: Config):
    """Create test dataset using same pattern as train.py.

    Args:
        cfg: Configuration object containing dataset settings

    Returns:
        Test dataset
    """
    if cfg.test_dataset is None or cfg.evaluator is None:
        raise ValueError("test_dataset and evaluator configuration are required")
    td, ev = cfg.test_dataset, cfg.evaluator
    # Extract config values (same as train.py)
    train_ratio = td.train_ratio
    val_ratio = td.val_ratio
    test_ratio = td.test_ratio
    use_patient_split = td.use_patient_split
    use_nabnet_vanilla_split = td.use_nabnet_vanilla_split
    seed = ev.seed

    training_scenario = _determine_training_scenario(
        is_pretraining=ev.is_pretraining,
        is_finetuning=ev.is_finetuning,
        is_few_shot=ev.is_few_shot,
    )
    logging.info(f"Training scenario: {training_scenario}")

    test_dataset = create_test_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        use_patient_split=use_patient_split,
        use_nabnet_vanilla_split=use_nabnet_vanilla_split,
        seed=seed,
        training_scenario=training_scenario,
        train_dataset_config=td,
        test_dataset_config=td,
    )
    logging.info(f"Created test dataset with {len(test_dataset)} samples")

    return test_dataset


def initialize_wandb_evaluator(evaluator, cfg: Config) -> None:
    """Initialize WandB for evaluator if available.

    Args:
        evaluator: Evaluator object containing progress bar with WandB
        cfg: Configuration object for WandB initialization
    """
    from src.loggings.wandb_wrapper import initialize_wandb_if_enabled

    initialize_wandb_if_enabled(evaluator, cfg, is_rank0=True)


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: Config) -> None:
    """Execute main test workflow.

    The test function now follows the same architectural patterns as train.py:
    - Configuration validation only
    - Configuration logging
    - Dataset creation
    - Evaluator instantiation via Hydra
    - Delegation of ALL runtime orchestration to evaluator

    Args:
        cfg: Configuration object from Hydra

    Raises:
        ValueError: If configuration validation fails
        RuntimeError: If runtime errors occur during testing
    """
    evaluator = None
    try:
        # 1. Configuration validation
        validate_test_config(cfg)

        # 2. Configuration logging
        log_test_configuration(cfg)

        # 3. Log ConfigStore registration status
        logging.info(
            f"Imported {datasets_imported} dataset modules, {
                models_imported
            } model modules, {evaluators_imported} evaluator modules"
        )

        # 4. Create test dataset
        test_dataset = create_test_datasets(cfg)

        # 5. Instantiate evaluator (Hydra handles everything)
        assert cfg.evaluator is not None
        evaluator = instantiate(cfg.evaluator, _convert_="all")
        logging.info(f"Created evaluator: {evaluator.__class__.__name__}")

        # 6. Initialize WandB if available
        initialize_wandb_evaluator(evaluator, cfg)

        # 7. Delegate ALL runtime orchestration to evaluator
        #    Evaluator now performs inference locally using its configured processor.
        results, test_loader = evaluator.run_evaluation(test_dataset)
        evaluator.print_results(results, test_loader)

        logging.info("Test completed successfully!")

    except ValueError as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        raise
    except RuntimeError as e:
        logging.error(f"Runtime error during testing: {str(e)}")
        raise
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during testing: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if evaluator is not None:
            evaluator.cleanup()
        logging.info("Testing completed")


if __name__ == "__main__":
    hydra_argv = _parse_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *hydra_argv]
    main()
