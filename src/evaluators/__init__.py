"""Evaluators package for MD-ViSCo testing.

This package provides focused evaluator classes for different types of model evaluation:
- WaveformReconstructionEvaluator: MAE and Pearson correlation evaluation
- BloodPressureEvaluator: SBP/DBP/MAP, BHS standards evaluation
- FeatureExtractionEvaluator: PPG/ECG feature extraction from model outputs
- AFClassificationEvaluator: AF classification evaluation
"""


def import_evaluators() -> int:
    """Import all evaluator submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported evaluator modules

    Call this explicitly from your bootstrap (e.g., test.py) before instantiation,
    to avoid import-time side-effects when used as a library.
    """
    from src.utils.module_utils import import_modules as _import_modules

    pkg: str = __package__ or "src.evaluators"
    return _import_modules(pkg, module_type="evaluators")


__all__ = ["import_evaluators"]
