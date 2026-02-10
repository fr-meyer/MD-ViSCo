"""MD-ViSCo Training Framework.

A comprehensive training framework supporting task-based trainers:
- WaveformReconstructionTrainer: For waveform regression tasks
- ScalarRegressionTrainer: For scalar regression tasks (BP, HR)
- ClassificationTrainer: For classification tasks (AF detection)
- GANTrainer: For GAN-based training
"""

from __future__ import annotations

from .trainer import BaseTrainer


def import_trainers() -> int:
    """Import all trainer submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported trainer modules

    Call this explicitly from your bootstrap (e.g., train.py) before instantiation,
    to avoid import-time side-effects when used as a library.
    """
    from src.utils.module_utils import import_modules as _import_modules

    if __package__ is not None:
        return _import_modules(__package__, module_type="trainer")
    return 0


__version__ = "1.0.0"
__all__ = ["BaseTrainer", "import_trainers"]
