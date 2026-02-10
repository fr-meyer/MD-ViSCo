"""Processor infrastructure for post-processing model outputs.

This module provides infrastructure for separating model forward pass logic
from post-processing operations (trimming, scalar extraction, denormalization).
Processors consume canonical model output dictionaries and batch metadata to
produce unified result payloads for trainers and evaluators.

Classes:
    - OutputProcessor: Abstract base class defining the processor contract
    - ProcessingMetadata: Dataclass encapsulating metadata for post-processing
    - ScalarExtractor: Base class for scalar extractors
    - MinMaxExtractor: Extractor for min/max scalar extraction
    - BPExtractor: Extractor for blood pressure (SBP/DBP/MAP) extraction

Functions:
    - import_processors: Import all processor submodules for ConfigStore registration
    - import_extractors: Import all extractor submodules for ConfigStore registration

Examples:
    >>> from src.processors import (
    ...     import_processors, OutputProcessor, ProcessingMetadata
    ... )
    >>> import_processors()
    >>> metadata = ProcessingMetadata.from_batch(batch)

See Also:
    - src.processors.waveform_processor: Waveform regression/reconstruction processor
    - src.processors.scalar_processor: Scalar output processor
    - src.processors.extractors: Scalar extractor implementations
"""

# Import only base classes and utilities
from src.processors.output_processor import OutputProcessor
from src.processors.output_processor import ProcessingMetadata


def import_processors() -> int:
    """Import all processor submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported processor modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    return _import_modules(__package__ or "src.processors", module_type="processor")


def import_extractors() -> int:
    """Import all extractor submodules to trigger ConfigStore registration.

    Returns:
        Number of successfully imported extractor modules
    """
    from src.utils.module_utils import import_modules as _import_modules

    return _import_modules(__package__ or "src.processors", module_type="extractor")


__all__ = [
    # Base classes
    "OutputProcessor",
    "ProcessingMetadata",
    # Dynamic import functions
    "import_processors",
    "import_extractors",
]
