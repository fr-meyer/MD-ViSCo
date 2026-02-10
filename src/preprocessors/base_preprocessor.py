"""Base preprocessor infrastructure for input preprocessing.

This module provides the foundational classes for separating raw input data
transformation from model forward pass logic. It establishes a clean interface
that subsequent phases can build upon without modification.

The preprocessor pattern mirrors the processor pattern but handles INPUT
transformation (raw input data → model-ready format) rather than OUTPUT
post-processing (model outputs → interpretable results).

Classes:
    - BasePreprocessorConfig: Base configuration dataclass for all preprocessors
    - BasePreprocessor: Abstract base class defining preprocessor contract
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BasePreprocessorConfig:
    """Base configuration for all preprocessors.

    This dataclass provides the foundational configuration structure for all
    preprocessors in the MD-ViSCo project. It follows the OmegaConf-compatible
    dataclass pattern used throughout the codebase.

    Subclasses extend this to add specific configuration fields required for
    their particular preprocessing logic (e.g., tokenizer_name, max_length for
    text encoders).

    The `_target_` field is used by Hydra for dynamic instantiation. It must be
    set by subclasses to point to the implementation class.

    The `active_fields` field allows filtering which input fields to process.
    This is useful when a dataset only includes a subset of available fields
    (e.g., only age and gender, but not height/weight/BMI).

    Attributes:
        _target_ (str): Class path for Hydra instantiation. Must be set by
            subclasses. Default: empty string.

        active_fields (Optional[List[str]]): List of field names that should be
            processed. If None, all available fields are processed. Default: None.

    Examples:
        >>> # Base config (should not be used directly)
        >>> base_config = BasePreprocessorConfig()
        >>> base_config._target_
        ''

        >>> # Subclass config (typical usage)
        >>> @dataclass
        ... class DemographicsTextEncoderConfig(BasePreprocessorConfig):
        ...     _target_: str = (
        ...         "src.preprocessors.demographics_text_encoder."
        ...         "DemographicsTextEncoder"
        ...     )
        ...     tokenizer_name: str = "distilbert-base-uncased"
        ...     max_length: int = 200
        ...     active_fields: Optional[List[str]] = None
        >>> config = DemographicsTextEncoderConfig()
        >>> config._target_
        'src.preprocessors.demographics_text_encoder.DemographicsTextEncoder'

    Note:
        This dataclass is designed to be extensible. Subclasses add their own
        fields without breaking existing code, as long as defaults are provided.
        Base classes are NOT registered with ConfigStore - only concrete
        implementations register their config dataclasses.
    """

    _target_: str = ""
    active_fields: list[str] | None = None


class BasePreprocessor(ABC):
    """Abstract base class for input preprocessors.

    This class defines the contract for all input preprocessors in the MD-ViSCo
    project. It separates raw input data transformation (preprocessing) from
    model forward pass logic (prediction).

    The preprocessor pattern provides several benefits:
    1. **Separation of Concerns**: Models focus on prediction, preprocessors
       handle input transformation. This follows the Single Responsibility Principle.
    2. **Reusability**: The same preprocessor can be shared across multiple models
       that require similar input formats (e.g., all models using demographics
       text encoding can use DemographicsTextEncoder).
    3. **Testability**: Preprocessing logic can be tested independently of
       model forward passes.
    4. **Flexibility**: Different preprocessing strategies can be swapped at runtime
       without modifying model code.

    This pattern is inspired by industry standards:
    - HuggingFace's Processor classes separate tokenization from model inference
    - sklearn's Pipeline pattern separates preprocessing from prediction
    - PyTorch Lightning's LightningDataModule separates data preparation from training

    **Role in the Pipeline:**

    Preprocessors transform raw input data into model-ready format:
    - Input: Raw data tensors (e.g., {"age_raw": tensor, "gender_raw": tensor})
    - Output: Model-ready tensors (e.g., {"input_ids": tensor, "attention_mask":
        tensor})

    **Contrast with Processors:**

    Preprocessors handle INPUT transformation (raw → model-ready):
    - Demographics to text encoding
    - Feature encoding and normalization
    - Data augmentation

    Processors handle OUTPUT post-processing (model outputs → interpretable):
    - Waveform trimming and denormalization
    - Scalar extraction (SBP/DBP from waveforms)
    - Classification output formatting

    **Examples of Use Cases:**

    1. **Demographics Text Encoding**: Transform numeric demographics (age, gender,
       height, weight, BMI) into text descriptions, then tokenize for language
       model input.

    2. **Feature Encoding**: Transform categorical features into embeddings or
       one-hot encodings.

    3. **Data Normalization**: Apply normalization transforms to input features
       (e.g., z-score normalization, min-max scaling).

    Subclasses must implement the `encode_batch()` abstract method to define
    their specific preprocessing logic. The `configure_from_include_list()` method
    is optional and can be overridden by subclasses that need field filtering.

    Examples:
        >>> # Configure preprocessor
        >>> preprocessor = DemographicsTextEncoder()
        >>> preprocessor.configure_from_include_list(["age_raw", "gender_raw"])

        >>> # Process batch
        >>> batch = {
        ...     "age_raw": torch.tensor([[25.0], [30.0]]),
        ...     "gender_raw": torch.tensor([[1.0], [0.0]])
        ... }
        >>> processed = preprocessor.encode_batch(batch)
        >>> processed.keys()
        dict_keys(['input_ids', 'attention_mask'])

    Note:
        This is a frozen interface that subsequent phases depend on. Changes to
        method signatures would break:
        - Phase 2+: Concrete preprocessor implementations
        - Phase 3+: Trainer integration
        - Phase 4+: Dataset integration
    """

    def __init__(self) -> None:
        """Initialize the preprocessor.

        Subclasses should call super().__init__() to initialize the base class. This
            sets up the
        active_fields attribute that can be used for field filtering.
        """
        self.active_fields: list[str] | None = None

    def configure_from_include_list(self, include_list: list[str]) -> None:
        """Configure active fields from dataset include list.

        This method allows preprocessors to filter which input fields to process
        based on what fields are actually present in the dataset. This is useful
        when datasets have varying field availability (e.g., some datasets include
        height/weight, others don't).

        This is an optional method - not all preprocessors need this functionality.
        Subclasses can override this method to implement their own filtering logic.

        Args:
            include_list (List[str]): List of field names that should be processed.
                For example, ["age_raw", "gender_raw"] indicates that only age
                and gender fields are available in the dataset.

        Examples:
            >>> # Demographics preprocessor filtering to available fields
            >>> preprocessor = DemographicsTextEncoder()
            >>> preprocessor.configure_from_include_list(["age_raw", "gender_raw"])
            >>> # Only age and gender are processed; height/weight/BMI ignored.

        Note:
            The default implementation sets active_fields to the include_list.
            Subclasses should override this method if they need custom filtering
            logic. See src/preprocessors/demographics_text_encoder.py for
            an example implementation.
        """
        self.active_fields = include_list

    @abstractmethod
    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Transform input batch into model-ready format.

        Main interface for preprocessing. Must be implemented by subclasses.
        Transforms raw input tensors into the format expected by the model.

        The method receives a dictionary of input data and returns a dictionary
        of processed data. The input format is typically raw data fields (e.g.,
        {"age_raw": tensor, "gender_raw": tensor}), and the output format is
        model-ready data (e.g., {"input_ids": tensor, "attention_mask": tensor}).
        While the examples show tensors, the implementation can handle other data
        types depending on the specific preprocessor use case.

        Args:
            batch (Dict[str, Any]): Dictionary of input data to process.
                Keys are field names (e.g., "age_raw", "gender_raw", "height_raw"),
                values can be torch tensors or other data types depending on the
                preprocessor implementation. Typical tensor shapes (when values are
                    tensors):
                - Scalar fields: [B, 1] where B=batch size
                - Sequence fields: [B, T] where T=sequence length
                - Multi-dimensional fields: [B, C, ...] where C=channels

        Returns:
            Dict[str, Any]: Dictionary of processed data ready for model input.
                Keys are model-specific (e.g., "input_ids", "attention_mask",
                    "features").
                Values can be torch tensors or other data types depending on the
                preprocessor implementation. Typical tensor shapes (when values are
                    tensors):
                - Text encoding: {"input_ids": [B, L], "attention_mask": [B, L]}
                  where L=sequence length
                - Feature encoding: {"features": [B, D]} where D=feature dimension
                - Multi-modal: {"input_ids": [B, L], "waveform": [B, C, T]}

        Raises:
            NotImplementedError: If subclass doesn't implement this method.
            ValueError: If batch format is invalid or missing required fields.
            RuntimeError: If preprocessing operations fail (e.g., tokenization
                fails, feature extraction fails).

        Examples:
            >>> # Demographics text encoding
            >>> preprocessor = DemographicsTextEncoder()
            >>> batch = {
            ...     "age_raw": torch.tensor([[25.0], [30.0]]),
            ...     "gender_raw": torch.tensor([[1.0], [0.0]])
            ... }
            >>> processed = preprocessor.encode_batch(batch)
            >>> processed.keys()
            dict_keys(['input_ids', 'attention_mask'])
            >>> processed['input_ids'].shape
            torch.Size([2, 200])  # batch_size=2, max_length=200

            >>> # Feature encoding
            >>> preprocessor = FeatureEncoder()
            >>> batch = {"features_raw": torch.randn(32, 10)}
            >>> processed = preprocessor.encode_batch(batch)
            >>> processed.keys()
            dict_keys(['features'])
            >>> processed['features'].shape
            torch.Size([32, 128])  # batch_size=32, feature_dim=128

        Note:
            Subclasses must implement this method to define their specific
            preprocessing logic. The implementation should handle batch processing
            efficiently, as it is called on every training/inference step.
        """
        raise NotImplementedError("Subclasses must implement encode_batch")
