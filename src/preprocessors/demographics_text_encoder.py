"""Demographics text encoder: numeric features to tokenized text.

This module provides a preprocessor that converts numeric demographic tensors (age,
gender, height, weight, BMI) into natural language text descriptions, then tokenizes
them using HuggingFace tokenizers for use as input to language models.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch

from src.preprocessors.base_preprocessor import BasePreprocessor
from src.preprocessors.base_preprocessor import BasePreprocessorConfig

try:
    # Hydra is optional at import time; only needed for config registration
    from hydra.core.config_store import ConfigStore
except ImportError:  # pragma: no cover - hydra may not be available in all contexts
    ConfigStore: Any = None  # Optional; set when Hydra config store is used


logger = logging.getLogger(__name__)


@dataclass
class DemographicsTextEncoderConfig(BasePreprocessorConfig):
    """Configuration options for :class:`DemographicsTextEncoder`.

    Attributes:
        _target_: Full path to the encoder class for Hydra instantiation
        tokenizer_name: HuggingFace tokenizer model name to use for text encoding.
            Default: "distilbert-base-uncased"
        max_length: Maximum sequence length for tokenized text. Default: 200
        padding: Padding strategy for tokenization. Default: "max_length"
        truncation: Whether to truncate sequences exceeding max_length. Default: True
    """

    _target_: str = (
        "src.preprocessors.demographics_text_encoder.DemographicsTextEncoder"
    )
    tokenizer_name: str = "distilbert-base-uncased"
    max_length: int = 200
    padding: str = "max_length"
    truncation: bool = True


class DemographicsTextEncoder(BasePreprocessor):
    """Text encoder for demographic features (age, gender, height, weight, BMI).

    This preprocessor transforms numeric demographic tensors into natural language
    text descriptions, then tokenizes them for use as input to language models.
    It handles batch processing and supports field filtering via ``active_fields``
    to process only available demographic fields.

    The encoder creates text descriptions in the format:
        "Patient Age: {age} years / Patient Gender: {gender} / Patient Height: "
        "{height} cm / Patient Weight: {weight} kg / Patient BMI: {bmi} kg/m2"

    Only fields present in ``active_fields`` are included in the text description.
    When a field value cannot be converted (e.g., NaN, invalid tensor), a warning
    is logged and the field defaults to "Unknown".

    Expected batch keys:
        - "age_raw": Tensor of shape [B, 1] with age values
        - "gender_raw": Tensor of shape [B, 1] with gender values (1=Male, 0=Female)
        - "height_raw": Tensor of shape [B, 1] with height values in cm
        - "weight_raw": Tensor of shape [B, 1] with weight values in kg
        - "bmi_raw": Tensor of shape [B, 1] with BMI values

    Output keys:
        - "input_ids": Tokenized input IDs tensor of shape [B, max_length]
        - "attention_mask": Attention mask tensor of shape [B, max_length]

    Examples:
        >>> encoder = DemographicsTextEncoder(tokenizer_name="distilbert-base-uncased")
        >>> batch = {
        ...     "age_raw": torch.tensor([[25.0], [30.0]]),
        ...     "gender_raw": torch.tensor([[1.0], [0.0]])
        ... }
        >>> encoder.configure_from_include_list(["age_raw", "gender_raw"])
        >>> output = encoder.encode_batch(batch)
        >>> output.keys()
        dict_keys(['input_ids', 'attention_mask'])
        >>> output['input_ids'].shape
        torch.Size([2, 200])
    """

    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 200,
        padding: str = "max_length",
        truncation: bool = True,
        active_fields: list[str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the DemographicsTextEncoder.

        Args:
            tokenizer_name: HuggingFace tokenizer model name to use for text encoding.
                Must be a valid model identifier available on HuggingFace Hub.
                Default: "distilbert-base-uncased"
            max_length: Maximum sequence length for tokenized text. Sequences longer
                than this will be truncated if truncation=True. Default: 200
            padding: Padding strategy for tokenization. Options: "max_length" (pad to
                max_length), "longest" (pad to longest sequence in batch), or
                    "do_not_pad".
                Default: "max_length"
            truncation: Whether to truncate sequences exceeding max_length. When True,
                sequences longer than max_length are truncated. Default: True
            active_fields: Optional list of field names to process. When None, all
                available fields (age_raw, gender_raw, height_raw, weight_raw, bmi_raw)
                are processed. When provided, only fields in this list are included in
                the text description. Default: None
            *args: Additional positional arguments passed to BasePreprocessor
            **kwargs: Additional keyword arguments passed to BasePreprocessor

        Raises:
            RuntimeError: If tokenizer cannot be loaded from HuggingFace Hub
            ValueError: If tokenizer_name is invalid or model not found
        """
        super().__init__(*args, **kwargs)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.active_fields = active_fields

    def configure_from_include_list(self, include_list: list[str]) -> None:
        """Configure active demographic fields from dataset include list.

        Filters demographic fields (age_raw, gender_raw, height_raw, weight_raw,
            bmi_raw)
        to only those present in the provided include_list.

        Args:
            include_list: List of field names available in the dataset.
        """
        demo_fields = ["age_raw", "gender_raw", "height_raw", "weight_raw", "bmi_raw"]
        self.active_fields = [f for f in demo_fields if f in include_list]
        logger.info(
            "DemographicsTextEncoder configured with fields: %s", self.active_fields
        )

    def _create_text_description(
        self, batch_demographics: dict[str, torch.Tensor]
    ) -> list[str]:
        """Create natural language text descriptions from demographic tensors.

        Converts numeric demographic values (age, gender, height, weight, BMI) into
        formatted text strings. Handles invalid values (NaN, non-finite, conversion
        errors) by logging warnings and falling back to "Unknown" for the affected
            field.

        Args:
            batch_demographics: Dictionary mapping field names to tensors. Expected
                keys:
                - "age_raw": Tensor of shape [B, 1] or [B] with age values
                - "gender_raw": Tensor of shape [B, 1] or [B] with gender (1=Male,
                    0=Female)
                - "height_raw": Tensor of shape [B, 1] or [B] with height in cm
                - "weight_raw": Tensor of shape [B, 1] or [B] with weight in kg
                - "bmi_raw": Tensor of shape [B, 1] or [B] with BMI values

        Returns:
            List of text descriptions, one per batch item. Format:
                "Patient Age: {age} years / Patient Gender: {gender} / "
                "Patient Height: {height} cm / Patient Weight: {weight} kg / "
                "Patient BMI: {bmi} kg/m2"
            Only fields present in ``active_fields`` are included. Invalid values
            are replaced with "Unknown" and a warning is logged.

        Raises:
            RuntimeError: If batch is empty or tensors have incompatible shapes
            ValueError: If tensor shapes are invalid (not [B, 1] or [B])
        """
        if self.active_fields is None:
            # If not configured, infer active fields from provided batch keys
            self.active_fields = [
                k
                for k in [
                    "age_raw",
                    "gender_raw",
                    "height_raw",
                    "weight_raw",
                    "bmi_raw",
                ]
                if k in batch_demographics
            ]

        # Ensure tensors are [B, 1] and get batch size
        if not batch_demographics:
            raise ValueError(
                "batch_demographics is empty. Cannot create text descriptions without "
                "demographic data. Ensure at least one demographic field (age_raw, "
                "gender_raw, height_raw, weight_raw, bmi_raw) is provided in the batch."
            )
        some_key = next(iter(batch_demographics))
        batch_size = int(batch_demographics[some_key].shape[0])

        # Squeeze to shape [B]
        squeezed: dict[str, torch.Tensor] = {}
        for key, tensor in batch_demographics.items():
            if tensor.ndim == 2 and tensor.shape[1] == 1:
                squeezed[key] = tensor.squeeze(1)
            else:
                squeezed[key] = tensor

        texts: list[str] = []
        for i in range(batch_size):
            parts: list[str] = []
            if "age_raw" in self.active_fields and "age_raw" in squeezed:
                val = squeezed["age_raw"][i]
                age_str = "Unknown"
                try:
                    v = val.item()
                    if math.isfinite(v):
                        age_str = f"{int(round(v))}"
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.warning(
                        "Failed to format age_raw value %s for batch item %d: %s. "
                        "Using default 'Unknown'.",
                        val,
                        i,
                        e,
                    )
                parts.append(f"Patient Age: {age_str} years")

            if "gender_raw" in self.active_fields and "gender_raw" in squeezed:
                val = squeezed["gender_raw"][i]
                gender_str = "Unknown"
                try:
                    v = float(val.item())
                    if v == 1:
                        gender_str = "Male"
                    elif v == 0:
                        gender_str = "Female"
                    else:
                        gender_str = "Unknown"
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.warning(
                        "Failed to format gender_raw value %s for batch item %d: %s. "
                        "Using default 'Unknown'.",
                        val,
                        i,
                        e,
                    )
                    gender_str = "Unknown"
                parts.append(f"Patient Gender: {gender_str}")

            if "height_raw" in self.active_fields and "height_raw" in squeezed:
                val = squeezed["height_raw"][i]
                height_str = "Unknown"
                try:
                    v = val.item()
                    if math.isfinite(v):
                        height_str = f"{round(v)}"
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.warning(
                        "Failed to format height_raw value %s for batch item %d: %s. "
                        "Using default 'Unknown'.",
                        val,
                        i,
                        e,
                    )
                parts.append(f"Patient Height: {height_str} cm")

            if "weight_raw" in self.active_fields and "weight_raw" in squeezed:
                val = squeezed["weight_raw"][i]
                weight_str = "Unknown"
                try:
                    v = val.item()
                    if math.isfinite(v):
                        weight_str = f"{round(v)}"
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.warning(
                        "Failed to format weight_raw value %s for batch item %d: %s. "
                        "Using default 'Unknown'.",
                        val,
                        i,
                        e,
                    )
                parts.append(f"Patient Weight: {weight_str} kg")

            if "bmi_raw" in self.active_fields and "bmi_raw" in squeezed:
                val = squeezed["bmi_raw"][i]
                bmi_str = "Unknown"
                try:
                    v = val.item()
                    if math.isfinite(v):
                        bmi_str = f"{round(v, 1)}"
                except (ValueError, RuntimeError, TypeError) as e:
                    logger.warning(
                        "Failed to format bmi_raw value %s for batch item %d: %s. "
                        "Using default 'Unknown'.",
                        val,
                        i,
                        e,
                    )
                parts.append(f"Patient BMI: {bmi_str} kg/m2")

            texts.append(" / ".join(parts))
        return texts

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transform demographic tensors into tokenized text encodings.

        This method implements the abstract method from BasePreprocessor. It converts
        numeric demographic values into text descriptions, then tokenizes them using
        the configured HuggingFace tokenizer.

        Args:
            batch: Dictionary mapping field names to tensors (alias for base
                'batch'). Expected keys:
                - "age_raw": Tensor of shape [B, 1] or [B] with age values
                - "gender_raw": Tensor of shape [B, 1] or [B] with gender (1=Male,
                    0=Female)
                - "height_raw": Tensor of shape [B, 1] or [B] with height in cm
                - "weight_raw": Tensor of shape [B, 1] or [B] with weight in kg
                - "bmi_raw": Tensor of shape [B, 1] or [B] with BMI values
                Only fields present in ``active_fields`` are processed.

        Returns:
            Dictionary with tokenized text encodings:
                - "input_ids": Token IDs tensor of shape [B, max_length]
                - "attention_mask": Attention mask tensor of shape [B, max_length]
                indicating which tokens are padding (0) vs. real tokens (1)

        Raises:
            ValueError: If batch is empty or missing required fields
            RuntimeError: If tokenization fails (e.g., tokenizer error, OOM)
            TypeError: If input tensors have invalid types or shapes
        """
        texts = self._create_text_description(batch)
        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


# Register with Hydra ConfigStore
if __name__ != "__main__" and ConfigStore is not None:
    cs = ConfigStore.instance()
    cs.store(
        group="input_preprocessor",
        name="base_demographics_text_encoder",
        node=DemographicsTextEncoderConfig,
    )
