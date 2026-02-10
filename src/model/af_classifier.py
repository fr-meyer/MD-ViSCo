"""PyTorch AF Classifier Models.

This module implements both MLP and CNN-based classifiers for atrial
fibrillation detection. The CNN-based classifier uses CNN encoder
components from MDViSCo's UNet_SwinUnet.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig

# =====================
# Original MLP Classifier
# =====================


@dataclass
class AFClassifierConfig(SingleStageModelConfig):
    """Configuration for PyTorch MLP AF Classifier."""

    _target_: str = "src.model.af_classifier.AFClassifier"
    model_name: str = "AFClassifier"

    # Architecture parameters
    hidden_sizes: tuple[int, ...] = (512, 256, 2)
    dropout: float = 0.2
    activation: str = "relu"  # "relu", "gelu", "tanh"

    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


class AFClassifier(SingleStageModel):
    """PyTorch MLP Classifier for Atrial Fibrillation Detection.

    Uses single output + sigmoid for binary classification, matching sklearn behavior.
    This approach is more efficient and natural for binary classification than using
    two outputs with softmax.

    Architecture:
        Input -> Hidden Layers -> Single Output -> Sigmoid -> Probabilities

    Attributes:
        input_length: Number of input features (ECG signal length).
        hidden_sizes: Sizes of hidden layers.
        dropout: Dropout probability.
        network: Sequential stack of linear, activation, dropout, and output layer.
    """

    def __init__(
        self,
        input_length: int = MISSING,
        hidden_sizes: tuple[int, ...] = (200, 100, 50),
        dropout: float = 0.1,
        activation: str = "relu",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize AFClassifier with configuration parameters.

        Args:
            input_length: Number of input features (ECG signal length)
            hidden_sizes: Sizes of hidden layers
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__(
            *args,
            supports_multi_directional=False,
            input_length=input_length,
            **kwargs,
        )

        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        # Build network layers
        layers = []
        prev_size = self.input_length

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        # Single output for binary classification (no sigmoid - use BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def forward(self, x: Any) -> dict[str, Any]:
        """Forward pass - returns logits for BCEWithLogitsLoss.

        Accepts either a NEW-format batch dict or a tensor.

        Args:
            x: Dict with NEW batch structure (uses extract_input) or tensor of
               shape [B, C, T] or [B, T].

        Returns:
            Dict[str, torch.Tensor]: Dictionary following the canonical model schema:
                - "predictions": Logits of shape [B, 1] for binary classification
                - "extras": Dictionary containing:
                    - "embeddings": Intermediate feature embeddings from the network
                      (before the final linear layer)
        """
        # Support dict-or-tensor input
        if isinstance(x, dict):
            x = self.extract_input(x)  # [B, C, T]

        # x shape: [B, C, T] -> [B, T] (flatten time dimension)
        if isinstance(x, torch.Tensor) and x.dim() == 3:
            x = x.view(x.size(0), -1)  # [B, C*T]
        elif isinstance(x, torch.Tensor) and x.dim() == 2:
            pass  # [B, T] already flattened

        embeddings = x
        for layer in self.network[:-1]:
            embeddings = layer(embeddings)
        logits = self.network[-1](embeddings)
        return {
            "predictions": logits,
            "extras": {"embeddings": embeddings},
        }

    def predict_proba(self, x: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities (sklearn-compatible).

        Args:
            x: Input tensor or batch dict

        Returns:
            torch.Tensor: Probabilities of shape [N, 2] for sklearn compatibility
                          [prob_class_0, prob_class_1]
        """
        with torch.no_grad():
            out = self.forward(x)
            logits = out["predictions"]
            probs = torch.sigmoid(logits)
            # Return shape [N, 2] for sklearn: [prob_class_0, prob_class_1]
            return torch.cat([1 - probs, probs], dim=1)

    def predict(self, x: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Get class predictions (0 or 1).

        Args:
            x: Input tensor or batch dict

        Returns:
            torch.Tensor: Class predictions of shape [N]
        """
        with torch.no_grad():
            out = self.forward(x)
            logits = out["predictions"]
            probs = torch.sigmoid(logits)
            return (probs > 0.5).long().squeeze(1)  # Threshold at 0.5

    def get_probabilities(
        self, x: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Get raw probabilities (single output).

        Args:
            x: Input tensor or batch dict

        Returns:
            torch.Tensor: Probabilities of shape [N, 1]
        """
        with torch.no_grad():
            out = self.forward(x)
            logits = out["predictions"]
            return torch.sigmoid(logits)


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_af_classifier", node=AFClassifierConfig, group="model")
