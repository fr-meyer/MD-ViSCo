"""P2E-WGAN Implementation.

This module implements P2E-WGAN (PPG to ECG Wasserstein GAN) for generative
waveform synthesis.

References:
- Paper: "P2E-WGAN: ECG waveform synthesis from PPG with conditional Wasserstein
    generative adversarial networks"
  https://dl.acm.org/doi/10.1145/3412841.3441979
- Original Implementation: https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction
- License: MIT

Note: This implementation is adapted from the original codebase for use in
the MD-ViSCo framework.
"""

import logging

# Standard library imports
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.autograd as autograd
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig
from src.utils.checkpoint_io import remove_ddp_prefix_from_state_dict

logger = logging.getLogger(__name__)

# Canonical checkpoint keys used when saving (must match _checkpoint_state_dict)
P2EWGAN_KEY_G = "model_G_state_dict"
P2EWGAN_KEY_D = "model_D_state_dict"
# Legacy keys that may appear in old checkpoints
P2EWGAN_LEGACY_KEY_G = "generator_state_dict"
P2EWGAN_LEGACY_KEY_D = "discriminator_state_dict"

# ----------
#  Configuration
# ----------


@dataclass
class P2EWGANConfig(SingleStageModelConfig):
    """Configuration for P2E-WGAN architecture parameters.

    This dataclass contains all the model configuration attributes that directly control
    the architecture and behavior of the P2E-WGAN model, independently of training or
    dataset.

    Training parameters (lambda_gp, n_critic) remain in dataset configs.
    """

    _target_: str = "src.model.p2e_wgan.P2EWGAN"
    supports_multi_directional: bool = False  # P2E-WGAN only supports single-direction
    model_name: str = "P2EWGAN"

    # Model Architecture configuration
    in_channels: int = 1
    out_channels: int = 1

    # Network capacity configuration
    generator_init_filters: int = (
        MISSING  # Controls generator network capacity (approximation default)
    )
    discriminator_init_filters: int = (
        MISSING  # Controls discriminator network capacity (approximation default)
    )


@dataclass
class GeneratorP2EWGANConfig(SingleStageModelConfig):
    """Configuration for P2E-WGAN generator U-Net architecture parameters.

    See P2EWGANConfig for full model configuration. This config specifically
    controls the generator sub-component (U-Net) architecture.
    """

    _target_: str = "src.model.p2e_wgan.GeneratorUNet"
    supports_multi_directional: bool = False  # P2E-WGAN only supports single-direction
    model_name: str = "P2EWGAN"

    # Model Architecture configuration
    in_channels: int = 1
    out_channels: int = 1

    init_filters: int = MISSING


# ----------
#  U-NET
# ----------


class UNetDown(nn.Module):
    """Downsampling block for U-Net architecture."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        ksize: int = 4,
        stride: int = 2,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize UNetDown block with configuration parameters.

        Args:
            in_size: Number of input channels
            out_size: Number of output channels
            ksize: Kernel size for convolution
            stride: Stride for convolution
            normalize: Whether to apply instance normalization
            dropout: Dropout probability
        """
        super().__init__()
        # Use padding='same' to maintain size/(stride) for any input size
        padding = ksize // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_size,
                out_size,
                kernel_size=ksize,
                stride=stride,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )
        ]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through downsampling block.

        Args:
            x: Input tensor.

        Returns:
            Downsampled tensor.
        """
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block for U-Net architecture."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        ksize: int = 4,
        stride: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize UNetUp block with configuration parameters.

        Args:
            in_size: Number of input channels
            out_size: Number of output channels
            ksize: Kernel size for transposed convolution
            stride: Stride for transposed convolution
            dropout: Dropout probability
        """
        super().__init__()
        # Use padding='same' for the transposed convolution
        padding = ksize // 2
        output_padding = stride - 1
        layers = [
            nn.ConvTranspose1d(
                in_size,
                out_size,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampling block with skip connection.

        Args:
            x: Input tensor to upsample.
            skip_input: Skip connection tensor from encoder.

        Returns:
            Upsampled tensor with skip connection.
        """
        x = self.model(x)

        # Center crop the larger tensor to match sizes
        if x.size(2) != skip_input.size(2):
            if x.size(2) > skip_input.size(2):
                diff = x.size(2) - skip_input.size(2)
                x = x[:, :, diff // 2 : -(diff - diff // 2)]
            else:
                diff = skip_input.size(2) - x.size(2)
                skip_input = skip_input[:, :, diff // 2 : -(diff - diff // 2)]

        x = torch.cat((x, skip_input), 1)
        return x


# ----------
#  Generator
# ----------


class GeneratorUNet(SingleStageModel):
    """U-Net generator for P2E-WGAN architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 128,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize GeneratorUNet with configuration parameters.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            init_filters: Initial filter size for the first layer
        """
        super().__init__(*args, **kwargs)

        # Parameter validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if init_filters <= 0:
            raise ValueError("init_filters must be positive")

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters

        f1 = init_filters  # First level filters
        f2 = init_filters * 2  # Second level filters
        f3 = init_filters * 4  # Third level filters
        f4 = init_filters * 4  # Bottom level filters

        # Downsampling path - will work with any input size
        self.down1 = UNetDown(in_channels, f1, normalize=False)  # size -> size/2
        self.down2 = UNetDown(f1, f2)  # size/2 -> size/4
        self.down3 = UNetDown(f2, f3, dropout=0.5)  # size/4 -> size/8
        self.down4 = UNetDown(f3, f4, dropout=0.5, normalize=False)  # size/8 -> size/16

        # Upsampling path
        self.up1 = UNetUp(f4, f3, dropout=0.5)  # size/16 -> size/8
        self.up2 = UNetUp(f3 * 2, f2)  # size/8 -> size/4
        self.up3 = UNetUp(f2 * 2, f1)  # size/4 -> size/2

        # Modified final convolution with dynamic padding
        final_conv_size = 4
        final_padding = final_conv_size // 2  # This will be 2 for kernel size 4

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                f1 * 2,
                out_channels,
                kernel_size=final_conv_size,
                padding=final_padding,
                padding_mode="replicate",
            ),
            nn.Tanh(),
        )

        # Automatically initialize weights on creation
        self.apply(weights_init_normal)

    def extract_input(
        self, batch_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract input for P2E-WGAN.

        This method handles the unified input processing for P2E-WGAN, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, 1, L] for P2E-WGAN

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask

        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        elif isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            raise TypeError("GeneratorUNet extract_input expected Tensor from parent")
        return {"x": x}

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass for GeneratorUNet. Expects a standardized batch dict.

        Args:
            batch_dict: Standardized batch dict produced by the DataLoader collate
                function (or equivalent dict in tests).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - predictions: Generated output tensor [B, out_channels, L]
                - extras: Empty dict for consistency with canonical schema
        """
        out = self.extract_input(batch_dict)
        x = out["x"]

        # Assertion to catch channel mismatch (S_max > 1 vs in_channels=1)
        assert x.shape[1] == self.in_channels, (
            f"Channel mismatch: extracted input has {x.shape[1]} channels "
            f"but model expects {self.in_channels} channels. "
            "If multi-source inputs are expected, update 'in_channels' in config "
            "to match S_max."
        )

        # Store original size
        original_size = x.size(2)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        output = self.final(u3)
        # Ensure output size matches input size
        if output.size(2) != original_size:
            # Center crop if needed
            if output.size(2) > original_size:
                diff = output.size(2) - original_size
                output = output[:, :, diff // 2 : diff // 2 + original_size]
            else:
                raise ValueError(
                    f"Output size {output.size(2)} is smaller than input size "
                    f"{original_size}"
                )

        return {
            "predictions": output,
            "extras": {},
        }


# --------------
#  Discriminator
# --------------


class Discriminator(SingleStageModel):
    """Discriminator network for P2E-WGAN architecture."""

    def __init__(
        self, in_channels: int = 1, init_filters: int = 128, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize Discriminator with configuration parameters.

        Args:
            in_channels: Number of input channels per signal
            init_filters: Initial filter size for the first layer
        """
        super().__init__(*args, **kwargs)

        # Parameter validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if init_filters <= 0:
            raise ValueError("init_filters must be positive")

        # Store parameters
        self.in_channels = in_channels
        self.init_filters = init_filters

        # Note: Paper (Sec. 2.5, Fig. 3) specifies discriminator filter length 5 and
        # stride 3; this implementation uses kernel 4 and stride 2 for consistency
        # with the codebase.
        def discriminator_block(
            in_filters, out_filters, ksize=4, stride=2, normalization=True
        ):
            """Return downsampling layers of each discriminator block."""
            padding = ksize // 2
            layers: list[nn.Module] = [
                nn.Conv1d(
                    in_filters,
                    out_filters,
                    ksize,
                    stride=stride,
                    padding=padding,
                    padding_mode="replicate",
                )
            ]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        f1 = init_filters  # First level filters
        f2 = init_filters * 2  # Second level filters
        f3 = init_filters * 4  # Third level filters
        f4 = init_filters * 8  # Fourth level filters

        self.model = nn.Sequential(
            *discriminator_block(
                in_channels * 2, f1, normalization=False
            ),  # size -> size/2
            *discriminator_block(f1, f2),  # size/2 -> size/4
            *discriminator_block(f2, f3),  # size/4 -> size/8
            *discriminator_block(f3, f4),  # size/8 -> size/16
            nn.Conv1d(
                f4, 1, 4, stride=1, padding=1, padding_mode="replicate"
            ),  # size/16 -> size/16
        )

        # Automatically initialize weights on creation
        self.apply(weights_init_normal)

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract input for P2E-WGAN.

        This method handles the unified input processing for P2E-WGAN, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, 1, L] for P2E-WGAN

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask

        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        elif isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            raise TypeError("Discriminator extract_input expected Tensor from parent")
        return x

    def forward(
        self, batch_dict: dict[str, torch.Tensor], signal_b: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """Forward pass for Discriminator using the standardized interface.

        Args:
            batch_dict: Output of DataLoader collate_fn (standardized dict) or
                pre-processed tensor
            signal_b: The target waveform (real or fake) [B, 1, L].

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - predictions: Discriminator output tensor
                - extras: Empty dict for consistency with canonical schema
        """
        if isinstance(batch_dict, torch.Tensor):
            signal_a = batch_dict
        else:
            x = self.extract_input(batch_dict)

            # Assertion to catch channel mismatch (S_max > 1 vs in_channels=1)
            assert x.shape[1] == self.in_channels, (
                f"Channel mismatch: extracted input has {x.shape[1]} channels "
                f"but model expects {self.in_channels} channels. "
                "If multi-source inputs are expected, update 'in_channels' in config "
                "to match S_max."
            )

            signal_a = x

        signal_input = torch.cat((signal_a, signal_b), 1)
        result: dict[str, torch.Tensor | dict[str, Any]] = {
            "predictions": self.model(signal_input),
            "extras": {},
        }
        return result


def _normalize_p2ewgan_checkpoint_keys(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Normalize P2E-WGAN checkpoint keys for backward compatibility.

    Old checkpoints may use generator_state_dict / discriminator_state_dict.
    This helper maps them to model_G_state_dict / model_D_state_dict and
    returns a copy of the checkpoint with canonical keys. Non-state keys
    (e.g. epoch, optimizer) are left unchanged.

    Args:
        checkpoint: Raw checkpoint dictionary (may contain legacy keys).

    Returns:
        Tuple of (normalized_checkpoint, old_format_detected). old_format_detected
        is True if any legacy key was present (caller may log a warning).
    """
    old_detected = False
    normalized = dict(checkpoint)
    if P2EWGAN_LEGACY_KEY_G in normalized and P2EWGAN_KEY_G not in normalized:
        normalized[P2EWGAN_KEY_G] = normalized.pop(P2EWGAN_LEGACY_KEY_G)
        old_detected = True
    if P2EWGAN_LEGACY_KEY_D in normalized and P2EWGAN_KEY_D not in normalized:
        normalized[P2EWGAN_KEY_D] = normalized.pop(P2EWGAN_LEGACY_KEY_D)
        old_detected = True
    return normalized, old_detected


def weights_init_normal(m: nn.Module) -> None:
    """Initialize weights using normal distribution for Conv and BatchNorm layers.

    Args:
        m: Module to initialize weights for (Conv or BatchNorm1d).
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(
    d: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    real_a: torch.Tensor,
    patch: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Calculate the gradient penalty loss for WGAN GP.

    Args:
        d: Discriminator model.
        real_samples: Real samples tensor.
        fake_samples: Fake samples tensor.
        real_a: Real input signal A.
        patch: Patch shape for gradient penalty (e.g. for torch.full).
        device: Device to compute on.

    Returns:
        Gradient penalty loss scalar tensor.
    """
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = d(real_a, interpolates)
    if isinstance(d_interpolates, dict):
        d_interpolates = d_interpolates["predictions"]
    fake = torch.full(
        (real_samples.shape[0], *patch), 1, dtype=torch.float, device=device
    )

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class P2EWGAN(SingleStageModel):
    """P2E-WGAN model combining generator and discriminator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        generator_init_filters: int = 128,
        discriminator_init_filters: int = 128,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize P2EWGAN with configuration parameters.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            generator_init_filters: Initial filter size for generator network
            discriminator_init_filters: Initial filter size for discriminator network
        """
        super().__init__(*args, **kwargs)

        # Parameter validation
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if generator_init_filters <= 0:
            raise ValueError("generator_init_filters must be positive")
        if discriminator_init_filters <= 0:
            raise ValueError("discriminator_init_filters must be positive")

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.generator_init_filters = generator_init_filters
        self.discriminator_init_filters = discriminator_init_filters

        self.generator = GeneratorUNet(
            in_channels, out_channels, generator_init_filters, *args, **kwargs
        )
        self.discriminator = Discriminator(
            in_channels, discriminator_init_filters, *args, **kwargs
        )

    def _unwrap(self, m: nn.Module) -> nn.Module:
        """Private helper to unwrap DDP wrappers.

        Args:
            m: Model instance (potentially wrapped in DDP)

        Returns:
            The unwrapped model if it was wrapped, otherwise the original model
        """
        return m.module if hasattr(m, "module") else m

    def set_layout(self, layout: dict[Any, int] | Any) -> None:
        """Set vital-to-channel layout for this module only.

        The trainer's recursive walker (recursively_set_layout) handles DDP
        unwrapping and propagation to child modules (GeneratorUNet and Discriminator).
        Each module manages only its own cached layout state.

        Args:
            layout: Dict[Vital, int] mapping from Vital enum to channel indices

        Note:
            The base class `BaseModel.set_layout` enforces that layout must be
            `Dict[Vital, int]` and raises `TypeError` for invalid layouts.
        """
        super().set_layout(layout)

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract input for GAN models with DDP-safe wrapper handling.

        This method provides a DDP-safe interface for extracting input from batches,
        unwrapping DDP wrappers before delegating to the generator's ``extract_input()``
        method. This follows the same pattern as ``set_layout()`` and ensures
            compatibility
        with distributed training setups.

        Args:
            batch_dict: Batch dictionary with unified structure containing:
                - "x": Source signal [B, C, T]
                - "src_idxs": Channel indices [B, S_max] (optional)
                - "src_mask": Channel mask [B, S_max] (optional)

        Returns:
            torch.Tensor: Processed input tensor with shape [B, S_max, T] with channels
                selected according to ``src_idxs`` and ``src_mask``.
        """
        generator = self._unwrap(self.generator)

        # Delegate to generator's extract_input() method
        return generator.extract_input(batch_dict)

    def _checkpoint_state_dict(self) -> dict[str, Any]:
        """Return checkpoint state dict for P2E-WGAN model.

        This method defines how the GAN model's state should be saved in checkpoints.
        GAN models save both generator and discriminator states under separate keys.

        Returns:
            Dict[str, Any]: Dictionary mapping checkpoint keys to state dicts

        Example:
            {
                'model_G_state_dict': {
                    'layer1.weight': tensor(...),
                    'layer1.bias': tensor(...),
                    # ... all generator parameters
                },
                'model_D_state_dict': {
                    'layer1.weight': tensor(...),
                    'layer1.bias': tensor(...),
                    # ... all discriminator parameters
                }
            }

        Note:
            This method is called by BaseTrainer.save_checkpoint() to get the
            model's checkpoint structure. It is the ONLY way to define how a
            GAN model's state is saved in checkpoints.
        """
        return {
            "model_G_state_dict": self.generator.state_dict(),
            "model_D_state_dict": self.discriminator.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load state dict with backward compatibility for old checkpoint keys.

        When state_dict is a checkpoint-style dict (contains model_G_state_dict /
        model_D_state_dict or legacy generator_state_dict / discriminator_state_dict),
        normalizes keys to the current format, loads generator and discriminator
        from the respective sub-dicts, and optionally warns when old-format
        checkpoints are detected. Otherwise delegates to the superclass.

        Args:
            state_dict: State dictionary or full checkpoint dict.
            strict: Whether to strictly enforce key matching. Default: True.
            assign: Whether to use assign() instead of load_state_dict().
                Default: False.

        Returns:
            IncompatibleKeys with combined missing/unexpected keys from G and D.
        """
        is_checkpoint_format = (
            P2EWGAN_KEY_G in state_dict
            or P2EWGAN_KEY_D in state_dict
            or P2EWGAN_LEGACY_KEY_G in state_dict
            or P2EWGAN_LEGACY_KEY_D in state_dict
        )
        if not is_checkpoint_format:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        norm, old_detected = _normalize_p2ewgan_checkpoint_keys(dict(state_dict))
        if old_detected:
            logger.warning(
                "Old-format P2E-WGAN checkpoint detected (generator_state_dict / "
                "discriminator_state_dict). Consider re-saving the checkpoint with "
                "current keys (model_G_state_dict / model_D_state_dict) for "
                "faster loading."
            )

        all_missing: list[str] = []
        all_unexpected: list[str] = []
        incompat_ref = None

        if P2EWGAN_KEY_G in norm:
            g_state = remove_ddp_prefix_from_state_dict(norm[P2EWGAN_KEY_G])
            inc = self.generator.load_state_dict(g_state, strict=False, assign=assign)
            incompat_ref = inc
            all_missing.extend(inc.missing_keys)
            all_unexpected.extend(inc.unexpected_keys)
        if P2EWGAN_KEY_D in norm:
            d_state = remove_ddp_prefix_from_state_dict(norm[P2EWGAN_KEY_D])
            inc = self.discriminator.load_state_dict(
                d_state, strict=False, assign=assign
            )
            if incompat_ref is None:
                incompat_ref = inc
            all_missing.extend(inc.missing_keys)
            all_unexpected.extend(inc.unexpected_keys)

        if incompat_ref is None:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        incompat = type(incompat_ref)(
            missing_keys=all_missing,
            unexpected_keys=all_unexpected,
        )
        if all_missing:
            logger.warning(
                "Missing keys when loading P2E-WGAN checkpoint: %s", all_missing
            )
        if all_unexpected:
            logger.warning(
                "Unexpected keys when loading P2E-WGAN checkpoint: %s", all_unexpected
            )
        if strict and (all_missing or all_unexpected):
            raise RuntimeError("Strict loading failed due to missing/unexpected keys.")
        if not all_missing and not all_unexpected:
            logger.info("P2E-WGAN checkpoint loaded successfully")
        return incompat


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_p2ewgan", group="model", node=P2EWGANConfig)
cs.store(name="base_generator_p2ewgan", group="model", node=GeneratorP2EWGANConfig)
