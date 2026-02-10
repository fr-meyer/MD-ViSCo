"""PPG2ABP Implementation.

This module implements PPG2ABP (UNetDS64, MultiResUNet1D) for PPG to ABP
waveform conversion.

References:
- Paper: "PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood
    Pressure (ABP) Waveforms using Fully Convolutional Neural Networks"
  https://www.mdpi.com/2306-5354/9/11/692
- Specific Citation: Section 3.3 'Effect of Number of Convolutional Filters', page 6
- Key Parameters: UNetDS64 base_channels=64, MultiResUNet1D alpha=2.5
- Original Implementation: https://github.com/nibtehaz/PPG2ABP
- License: MIT

Note: This implementation is adapted from the original codebase for use in
the MD-ViSCo framework.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports
from src.model.single_stage_model import SingleStageModel
from src.model.single_stage_model import SingleStageModelConfig


@dataclass
class UNetDS64Config(SingleStageModelConfig):
    """Configuration class for UNetDS64 model architecture parameters."""

    _target_: str = "src.model.ppg2abp.UNetDS64"
    supports_multi_directional: bool = False  # UNetDS64 only supports single-direction
    model_name: str = "UNetDS64"

    # Model Architecture configuration
    base_channels: int = (
        64  # Paper value from Section 3.3: filters [64, 128, 256, 512, 1024]
    )
    in_channels: int = 1  # Number of input signal channels
    out_channels: int = 1  # Number of output channels


@dataclass
class MultiResUNet1DConfig(SingleStageModelConfig):
    """Configuration for MultiResUNet1D and PPG2ABP refinement models (PulseDB/UCI).

    Contains all model architecture attributes that define the structure and capacity of
        the neural
    network model, not training or runtime behavior.
    """

    _target_: str = "src.model.ppg2abp.MultiResUNet1D"
    supports_multi_directional: bool = (
        False  # MultiResUNet1D only supports single-direction
    )
    model_name: str = "MultiResUNet1D"

    alpha: float = (
        2.5  # Weight multiplier (paper value from Section 3.3: alpha limited to 2.5)
    )
    in_channels: int = 1  # Canonical number of input channels to the model
    out_channels: int = 1  # Number of output channels (ABP)


class UNetDS64(SingleStageModel):
    """Deeply supervised U-Net with kernels multiples of 64.

    Based on PPG2ABP paper (Ibtehaz et al. 2022, Section 3.3).

    Args:
        base_channels (int): Base number of channels (default: 64, paper value from
            Section 3.3)
        in_channels (int): Number of input signal channels
        out_channels (int): Number of output channels
    """

    def __init__(
        self,
        base_channels: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize UNetDS64 with configuration parameters.

        Args:
            base_channels: Base number of channels (default: 64, paper value from
                Section 3.3)
            in_channels: Number of input signal channels
            out_channels: Number of output channels
        """
        # UNetDS64 only supports single-direction training
        super().__init__(*args, **kwargs)

        # Use base_channels as the conv_channel parameter
        x = base_channels

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU(),
            nn.Conv1d(x, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(x, x * 2, 3, padding=1),
            nn.BatchNorm1d(x * 2),
            nn.ReLU(),
            nn.Conv1d(x * 2, x * 2, 3, padding=1),
            nn.BatchNorm1d(x * 2),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(x * 2, x * 4, 3, padding=1),
            nn.BatchNorm1d(x * 4),
            nn.ReLU(),
            nn.Conv1d(x * 4, x * 4, 3, padding=1),
            nn.BatchNorm1d(x * 4),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Sequential(
            nn.Conv1d(x * 4, x * 8, 3, padding=1),
            nn.BatchNorm1d(x * 8),
            nn.ReLU(),
            nn.Conv1d(x * 8, x * 8, 3, padding=1),
            nn.BatchNorm1d(x * 8),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Sequential(
            nn.Conv1d(x * 8, x * 16, 3, padding=1),
            nn.BatchNorm1d(x * 16),
            nn.ReLU(),
            nn.Conv1d(x * 16, x * 16, 3, padding=1),
            nn.BatchNorm1d(x * 16),
            nn.ReLU(),
        )

        # Deep supervision outputs
        self.level4 = nn.Conv1d(x * 16, out_channels, 1)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv6 = nn.Sequential(
            nn.Conv1d(x * 16 + x * 8, x * 8, 3, padding=1),
            nn.BatchNorm1d(x * 8),
            nn.ReLU(),
            nn.Conv1d(x * 8, x * 8, 3, padding=1),
            nn.BatchNorm1d(x * 8),
            nn.ReLU(),
        )
        self.level3 = nn.Conv1d(x * 8, out_channels, 1)

        self.up7 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv7 = nn.Sequential(
            nn.Conv1d(x * 8 + x * 4, x * 4, 3, padding=1),
            nn.BatchNorm1d(x * 4),
            nn.ReLU(),
            nn.Conv1d(x * 4, x * 4, 3, padding=1),
            nn.BatchNorm1d(x * 4),
            nn.ReLU(),
        )
        self.level2 = nn.Conv1d(x * 4, out_channels, 1)

        self.up8 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv8 = nn.Sequential(
            nn.Conv1d(x * 4 + x * 2, x * 2, 3, padding=1),
            nn.BatchNorm1d(x * 2),
            nn.ReLU(),
            nn.Conv1d(x * 2, x * 2, 3, padding=1),
            nn.BatchNorm1d(x * 2),
            nn.ReLU(),
        )
        self.level1 = nn.Conv1d(x * 2, out_channels, 1)

        self.up9 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv9 = nn.Sequential(
            nn.Conv1d(x * 2 + x, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU(),
            nn.Conv1d(x, x, 3, padding=1),
            nn.BatchNorm1d(x),
            nn.ReLU(),
        )
        self.out = nn.Conv1d(x, out_channels, 1)

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare input for UNetDS64 from unified batch structure.

        This method handles the unified input processing for UNetDS64, including:
        - Channel selection using src_idxs and src_mask
        - Shape formatting to [B, 1, L] for UNetDS64

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs

        Returns:
            torch.Tensor: Prepared input tensor of shape (batch_size, 1, signal_length)
        """
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("UNetDS64 extract_input expected Tensor from parent")

        if x.dim() != 3:
            raise ValueError(
                f"UNetDS64 expects input tensor of shape [B, C, L]; received {x.shape}."
            )
        if x.size(1) != 1:
            raise ValueError(
                "UNetDS64 is configured for single-source inputs. "
                "Ensure input_preprocessing['source'] defines a single vital."
            )
        return x

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass of the UNetDS64 model.

        Args:
            batch_dict: Unified batch dict from DataLoader collate_fn

        Returns:
            Dict[str, torch.Tensor]: Dictionary following the canonical model schema:
                - "predictions": Tuple of deep supervision outputs (out, level1, level2,
                    level3, level4)
                  where each element is a tensor of shape [B, out_channels, T]
                      representing
                  predictions at different decoder levels
                - "extras": Empty dictionary for auxiliary outputs
        """
        if not isinstance(batch_dict, dict):
            raise TypeError(
                "UNetDS64.forward expects a unified batch dictionary produced by the "
                f"collate pipeline; received {type(batch_dict).__name__}."
            )
        x = self.extract_input(batch_dict)

        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        level4 = self.level4(conv5)

        # Decoder
        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)
        level3 = self.level3(conv6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        level2 = self.level2(conv7)

        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        level1 = self.level1(conv8)

        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        out = self.out(conv9)

        return {
            "predictions": (out, level1, level2, level3, level4),
            "extras": {},
        }


class MultiResUNet1D(SingleStageModel):
    """1D MultiResUNet.

    Based on PPG2ABP paper (Ibtehaz et al. 2022, Section 3.3).

    Args:
        alpha (float): Weight multiplier for filter calculation (default: 2.5, paper
            value from Section 3.3)
        in_channels (int): number of channels in the input (default: 1)
        out_channels (int): number of output channels (default: 1)

    The canonical constructor argument for input channels is ``in_channels``.

    Note:
        This model assumes that the input pipeline (dataset + collate + direction)
        provides the correct number and ordering of input channels. It does not
        perform any additional channel selection, reordering, or filtering.
        Instead, the dataset and collate functions are the single source of truth
        for channel layout and selection, and `in_channels` is treated purely as
        an architectural parameter defining the expected input structure.
    """

    def __init__(
        self,
        alpha: float = 2.5,
        in_channels: int = 1,
        out_channels: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize MultiResUNet1D with configuration parameters.

        Args:
            alpha: Weight multiplier for filter calculation (default: 2.5, paper value
                from Section 3.3)
            in_channels: Number of channels in the input (default: 1)
            out_channels: Number of output channels (default: 1)
        """
        # MultiResUNet1D only supports single-direction training
        super().__init__(*args, **kwargs)

        # Store parameters
        self.alpha = alpha
        self.in_channels = in_channels

        def conv_bn(in_channels, out_channels, num_row=3, num_col=3, activation=True):
            # Match Keras conv2d_bn signature
            kernel_size = num_row  # For 1D, we use num_row
            padding = kernel_size // 2
            layers = [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
            ]
            if activation:
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        class TransConvBN(nn.Module):
            """Upsample block; BatchNorm is applied after concatenation in caller."""

            def __init__(self, in_channels: int, skip_channels: int) -> None:
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode="nearest")
                # BatchNorm after concatenation needs in_channels + skip_channels
                self.bn = nn.BatchNorm1d(in_channels + skip_channels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Upsample input tensor.

                Args:
                    x: Input tensor to upsample.

                Returns:
                    Upsampled tensor. BatchNorm is applied after concatenation
                    in the caller, not here.
                """
                x = self.up(x)
                return (
                    x  # Don't apply BatchNorm here, will be applied after concatenation
                )

        def multi_res_block(u, in_channels):
            w = int(self.alpha * u)

            out_3x3 = int(w * 0.167)
            out_5x5 = int(w * 0.333)
            out_7x7 = int(w * 0.5)
            concat_channels = out_3x3 + out_5x5 + out_7x7

            return nn.ModuleDict(
                {
                    "shortcut": conv_bn(
                        in_channels,
                        concat_channels,
                        num_row=1,
                        num_col=1,
                        activation=False,
                    ),
                    "conv3x3": conv_bn(in_channels, out_3x3, num_row=3, num_col=3),
                    "conv5x5": nn.Sequential(
                        conv_bn(out_3x3, out_5x5, num_row=3, num_col=3),
                        conv_bn(out_5x5, out_5x5, num_row=3, num_col=3),
                    ),
                    "conv7x7": nn.Sequential(
                        conv_bn(out_5x5, out_7x7, num_row=3, num_col=3),
                        conv_bn(out_7x7, out_7x7, num_row=3, num_col=3),
                        conv_bn(out_7x7, out_7x7, num_row=3, num_col=3),
                    ),
                    "bn_concat": nn.BatchNorm1d(concat_channels),
                    "final_bn": nn.BatchNorm1d(concat_channels),
                    "proj": nn.Conv1d(
                        concat_channels, u, 1
                    ),  # Project back to U channels
                }
            )

        def res_path(filters, length):
            layers = []
            for _i in range(length):
                layers.append(
                    nn.ModuleDict(
                        {
                            "shortcut": conv_bn(
                                filters, filters, num_row=1, num_col=1, activation=False
                            ),
                            "conv": conv_bn(filters, filters, num_row=3, num_col=3),
                            "bn": nn.BatchNorm1d(filters),
                        }
                    )
                )
            return nn.ModuleList(layers)

        # Encoder path
        self.mresblock1 = multi_res_block(32, in_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.respath1 = res_path(32, 4)

        self.mresblock2 = multi_res_block(64, 32)
        self.pool2 = nn.MaxPool1d(2)
        self.respath2 = res_path(64, 3)

        self.mresblock3 = multi_res_block(128, 64)
        self.pool3 = nn.MaxPool1d(2)
        self.respath3 = res_path(128, 2)

        self.mresblock4 = multi_res_block(256, 128)
        self.pool4 = nn.MaxPool1d(2)
        self.respath4 = res_path(256, 1)

        self.mresblock5 = multi_res_block(512, 256)

        # Decoder path with correct channel counts for BatchNorm
        self.up6 = TransConvBN(512, 256)  # mres5(512) + mres4(256) channels
        self.mresblock6 = multi_res_block(256, 512 + 256)

        self.up7 = TransConvBN(256, 128)  # mres6(256) + mres3(128) channels
        self.mresblock7 = multi_res_block(128, 256 + 128)

        self.up8 = TransConvBN(128, 64)  # mres7(128) + mres2(64) channels
        self.mresblock8 = multi_res_block(64, 128 + 64)

        self.up9 = TransConvBN(64, 32)  # mres8(64) + mres1(32) channels
        self.mresblock9 = multi_res_block(32, 64 + 32)

        self.conv10 = nn.Conv1d(32, out_channels, 1)

    def extract_input(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare input for MultiResUNet1D from the unified batch.

        This method delegates entirely to the base class
            `SingleStageModel.extract_input`,
        which is responsible for interpreting the dataset/collate/direction logic,
        validating required fields, and applying any source indexing or masking needed
        to construct the model input tensor.

        MultiResUNet1D does not reinterpret, reorder, or filter channels beyond what
        the input pipeline already specifies, to avoid introducing multiple sources
        of truth for channel layout. The dataset and collate functions are considered
        authoritative for how channels are arranged and which channels are included.

        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask, tgt_idxs.

        Returns:
            torch.Tensor: Prepared input tensor as produced by
                `SingleStageModel.extract_input`.
        """
        x = super().extract_input(batch_dict)
        if isinstance(x, dict):
            x = x["x"]
        elif isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            raise TypeError("MultiResUNet1D extract_input expected Tensor from parent")
        return x

    def forward(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass of the MultiResUNet1D model.

        Args:
            batch_dict: Unified batch dict from DataLoader collate_fn

        Returns:
            Dict[str, torch.Tensor]: Dictionary following the canonical model schema:
                - "predictions": Output waveform tensor of shape [B, out_channels, T]
                    where
                  B is batch size and T is the sequence length
                - "extras": Empty dictionary for auxiliary outputs
        """
        if not isinstance(batch_dict, dict):
            raise TypeError(
                "MultiResUNet1D.forward expects a unified batch dictionary produced by "
                f"the collate pipeline; received {type(batch_dict).__name__}."
            )
        x = self.extract_input(batch_dict)

        def mres_forward(block, x):
            shortcut = block["shortcut"](x)

            conv3x3 = block["conv3x3"](x)
            conv5x5 = block["conv5x5"](conv3x3)
            conv7x7 = block["conv7x7"](conv5x5)

            concat = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)  # Channel dimension
            concat = block["bn_concat"](concat)

            out = concat + shortcut
            out = torch.relu(out)
            out = block["final_bn"](out)
            out = block["proj"](out)  # Project to U channels before ResPath

            return out

        def respath_forward(path, x):
            out = x
            for block in path:
                shortcut = block["shortcut"](out)

                conv = block["conv"](out)

                out = conv + shortcut
                out = torch.relu(out)
                out = block["bn"](out)

            return out

        # Encoder path with direct ResPath assignment like Keras
        mres1 = mres_forward(self.mresblock1, x)
        pool1 = self.pool1(mres1)
        mres1 = respath_forward(self.respath1, mres1)  # Direct assignment

        mres2 = mres_forward(self.mresblock2, pool1)
        pool2 = self.pool2(mres2)
        mres2 = respath_forward(self.respath2, mres2)  # Direct assignment

        mres3 = mres_forward(self.mresblock3, pool2)
        pool3 = self.pool3(mres3)
        mres3 = respath_forward(self.respath3, mres3)  # Direct assignment

        mres4 = mres_forward(self.mresblock4, pool3)
        pool4 = self.pool4(mres4)
        mres4 = respath_forward(self.respath4, mres4)  # Direct assignment

        mres5 = mres_forward(self.mresblock5, pool4)

        # Decoder path matching Keras order
        up6 = self.up6.up(mres5)  # Just upsample
        merge6 = torch.cat([up6, mres4], dim=1)  # Concatenate
        merge6 = self.up6.bn(merge6)  # BatchNorm after concatenation
        mres6 = mres_forward(self.mresblock6, merge6)

        up7 = self.up7.up(mres6)
        merge7 = torch.cat([up7, mres3], dim=1)
        merge7 = self.up7.bn(merge7)
        mres7 = mres_forward(self.mresblock7, merge7)

        up8 = self.up8.up(mres7)
        merge8 = torch.cat([up8, mres2], dim=1)
        merge8 = self.up8.bn(merge8)
        mres8 = mres_forward(self.mresblock8, merge8)

        up9 = self.up9.up(mres8)
        merge9 = torch.cat([up9, mres1], dim=1)
        merge9 = self.up9.bn(merge9)
        mres9 = mres_forward(self.mresblock9, merge9)

        out = self.conv10(mres9)

        return {
            "predictions": out,
            "extras": {},
        }


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_unet_ds64", group="model", node=UNetDS64Config)
cs.store(name="base_multires_unet_1d", group="model", node=MultiResUNet1DConfig)
