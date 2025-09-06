from dataclasses import dataclass
from omegaconf import MISSING
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    """Base configuration for all models"""
    _target_: str = MISSING
    supports_multi_directional: bool = False
    model_name: str = MISSING # Used for model identification (e.g., 'nabnet', 'ppg2abp', 'patchtst', 'mdvisco')
    
    # Common input length parameter - used by all models
    input_length: Optional[int] = None
    dictionary_key: str = MISSING
    """Length of input signal sequences. This is a common parameter used by all models:
    - MDViSCo: Used for UNet-Swin Transformer architecture sizing
    - NABNet: Used for feature extraction calculations  
    - PatchTST: Used for transformer context length configuration
    """

class BaseModel(nn.Module):
    """Base class for all models in MD-ViSCo project"""
    
    def __init__(self, supports_multi_directional: bool = False, model_name: str = MISSING, input_length: int = MISSING, dictionary_key: str = MISSING, *args, **kwargs):
        super().__init__()
        self._supports_multi_directional = supports_multi_directional
        self._model_name = model_name # Model identifier
        self._input_length = input_length
        self._dictionary_key = dictionary_key
        self.vital2chan = None
        self.vital_names = None
        self.num_channels = None

    @property
    def model_name(self) -> str:
        """Get the model name, handling DDP wrapping automatically"""
        return self._model_name
    
    @property
    def supports_multi_directional(self) -> bool:
        """Get multi-directional support flag, handling DDP wrapping automatically"""
        return self._supports_multi_directional
    
    @property
    def input_length(self) -> int:
        """Get input length parameter, handling DDP wrapping automatically"""
        return self._input_length

    def set_layout(self, vitals_dataset):
        """Cache vital-to-channel mapping for efficient lookup
        
        Args:
            vitals_dataset: VitalsDataset instance that provides vital-to-channel mapping
        """
        
        vital_names = [v.name for v in vitals_dataset.vitals()]
        channel_indices = [vitals_dataset.chan(v) for v in vitals_dataset.vitals()]
        
        self.vital2chan = torch.tensor(channel_indices)
        self.vital_names = vital_names
        self.num_channels = len(vital_names)
        
        logger.info(f"Model layout set: {dict(zip(vital_names, channel_indices))}")

    def extract_input(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and prepare input for models using unified batch structure.
        
        Common functionality for all models that need source extraction.
        
        Args:
            batch_dict: Unified batch dict with src_idxs, src_mask
            
        Returns:
            torch.Tensor: Prepared input tensor [B, S_max, T] with inactive sources zeroed
        """
        # Validate required fields
        required_fields = ["x", "src_idxs", "src_mask"]
        missing_fields = [field for field in required_fields if field not in batch_dict]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}. Expected unified batch structure.")

        # Extract waveform source
        waveform = batch_dict["x"]
        device = waveform.device
        B, C, T = waveform.shape

        # ---- indices/masks (types & device) ----
        src_idxs = batch_dict["src_idxs"].to(device=device, dtype=torch.long)  # [B, S_max]
        src_mask = batch_dict["src_mask"].to(device=device, dtype=torch.bool)  # [B, S_max]

        S_max = src_idxs.size(1)

        # ---- source extraction (fixed [B, S_max, T]) ----
        idxs_3d = src_idxs.unsqueeze(-1).expand(B, S_max, T)                   # [B, S_max, T]
        x_gathered = waveform.gather(dim=1, index=idxs_3d)                     # [B, S_max, T]
        x_real = x_gathered * src_mask.unsqueeze(-1)                           # [B, S_max, T]

        return x_real
