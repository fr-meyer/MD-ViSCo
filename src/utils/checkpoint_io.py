"""Centralized checkpoint I/O abstraction for MD-ViSCo.

This module provides a unified interface for checkpoint loading and state
extraction covering the canonical v3 schema. Checkpoints created before v3
are intentionally unsupported—`CheckpointIO` does not attempt on-the-fly
conversion and instead raises clear errors so callers can retrain or
re-export with the modern pipeline. Security guidance (SafeTensors,
``weights_only`` for torch.load) is baked into the public API.

Canonical Checkpoint Schema:
    Single model:
        {
            'model_state_dict': {...},
            'optimizer_state_dict': {...},
            'scheduler_state_dict': {...},
            'early_stopping_state': {...},
            'epoch': int,
            'timestamp': str,
            'flags': {...},
            'best_loss': float
        }

    Multi-model GAN:
        {
            'model_G_state_dict': {...},  # Generator
            'model_D_state_dict': {...},  # Discriminator
            'optimizer_state_dict': {...} or {'optim_G': {...},
                'optim_D': {...}},
            ...
        }

    Multi-model cascade:
        {
            'model_state_dict': {
                'stage1_model': {...},
                'stage2_model': {...}
            },
            'optimizer_state_dict': {...},
            ...
        }

Security:
    - SafeTensors: Preferred format for weight-only artifacts (no arbitrary
      code execution)
    - weights_only=True: Used for PyTorch 2.5+ to reduce deserialization
      risk
    - safe_only mode: Enforce SafeTensors-only loading via config flag

Saving:
    - save() method: Symmetric API to load() for checkpoint persistence
    - Atomic writes: Temp file + rename pattern prevents corruption
    - Format support: PyTorch (.pt/.pth) for full checkpoints, SafeTensors
      (.safetensors) for tensors only
    - SafeTensors limitation: Only model weights can be saved (no metadata
      like epoch, optimizer)
    - safe_only enforcement: Prevents saving PyTorch format when
      safe_only=True

Future Extensions:
    - Sharded checkpoints: Currently raises NotImplementedError with guidance
    - Cloud storage: Can be added via path preprocessing
    - Streaming reads: For memory-constrained environments
    - Compression: Support for compressed checkpoint formats
"""

# Standard library imports
import contextlib
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import torch
from hydra.core.config_store import ConfigStore

# Local imports
from src.utils.constants import CHECKPOINT_EARLY_STOPPING_STATE_KEY
from src.utils.constants import CHECKPOINT_MODEL_STATE_KEY
from src.utils.constants import CHECKPOINT_OPTIMIZER_STATE_KEY
from src.utils.constants import CHECKPOINT_SCHEDULER_STATE_KEY
from src.utils.constants import DDP_MODULE_PREFIX

logger = logging.getLogger(__name__)


def remove_ddp_prefix_from_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = DDP_MODULE_PREFIX,
) -> dict[str, Any]:
    """Remove DDP 'module.' prefix from state dict keys.

    Args:
        state_dict: State dictionary potentially containing DDP prefixes
        prefix: Prefix to remove (default: 'module.')

    Returns:
        State dictionary with prefixes removed

    Examples:
        >>> state = {'module.layer1.weight': tensor(...)}
        >>> remove_ddp_prefix_from_state_dict(state)
        {'layer1.weight': tensor(...)}
    """
    if not any(k.startswith(prefix) for k in state_dict):
        return dict(state_dict)

    logger.debug(f"Removing '{prefix}' prefix from {len(state_dict)} keys")
    return {k.replace(prefix, "", 1): v for k, v in state_dict.items()}


@dataclass
class CheckpointIOConfig:
    """Configuration for CheckpointIO.

    Args:
        safe_only: If True, only SafeTensors files are allowed (raises
            ValueError for .pt/.pth)
    """

    _target_: str = "src.utils.checkpoint_io.CheckpointIO"
    safe_only: bool = False


class CheckpointIO:
    """Centralized checkpoint I/O handler for loading and extracting checkpoint states.

    This class provides a unified interface for:
    - Format detection (PyTorch .pt/.pth, SafeTensors .safetensors)
    - Secure loading (weights_only for PyTorch 2.5+, SafeTensors support)
    - Checkpoint saving (save() method with atomic writes)
    - Format-specific serialization (PyTorch full checkpoints, SafeTensors
      tensors only)
    - State extraction (model, optimizer, scheduler, early stopping)
    - Sharded checkpoint detection

    Args:
        safe_only: If True, only SafeTensors files are allowed (raises
            ValueError for .pt/.pth)

    Note:
        SafeTensors files are automatically wrapped in the canonical
        checkpoint schema. When loading a .safetensors file, the returned
        dictionary will be: {'model_state_dict': <state_dict>} to align with
        PyTorch checkpoint format.

    Example:
        >>> io = CheckpointIO()
        >>>
        >>> # Loading checkpoints
        >>> ckpt = io.load('checkpoint.pt')
        >>> model_state = io.extract_model_state(ckpt)
        >>> optimizer_state = io.extract_optimizer(ckpt)
        >>>
        >>> # Saving checkpoints
        >>> checkpoint = {'model_state_dict': model.state_dict(),
        ...               'epoch': 10}
        >>> io.save('checkpoint.pt', checkpoint)  # PyTorch format
        >>> io.save('model.safetensors', checkpoint,
        ...         use_safetensors=True)  # SafeTensors (tensors only)
    """

    def __init__(self, safe_only: bool = False) -> None:
        """Initialize CheckpointIO.

        Args:
            safe_only: If True, enforce SafeTensors-only loading for security
        """
        self.safe_only = safe_only

    def load(self, path: str, map_location: str = "cpu") -> dict[str, Any]:
        """Load a checkpoint from disk with automatic format detection.

        Supports:
        - PyTorch files (.pt, .pth): Uses torch.load() with weights_only=True
          for security
        - SafeTensors files (.safetensors): Uses
          safetensors.torch.load_file()

        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to (default: 'cpu')

        Returns:
            Dictionary containing checkpoint data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If safe_only=True but file is not SafeTensors
            NotImplementedError: If checkpoint is sharded (not yet supported)
            ImportError: If SafeTensors is required but not installed

        Example:
            >>> io = CheckpointIO()
            >>> ckpt = io.load('model.pt', map_location='cuda:0')
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # Detect sharded checkpoints
        if self._detect_sharded_checkpoint(path):
            raise NotImplementedError(
                "Sharded checkpoints not yet supported. Please use a "
                "single checkpoint file or implement sequential shard "
                "loading."
            )

        ext = path_obj.suffix.lower()

        if ext == ".safetensors":
            try:
                from safetensors.torch import load_file

                logger.info(f"Loading SafeTensors checkpoint: {path}")
                state_dict = load_file(path, device=map_location)
                # Wrap in canonical schema
                return {"model_state_dict": state_dict}
            except ImportError as e:
                raise ImportError(
                    "SafeTensors not installed. Run: pip install safetensors"
                ) from e

        elif ext in [".pt", ".pth"]:
            # Enforce safe_only mode
            if self.safe_only:
                raise ValueError(
                    f"safe_only mode is enabled, but file is not "
                    f"SafeTensors: {path}. Please convert to SafeTensors "
                    f"format or disable safe_only mode."
                )

            logger.info(f"Loading PyTorch checkpoint: {path}")

            # Try with weights_only=True for PyTorch 2.5+ security
            try:
                checkpoint = torch.load(
                    path, map_location=map_location, weights_only=True
                )
                logger.debug("Loaded checkpoint with weights_only=True")
                return checkpoint
            except (TypeError, AttributeError) as e:
                # Fallback for older PyTorch versions or checkpoints with
                # non-tensor objects
                logger.warning(
                    f"Failed to load with weights_only=True (PyTorch 2.5+ "
                    f"feature), falling back to standard load: {e}"
                )
                checkpoint = torch.load(path, map_location=map_location)
                return checkpoint

        else:
            raise ValueError(
                f"Unsupported checkpoint format: {ext}. "
                f"Supported formats: .pt, .pth, .safetensors"
            )

    def save(
        self,
        path: str,
        checkpoint_dict: dict[str, Any],
        use_safetensors: bool = False,
    ) -> None:
        """Save a checkpoint to disk with automatic format detection and atomic writes.

        Provides symmetric API to load() method. Handles format-specific
        serialization (PyTorch or SafeTensors), enforces safe_only mode,
        implements atomic writes using temp file + rename pattern, and
        provides proper error handling.

        Supported formats:
        - PyTorch (.pt, .pth): Full checkpoint with all metadata (epoch,
          optimizer, scheduler, etc.)
        - SafeTensors (.safetensors): Tensors only (model weights), metadata
          is lost

        Args:
            path: Path to save checkpoint file. Format is detected from
                extension.
            checkpoint_dict: Pre-assembled checkpoint dictionary from caller
                (trainer). For PyTorch format, can contain any picklable
                objects. For SafeTensors format, only tensors will be saved
                (metadata is lost).
            use_safetensors: Force SafeTensors format regardless of file
                extension (default: False)

        Raises:
            FileNotFoundError: If parent directory doesn't exist
            ValueError: If safe_only=True but attempting to save PyTorch
                format, or if file extension is unsupported
            ImportError: If SafeTensors is required but not installed
            TypeError: If SafeTensors format but checkpoint contains
                non-tensor data
            PermissionError: If insufficient permissions to write file
            OSError: If disk full or other I/O errors

        Note:
            - Atomic writes: Uses temp file + rename pattern to prevent
              corruption. If save fails mid-write, original file (if exists)
              remains unchanged.
            - SafeTensors limitation: Only model weights can be saved, not
              metadata. A warning is logged when metadata is present but will
              be lost.
            - Trainers should assemble checkpoint dictionary before calling
              save().
            - Compatible with all checkpoint schemas (single-model, GAN,
              cascade).

        Example:
            >>> io = CheckpointIO()
            >>>
            >>> # PyTorch format with full checkpoint
            >>> checkpoint = {
            ...     'model_state_dict': model.state_dict(),
            ...     'optimizer_state_dict': optimizer.state_dict(),
            ...     'epoch': 10,
            ...     'best_loss': 0.5
            ... }
            >>> io.save('checkpoint.pt', checkpoint)
            >>>
            >>> # SafeTensors format (tensors only)
            >>> io.save('model.safetensors', checkpoint)
            ... # Warning: metadata lost
            >>>
            >>> # Explicit SafeTensors format
            >>> io.save('model.safetensors', checkpoint,
            ...         use_safetensors=True)
            >>>
            >>> # safe_only mode enforcement
            >>> io_safe = CheckpointIO(safe_only=True)
            >>> io_safe.save('model.safetensors', checkpoint)  # OK
            >>> io_safe.save('model.pt', checkpoint)
            ... # Raises ValueError
        """
        path_obj = Path(path)
        parent_dir = path_obj.parent

        if not parent_dir.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

        # Detect format
        if use_safetensors:
            if path_obj.suffix.lower() != ".safetensors":
                raise ValueError(
                    f"use_safetensors=True requires a .safetensors "
                    f"extension, got: {path_obj.suffix}. Change the path "
                    f"to end with .safetensors."
                )
            format_type = "safetensors"
        else:
            ext = path_obj.suffix.lower()
            if ext == ".safetensors":
                format_type = "safetensors"
            elif ext in [".pt", ".pth"]:
                format_type = "pytorch"
            else:
                raise ValueError(
                    f"Unsupported checkpoint format: {ext}. "
                    f"Supported formats: .pt, .pth, .safetensors"
                )

        # Enforce safe_only mode
        if self.safe_only and format_type == "pytorch":
            raise ValueError(
                "safe_only mode is enabled, but attempting to save PyTorch "
                "format. Use SafeTensors format (.safetensors) or disable "
                "safe_only mode."
            )

        if format_type == "pytorch":
            self._save_pytorch(path, checkpoint_dict)
        else:  # safetensors
            self._save_safetensors(path, checkpoint_dict)

    def _save_pytorch(self, path: str, checkpoint_dict: dict[str, Any]) -> None:
        """Save checkpoint in PyTorch format with atomic write.

        Args:
            path: Target file path
            checkpoint_dict: Complete checkpoint dictionary
        """
        path_obj = Path(path)
        parent_dir = path_obj.parent

        logger.info(f"Saving PyTorch checkpoint: {path}")

        temp_path = None
        try:
            # Temp file in same dir so os.replace is atomic on same filesystem.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=parent_dir,
                prefix=".tmp_",
                suffix=".pt",
            ) as temp_file:
                temp_path = temp_file.name

            # Atomic write: save to temp then rename
            torch.save(checkpoint_dict, temp_path)

            # Atomic rename
            os.replace(temp_path, path)
            temp_path = None  # Successfully moved, don't clean up

            logger.info(f"Checkpoint saved successfully: {path}")

        except PermissionError as e:
            raise PermissionError(
                f"Permission denied when saving checkpoint: {path}"
            ) from e
        except OSError as e:
            raise OSError(f"I/O error when saving checkpoint: {path}") from e
        except Exception as e:
            logger.exception("Unexpected error when saving checkpoint: %s", path)
            raise RuntimeError(
                f"Unexpected error when saving checkpoint: {path}"
            ) from e
        finally:
            # Clean up temp file if it still exists
            if temp_path and os.path.exists(temp_path):
                with contextlib.suppress(Exception):
                    os.unlink(temp_path)

    def _save_safetensors(self, path: str, checkpoint_dict: dict[str, Any]) -> None:
        """Save checkpoint in SafeTensors format with atomic write.

        Only tensors are saved. Metadata (epoch, optimizer, scheduler,
        etc.) is lost.

        Args:
            path: Target file path
            checkpoint_dict: Complete checkpoint dictionary (tensors will be
                extracted)
        """
        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise ImportError(
                "SafeTensors not installed. Run: pip install safetensors"
            ) from e

        path_obj = Path(path)
        parent_dir = path_obj.parent

        # Extract tensors
        tensor_dict, has_metadata = self._extract_tensors_for_safetensors(
            checkpoint_dict
        )

        # Warn if metadata will be lost
        if has_metadata:
            logger.warning(
                "SafeTensors format only supports tensors. Metadata "
                "(epoch, optimizer, scheduler, etc.) will not be saved. "
                "Use PyTorch format (.pt/.pth) to save complete checkpoint."
            )

        logger.info(f"Saving SafeTensors checkpoint: {path}")

        temp_path = None
        try:
            # Temp file in same dir so os.replace is atomic on same filesystem.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=parent_dir,
                prefix=".tmp_",
                suffix=".safetensors",
            ) as temp_file:
                temp_path = temp_file.name

            # Atomic write: save to temp then rename
            save_file(tensor_dict, temp_path)

            # Atomic rename
            os.replace(temp_path, path)
            temp_path = None  # Successfully moved, don't clean up

            logger.info(f"Checkpoint saved successfully: {path}")

        except PermissionError as e:
            raise PermissionError(
                f"Permission denied when saving checkpoint: {path}"
            ) from e
        except OSError as e:
            raise OSError(f"I/O error when saving checkpoint: {path}") from e
        except Exception as e:
            logger.exception("Unexpected error when saving checkpoint: %s", path)
            raise RuntimeError(
                f"Unexpected error when saving checkpoint: {path}"
            ) from e
        finally:
            # Clean up temp file if it still exists
            if temp_path and os.path.exists(temp_path):
                with contextlib.suppress(Exception):
                    os.unlink(temp_path)

    def _flatten_tensors(
        self, d: dict[str, Any], prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Recursively flatten nested dictionaries of tensors with dot-separated keys.

        Args:
            d: Dictionary to flatten (may contain nested dicts or tensors)
            prefix: Current key prefix for nested structures

        Returns:
            Flattened dictionary with dot-separated keys

        Raises:
            TypeError: If a value is not a torch.Tensor or dict
        """
        out = {}
        for k, v in d.items():
            name = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                out.update(self._flatten_tensors(v, name))
            else:
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"Expected torch.Tensor but got {type(v)} for key '{name}'"
                    )
                out[name] = v
        return out

    def _extract_tensors_for_safetensors(
        self, checkpoint_dict: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Extract tensor-only data from checkpoint for SafeTensors format.

        Handles multiple checkpoint schemas:
        - Single-model: Extracts 'model_state_dict'
        - Cascade: Flattens nested 'model_state_dict' with prefixes
          (stage1_model., stage2_model.)
        - GAN: Flattens 'model_G_state_dict' and 'model_D_state_dict' with
          prefixes (G., D.)

        Args:
            checkpoint_dict: Complete checkpoint dictionary

        Returns:
            Tuple of (tensor_dict, has_metadata) where:
                - tensor_dict: Flattened dictionary of tensors only
                - has_metadata: True if non-tensor keys were present

        Raises:
            TypeError: If extracted values are not torch.Tensor instances
        """
        tensor_dict = {}
        has_metadata = False

        non_tensor_keys = []
        for key in checkpoint_dict:
            if key not in [
                "model_state_dict",
                "model_G_state_dict",
                "model_D_state_dict",
            ]:
                non_tensor_keys.append(key)

        if non_tensor_keys:
            has_metadata = True

        # Handle single-model format
        if "model_state_dict" in checkpoint_dict:
            model_state = checkpoint_dict["model_state_dict"]
            if isinstance(model_state, dict):
                # Flatten whether single-model or nested (cascade)
                tensor_dict.update(self._flatten_tensors(model_state))

        # Handle GAN format (flat structure)
        if "model_G_state_dict" in checkpoint_dict:
            model_g_state = checkpoint_dict["model_G_state_dict"]
            if isinstance(model_g_state, dict):
                tensor_dict.update(self._flatten_tensors(model_g_state, prefix="G"))

        if "model_D_state_dict" in checkpoint_dict:
            model_d_state = checkpoint_dict["model_D_state_dict"]
            if isinstance(model_d_state, dict):
                tensor_dict.update(self._flatten_tensors(model_d_state, prefix="D"))

        if not tensor_dict:
            raise ValueError(
                "No tensors found in checkpoint. Expected "
                "'model_state_dict', 'model_G_state_dict', or "
                "'model_D_state_dict' keys."
            )

        return tensor_dict, has_metadata

    def extract_model_state(
        self,
        ckpt: dict[str, Any],
        key: str = CHECKPOINT_MODEL_STATE_KEY,
    ) -> dict[str, torch.Tensor] | None:
        """Extract model state dictionary from checkpoint.

        When the requested key is absent, returns None so callers can fall
        back to checkpoint.get(key) or treat the entire checkpoint as a raw
        state dict.

        Args:
            ckpt: Checkpoint dictionary
            key: Key to extract (default: 'model_state_dict')

        Returns:
            Model state dictionary if found, None otherwise

        Example:
            >>> model_state = io.extract_model_state(ckpt)
            >>> model_state = io.extract_model_state(
            ...     ckpt, key='model_G_state_dict'
            ... )
        """
        logger.debug(f"Extracting model state with key: {key}")

        if key not in ckpt:
            logger.warning(
                f"Checkpoint is missing expected key '{key}'. Callers may "
                f"fall back to checkpoint.get(key) or treat the entire "
                f"checkpoint as a raw state dict."
            )
            return None

        return ckpt[key]

    def extract_multi_model_state(
        self,
        ckpt: dict[str, Any],
        keys: dict[str, str],
    ) -> dict[str, dict[str, torch.Tensor] | None]:
        """Extract multiple model state dictionaries from checkpoint.

        Handles both flat structure (GAN) and nested structure (cascade)
        within the canonical schema.

        Args:
            ckpt: Checkpoint dictionary
            keys: Mapping of model names to checkpoint keys
                e.g., {'stage1': 'stage1_model_state_dict',
                'stage2': 'model_state_dict'}

        Returns:
            Dictionary mapping model names to their state dictionaries

        Example:
            >>> states = io.extract_multi_model_state(ckpt, {
            ...     'generator': 'model_G_state_dict',
            ...     'discriminator': 'model_D_state_dict'
            ... })
        """
        logger.debug(f"Extracting multi-model states with keys: {keys}")

        result = {}

        for model_name, checkpoint_key in keys.items():
            # First try direct extraction
            try:
                state = self.extract_model_state(ckpt, key=checkpoint_key)
            except KeyError:
                state = None

            # If not found and we're looking for a nested structure
            if state is None and checkpoint_key != "model_state_dict":
                # Cascade format nests model state under model_state_dict.
                model_state_dict = ckpt.get("model_state_dict", {})
                if (
                    isinstance(model_state_dict, dict)
                    and checkpoint_key in model_state_dict
                ):
                    state = model_state_dict[checkpoint_key]
                    logger.debug(
                        f"Found nested model state at model_state_dict.{checkpoint_key}"
                    )
                # Also check with the model_name directly
                elif (
                    isinstance(model_state_dict, dict)
                    and model_name in model_state_dict
                ):
                    state = model_state_dict[model_name]
                    logger.debug(
                        f"Found nested model state at model_state_dict.{model_name}"
                    )

            result[model_name] = state

            if state is None:
                logger.warning(
                    f"Model state for '{model_name}' (key: "
                    f"'{checkpoint_key}') not found"
                )

        return result

    @staticmethod
    def _is_likely_state_dict(d: dict[str, Any]) -> bool:
        """Return True if d looks like a raw state dict (all values are tensors).

        Used to decide whether to treat the full checkpoint as state dict.
        """
        if not d or not isinstance(d, dict):
            return False
        return all(isinstance(v, torch.Tensor) for v in d.values())

    @staticmethod
    def _strip_ddp_prefix(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remove DDP 'module.' prefix from checkpoint keys if present.

        Args:
            state_dict: State dictionary potentially containing
                DDP-prefixed keys.

        Returns:
            Dict with normalized keys (prefix removed) or original dict if
            no prefix found.
        """
        if not state_dict:
            return state_dict
        return remove_ddp_prefix_from_state_dict(state_dict, prefix=DDP_MODULE_PREFIX)

    def load_state_dict_from_path(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        *,
        map_location: str = "cpu",
        key: str = "model_state_dict",
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a model's state dict directly from a checkpoint path.

        Args:
            model: Model instance to load weights into.
            checkpoint_path: Path to checkpoint file on disk.
            map_location: Device mapping for torch.load / safetensors load.
            key: Checkpoint key containing the model state dict.
            strict: Whether to enforce strict key matching in
                load_state_dict().

        Returns:
            The full checkpoint dictionary that was loaded from disk.
        """
        checkpoint = self.load(checkpoint_path, map_location=map_location)
        state_dict = self.extract_model_state(checkpoint, key=key)
        if state_dict is None:
            state_dict = checkpoint.get(key)
        if state_dict is None and self._is_likely_state_dict(checkpoint):
            # Accept raw state dict payloads when callers persisted tensors
            # directly.
            state_dict = checkpoint
        if state_dict is None:
            raise KeyError(
                f"Checkpoint has no key '{key}' and is not a raw state "
                f"dict. Provide a canonical checkpoint with "
                f"model_state_dict or a raw state dict file."
            )
        state_dict = self._strip_ddp_prefix(state_dict)
        model.load_state_dict(state_dict, strict=strict)
        return checkpoint

    def load_components(
        self,
        checkpoint_path: str,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        key_map: dict[str, str] | None = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load model/optimizer/scheduler from checkpoint path into instances.

        Args:
            checkpoint_path: Path to checkpoint file on disk.
            model: Model instance to load weights into (optional).
            optimizer: Optimizer instance to restore (optional).
            scheduler: Scheduler instance to restore (optional).
            key_map: Optional override for checkpoint keys. Defaults to
                canonical schema.
            map_location: Device mapping for torch.load / safetensors load.
            strict: Whether to enforce strict key matching when loading
                model weights.

        Returns:
            The full checkpoint dictionary that was loaded.
        """
        checkpoint = self.load(checkpoint_path, map_location=map_location)
        keys = key_map or {
            "model": "model_state_dict",
            "optimizer": "optimizer_state_dict",
            "scheduler": "scheduler_state_dict",
        }

        if model is not None:
            model_key = keys.get("model", "model_state_dict")
            model_state = self.extract_model_state(checkpoint, key=model_key)
            if model_state is None:
                model_state = checkpoint.get(model_key)
            if model_state is None and self._is_likely_state_dict(checkpoint):
                # Accept raw state dict payloads when callers persisted
                # tensors directly.
                model_state = checkpoint
            if model_state is None:
                logger.warning(
                    "Model state not found in checkpoint while attempting "
                    f"to restore weights (expected key '{model_key}')."
                )
            else:
                model_state = self._strip_ddp_prefix(model_state)
                model.load_state_dict(model_state, strict=strict)

        if optimizer is not None:
            optimizer_state = self.extract_optimizer(
                checkpoint, key=keys.get("optimizer", "optimizer_state_dict")
            )
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            else:
                logger.warning(
                    "Optimizer state not found in checkpoint while "
                    f"attempting to restore optimizer (expected key "
                    f"'{keys.get('optimizer', 'optimizer_state_dict')}')."
                )

        if scheduler is not None:
            scheduler_state = self.extract_scheduler(
                checkpoint, key=keys.get("scheduler", "scheduler_state_dict")
            )
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            else:
                logger.info(
                    "No scheduler state found in checkpoint during restore "
                    f"(expected key "
                    f"'{keys.get('scheduler', 'scheduler_state_dict')}')."
                )

        return checkpoint

    def extract_optimizer(
        self,
        ckpt: dict[str, Any],
        key: str = CHECKPOINT_OPTIMIZER_STATE_KEY,
    ) -> dict[str, Any] | None:
        """Extract optimizer state dictionary from checkpoint.

        Applies legacy key adapter if requested key not found.

        Args:
            ckpt: Checkpoint dictionary
            key: Key to extract (default: 'optimizer_state_dict')

        Returns:
            Optimizer state dictionary if found, None otherwise

        Example:
            >>> optimizer_state = io.extract_optimizer(ckpt)
        """
        logger.debug(f"Extracting optimizer state with key: {key}")

        if key in ckpt:
            logger.debug(f"Found optimizer state at key: {key}")
            return ckpt[key]

        logger.debug(f"Optimizer state key '{key}' not found in checkpoint")
        return None

    def extract_scheduler(
        self,
        ckpt: dict[str, Any],
        key: str = CHECKPOINT_SCHEDULER_STATE_KEY,
    ) -> dict[str, Any] | None:
        """Extract scheduler state dictionary from checkpoint.

        Args:
            ckpt: Checkpoint dictionary
            key: Key to extract (default: 'scheduler_state_dict')

        Returns:
            Scheduler state dictionary if found, None otherwise

        Example:
            >>> scheduler_state = io.extract_scheduler(ckpt)
        """
        logger.debug(f"Extracting scheduler state with key: {key}")

        if key in ckpt:
            logger.debug(f"Found scheduler state at key: {key}")
            return ckpt[key]

        logger.debug(f"Scheduler state key '{key}' not found in checkpoint")
        return None

    def extract_early_stopping(
        self,
        ckpt: dict[str, Any],
        key: str = CHECKPOINT_EARLY_STOPPING_STATE_KEY,
    ) -> dict[str, Any] | None:
        """Extract early stopping state dictionary from checkpoint.

        Args:
            ckpt: Checkpoint dictionary
            key: Key to extract (default: 'early_stopping_state')

        Returns:
            Early stopping state dictionary if found, None otherwise

        Example:
            >>> early_stopping_state = io.extract_early_stopping(ckpt)
        """
        logger.debug(f"Extracting early stopping state with key: {key}")

        if key in ckpt:
            logger.debug(f"Found early stopping state at key: {key}")
            return ckpt[key]

        logger.debug(f"Early stopping state key '{key}' not found in checkpoint")
        return None

    def _detect_sharded_checkpoint(self, path: str) -> bool:
        """Detect if checkpoint is sharded (not yet supported).

        Args:
            path: Path to checkpoint file

        Returns:
            True if checkpoint appears to be sharded, False otherwise
        """
        path_obj = Path(path)

        if path_obj.name.endswith(".index.json"):
            return True

        index_path = path_obj.with_suffix(".index.json")
        if index_path.exists():
            return True

        appended_index = Path(str(path_obj) + ".index.json")
        return appended_index.exists()


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_checkpoint_io",
    node=CheckpointIOConfig,
    group="checkpoint_io",
)
