"""Checkpoint path and directory management.

This module builds checkpoint paths and directories; load/save logic is
delegated to :class:`src.utils.checkpoint_io.CheckpointIO`. Supports
build mode (placeholders like {epoch}, {seed}) and concat mode for
subpath + filename construction.

Classes:
    - CheckpointManagerConfig: Hydra-compatible config for CheckpointManager
    - CheckpointManager: Manages checkpoint paths and naming

Examples:
    >>> manager = CheckpointManager(base_dir="./weights/", ...)
    >>> path = manager.build_path(key='save', epoch=100, direction='ppg2abp', ...)

See Also:
    - src.utils.checkpoint_io: CheckpointIO for load/save
    - README.md: "Checkpoint Management" section for build vs concat mode details
"""

# Standard library imports
import glob
import logging
import os
from dataclasses import dataclass
from typing import Any

# Third-party imports
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)


@dataclass
class CheckpointManagerConfig:
    """Configuration for :class:`CheckpointManager` with Hydra-compatible defaults.

    Attributes:
        _target_: Full path to CheckpointManager for Hydra instantiation.
        base_dir: Root directory for checkpoint files (e.g. "./weights/").
        file_ext: Checkpoint file extension (e.g. ".pt").
        path_format: Format string for subfolder path; placeholders like
            {trainer_name}, {dataset_name}, {model_name}, {batch_size}, etc.
        filename_format: Format string for checkpoint filename; placeholders
            like {direction}, {seed}, {epoch_suffix}, {file_ext}.
    """

    _target_: str = "src.utils.checkpoint_manager.CheckpointManager"
    base_dir: str = "./weights/"
    file_ext: str = ".pt"
    path_format: str = (
        "{trainer_name}/{dataset_name}/{trainer_name}_{model_name}_BS_{batch_size}_"
        "E_{num_epochs}_LR_{learning_rate}{optional_params}_P_{scheduler_patience}"
        "_ES_{early_stopping_patience}{extra_flags}"
    )
    filename_format: str = "{direction}checkpoint_S_{seed}{epoch_suffix}{file_ext}"


class CheckpointManager:
    """Manages checkpoint paths and naming; load/save logic lives in CheckpointIO.

    Responsibility: build paths via kwargs-based format substitution. Supports
    **build mode** (placeholders like {epoch}, {seed}) and **concat mode**
    (literal subpath + filename). Inputs: key, epoch, direction, seed, overwrite,
    plus any kwargs required by path_format/filename_format. Outputs: full path.
    For key='save', raises ValueError if file exists and overwrite=False.

    Example:
        path = manager.build_path(
            key='save', epoch=100, direction='ppg2abp', seed=42,
            model_name='nabnet', trainer_name='approximation', dataset_name='uci',
            batch_size=32, num_epochs=100, learning_rate=0.001,
            scheduler_patience=10, early_stopping_patience=20, **opts
        )

    See Also:
        README.md "Checkpoint Management" for build vs concat mode details and examples.
    """

    def __init__(
        self,
        base_dir: str = "./weights/",
        file_ext: str = ".pt",
        path_format: str = (
            "{trainer_name}/{dataset_name}/{trainer_name}_{model_name}_BS_"
            "{batch_size}_E_{num_epochs}_LR_{learning_rate}{optional_params}_P_"
            "{scheduler_patience}_ES_{early_stopping_patience}{extra_flags}"
        ),
        filename_format: str = "{direction}checkpoint_S_{seed}{epoch_suffix}{file_ext}",
    ):
        """Initialize with path management parameters only.

        Args:
            base_dir: Base directory for checkpoints
            file_ext: File extension for checkpoint files
            path_format: Format string for subfolder path construction
            filename_format: Format string for filename construction
        """
        self.base_dir = base_dir
        self.file_ext = file_ext
        self.path_format = path_format
        self.filename_format = filename_format
        self._validate_config()
        logger.debug(f"Initialized CheckpointManager: {self}")

    def __repr__(self) -> str:
        """Return string representation for logging and debugging."""
        return (
            f"<CheckpointManager(base_dir='{self.base_dir}', "
            f"file_ext='{self.file_ext}')>"
        )

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.base_dir:
            raise ValueError("base_dir cannot be empty")
        if not self.file_ext:
            raise ValueError("file_ext cannot be empty")

    def build_path(
        self,
        key: str | None = None,
        epoch: int | None = None,
        direction: str = "",
        seed: int = 0,
        overwrite: bool = False,
        **kwargs,
    ) -> str:
        """Build checkpoint path with kwargs-based format substitution (uniform logic).

        Kwargs should include all parameters needed for format substitution.
        For 'save' operations, pass full config (including train_ratio if needed).
        For loading, pass minimal params (epoch, direction, seed, and train_ratio
        only if format requires it). Path building is uniform; directory creation
        and existence checks handled by caller.

        Args:
            key: Optional checkpoint key (e.g., 'save', 'load', 'stage1', 'stage2')
            epoch: Epoch number. If ``None``, results in a suffix-free filename
                (used for best-checkpoint loading). If an integer, includes
                ``_epoch_{epoch}`` suffix.
            direction: Direction for multi-directional training
            seed: Seed for reproducibility
            overwrite: If True, allow overwriting existing checkpoint for
                'save' key. If False (default), raise ValueError if checkpoint
                exists.
            **kwargs: All parameters needed for format substitution (e.g., model_name,
                trainer_name, dataset_name, batch_size, num_epochs, learning_rate,
                scheduler_patience, early_stopping_patience, is_finetuning,
                use_patient_split, use_patient_information, use_wcl, train_ratio, etc.)

        Returns:
            str: Full checkpoint path

        Raises:
            KeyError: If format strings reference parameters that don't exist in kwargs
            ValueError: If key is 'save', checkpoint exists, and overwrite is False
        """
        params = self._extract_all_parameters(
            epoch=epoch, direction=direction, seed=seed, **kwargs
        )
        subfolder = self._build_subfolder(params)
        filename = self._build_filename(params)
        full_path = os.path.join(params["base_dir"], subfolder, filename)

        if key == "save" and os.path.exists(full_path) and not overwrite:
            raise ValueError(
                f"Checkpoint exists at {full_path} and overwrite is False. "
                f"Set overwrite=True to overwrite existing checkpoint."
            )

        logger.debug(f"Built checkpoint path: {full_path}")
        return full_path

    def find_checkpoint(
        self,
        key: str | None = None,
        epoch: int | None = None,
        direction: str = "",
        seed: int = 0,
        **kwargs,
    ) -> str | None:
        """Find checkpoint using kwargs-based format substitution.

        Kwargs should include all parameters needed for format substitution.
        For loading, pass minimal params (epoch, direction, seed, and train_ratio
        only if format requires it).

        Args:
            key: Optional checkpoint key. If 'save', uses glob search.
                Otherwise, uses direct path check.
            epoch: Epoch number. If ``None``, results in a suffix-free filename
                (used for best-checkpoint loading). If an integer, includes
                ``_epoch_{epoch}`` suffix.
            direction: Direction for multi-directional training
            seed: Seed for reproducibility
            **kwargs: All parameters needed for format substitution (e.g., model_name,
                trainer_name, dataset_name, batch_size, num_epochs, learning_rate,
                scheduler_patience, early_stopping_patience, is_finetuning,
                use_patient_split, use_patient_information, use_wcl, train_ratio, etc.)

        Returns:
            Optional[str]: Checkpoint path if found, None otherwise
        """
        if key == "save":
            try:
                params = self._extract_all_parameters(
                    epoch=epoch, direction=direction, seed=seed, **kwargs
                )
                subfolder = self._build_subfolder(params)
                format_params = params.copy()
                if format_params.get("direction") is not None:
                    format_params["direction"] += (
                        "_" if format_params["direction"] != "" else ""
                    )
                epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""
                format_params["epoch_suffix"] = epoch_suffix
                format_params["file_ext"] = params["file_ext"]
                filename_pattern = self.filename_format.format(**format_params)
                search_path = os.path.join(
                    params["base_dir"], subfolder, filename_pattern
                )
                matches = glob.glob(search_path)
                if matches:
                    return (
                        matches[0]
                        if len(matches) == 1
                        else max(matches, key=os.path.getmtime)
                    )
            except (KeyError, ValueError) as e:
                logger.debug(f"Failed to build path for glob search: {e}")
        else:
            try:
                candidate_path = self.build_path(
                    key=key,
                    epoch=epoch,
                    direction=direction,
                    seed=seed,
                    overwrite=False,
                    **kwargs,
                )
                return candidate_path if os.path.exists(candidate_path) else None
            except (KeyError, ValueError) as e:
                logger.debug(f"Failed to build path: {e}")

        return None

    def _extract_all_parameters(
        self, epoch: int | None, direction: str, seed: int, **kwargs
    ) -> dict[str, Any]:
        """Extract parameters for path construction without business logic validation.

        Args:
            epoch: Epoch number (optional)
            direction: Direction for multi-directional training
            seed: Seed for reproducibility
            **kwargs: All other parameters needed for format substitution

        Returns:
            Dict containing base_dir, file_ext, epoch, direction, seed, and all kwargs
        """
        params = {
            "base_dir": self.base_dir,
            "file_ext": self.file_ext,
            "epoch": epoch,
            "direction": direction,
            "seed": seed,
            **kwargs,
        }

        return params

    def _build_subfolder(self, params: dict[str, Any]) -> str:
        """Build subfolder using configurable format.

        Args:
            params: Complete parameter dictionary from _extract_all_parameters

        Returns:
            str: Subfolder path with finetuning/patient split suffixes

        Raises:
            KeyError: If path_format references a parameter that doesn't exist in params
        """
        format_params = params.copy()
        optional_params = ""
        extra_flags = ""

        model_name = format_params.get("model_name")
        if model_name is not None:
            if model_name == "BPModel":
                optional_params += f"_WCL_{format_params.get('use_wcl', False)}"
                extra_flags += "_BP_NORM"
            if model_name in ["BPModel", "PatchTST"]:
                optional_params += (
                    f"_PI_{format_params.get('use_patient_information', False)}"
                )

        if format_params.get("is_finetuning", False):
            extra_flags += "_finetuning"

        if format_params.get("use_patient_split", False):
            extra_flags += "_Patient_Split"

        format_params["optional_params"] = optional_params
        format_params["extra_flags"] = extra_flags

        # Omit _P_{scheduler_patience} segment when None for cleaner paths
        segment = "_P_{scheduler_patience}"
        path_format = self.path_format
        if format_params.get("scheduler_patience") is None:
            path_format = path_format.replace(segment, "")

        try:
            subfolder = path_format.format(**format_params)
        except KeyError as e:
            raise KeyError(
                f"Missing parameter '{e}' for path format. Available: {
                    list(format_params.keys())
                }"
            ) from e

        return subfolder

    def _build_filename(self, params: dict[str, Any]) -> str:
        """Build filename using configurable format.

        Args:
            params: Complete parameter dictionary from _extract_all_parameters

        Returns:
            str: Filename

        Raises:
            KeyError: If filename_format references a parameter that doesn't exist
                in params
        """
        format_params = params.copy()
        if params.get("direction") is not None:
            format_params["direction"] += "_" if params["direction"] != "" else ""

        epoch_suffix = ""
        if params.get("epoch") is not None:
            epoch_suffix = f"_epoch_{params['epoch']}"

        format_params["epoch_suffix"] = epoch_suffix
        format_params["file_ext"] = params["file_ext"]

        try:
            return self.filename_format.format(**format_params)
        except KeyError as e:
            raise KeyError(
                f"Missing parameter '{e}' for filename format. Available: {
                    list(format_params.keys())
                }"
            ) from e


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(
    name="base_checkpoint_manager",
    node=CheckpointManagerConfig,
    group="checkpoint_manager",
)
