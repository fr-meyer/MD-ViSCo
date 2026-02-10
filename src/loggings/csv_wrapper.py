"""CSV logging wrapper with configurable overwrite and run-level state.

Main classes: CSVWrapperConfig, CSVWrapper. See README and src/conf/ for configuration.

See Also:
    src.conf.csv_wrapper: Hydra config for CSVWrapper.
    README: Usage and configuration overview.
"""

# Standard library imports
import csv
import logging
import os
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)


def _get_config_defaults() -> "CSVWrapperConfig":
    """Return default config instance (single source of truth for defaults)."""
    return CSVWrapperConfig()


@dataclass
class CSVWrapperConfig:
    """Configuration for CSVWrapper with unified behavior control.

    Single source of truth for default values; CSVWrapper derives from this.
    """

    _target_: str = "src.loggings.csv_wrapper.CSVWrapper"
    output_dir: str = "./results"
    flush_every_n: int = 100
    filename_template: str = (
        "{dataset}_{method}_{direction}_{seed}_{result_type}_results.csv"
    )
    subdirectory: str = "evaluation_results"
    overwrite: bool = False
    use_timestamp: bool = False


class CSVWrapper:
    """Unified CSV wrapper with configurable overwrite behavior.

    This class handles CSV file operations with support for:
    - Configurable overwrite/append behavior
    - Run-level state management
    - Timestamped or deterministic filenames

    Instantiation:
        Objects are instantiated via Hydra's automatic instantiation system
        using the _target_ field in configuration files. Constructor
        defaults are derived from CSVWrapperConfig (single source of truth).
    """

    # Configuration attributes (set from CSVWrapperConfig)
    output_dir: str
    flush_every_n: int
    filename_template: str
    subdirectory: str
    overwrite: bool
    use_timestamp: bool

    # Run-level state tracking
    _active_files: dict[str, Any]
    _buffers: dict[str, list[dict[str, Any]]]
    _fieldnames: dict[str, set[str]]
    _file_created: dict[str, bool]
    _run_started: dict[str, bool]
    _overwrite_done: dict[str, bool]
    _header_fieldnames: dict[str, list[str]]
    _metrics_path_kwargs: dict[str, dict[str, Any]]
    _metrics_schema_version: dict[str, int]
    _active_metrics_path: dict[str, str]

    def __init__(self, config: CSVWrapperConfig | None = None, **kwargs):
        """Initialize from config or kwargs; config is the single source of truth.

        Args:
            config: Optional config instance. If None, CSVWrapperConfig() is used.
            **kwargs: Override any config field (e.g. output_dir, flush_every_n,
                filename_template, subdirectory, overwrite, use_timestamp).
        """
        base = config if config is not None else _get_config_defaults()
        for f in dataclass_fields(CSVWrapperConfig):
            if f.name.startswith("_"):
                continue
            setattr(self, f.name, kwargs.get(f.name, getattr(base, f.name)))

        # Run-level state tracking
        self._active_files: dict[str, Any] = {}
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._fieldnames: dict[str, set[str]] = {}
        self._file_created: dict[str, bool] = {}
        self._run_started: dict[str, bool] = {}
        self._overwrite_done: dict[str, bool] = {}
        self._header_fieldnames: dict[str, list[str]] = {}  # schema written in header
        self._metrics_path_kwargs: dict[
            str, dict[str, Any]
        ] = {}  # template kwargs per file_key
        self._metrics_schema_version: dict[
            str, int
        ] = {}  # rotation when schema changes
        self._active_metrics_path: dict[
            str, str
        ] = {}  # path of current metrics file per file_key

        self._validate_config()
        logger.debug(f"Initialized CSVWrapper: {self}")

    def __repr__(self) -> str:
        """Return string representation for logging and debugging."""
        return (
            f"<CSVWrapper(output_dir='{self.output_dir}', "
            f"flush_every_n={self.flush_every_n})>"
        )

    def _validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.flush_every_n <= 0:
            raise ValueError("flush_every_n must be positive")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        file_key: str = "default",
        is_rank0: bool = False,
        dataset: str | None = None,
        method: str | None = None,
        direction: str | None = None,
        seed: int | None = None,
        result_type: str | None = None,
    ):
        """Unified metrics logging with run-level overwrite control.

        Path is built from configured filename_template, subdirectory, and use_timestamp
            (same
        config surface as CSVWrapperConfig). Optional
            dataset/method/direction/seed/result_type are
        stored on first call per file_key and used to format the template; when absent,
            defaults
        (e.g. dataset=file_key, result_type='metrics') are used so progress-bar and
            other callers
        respect the same config.
        """
        if not is_rank0:
            return

        if file_key not in self._run_started or not self._run_started[file_key]:
            self.start_new_run(file_key)

        # Store template kwargs on first use so path is consistent for this file_key
        if file_key not in self._metrics_path_kwargs:
            self._metrics_path_kwargs[file_key] = {
                "dataset": dataset if dataset is not None else file_key,
                "method": method if method is not None else "",
                "direction": direction if direction is not None else "",
                "seed": seed if seed is not None else 0,
                "result_type": result_type if result_type is not None else "metrics",
            }

        if file_key not in self._buffers:
            self._buffers[file_key] = []
            self._fieldnames[file_key] = set()
            self._file_created[file_key] = False

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step or 0,
            **metrics,
        }
        self._fieldnames[file_key].update(log_entry.keys())
        self._buffers[file_key].append(log_entry)

        if len(self._buffers[file_key]) >= self.flush_every_n:
            self._flush_buffer(file_key)

    def generate_filepath(
        self, dataset: str, method: str, direction: str, seed: int, result_type: str
    ) -> str:
        """Generate filepath with optional timestamp support.

        Args:
            dataset: Dataset name
            method: Model method name
            direction: Direction string (e.g., 'PPG2ECG')
            seed: Seed value
            result_type: Type of results (e.g., 'waveform')

        Returns:
            str: Full filepath for the CSV file
        """
        # Add timestamp if requested
        if self.use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            template = self.filename_template.replace(".csv", f"_{timestamp}.csv")
        else:
            template = self.filename_template

        filename = template.format(
            dataset=dataset,
            method=method,
            direction=direction,
            seed=seed,
            result_type=result_type,
        )
        return os.path.join(self.output_dir, self.subdirectory, filename)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        filepath: str,
        file_key: str = "default",
        is_rank0: bool = False,
    ) -> None:
        """Save DataFrame with run-level overwrite control."""
        if not is_rank0:
            return

        # Auto-start run if not started
        if file_key not in self._run_started or not self._run_started[file_key]:
            self.start_new_run(file_key)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Determine write mode and header
        if self.overwrite and not self._overwrite_done.get(file_key, False):
            # First write in this run - overwrite mode
            mode = "w"
            write_header = True
            self._overwrite_done[file_key] = True
            logger.info(f"First write for {file_key} (overwrite mode)")
        else:
            # Subsequent writes - append mode
            mode = "a"
            write_header = not os.path.exists(filepath)
            logger.info(f"Subsequent write for {file_key} (append mode)")

        df.to_csv(filepath, mode=mode, index=False, header=write_header)
        logger.info(f"Saved CSV to {filepath} (mode: {mode}, header: {write_header})")

    def _get_metrics_filepath(
        self, file_key: str, schema_version: int | None = None
    ) -> str:
        """Build metrics filepath from filename_template, subdirectory, use_timestamp.

        Uses _metrics_path_kwargs[file_key] for template placeholders; when absent,
            defaults (e.g.
        dataset=file_key, result_type='metrics') are used so progress-bar path respects
            the same
        config surface as CSVWrapperConfig.
        """
        kwargs = self._metrics_path_kwargs.get(file_key) or {
            "dataset": file_key,
            "method": "",
            "direction": "",
            "seed": 0,
            "result_type": "metrics",
        }
        if self.use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            template = self.filename_template.replace(".csv", f"_{timestamp}.csv")
        else:
            template = self.filename_template
        filename = template.format(**kwargs)
        if schema_version is not None and schema_version > 0:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_v{schema_version}{ext}"
        return os.path.join(self.output_dir, self.subdirectory, filename)

    def _flush_buffer(self, file_key: str):
        """Flush buffered data to CSV using configured path and overwrite semantics.

        Detects when fieldnames change after the initial header write and rotates to a
            new file so
        the header matches the current schema before writing more rows.
        """
        if file_key not in self._buffers or not self._buffers[file_key]:
            return

        current_fieldnames = sorted(self._fieldnames[file_key])

        # Schema changed after header was written: rotate to new file so header
        # matches schema
        if (
            file_key in self._header_fieldnames
            and self._header_fieldnames[file_key] != current_fieldnames
        ):
            if file_key in self._active_files:
                try:
                    self._active_files[file_key].close()
                except Exception as e:
                    logger.warning(f"Error closing file handle for {file_key}: {e}")
                del self._active_files[file_key]
            if file_key in self._active_metrics_path:
                del self._active_metrics_path[file_key]
            self._metrics_schema_version[file_key] = (
                self._metrics_schema_version.get(file_key, 0) + 1
            )
            self._header_fieldnames.pop(file_key, None)

        if file_key not in self._active_files:
            schema_version = self._metrics_schema_version.get(file_key, 0)
            filepath = self._get_metrics_filepath(
                file_key, schema_version=schema_version if schema_version > 0 else None
            )
            self._active_metrics_path[file_key] = filepath
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if self.overwrite and not self._overwrite_done.get(file_key, False):
                mode = "w"
                self._overwrite_done[file_key] = True
            else:
                mode = "a"
            # File kept open for incremental writes until flush/close; noqa SIM115
            self._active_files[file_key] = open(filepath, mode, newline="")  # noqa: SIM115

        writer = csv.DictWriter(
            self._active_files[file_key], fieldnames=current_fieldnames
        )

        # Write header only when schema is first established or after rotation
        if file_key not in self._header_fieldnames:
            writer.writeheader()
            self._header_fieldnames[file_key] = current_fieldnames

        writer.writerows(self._buffers[file_key])
        self._active_files[file_key].flush()
        self._buffers[file_key] = []

    def finish(self):
        """Flush all buffers and close files."""
        exceptions = []

        # Flush all buffers with error collection
        for file_key in list(self._buffers.keys()):
            try:
                self._flush_buffer(file_key)
            except Exception as e:
                logger.warning(f"Error flushing buffer for {file_key}: {e}")
                exceptions.append(e)

        # Close all files with error collection
        for file_key, file_handle in list(self._active_files.items()):
            try:
                file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing file for {file_key}: {e}")
                exceptions.append(e)

        # Clear state
        self._active_files.clear()

        # Re-raise first exception if any occurred
        if exceptions:
            raise exceptions[0]

    def start_new_run(self, file_key: str = "default"):
        """Start a new run and clear existing file if overwrite is enabled.

        Args:
            file_key: Key to identify the file/run
        """
        if self.overwrite and file_key in self._overwrite_done:
            # Reset run state for new run
            self._overwrite_done[file_key] = False
            self._run_started[file_key] = False

        # Clear any existing file if overwrite is enabled; first flush will use 'w'
        if self.overwrite:
            self._clear_file_if_exists(file_key)
            self._overwrite_done[file_key] = False  # first flush truncates
            self._run_started[file_key] = True
            logger.info(f"Started new run for {file_key} (overwrite mode)")
        else:
            self._run_started[file_key] = True
            logger.info(f"Started new run for {file_key} (append mode)")

    def _clear_file_if_exists(self, file_key: str):
        """Clear existing metrics file for overwrite mode using configured path."""
        if file_key in self._active_files:
            try:
                self._active_files[file_key].close()
            except Exception as e:
                logger.warning(f"Error closing file handle for {file_key}: {e}")
            finally:
                del self._active_files[file_key]
        self._active_metrics_path.pop(file_key, None)
        self._header_fieldnames.pop(file_key, None)
        self._metrics_schema_version[file_key] = 0

        filepath = self._get_metrics_filepath(file_key)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleared existing file for overwrite: {filepath}")
            except OSError as e:
                logger.warning(f"Could not remove file {filepath}: {e}")

    def _generate_filepath_for_key(self, file_key: str) -> str:
        """Generate metrics filepath for file_key (same config as CSVWrapperConfig)."""
        return self._get_metrics_filepath(file_key)

    def finish_run(self, file_key: str = "default"):
        """Finish current run and cleanup.

        Args:
            file_key: Key to identify the file/run
        """
        if file_key in self._active_files:
            self._flush_buffer(file_key)
            try:
                self._active_files[file_key].close()
            except Exception as e:
                logger.warning(f"Error closing file handle for {file_key}: {e}")
            finally:
                del self._active_files[file_key]
        self._active_metrics_path.pop(file_key, None)

        self._run_started[file_key] = False
        self._overwrite_done[file_key] = False
        logger.info(f"Finished run for {file_key}")


# Register with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_csv_wrapper", node=CSVWrapperConfig, group="csv_wrapper")
