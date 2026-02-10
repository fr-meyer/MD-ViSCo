"""Shared utilities for module discovery and import.

This module provides a robust, package-agnostic way to discover and import
submodules recursively, with proper error handling and logging. Used by
processors, datasets, criterions, trainers, and models for ConfigStore
registration via import_processors(), import_datasets(), etc.

Functions:
    - import_modules: Import all submodules of a package for ConfigStore
      registration
    - _iter_submodules: Yield fully-qualified submodule names (internal)

Examples:
    >>> import_modules("src.trainers", module_type="trainer")
    >>> import_modules("src.model", module_type="model")

See Also:
    - src.processors: import_processors, import_extractors
    - src.dataset: import_datasets
    - src.criterions: import_criterions
"""

from __future__ import annotations

# Standard library imports
import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


def _iter_submodules(package_name: str) -> Iterable[str]:
    """Yield fully-qualified submodule names under this package (recursive).

    Args:
        package_name: The package to scan (e.g., "src.trainers")

    Yields:
        Fully qualified module names (e.g.,
        "src.trainers.waveform_reconstruction_trainer")
    """
    try:
        pkg = importlib.import_module(package_name)
        if not hasattr(pkg, "__path__"):
            return  # not a package
        for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_name}."):
            name = m.name
            # Skip dunders and our own __init__
            if name.rsplit(".", 1)[-1].startswith("_"):
                continue
            yield name
    except ImportError as e:
        log.warning(f"Could not import package {package_name}: {e}")
        return


def import_modules(package_name: str, module_type: str = "modules") -> int:
    """Import all submodules of a package to trigger ConfigStore registration.

    Args:
        package_name: The package to scan (e.g., "src.trainers")
        module_type: Human-readable name for logging (e.g., "trainers",
            "models", "criterions")

    Returns:
        Number of successfully imported modules

    Example:
        >>> import_modules("src.trainers", module_type="trainer")
        >>> import_modules("src.model", module_type="model")
    """
    log.info(f"Importing {module_type} from package: {package_name}")
    imported_count = 0
    failed_count = 0

    for name in _iter_submodules(package_name):
        try:
            importlib.import_module(name)
            log.debug("Imported %s module: %s", module_type, name)
            imported_count += 1
        except Exception as e:
            # Keep going; one bad module shouldn't block others
            log.warning("Could not import %s module %s: %s", module_type, name, e)
            failed_count += 1

    log.info(
        f"Successfully imported {imported_count} {module_type} modules"
        + (f" ({failed_count} failed)" if failed_count > 0 else "")
    )

    return imported_count
