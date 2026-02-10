"""Configuration package for MD-ViSCo project.

This package contains all configuration classes and utilities for the MD-ViSCo project,
including the main Config class, custom resolvers, and dataset-specific configurations.

Hydra Configuration System
-------------------------
This package integrates with Hydra's configuration composition system:

1. **Config Registration**: The main Config dataclass is registered with Hydra's
   ConfigStore as "base_config" at module import time. This enables Hydra to
   validate and compose configuration groups.

2. **Resolver Registration**: Custom OmegaConf resolvers (defined in resolvers.py)
   must be registered before Hydra composes configs. Call register_resolvers() at
   module level in entry points (train.py, test.py).

3. **Config Group Registration**: Config groups (early_stopping, optimizers) are
   registered via imports that trigger ConfigStore registration in their respective
   modules. This allows Hydra to resolve defaults like base_early_stopping, base_adam.

Composition Flow
----------------
1. Entry point imports this module (triggers Config registration)
2. Entry point calls register_resolvers() (enables custom resolvers)
3. Entry point imports config groups (triggers group registration)
4. Hydra decorator composes configs using registered schemas and resolvers
5. Hydra instantiates Config with composed configuration groups

Example Entry Point Setup
--------------------------
    from src.conf.config import Config
    from src.conf.resolvers import register_resolvers

    register_resolvers()  # Must be called before @hydra.main

    @hydra.main(config_path="conf", config_name="config")
    def main(cfg: Config):
        ...

See Also
--------
- resolvers.py : Custom OmegaConf resolvers for dynamic config values
- config.py : Main Config dataclass definition
- README.md : Configuration section for usage and options
"""

# Third-party imports
from hydra.core.config_store import ConfigStore

# Local imports (side-effect imports for ConfigStore registration)
from src.optimizers import adam  # noqa: F401  # ConfigStore registration for base_adam

# Register config-group schemas so Hydra can resolve defaults
# (e.g. base_early_stopping, base_adam, base_sgd). These imports trigger
# ConfigStore registration in their respective modules.
from src.utils import train_utils  # noqa: F401  # EarlyStoppingConfig registration

from .config import Config

# Register with Hydra ConfigStore
# Register config at module import time (consistent with dataset configs).
# This enables Hydra to resolve "base_config" as the root config schema.
if __name__ != "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)  # Generic name, not matching YAML file

__all__ = ["Config"]
