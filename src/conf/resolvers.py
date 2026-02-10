"""Custom OmegaConf resolvers for MD-ViSCo Hydra configuration.

This module provides custom resolvers that extend OmegaConf's interpolation capabilities
for Hydra configuration files. Resolvers enable dynamic value computation based on
other configuration values.

Resolvers must be registered before Hydra composes configs that reference them.
Call register_resolvers() at module level in every Hydra entry point (train.py,
test.py) that may load configs using these resolvers.

Available Resolvers
-------------------
encoder_output_size(input_size: int) -> int
    Computes CNNEncoder output dimension based on input length.
    Formula: 256 * (input_length / 8)
    Used by: af_classifier_cascade.yaml for stage2_model.input_length
    Example: ${encoder_output_size:1024} resolves to 32768

Registration
------------
Resolvers are registered automatically when register_resolvers() is called.
This function is idempotent and safe to call multiple times.

Usage in Entry Points
---------------------
In train.py and test.py, call register_resolvers() at module level before
Hydra decorators:

    from src.conf.resolvers import register_resolvers
    register_resolvers()

    @hydra.main(...)
    def main(cfg: Config):
        ...

Usage in Config Files
---------------------
Reference resolvers in YAML configs using OmegaConf interpolation syntax:

    stage2_model:
      input_size: ${encoder_output_size:1024}
"""

# Third-party imports
from omegaconf import OmegaConf


def register_resolvers() -> None:
    """Register custom OmegaConf resolvers for Hydra configuration.

    This function registers all custom resolvers used by MD-ViSCo configuration files.
    It must be called before Hydra composes any configs that reference these resolvers.

    The function is idempotent (safe to call multiple times) and uses replace=True
    to allow re-registration during testing or module reloading.

    Examples:
        Register resolvers at module level in entry points:

        >>> from src.conf.resolvers import register_resolvers
        >>> register_resolvers()

        Use resolvers in YAML configs:

        >>> # In af_classifier_cascade.yaml:
        >>> # stage2_model.input_length: ${encoder_output_size:1024}
    """
    OmegaConf.register_new_resolver(
        "encoder_output_size",
        lambda input_size: int(256 * (input_size / 8)),
        replace=True,
        use_cache=True,
    )
