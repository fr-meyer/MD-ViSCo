"""Core components of the MD-ViSCo framework.

register_core() is called during app startup (e.g. from the main entry
point) to register with Hydra's ConfigStore: directions (source→target
configs), vital dataset metadata, and single-direction configs.
"""

__all__: list[str] = ["register_core"]


def register_core():
    """Register core components with Hydra's ConfigStore.

    Registers directions, vital dataset metadata, and single-direction
    configs. Call once at startup before composing config.
    """
    from src.core.direction import register_directions
    from src.core.domain import register_direction
    from src.dataset.base_dataset import register_vital

    register_directions()
    register_vital()
    register_direction()
