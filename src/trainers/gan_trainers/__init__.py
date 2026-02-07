# Import all GAN trainers
from .gan_trainer import GANTrainer
from .approximation_gan_trainer import ApproximationGANTrainer
from .refinement_gan_trainer import RefinementGANTrainer

__all__ = [
    'GANTrainer',
    'ApproximationGANTrainer', 
    'RefinementGANTrainer'
]
