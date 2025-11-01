"""
Training modules for the diffusion model.
"""

from .trainer import train_model
from .config import TrainingConfig

__all__ = ['train_model', 'TrainingConfig']