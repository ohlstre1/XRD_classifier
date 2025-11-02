"""
Training configuration for XRD diffusion model.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    # Model parameters
    num_timesteps: int = 1000
    hidden_channels: int = 16
    time_embedding_dim: int = 256
    num_res_blocks: int = 2
    attention_levels: List[int] = field(default_factory=lambda: [1, 2])
    num_levels: int = 2

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Data parameters
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Paths
    save_path: str = "./models/xrd_diffusion"

    def __post_init__(self):
        if self.attention_levels is None:
            self.attention_levels = [1, 2]