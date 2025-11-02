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
    weight_decay: float = 5e-6

    # Data parameters
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Paths
    save_path: str = "./models/xrd_diffusion"

    # Checkpointing
    save_every_n_epochs: int = 10
    keep_top_k_models: int = 3
    auto_resume: bool = True

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "xrd-diffusion"
    wandb_entity: str = None  # Use default entity
    wandb_run_name: str = None  # Auto-generate if None

    def __post_init__(self):
        if self.attention_levels is None:
            self.attention_levels = [1, 2]