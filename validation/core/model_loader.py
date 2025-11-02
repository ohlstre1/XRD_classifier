"""
Core model loading functionality for XRD diffusion validation.
"""

import torch
from pathlib import Path
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add diffusion to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "diffusion"))

from diffusion.models.complete_model import DiffusionAugmentor
from diffusion.diffusion.process import DiffusionProcess
from diffusion.training.config import TrainingConfig


class ModelLoader:
    """
    Handles loading of trained diffusion models and related components.
    """

    def __init__(self, device: str = 'auto'):
        """
        Initialize model loader.

        Args:
            device: Device to load model to ('auto', 'cuda', 'cpu')
        """
        self.device = self._get_device(device)
        self.config = TrainingConfig()

    def _get_device(self, device: str) -> str:
        """Get the appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def load_diffusion_model(self, model_path: str = "diffusion/models/xrd_diffusion/best_model.pth") -> DiffusionAugmentor:
        """
        Load the trained diffusion model.

        Args:
            model_path: Path to the model checkpoint

        Returns:
            Loaded DiffusionAugmentor model
        """
        print("Loading trained diffusion model...")

        # Initialize model architecture
        model = DiffusionAugmentor(
            in_channels=1,
            hidden_channels=self.config.hidden_channels,
            time_embedding_dim=self.config.time_embedding_dim,
            num_res_blocks=self.config.num_res_blocks,
            attention_levels=self.config.attention_levels,
            num_levels=self.config.num_levels,
            temperature_condition=True
        ).to(self.device)

        # Load trained weights
        model_path = Path(model_path)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {model_path}")
            print(f"  Trained for {checkpoint['epoch']} epochs")
            print(f"  Best validation loss: {checkpoint['val_loss']:.6f}")
        except FileNotFoundError:
            print(f"⚠️  Model file not found at {model_path}")
            print("   Will use randomly initialized model for demonstration")
        except KeyError as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("   Will use randomly initialized model")

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def load_diffusion_process(self) -> DiffusionProcess:
        """
        Load the diffusion process.

        Returns:
            Configured DiffusionProcess
        """
        diffusion = DiffusionProcess(
            num_timesteps=self.config.num_timesteps,
            schedule_type='cosine',
            device=self.device
        )

        print(f"Diffusion timesteps: {diffusion.num_timesteps}")

        return diffusion

    def load_complete_setup(self, model_path: str = "diffusion/models/xrd_diffusion/best_model.pth") -> Tuple[DiffusionAugmentor, DiffusionProcess]:
        """
        Load both model and diffusion process.

        Args:
            model_path: Path to the model checkpoint

        Returns:
            Tuple of (model, diffusion_process)
        """
        model = self.load_diffusion_model(model_path)
        diffusion = self.load_diffusion_process()

        return model, diffusion

    def get_model_info(self, model: DiffusionAugmentor) -> dict:
        """
        Get information about the loaded model.

        Args:
            model: Loaded model

        Returns:
            Dictionary with model information
        """
        return {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': next(model.parameters()).device,
            'hidden_channels': self.config.hidden_channels,
            'time_embedding_dim': self.config.time_embedding_dim,
            'num_res_blocks': self.config.num_res_blocks,
            'attention_levels': self.config.attention_levels,
            'num_levels': self.config.num_levels
        }


def load_diffusion_setup(model_path: str = "diffusion/models/xrd_diffusion/best_model.pth",
                        device: str = 'auto') -> Tuple[DiffusionAugmentor, DiffusionProcess, ModelLoader]:
    """
    Convenience function to load complete diffusion setup.

    Args:
        model_path: Path to model checkpoint
        device: Device to use

    Returns:
        Tuple of (model, diffusion_process, model_loader)
    """
    loader = ModelLoader(device)
    model, diffusion = loader.load_complete_setup(model_path)

    print(f"Using device: {loader.device}")
    print(f"PyTorch version: {torch.__version__}")

    return model, diffusion, loader