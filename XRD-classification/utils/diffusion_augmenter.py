#!/usr/bin/env python3
"""
Diffusion XRD Pattern Augmentation Wrapper
==========================================

This module provides a wrapper for the diffusion-based XRD pattern augmentation
system, integrating with the existing diffusion model infrastructure.

Features:
- Compatible interface with DualXRDAugmenter
- Uses trained diffusion models from the diffusion directory
- Automatic path resolution and error handling
- Fallback mechanisms for robust operation
"""

import torch
import numpy as np
import os
import sys
import warnings
from typing import Dict, Any, Tuple, Optional


class DiffusionXRDAugmenter:
    """
    Wrapper for diffusion-based XRD pattern augmentation.

    Integrates with the existing diffusion model infrastructure while
    providing a compatible interface for the dual augmentation system.
    """

    def __init__(self, model_path: str, device: str = 'auto', verbose: bool = True):
        """
        Initialize the diffusion XRD augmenter.

        Args:
            model_path: Path to the trained diffusion model
            device: Device to run on ('auto', 'cpu', 'cuda')
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.model_path = model_path
        self.model = None
        self.diffusion_process = None
        self.is_available = False

        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.verbose:
            print(f"🖥️  Diffusion augmenter using device: {self.device}")

        # Try to initialize the diffusion system
        self._initialize_diffusion_system()

    def _initialize_diffusion_system(self):
        """Initialize the diffusion model and process."""
        try:
            # Add diffusion modules to path
            diffusion_base = '/home/bert_25/XRD_calssifier/diffusion'
            if diffusion_base not in sys.path:
                sys.path.insert(0, diffusion_base)

            # Import diffusion components
            from diffusion.process import DiffusionProcess

            # Import model directly
            sys.path.append(os.path.join(diffusion_base, 'models'))
            from complete_model import ImprovedDiffusionDenoiser

            # Check if model file exists
            if not os.path.exists(self.model_path):
                if self.verbose:
                    print(f"⚠️  Diffusion model not found at {self.model_path}")
                return

            # Initialize diffusion process
            self.diffusion_process = DiffusionProcess(
                num_timesteps=1000,
                schedule_type='cosine',
                device=self.device
            )

            # Initialize model architecture
            self.model = ImprovedDiffusionDenoiser(
                in_channels=1,
                hidden_channels=16,
                time_embedding_dim=256,
                num_res_blocks=2,
                attention_levels=[1, 2],
                num_levels=2,
                temperature_condition=True
            ).to(self.device)

            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.is_available = True

            if self.verbose:
                epoch = checkpoint.get('epoch', 'Unknown')
                loss = checkpoint.get('loss', 'Unknown')
                print(f"✅ Diffusion augmenter initialized successfully!")
                print(f"   Model loaded from epoch {epoch} with loss {loss}")

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to initialize diffusion augmenter: {e}")
            self.is_available = False

    def generate_temperature_conditions(self, num_samples: int, temp_range: tuple = (0.1, 2.0),
                                      temp_mode: str = 'random') -> torch.Tensor:
        """
        Generate temperature conditioning values.

        Args:
            num_samples: Number of temperature values to generate
            temp_range: Temperature range (min, max)
            temp_mode: Temperature sampling mode ('random', 'linear', 'high', 'low')

        Returns:
            Temperature tensor
        """
        temp_min, temp_max = temp_range

        if temp_mode == 'random':
            temps = torch.rand(num_samples) * (temp_max - temp_min) + temp_min
        elif temp_mode == 'linear':
            temps = torch.linspace(temp_min, temp_max, num_samples)
        elif temp_mode == 'high':
            temps = torch.full((num_samples,), temp_max)
        elif temp_mode == 'low':
            temps = torch.full((num_samples,), temp_min)
        else:
            raise ValueError(f"Unknown temp_mode: {temp_mode}")

        return temps.to(self.device)

    def augment_pattern(self, synth_pattern: torch.Tensor, num_samples: int = 5,
                       temp_range: tuple = (0.1, 2.0), temp_mode: str = 'random',
                       noise_timestep_range: tuple = (0, 50), base_seed: Optional[int] = None,
                       return_metadata: bool = False) -> torch.Tensor:
        """
        Augment a single synthetic XRD pattern using diffusion model.

        Args:
            synth_pattern: Input synthetic pattern [1, L] or [1, 1, L]
            num_samples: Number of augmented samples to generate
            temp_range: Temperature conditioning range
            temp_mode: Temperature sampling mode
            noise_timestep_range: Range of diffusion timesteps for noise addition
            base_seed: Base random seed (None for random)
            return_metadata: Whether to return augmentation metadata

        Returns:
            Augmented patterns [num_samples, 1, L]
            Metadata dict (if return_metadata=True)
        """
        if not self.is_available:
            raise RuntimeError("Diffusion augmenter is not available")

        # Ensure proper input shape
        if synth_pattern.dim() == 2:
            synth_pattern = synth_pattern.unsqueeze(1)  # [1, 1, L]
        elif synth_pattern.dim() == 1:
            synth_pattern = synth_pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, L]

        synth_pattern = synth_pattern.to(self.device)

        # Generate conditions
        temperatures = self.generate_temperature_conditions(num_samples, temp_range, temp_mode)

        # Generate random seeds if base_seed provided
        if base_seed is not None:
            seeds = [base_seed + i for i in range(num_samples)]
        else:
            seeds = [None] * num_samples

        # Storage for results
        augmented_patterns = []
        metadata = {
            'temperatures': temperatures.cpu().numpy(),
            'seeds': seeds,
            'timesteps': [],
            'noise_levels': []
        }

        self.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                # Set seed for this sample
                if seeds[i] is not None:
                    torch.manual_seed(seeds[i])
                    np.random.seed(seeds[i])

                # Random timestep for noise addition
                timestep = torch.randint(
                    noise_timestep_range[0],
                    noise_timestep_range[1] + 1,
                    (1,),
                    device=self.device
                )

                # Get temperature for this sample
                temp = temperatures[i:i+1]

                # Apply augmentation process
                # Direct noise prediction and application
                noise_pred = self.model(synth_pattern, timestep, temp)

                # Add predicted noise to create realistic experimental pattern
                noise_scale = 0.1 + (timestep.float() / 1000.0) * 0.3  # Scale noise by timestep
                augmented = synth_pattern + noise_pred * noise_scale

                # Clamp to reasonable values
                augmented = torch.clamp(augmented, 0, None)

                augmented_patterns.append(augmented)

                # Store metadata
                metadata['timesteps'].append(timestep.item())
                metadata['noise_levels'].append(noise_scale.item())

        # Combine results
        augmented_patterns = torch.cat(augmented_patterns, dim=0)

        if return_metadata:
            return augmented_patterns, metadata
        else:
            return augmented_patterns

    def augment_batch(self, patterns: torch.Tensor, samples_per_pattern: int,
                     **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Augment a batch of patterns.

        Args:
            patterns: Batch of XRD patterns [batch_size, L] or [batch_size, 1, L]
            samples_per_pattern: Number of augmented samples per input pattern
            **kwargs: Additional arguments for augment_pattern

        Returns:
            Tuple of (augmented_batch, metadata)
        """
        if not self.is_available:
            raise RuntimeError("Diffusion augmenter is not available")

        batch_size = patterns.shape[0]
        all_augmented = []

        for i in range(batch_size):
            pattern = patterns[i]
            augmented = self.augment_pattern(pattern, samples_per_pattern, **kwargs)
            all_augmented.append(augmented)

        # Combine all augmented patterns
        final_augmented = torch.cat(all_augmented, dim=0)  # [batch_size * samples_per_pattern, 1, L]

        metadata = {
            'original_batch_size': batch_size,
            'samples_per_pattern': samples_per_pattern,
            'total_samples': final_augmented.shape[0],
            'augmentation_methods': ['diffusion'] * final_augmented.shape[0]
        }

        return final_augmented, metadata

    def get_available_methods(self) -> list:
        """Get list of available augmentation methods."""
        if self.is_available:
            return ['diffusion']
        else:
            return []


def test_diffusion_augmenter():
    """Test function for the diffusion augmentation system."""
    print("Testing Diffusion XRD Augmenter...")

    # Use the actual model path
    model_path = "/home/bert_25/XRD_calssifier/diffusion/models/xrd_diffusion/improved_diffusion_model_best.pth"

    # Create augmenter
    try:
        augmenter = DiffusionXRDAugmenter(model_path, device='auto', verbose=True)

        if augmenter.is_available:
            # Test with synthetic pattern
            test_pattern = torch.randn(4500)  # Synthetic XRD pattern

            print("Testing single pattern augmentation...")
            augmented = augmenter.augment_pattern(test_pattern, num_samples=3)

            print(f"Original pattern shape: {test_pattern.shape}")
            print(f"Augmented patterns shape: {augmented.shape}")
            print(f"Available methods: {augmenter.get_available_methods()}")
            print("✅ Diffusion augmentation test completed successfully!")
        else:
            print("⚠️  Diffusion augmentation is not available")

    except Exception as e:
        print(f"❌ Diffusion augmentation test failed: {e}")


if __name__ == "__main__":
    test_diffusion_augmenter()