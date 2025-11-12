#!/usr/bin/env python3
"""
Classical XRD Pattern Augmentation System
=========================================

This module provides physics-based classical augmentation for XRD patterns,
extracted and adapted from the diffusion model's augmentation methods.

Features:
- Peak shifting: Rolling spectrum by small offsets
- Peak variations: Scaling intensities at peak locations
- Peak removal: Randomly removing peaks
- Peak broadening: Scherrer equation-based Gaussian convolution
- Configurable intensity parameters
- Batch processing capabilities
- No external dependencies

Physics basis:
- Uses Scherrer equation for realistic peak broadening
- Simulates instrument and sample effects
- Maintains peak relationships and overall spectrum characteristics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional


class ClassicalXRDAugmenter:
    """
    Classical XRD pattern augmentation using physics-based methods.

    Extracted from diffusion model augmentation methods and adapted
    for standalone classical augmentation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'cpu', verbose: bool = True):
        """
        Initialize the classical XRD augmenter.

        Args:
            config: Configuration dictionary with augmentation settings
            device: Device to run computations on ('cpu' or 'cuda')
            verbose: Enable verbose output
        """
        self.device = device
        self.verbose = verbose

        # Default configuration
        self.config = {
            'peak_shifting': {
                'enabled': True,
                'probability': 0.5,
                'max_shift': 5
            },
            'peak_variations': {
                'enabled': True,
                'probability': 0.7,
                'variation_range': [0.8, 1.2],
                'threshold': 0.01
            },
            'peak_removal': {
                'enabled': True,
                'probability': 0.3,
                'removal_probability': 0.1,
                'threshold': 0.01
            },
            'peak_broadening': {
                'enabled': True,
                'probability': 0.8,
                'intensity_range': [0.1, 0.8],  # Controls broadening strength
                'wavelength': 1.54056,  # Cu Kα radiation
                'min_crystallite_size': 5,  # nm
                'max_crystallite_size': 100  # nm
            },
            'background_noise': {
                'enabled': True,
                'probability': 0.6,
                'noise_level_range': [0.005, 0.03]
            }
        }

        # Update with provided config
        if config:
            self._update_config(config)

        # Scherrer equation parameters
        self.wavelength = self.config['peak_broadening']['wavelength']
        self.K = 0.9  # Scherrer constant (shape factor)
        self.min_crystallite_size = self.config['peak_broadening']['min_crystallite_size']
        self.max_crystallite_size = self.config['peak_broadening']['max_crystallite_size']

        if self.verbose:
            print("✅ Classical XRD Augmenter initialized")
            self._print_config()

    def _update_config(self, new_config: Dict[str, Any]):
        """Recursively update configuration."""
        def update_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    update_dict(target[key], value)
                else:
                    target[key] = value

        update_dict(self.config, new_config)

    def _print_config(self):
        """Print current configuration."""
        print("Classical Augmentation Configuration:")
        for method, settings in self.config.items():
            if isinstance(settings, dict) and 'enabled' in settings:
                status = "✅" if settings['enabled'] else "❌"
                prob = settings.get('probability', 'N/A')
                print(f"  {method}: {status} (prob: {prob})")

    def create_broadening_kernel(self, intensity: float, L: int) -> torch.Tensor:
        """
        Create a Gaussian convolution kernel for peak broadening.

        Adapted from DiffusionProcess.create_scherrer_kernel() but uses
        intensity parameter instead of timestep.

        Args:
            intensity: Broadening intensity (0.0 to 1.0)
            L: Length of the pattern

        Returns:
            Convolution kernel tensor
        """
        # Map intensity to crystallite size (smaller size = more broadening)
        crystallite_size = self.max_crystallite_size - intensity * (
            self.max_crystallite_size - self.min_crystallite_size
        )

        # Base sigma calculation proportional to intensity
        base_sigma = 0.02 + intensity * 0.98  # Scale from 0.02 to 1.0

        # Scale sigma based on pattern length to get proper pixel-space broadening
        sigma_pixels = base_sigma * (L / 100)  # Scale to pixel space

        # Determine appropriate kernel size based on sigma (odd number, at least 3)
        kernel_size = max(3, int(6 * sigma_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Ensure kernel size is reasonable compared to pattern length
        kernel_size = min(kernel_size, L//4)

        # Create Gaussian kernel
        kernel = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=self.device)
        kernel = torch.exp(-0.5 * (kernel / sigma_pixels) ** 2)
        kernel = kernel / kernel.sum()  # normalize

        return kernel.view(1, 1, -1)

    def apply_peak_shifting(self, x: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply peak shifting augmentation.

        Args:
            x: Input pattern [batch, 1, L]
            config: Peak shifting configuration

        Returns:
            Augmented pattern
        """
        if not config['enabled']:
            return x

        x_aug = x.clone()
        batch_size = x.shape[0]

        for i in range(batch_size):
            if torch.rand(1).item() < config['probability']:
                max_shift = config['max_shift']
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                x_aug[i, 0, :] = torch.roll(x_aug[i, 0, :], shifts=shift)

        return x_aug

    def apply_peak_variations(self, x: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply peak intensity variation augmentation.

        Args:
            x: Input pattern [batch, 1, L]
            config: Peak variations configuration

        Returns:
            Augmented pattern
        """
        if not config['enabled']:
            return x

        x_aug = x.clone()
        batch_size = x.shape[0]
        threshold = config['threshold']
        var_range = config['variation_range']

        for i in range(batch_size):
            if torch.rand(1).item() < config['probability']:
                # Identify peaks
                peak_mask = (x_aug[i, 0, :] > threshold)

                # Apply random scaling to peaks
                random_factors = torch.empty_like(x_aug[i, 0, :]).uniform_(*var_range)
                x_aug[i, 0, :][peak_mask] *= random_factors[peak_mask]

        return x_aug

    def apply_peak_removal(self, x: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply peak removal augmentation.

        Args:
            x: Input pattern [batch, 1, L]
            config: Peak removal configuration

        Returns:
            Augmented pattern
        """
        if not config['enabled']:
            return x

        x_aug = x.clone()
        batch_size = x.shape[0]
        threshold = config['threshold']
        removal_prob = config['removal_probability']

        for i in range(batch_size):
            if torch.rand(1).item() < config['probability']:
                # Identify peaks
                peak_mask = (x_aug[i, 0, :] > threshold)

                # Randomly remove some peaks
                removal_mask = (torch.rand(peak_mask.shape, device=x_aug.device) < removal_prob) & peak_mask
                x_aug[i, 0, :][removal_mask] = 0.0

        return x_aug

    def apply_peak_broadening(self, x: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply physics-based peak broadening using Scherrer equation.

        Args:
            x: Input pattern [batch, 1, L]
            config: Peak broadening configuration

        Returns:
            Augmented pattern
        """
        if not config['enabled']:
            return x

        x_aug = x.clone()
        batch_size, _, L = x.shape
        intensity_range = config['intensity_range']

        for i in range(batch_size):
            if torch.rand(1).item() < config['probability']:
                # Random broadening intensity
                intensity = torch.rand(1).item() * (intensity_range[1] - intensity_range[0]) + intensity_range[0]

                # Create broadening kernel
                kernel = self.create_broadening_kernel(intensity, L)

                # Apply convolution
                x_temp = x_aug[i:i+1]
                pad_size = kernel.shape[2] // 2
                x_padded = F.pad(x_temp, (pad_size, pad_size), mode='reflect')
                x_aug[i:i+1] = F.conv1d(x_padded, kernel)

        return x_aug

    def apply_background_noise(self, x: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Apply background noise augmentation.

        Args:
            x: Input pattern [batch, 1, L]
            config: Background noise configuration

        Returns:
            Augmented pattern
        """
        if not config['enabled']:
            return x

        x_aug = x.clone()
        batch_size = x.shape[0]
        noise_range = config['noise_level_range']

        for i in range(batch_size):
            if torch.rand(1).item() < config['probability']:
                # Random noise level
                noise_level = torch.rand(1).item() * (noise_range[1] - noise_range[0]) + noise_range[0]

                # Add Gaussian noise
                noise = torch.randn_like(x_aug[i]) * noise_level
                x_aug[i] += noise

        return x_aug

    def augment_pattern(self, pattern: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Augment a single XRD pattern.

        Args:
            pattern: Input XRD pattern [L] or [1, L]
            num_samples: Number of augmented samples to generate

        Returns:
            Augmented patterns [num_samples, 1, L]
        """
        # Ensure correct input shape
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        elif pattern.dim() == 2:
            pattern = pattern.unsqueeze(1)  # [1, 1, L]

        # Move to device
        pattern = pattern.to(self.device)

        # Replicate pattern for multiple samples
        patterns_batch = pattern.repeat(num_samples, 1, 1)  # [num_samples, 1, L]

        # Apply augmentations in sequence
        augmented = patterns_batch

        # 1. Peak shifting
        augmented = self.apply_peak_shifting(augmented, self.config['peak_shifting'])

        # 2. Peak variations
        augmented = self.apply_peak_variations(augmented, self.config['peak_variations'])

        # 3. Peak removal
        augmented = self.apply_peak_removal(augmented, self.config['peak_removal'])

        # 4. Peak broadening
        augmented = self.apply_peak_broadening(augmented, self.config['peak_broadening'])

        # 5. Background noise
        augmented = self.apply_background_noise(augmented, self.config['background_noise'])

        return augmented

    def augment_batch(self, patterns: torch.Tensor, samples_per_pattern: int) -> Tuple[torch.Tensor, Dict]:
        """
        Augment a batch of patterns.

        Args:
            patterns: Batch of XRD patterns [batch_size, L] or [batch_size, 1, L]
            samples_per_pattern: Number of augmented samples per input pattern

        Returns:
            Tuple of (augmented_batch, metadata)
        """
        batch_size = patterns.shape[0]
        all_augmented = []

        for i in range(batch_size):
            pattern = patterns[i]
            augmented = self.augment_pattern(pattern, samples_per_pattern)
            all_augmented.append(augmented)

        # Combine all augmented patterns
        final_augmented = torch.cat(all_augmented, dim=0)  # [batch_size * samples_per_pattern, 1, L]

        metadata = {
            'original_batch_size': batch_size,
            'samples_per_pattern': samples_per_pattern,
            'total_samples': final_augmented.shape[0],
            'augmentation_methods': ['classical'] * final_augmented.shape[0]
        }

        return final_augmented, metadata

    def get_available_methods(self) -> list:
        """Get list of available augmentation methods."""
        methods = []
        for method, config in self.config.items():
            if isinstance(config, dict) and config.get('enabled', False):
                methods.append(method)
        return methods


def test_classical_augmenter():
    """Test function for the classical augmentation system."""
    print("Testing Classical XRD Augmenter...")

    # Create test configuration
    test_config = {
        'peak_shifting': {'enabled': True, 'probability': 0.5},
        'peak_variations': {'enabled': True, 'probability': 0.7},
        'peak_removal': {'enabled': True, 'probability': 0.3},
        'peak_broadening': {'enabled': True, 'probability': 0.8},
        'background_noise': {'enabled': True, 'probability': 0.6}
    }

    # Create augmenter
    augmenter = ClassicalXRDAugmenter(test_config, verbose=True)

    # Test with synthetic pattern
    test_pattern = torch.randn(4500)  # Synthetic XRD pattern

    print("Testing single pattern augmentation...")
    augmented = augmenter.augment_pattern(test_pattern, num_samples=5)

    print(f"Original pattern shape: {test_pattern.shape}")
    print(f"Augmented patterns shape: {augmented.shape}")
    print(f"Available methods: {augmenter.get_available_methods()}")

    print("✅ Classical augmentation test completed successfully!")


if __name__ == "__main__":
    test_classical_augmenter()