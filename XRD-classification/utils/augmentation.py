#!/usr/bin/env python3
"""
Dual XRD Pattern Augmentation System
====================================

This module combines classical signal processing augmentation with diffusion model
augmentation to create highly diverse and realistic XRD pattern variations.

Features:
- Classical augmentation (immediate, no training required)
- Diffusion model augmentation (uses trained models)
- Configurable mixing ratios between methods
- Beta distribution for realistic noise levels
- Fallback mechanisms for robustness
"""

import torch
import numpy as np
import os
import sys
from typing import Tuple, List, Optional, Dict, Any
import warnings

# Add legacy modules to path
sys.path.append('../legacy')

try:
    from classical_xrd_augmenter import ClassicalXRDAugmenter
    CLASSICAL_AVAILABLE = True
except ImportError:
    CLASSICAL_AVAILABLE = False
    warnings.warn("Classical augmenter not available. Using fallback.")

try:
    from xrd_pattern_augmenter_refactored import XRDPatternAugmenter
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    warnings.warn("Diffusion augmenter not available. Using fallback.")


class DualXRDAugmenter:
    """
    Dual augmentation system combining classical and diffusion methods.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        """
        Initialize the dual augmentation system.

        Args:
            config: Configuration dictionary with augmentation settings
            verbose: Enable verbose output
        """
        self.config = config
        self.verbose = verbose

        # Initialize augmenters
        self.classical_augmenter = None
        self.diffusion_augmenter = None

        # Setup classical augmenter
        if config['augmentation']['classical']['enabled'] and CLASSICAL_AVAILABLE:
            self._setup_classical_augmenter()

        # Setup diffusion augmenter
        if config['augmentation']['diffusion']['enabled'] and DIFFUSION_AVAILABLE:
            self._setup_diffusion_augmenter()

        if self.verbose:
            self._print_status()

    def _setup_classical_augmenter(self):
        """Setup classical augmentation system."""
        try:
            self.classical_augmenter = ClassicalXRDAugmenter(verbose=self.verbose)
            if self.verbose:
                print("✅ Classical augmenter initialized")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Classical augmenter failed to initialize: {e}")
            self.classical_augmenter = None

    def _setup_diffusion_augmenter(self):
        """Setup diffusion model augmentation system."""
        try:
            model_path = self.config['augmentation']['diffusion']['model_path']
            if os.path.exists(model_path):
                self.diffusion_augmenter = XRDPatternAugmenter(
                    model_path=model_path,
                    device='auto',
                    verbose=self.verbose
                )
                if self.verbose:
                    print("✅ Diffusion augmenter initialized")
            else:
                if self.verbose:
                    print(f"⚠️ Diffusion model not found at {model_path}")
                self.diffusion_augmenter = None
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Diffusion augmenter failed to initialize: {e}")
            self.diffusion_augmenter = None

    def _print_status(self):
        """Print augmentation system status."""
        print("\n🔬 Dual XRD Augmentation System Status:")
        print(f"   Classical Augmentation: {'✅ Available' if self.classical_augmenter else '❌ Unavailable'}")
        print(f"   Diffusion Augmentation: {'✅ Available' if self.diffusion_augmenter else '❌ Unavailable'}")

        if not self.classical_augmenter and not self.diffusion_augmenter:
            print("   ⚠️ No augmentation methods available!")

        print()

    def generate_noise_levels(self, n_samples: int) -> np.ndarray:
        """
        Generate noise levels using Beta distribution (biased toward lower noise).

        Args:
            n_samples: Number of noise level samples to generate

        Returns:
            Array of noise levels
        """
        alpha = self.config['augmentation']['noise_beta_alpha']
        beta = self.config['augmentation']['noise_beta_beta']
        max_noise = self.config['augmentation']['max_noise_level']

        # Beta distribution biased toward lower values
        noise_levels = np.random.beta(alpha, beta, size=n_samples) * max_noise
        return noise_levels

    def augment_pattern_classical(self, pattern: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Augment pattern using classical methods.

        Args:
            pattern: Input XRD pattern [1, L] or [L]
            num_samples: Number of augmented samples to generate

        Returns:
            Augmented patterns [num_samples, 1, L]
        """
        if self.classical_augmenter is None:
            # Fallback: simple noise addition
            return self._fallback_augmentation(pattern, num_samples)

        augmented_samples = []
        config = self.config['augmentation']['classical']

        for i in range(num_samples):
            # Start with original pattern
            augmented = pattern.clone() if torch.is_tensor(pattern) else torch.tensor(pattern)

            # Ensure correct shape [L] for classical augmenter
            if augmented.dim() > 1:
                augmented = augmented.squeeze()

            # Apply classical augmentations
            try:
                # Peak broadening
                if config.get('peak_broadening', {}).get('enabled', True):
                    broadening_range = config.get('peak_broadening', {}).get('broadening_range', [0.5, 2.0])
                    broadening_factor = np.random.uniform(*broadening_range)
                    augmented = self.classical_augmenter.add_peak_broadening(
                        augmented, broadening_factor=broadening_factor
                    )

                # Intensity variations
                if config.get('intensity_variation', {}).get('enabled', True):
                    scale_range = config.get('intensity_variation', {}).get('scale_range', [0.8, 1.2])
                    scale_factor = np.random.uniform(*scale_range)
                    augmented = augmented * scale_factor

                # Background noise
                if config.get('background_noise', {}).get('enabled', True):
                    noise_range = config.get('background_noise', {}).get('noise_level_range', [0.01, 0.1])
                    noise_level = np.random.uniform(*noise_range)
                    noise = torch.randn_like(augmented) * noise_level
                    augmented = augmented + noise

                # Ensure proper shape [1, L]
                if augmented.dim() == 1:
                    augmented = augmented.unsqueeze(0)

                augmented_samples.append(augmented)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Classical augmentation failed for sample {i}: {e}")
                # Use fallback
                fallback = self._fallback_augmentation(pattern, 1)
                augmented_samples.append(fallback[0])

        return torch.stack(augmented_samples)

    def augment_pattern_diffusion(self, pattern: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Augment pattern using diffusion model.

        Args:
            pattern: Input XRD pattern [1, L] or [L]
            num_samples: Number of augmented samples to generate

        Returns:
            Augmented patterns [num_samples, 1, L]
        """
        if self.diffusion_augmenter is None:
            # Use classical fallback if enabled
            if self.config['augmentation']['diffusion'].get('use_classical_fallback', True):
                return self.augment_pattern_classical(pattern, num_samples)
            else:
                return self._fallback_augmentation(pattern, num_samples)

        try:
            # Ensure correct input shape for diffusion model
            if pattern.dim() == 1:
                pattern = pattern.unsqueeze(0)  # [1, L]

            config = self.config['augmentation']['diffusion']

            # Use diffusion augmenter
            augmented = self.diffusion_augmenter.augment_pattern(
                synth_pattern=pattern,
                num_samples=num_samples,
                temp_range=tuple(config.get('temp_range', [0.1, 2.0])),
                temp_mode=config.get('temp_mode', 'random'),
                noise_timestep_range=tuple(config.get('noise_timestep_range', [0, 50])),
                base_seed=None,
                return_metadata=False
            )

            return augmented  # Should be [num_samples, 1, L]

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Diffusion augmentation failed: {e}")

            # Use classical fallback if enabled
            if self.config['augmentation']['diffusion'].get('use_classical_fallback', True):
                return self.augment_pattern_classical(pattern, num_samples)
            else:
                return self._fallback_augmentation(pattern, num_samples)

    def _fallback_augmentation(self, pattern: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Simple fallback augmentation using basic noise addition.

        Args:
            pattern: Input pattern
            num_samples: Number of samples to generate

        Returns:
            Augmented patterns [num_samples, 1, L]
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)  # [1, L]

        augmented_samples = []

        for i in range(num_samples):
            # Add Gaussian noise
            noise_level = np.random.uniform(0.01, 0.1)
            noise = torch.randn_like(pattern) * noise_level

            # Add intensity scaling
            scale = np.random.uniform(0.9, 1.1)

            augmented = pattern * scale + noise
            augmented_samples.append(augmented)

        return torch.stack(augmented_samples)

    def augment_pattern_mixed(self, pattern: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Augment pattern using mixed classical and diffusion methods.

        Args:
            pattern: Input XRD pattern [1, L] or [L]
            num_samples: Total number of augmented samples to generate

        Returns:
            Tuple of (augmented_patterns [num_samples, 1, L], method_labels)
        """
        classical_ratio = self.config['augmentation']['classical'].get('samples_ratio', 0.5)
        diffusion_ratio = self.config['augmentation']['diffusion'].get('samples_ratio', 0.5)

        # Normalize ratios
        total_ratio = classical_ratio + diffusion_ratio
        if total_ratio > 0:
            classical_ratio /= total_ratio
            diffusion_ratio /= total_ratio
        else:
            classical_ratio = diffusion_ratio = 0.5

        # Calculate sample counts
        classical_samples = int(num_samples * classical_ratio)
        diffusion_samples = num_samples - classical_samples

        augmented_patterns = []
        method_labels = []

        # Generate classical augmented samples
        if classical_samples > 0:
            classical_aug = self.augment_pattern_classical(pattern, classical_samples)
            augmented_patterns.append(classical_aug)
            method_labels.extend(['classical'] * classical_samples)

        # Generate diffusion augmented samples
        if diffusion_samples > 0:
            diffusion_aug = self.augment_pattern_diffusion(pattern, diffusion_samples)
            augmented_patterns.append(diffusion_aug)
            method_labels.extend(['diffusion'] * diffusion_samples)

        # Combine all augmented patterns
        if augmented_patterns:
            all_augmented = torch.cat(augmented_patterns, dim=0)
        else:
            all_augmented = self._fallback_augmentation(pattern, num_samples)
            method_labels = ['fallback'] * num_samples

        return all_augmented, method_labels

    def augment_batch(self, patterns: torch.Tensor, samples_per_pattern: int) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Augment a batch of patterns.

        Args:
            patterns: Batch of XRD patterns [batch_size, L] or [batch_size, 1, L]
            samples_per_pattern: Number of augmented samples per input pattern

        Returns:
            Tuple of (augmented_batch, metadata_list)
        """
        batch_size = patterns.shape[0]
        all_augmented = []
        all_metadata = []

        for i in range(batch_size):
            pattern = patterns[i]

            # Generate augmented samples
            augmented, methods = self.augment_pattern_mixed(pattern, samples_per_pattern)
            all_augmented.append(augmented)

            # Create metadata
            metadata = {
                'original_index': i,
                'augmentation_methods': methods,
                'num_samples': samples_per_pattern
            }
            all_metadata.append(metadata)

        # Combine all augmented patterns
        final_augmented = torch.cat(all_augmented, dim=0)  # [batch_size * samples_per_pattern, 1, L]

        return final_augmented, all_metadata

    def get_available_methods(self) -> List[str]:
        """Get list of available augmentation methods."""
        methods = []
        if self.classical_augmenter is not None:
            methods.append('classical')
        if self.diffusion_augmenter is not None:
            methods.append('diffusion')
        if not methods:
            methods.append('fallback')
        return methods


def test_dual_augmenter():
    """Test function for the dual augmentation system."""
    # Simple test configuration
    test_config = {
        'augmentation': {
            'classical': {
                'enabled': True,
                'samples_ratio': 0.5,
                'peak_broadening': {'enabled': True, 'broadening_range': [0.5, 2.0]},
                'intensity_variation': {'enabled': True, 'scale_range': [0.8, 1.2]},
                'background_noise': {'enabled': True, 'noise_level_range': [0.01, 0.1]}
            },
            'diffusion': {
                'enabled': True,
                'samples_ratio': 0.5,
                'model_path': '../models/xrd_diffusion/improved_diffusion_model_best.pth',
                'temp_range': [0.1, 2.0],
                'temp_mode': 'random',
                'noise_timestep_range': [0, 50],
                'use_classical_fallback': True
            },
            'noise_beta_alpha': 2,
            'noise_beta_beta': 5,
            'max_noise_level': 1000
        }
    }

    # Create augmenter
    augmenter = DualXRDAugmenter(test_config, verbose=True)

    # Test with synthetic pattern
    test_pattern = torch.randn(4500)  # Synthetic XRD pattern

    print("Testing augmentation...")
    augmented, methods = augmenter.augment_pattern_mixed(test_pattern, num_samples=6)

    print(f"Original pattern shape: {test_pattern.shape}")
    print(f"Augmented patterns shape: {augmented.shape}")
    print(f"Methods used: {methods}")
    print(f"Available methods: {augmenter.get_available_methods()}")


if __name__ == "__main__":
    test_dual_augmenter()