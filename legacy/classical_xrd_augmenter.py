#!/usr/bin/env python3
"""
Classical XRD Pattern Augmenter (No Training Required)
=====================================================

This script provides XRD pattern augmentation using classical signal processing
techniques without requiring a trained model. Perfect for immediate use on synthetic
XRD patterns to create realistic experimental-like variations.

Features:
- Peak broadening using Gaussian/Lorentzian convolution
- Intensity variations and scaling
- Background noise addition
- Peak shifting and position jitter
- Baseline drift simulation
- Multiple samples per input pattern
- Configurable parameters for realistic experimental simulation

Usage:
    python classical_xrd_augmenter.py --input_file data.pt --output_dir ./augmented --samples_per_pattern 5
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from tqdm import tqdm
import time
from datetime import datetime
from scipy import ndimage
from scipy.signal import convolve
import warnings
warnings.filterwarnings('ignore')

class ClassicalXRDAugmenter:
    """
    Classical XRD pattern augmenter using signal processing techniques.
    No training required - works immediately with any synthetic XRD patterns.
    """

    def __init__(self, verbose=True):
        """
        Initialize the Classical XRD Augmenter.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

        if self.verbose:
            print("🔬 Classical XRD Pattern Augmenter initialized!")
            print("   No training required - ready to augment patterns immediately")

    def add_peak_broadening(self, pattern, broadening_factor=1.0, profile='gaussian'):
        """
        Add peak broadening to simulate instrument effects and crystallite size.

        Args:
            pattern: XRD pattern [L] or [1, L]
            broadening_factor: Amount of broadening (0.5=narrow, 2.0=broad)
            profile: 'gaussian', 'lorentzian', or 'voigt'

        Returns:
            torch.Tensor: Broadened pattern
        """
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().flatten()
        else:
            pattern_np = np.array(pattern).flatten()

        # Calculate sigma based on broadening factor
        base_sigma = 0.5 + broadening_factor * 1.5
        sigma = max(0.3, base_sigma)

        if profile == 'gaussian':
            # Gaussian broadening (most common)
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            kernel = np.exp(-0.5 * ((np.arange(kernel_size) - kernel_size // 2) / sigma) ** 2)
            kernel = kernel / kernel.sum()

        elif profile == 'lorentzian':
            # Lorentzian broadening
            gamma = sigma * 0.5  # HWHM
            kernel_size = int(12 * gamma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            x = np.arange(kernel_size) - kernel_size // 2
            kernel = gamma / (np.pi * (x**2 + gamma**2))
            kernel = kernel / kernel.sum()

        elif profile == 'voigt':
            # Voigt profile (convolution of Gaussian and Lorentzian)
            # Simplified approximation
            sigma_g = sigma * 0.7
            gamma_l = sigma * 0.3

            kernel_size = int(8 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            x = np.arange(kernel_size) - kernel_size // 2

            # Gaussian component
            gaussian = np.exp(-0.5 * (x / sigma_g) ** 2)
            # Lorentzian component
            lorentzian = gamma_l / (x**2 + gamma_l**2)

            # Combine (simplified Voigt)
            kernel = gaussian * lorentzian
            kernel = kernel / kernel.sum()

        else:
            raise ValueError(f"Unknown profile: {profile}")

        # Apply convolution
        broadened = convolve(pattern_np, kernel, mode='same')

        return torch.tensor(broadened, dtype=torch.float32)

    def add_intensity_variations(self, pattern, variation_factor=0.1, peak_threshold=0.1):
        """
        Add intensity variations to simulate experimental measurement variations.

        Args:
            pattern: XRD pattern
            variation_factor: Amount of variation (0.0-1.0)
            peak_threshold: Minimum intensity to consider as a peak

        Returns:
            torch.Tensor: Pattern with intensity variations
        """
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().flatten()
        else:
            pattern_np = np.array(pattern).flatten()

        # Create variation mask (stronger variation at peaks)
        peak_mask = pattern_np > (pattern_np.max() * peak_threshold)

        # Generate random variations
        variations = np.random.normal(1.0, variation_factor, pattern_np.shape)

        # Apply stronger variations to peaks
        enhanced_variations = np.where(peak_mask,
                                     variations,
                                     1.0 + (variations - 1.0) * 0.3)  # Reduced variation for background

        varied_pattern = pattern_np * enhanced_variations

        # Ensure non-negative
        varied_pattern = np.maximum(varied_pattern, 0)

        return torch.tensor(varied_pattern, dtype=torch.float32)

    def add_background_noise(self, pattern, noise_level=0.05, noise_type='gaussian'):
        """
        Add background noise to simulate detector noise and measurement uncertainty.

        Args:
            pattern: XRD pattern
            noise_level: Noise amplitude relative to signal
            noise_type: 'gaussian', 'poisson', or 'uniform'

        Returns:
            torch.Tensor: Pattern with added noise
        """
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().flatten()
        else:
            pattern_np = np.array(pattern).flatten()

        signal_level = np.mean(pattern_np)
        noise_amplitude = signal_level * noise_level

        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_amplitude, pattern_np.shape)
        elif noise_type == 'poisson':
            # Poisson noise (detector-like)
            # Scale pattern for Poisson, then scale back
            scaled_pattern = pattern_np * 1000
            noisy_scaled = np.random.poisson(scaled_pattern)
            noise = (noisy_scaled - scaled_pattern) / 1000
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_amplitude, noise_amplitude, pattern_np.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        noisy_pattern = pattern_np + noise

        # Ensure non-negative
        noisy_pattern = np.maximum(noisy_pattern, 0)

        return torch.tensor(noisy_pattern, dtype=torch.float32)

    def add_peak_shifting(self, pattern, max_shift=3):
        """
        Add small peak position shifts to simulate sample positioning errors.

        Args:
            pattern: XRD pattern
            max_shift: Maximum shift in data points

        Returns:
            torch.Tensor: Pattern with peak shifts
        """
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().flatten()
        else:
            pattern_np = np.array(pattern).flatten()

        # Random shift
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift != 0:
            shifted_pattern = np.roll(pattern_np, shift)

            # Handle edge effects
            if shift > 0:
                shifted_pattern[:shift] = pattern_np[0]  # Repeat first value
            else:
                shifted_pattern[shift:] = pattern_np[-1]  # Repeat last value
        else:
            shifted_pattern = pattern_np.copy()

        return torch.tensor(shifted_pattern, dtype=torch.float32)

    def add_baseline_drift(self, pattern, drift_amplitude=0.02, drift_frequency=1.0):
        """
        Add baseline drift to simulate instrument instability.

        Args:
            pattern: XRD pattern
            drift_amplitude: Amplitude of drift relative to signal
            drift_frequency: Frequency of drift oscillation

        Returns:
            torch.Tensor: Pattern with baseline drift
        """
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().flatten()
        else:
            pattern_np = np.array(pattern).flatten()

        n_points = len(pattern_np)
        signal_level = np.mean(pattern_np)

        # Create drift pattern
        x = np.linspace(0, drift_frequency * 2 * np.pi, n_points)
        drift = signal_level * drift_amplitude * (
            np.sin(x) + 0.3 * np.sin(3 * x) + 0.1 * np.sin(7 * x)
        )

        # Add linear component
        linear_drift = signal_level * drift_amplitude * 0.5 * np.linspace(-1, 1, n_points)

        total_drift = drift + linear_drift
        drifted_pattern = pattern_np + total_drift

        return torch.tensor(drifted_pattern, dtype=torch.float32)

    def add_preferred_orientation(self, pattern, orientation_factor=0.1):
        """
        Simulate preferred orientation effects by selectively enhancing/reducing certain peaks.

        Args:
            pattern: XRD pattern
            orientation_factor: Strength of preferred orientation effect

        Returns:
            torch.Tensor: Pattern with preferred orientation effects
        """
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy().flatten()
        else:
            pattern_np = np.array(pattern).flatten()

        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(pattern_np, height=pattern_np.max() * 0.1)

        # Create orientation effects
        oriented_pattern = pattern_np.copy()

        for peak in peaks:
            # Random enhancement or reduction
            if np.random.random() < 0.3:  # 30% chance of significant change
                factor = 1.0 + np.random.uniform(-orientation_factor, orientation_factor * 2)
                factor = max(0.1, factor)  # Don't completely eliminate peaks

                # Apply to peak region
                peak_width = 10  # Approximate peak width
                start = max(0, peak - peak_width)
                end = min(len(oriented_pattern), peak + peak_width)

                # Gaussian weighting around peak
                peak_region = np.arange(start, end)
                weights = np.exp(-0.5 * ((peak_region - peak) / (peak_width / 3)) ** 2)
                weights = weights * (factor - 1) + 1

                oriented_pattern[start:end] *= weights

        return torch.tensor(oriented_pattern, dtype=torch.float32)

    def augment_pattern(self, pattern, num_samples=5, augmentation_config=None, base_seed=None):
        """
        Augment a single XRD pattern using classical techniques.

        Args:
            pattern: Input XRD pattern [L] or [1, L]
            num_samples: Number of augmented samples to generate
            augmentation_config: Dictionary of augmentation parameters
            base_seed: Base random seed for reproducibility

        Returns:
            torch.Tensor: Augmented patterns [num_samples, L]
            dict: Metadata about augmentation parameters used
        """
        # Default augmentation configuration
        default_config = {
            'broadening': {
                'enabled': True,
                'factor_range': (0.5, 2.0),
                'profile': 'gaussian'  # 'gaussian', 'lorentzian', 'voigt'
            },
            'intensity_variation': {
                'enabled': True,
                'variation_range': (0.05, 0.2),
                'peak_threshold': 0.1
            },
            'background_noise': {
                'enabled': True,
                'noise_level_range': (0.01, 0.1),
                'noise_type': 'gaussian'  # 'gaussian', 'poisson', 'uniform'
            },
            'peak_shifting': {
                'enabled': True,
                'max_shift_range': (1, 5)
            },
            'baseline_drift': {
                'enabled': True,
                'amplitude_range': (0.01, 0.05),
                'frequency_range': (0.5, 2.0)
            },
            'preferred_orientation': {
                'enabled': True,
                'factor_range': (0.05, 0.3)
            }
        }

        if augmentation_config is None:
            augmentation_config = default_config
        else:
            # Merge with defaults
            for key, value in default_config.items():
                if key not in augmentation_config:
                    augmentation_config[key] = value

        # Ensure pattern is 1D
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.cpu().numpy().flatten()
        else:
            pattern = np.array(pattern).flatten()

        augmented_patterns = []
        metadata = {
            'parameters_used': [],
            'seeds': []
        }

        for i in range(num_samples):
            # Set seed for this sample
            if base_seed is not None:
                sample_seed = base_seed + i
                np.random.seed(sample_seed)
                metadata['seeds'].append(sample_seed)
            else:
                metadata['seeds'].append(None)

            # Start with original pattern
            augmented = torch.tensor(pattern, dtype=torch.float32)

            # Track parameters used for this sample
            sample_params = {}

            # Apply augmentations in sequence
            if augmentation_config['broadening']['enabled']:
                factor = np.random.uniform(*augmentation_config['broadening']['factor_range'])
                profile = augmentation_config['broadening']['profile']
                augmented = self.add_peak_broadening(augmented, factor, profile)
                sample_params['broadening'] = {'factor': factor, 'profile': profile}

            if augmentation_config['intensity_variation']['enabled']:
                variation = np.random.uniform(*augmentation_config['intensity_variation']['variation_range'])
                threshold = augmentation_config['intensity_variation']['peak_threshold']
                augmented = self.add_intensity_variations(augmented, variation, threshold)
                sample_params['intensity_variation'] = {'factor': variation, 'threshold': threshold}

            if augmentation_config['background_noise']['enabled']:
                noise_level = np.random.uniform(*augmentation_config['background_noise']['noise_level_range'])
                noise_type = augmentation_config['background_noise']['noise_type']
                augmented = self.add_background_noise(augmented, noise_level, noise_type)
                sample_params['background_noise'] = {'level': noise_level, 'type': noise_type}

            if augmentation_config['peak_shifting']['enabled']:
                max_shift = np.random.randint(*augmentation_config['peak_shifting']['max_shift_range'])
                augmented = self.add_peak_shifting(augmented, max_shift)
                sample_params['peak_shifting'] = {'max_shift': max_shift}

            if augmentation_config['baseline_drift']['enabled']:
                amplitude = np.random.uniform(*augmentation_config['baseline_drift']['amplitude_range'])
                frequency = np.random.uniform(*augmentation_config['baseline_drift']['frequency_range'])
                augmented = self.add_baseline_drift(augmented, amplitude, frequency)
                sample_params['baseline_drift'] = {'amplitude': amplitude, 'frequency': frequency}

            if augmentation_config['preferred_orientation']['enabled']:
                factor = np.random.uniform(*augmentation_config['preferred_orientation']['factor_range'])
                augmented = self.add_preferred_orientation(augmented, factor)
                sample_params['preferred_orientation'] = {'factor': factor}

            augmented_patterns.append(augmented)
            metadata['parameters_used'].append(sample_params)

        # Stack patterns
        augmented_patterns = torch.stack(augmented_patterns, dim=0)

        return augmented_patterns, metadata

    def augment_batch(self, patterns, samples_per_pattern=5, augmentation_config=None,
                     base_seed=42, progress_bar=True):
        """
        Augment a batch of XRD patterns.

        Args:
            patterns: Input patterns [N, L] or [N, 1, L]
            samples_per_pattern: Number of augmented samples per input
            augmentation_config: Augmentation configuration
            base_seed: Base random seed
            progress_bar: Show progress bar

        Returns:
            torch.Tensor: All augmented patterns [N*samples_per_pattern, L]
            list: Metadata for each input pattern
        """
        if isinstance(patterns, torch.Tensor):
            patterns = patterns.cpu().numpy()

        if patterns.ndim == 3:
            patterns = patterns.squeeze(1)  # Remove channel dimension if present

        num_patterns = patterns.shape[0]
        all_augmented = []
        all_metadata = []

        iterator = range(num_patterns)
        if progress_bar:
            iterator = tqdm(iterator, desc="Augmenting patterns")

        for i in iterator:
            pattern_seed = base_seed + i * 1000 if base_seed else None

            augmented, metadata = self.augment_pattern(
                pattern=patterns[i],
                num_samples=samples_per_pattern,
                augmentation_config=augmentation_config,
                base_seed=pattern_seed
            )

            all_augmented.append(augmented)
            all_metadata.append(metadata)

        # Combine all results
        all_augmented = torch.cat(all_augmented, dim=0)
        return all_augmented, all_metadata

    def save_results(self, augmented_patterns, metadata_list, original_patterns,
                    output_dir, prefix="classical_augmented"):
        """Save augmented patterns and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save patterns
        patterns_file = os.path.join(output_dir, f"{prefix}_patterns_{timestamp}.pt")
        torch.save({
            'augmented_patterns': augmented_patterns,
            'original_patterns': original_patterns,
            'metadata': metadata_list,
            'generation_info': {
                'timestamp': timestamp,
                'method': 'classical_augmentation',
                'num_original': original_patterns.shape[0] if hasattr(original_patterns, 'shape') else len(original_patterns),
                'num_augmented': augmented_patterns.shape[0],
                'samples_per_pattern': len(metadata_list[0]['parameters_used']) if metadata_list else 0
            }
        }, patterns_file)

        if self.verbose:
            print(f"💾 Saved patterns to: {patterns_file}")

        # Save metadata as JSON
        metadata_file = os.path.join(output_dir, f"{prefix}_metadata_{timestamp}.json")

        # Convert numpy types to JSON-serializable types
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj

        json_metadata = convert_for_json(metadata_list)

        with open(metadata_file, 'w') as f:
            json.dump({
                'metadata': json_metadata,
                'generation_info': {
                    'timestamp': timestamp,
                    'method': 'classical_augmentation',
                    'num_patterns': len(json_metadata)
                }
            }, f, indent=2)

        if self.verbose:
            print(f"📄 Saved metadata to: {metadata_file}")

    def visualize_augmentation(self, original_pattern, augmented_patterns,
                              metadata=None, save_path=None, show_plot=True):
        """Visualize original pattern and its augmented versions."""
        # Prepare data
        if isinstance(original_pattern, torch.Tensor):
            orig = original_pattern.cpu().numpy().flatten()
        else:
            orig = np.array(original_pattern).flatten()

        if isinstance(augmented_patterns, torch.Tensor):
            aug_data = augmented_patterns.cpu().numpy()
        else:
            aug_data = np.array(augmented_patterns)

        if aug_data.ndim == 1:
            aug_data = aug_data.reshape(1, -1)

        # Create x-axis (assume 2theta range)
        x_axis = np.linspace(10, 80, len(orig))

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Original vs all augmented
        axes[0,0].plot(x_axis, orig, 'k-', linewidth=2, label='Original (Synthetic)', alpha=0.9)
        for i, aug in enumerate(aug_data):
            alpha = max(0.3, 1.0 - i * 0.1)
            axes[0,0].plot(x_axis, aug, alpha=alpha, linewidth=1,
                          label=f'Augmented {i+1}' if i < 3 else '')
        axes[0,0].set_title('Original vs Augmented Patterns')
        axes[0,0].set_xlabel('2θ (degrees)')
        axes[0,0].set_ylabel('Intensity (a.u.)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2: Difference from original
        axes[0,1].plot(x_axis, np.zeros_like(x_axis), 'k--', alpha=0.5, label='Zero difference')
        for i, aug in enumerate(aug_data):
            diff = aug - orig
            axes[0,1].plot(x_axis, diff, alpha=0.7, linewidth=1, label=f'Diff {i+1}' if i < 3 else '')
        axes[0,1].set_title('Difference from Original')
        axes[0,1].set_xlabel('2θ (degrees)')
        axes[0,1].set_ylabel('Intensity Difference')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Statistics
        mean_aug = np.mean(aug_data, axis=0)
        std_aug = np.std(aug_data, axis=0)

        axes[1,0].plot(x_axis, orig, 'k-', linewidth=2, label='Original')
        axes[1,0].plot(x_axis, mean_aug, 'r-', linewidth=2, label='Augmented Mean')
        axes[1,0].fill_between(x_axis, mean_aug - std_aug, mean_aug + std_aug,
                              alpha=0.3, color='red', label='±1 Std')
        axes[1,0].set_title('Statistical Summary')
        axes[1,0].set_xlabel('2θ (degrees)')
        axes[1,0].set_ylabel('Intensity (a.u.)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Augmentation summary
        if metadata is not None and 'parameters_used' in metadata:
            # Collect parameter statistics
            param_summary = {}
            for params in metadata['parameters_used']:
                for key, value in params.items():
                    if key not in param_summary:
                        param_summary[key] = []
                    if isinstance(value, dict):
                        # Extract numerical values
                        for subkey, subval in value.items():
                            if isinstance(subval, (int, float)):
                                param_key = f"{key}_{subkey}"
                                if param_key not in param_summary:
                                    param_summary[param_key] = []
                                param_summary[param_key].append(subval)

            # Plot parameter distribution
            param_names = []
            param_values = []
            for key, values in param_summary.items():
                if values and isinstance(values[0], (int, float)):
                    param_names.append(key.replace('_', '\n'))
                    param_values.append(np.mean(values))

            if param_names:
                axes[1,1].bar(range(len(param_names)), param_values, alpha=0.7)
                axes[1,1].set_xticks(range(len(param_names)))
                axes[1,1].set_xticklabels(param_names, rotation=45, ha='right')
                axes[1,1].set_title('Average Augmentation Parameters')
                axes[1,1].set_ylabel('Parameter Value')
            else:
                axes[1,1].text(0.5, 0.5, 'No numerical\nparameters found',
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Parameters')
        else:
            axes[1,1].text(0.5, 0.5, 'No metadata available',
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Metadata')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"📊 Saved visualization to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

def load_synthetic_patterns(file_path, max_patterns=None):
    """Load synthetic XRD patterns from various file formats."""
    if file_path.endswith('.pt'):
        data = torch.load(file_path, map_location='cpu')
        if isinstance(data, dict):
            # Try common keys
            for key in ['synth_xrd', 'synthetic_patterns', 'patterns', 'data']:
                if key in data:
                    patterns = data[key]
                    break
            else:
                # Use first tensor-like value
                patterns = next(iter(data.values()))
        else:
            patterns = data
    elif file_path.endswith('.npy'):
        patterns = torch.from_numpy(np.load(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if max_patterns is not None:
        patterns = patterns[:max_patterns]

    return patterns.float()

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Classical XRD Pattern Augmenter (No Training Required)')

    # Input/Output
    parser.add_argument('--input_file', required=True, help='Input synthetic patterns file')
    parser.add_argument('--output_dir', default='./classical_augmented', help='Output directory')

    # Augmentation parameters
    parser.add_argument('--samples_per_pattern', type=int, default=5,
                       help='Number of augmented samples per input pattern')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed')

    # Processing parameters
    parser.add_argument('--max_patterns', type=int, help='Maximum patterns to process')

    # Augmentation control
    parser.add_argument('--disable_broadening', action='store_true', help='Disable peak broadening')
    parser.add_argument('--disable_intensity_var', action='store_true', help='Disable intensity variations')
    parser.add_argument('--disable_noise', action='store_true', help='Disable background noise')
    parser.add_argument('--disable_shifting', action='store_true', help='Disable peak shifting')
    parser.add_argument('--disable_drift', action='store_true', help='Disable baseline drift')
    parser.add_argument('--disable_orientation', action='store_true', help='Disable preferred orientation')

    # Output options
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--vis_samples', type=int, default=3, help='Number of patterns to visualize')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Initialize augmenter
    print("🔬 Classical XRD Pattern Augmenter (No Training Required)")
    print("=" * 60)

    augmenter = ClassicalXRDAugmenter(verbose=not args.quiet)

    # Create augmentation configuration
    augmentation_config = {
        'broadening': {'enabled': not args.disable_broadening},
        'intensity_variation': {'enabled': not args.disable_intensity_var},
        'background_noise': {'enabled': not args.disable_noise},
        'peak_shifting': {'enabled': not args.disable_shifting},
        'baseline_drift': {'enabled': not args.disable_drift},
        'preferred_orientation': {'enabled': not args.disable_orientation}
    }

    # Load synthetic patterns
    print(f"\n📂 Loading synthetic patterns from: {args.input_file}")
    try:
        synth_patterns = load_synthetic_patterns(args.input_file, args.max_patterns)
        print(f"   Loaded {synth_patterns.shape[0]} patterns with {synth_patterns.shape[-1]} points each")
    except Exception as e:
        print(f"   Error loading patterns: {e}")
        return

    # Run augmentation
    print(f"\n🎯 Starting classical augmentation...")
    print(f"   Samples per pattern: {args.samples_per_pattern}")
    print(f"   Enabled augmentations: {[k for k, v in augmentation_config.items() if v['enabled']]}")

    start_time = time.time()

    augmented_patterns, metadata_list = augmenter.augment_batch(
        patterns=synth_patterns,
        samples_per_pattern=args.samples_per_pattern,
        augmentation_config=augmentation_config,
        base_seed=args.base_seed,
        progress_bar=not args.quiet
    )

    augmentation_time = time.time() - start_time
    print(f"\n⏱️  Augmentation completed in {augmentation_time:.2f} seconds")
    print(f"   Generated {augmented_patterns.shape[0]} augmented patterns")
    print(f"   Average time per pattern: {augmentation_time / synth_patterns.shape[0]:.3f} seconds")

    # Save results
    print(f"\n💾 Saving results to: {args.output_dir}")
    augmenter.save_results(
        augmented_patterns=augmented_patterns,
        metadata_list=metadata_list,
        original_patterns=synth_patterns,
        output_dir=args.output_dir
    )

    # Create visualizations
    if args.visualize:
        print(f"\n📊 Creating visualizations...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for i in range(min(args.vis_samples, synth_patterns.shape[0])):
            start_idx = i * args.samples_per_pattern
            end_idx = start_idx + args.samples_per_pattern

            vis_path = os.path.join(vis_dir, f"classical_augmentation_example_{i+1}.png")
            augmenter.visualize_augmentation(
                original_pattern=synth_patterns[i],
                augmented_patterns=augmented_patterns[start_idx:end_idx],
                metadata=metadata_list[i],
                save_path=vis_path,
                show_plot=False
            )

        print(f"   Saved {args.vis_samples} visualization(s) to: {vis_dir}")

    print(f"\n✅ Classical XRD Pattern Augmentation Complete!")
    print(f"📁 Check output directory: {args.output_dir}")

if __name__ == "__main__":
    main()