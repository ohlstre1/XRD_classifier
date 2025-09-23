#!/usr/bin/env python3
"""
Demo: XRD Pattern Augmentation Without Training
==============================================

This script demonstrates how to use the XRD augmenter without needing
a pre-trained model. Perfect for immediate use with synthetic XRD patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from xrd_pattern_augmenter import XRDPatternAugmenter
import os

def create_demo_xrd_patterns(n_patterns=5):
    """Create realistic demo XRD patterns."""
    print("📝 Creating demo XRD patterns...")

    n_points = 1000
    two_theta = np.linspace(10, 80, n_points)
    patterns = []

    for i in range(n_patterns):
        # Start with baseline
        pattern = np.full(n_points, 10.0)  # Background

        # Add multiple peaks with different characteristics
        peak_data = [
            (20 + i * 0.5, 100, 1.0),   # Main peak
            (35 + i * 0.3, 80, 1.2),    # Secondary peak
            (50 + i * 0.2, 60, 0.8),    # Tertiary peak
            (65 + i * 0.1, 40, 1.5),    # Quaternary peak
        ]

        for pos, intensity, width in peak_data:
            # Gaussian peaks
            peak = intensity * np.exp(-((two_theta - pos) / width) ** 2)
            pattern += peak

        # Add minor crystalline peaks
        for j in range(3):
            minor_pos = 25 + i * 2 + j * 8
            minor_intensity = np.random.uniform(15, 25)
            minor_width = np.random.uniform(0.5, 1.0)
            minor_peak = minor_intensity * np.exp(-((two_theta - minor_pos) / minor_width) ** 2)
            pattern += minor_peak

        # Ensure non-negative
        pattern = np.maximum(pattern, 0)
        patterns.append(pattern)

    patterns = torch.tensor(patterns, dtype=torch.float32)
    print(f"   Created {n_patterns} demo patterns with {n_points} points each")

    return patterns, two_theta

def demonstrate_no_training_augmentation():
    """Demonstrate XRD augmentation without training."""
    print("🔬 XRD Pattern Augmentation Demo - No Training Required")
    print("=" * 60)

    # Create demo patterns
    demo_patterns, two_theta_axis = create_demo_xrd_patterns(3)

    print(f"\n🎯 Demo Configuration:")
    print(f"   Input patterns: {demo_patterns.shape[0]}")
    print(f"   Points per pattern: {demo_patterns.shape[1]}")
    print(f"   2θ range: {two_theta_axis[0]:.1f}° - {two_theta_axis[-1]:.1f}°")

    # Initialize augmenter in no-training mode
    print(f"\n🤖 Initializing Classical Augmenter (No Training Required)...")

    augmenter = XRDPatternAugmenter(
        model_path=None,        # No model needed!
        use_classical=True,     # Use classical augmentation
        verbose=True
    )

    # Demo 1: Basic augmentation
    print(f"\n📋 Demo 1: Basic Pattern Augmentation")
    print("-" * 40)

    single_pattern = demo_patterns[0:1]
    print(f"Input pattern shape: {single_pattern.shape}")

    # Generate 5 augmented versions
    augmented_basic = augmenter.augment_pattern(
        synth_pattern=single_pattern,
        num_samples=5,
        base_seed=42
    )

    print(f"Output shape: {augmented_basic.shape}")  # Should be [5, 1, 1000]
    print(f"Generated 5 augmented versions from 1 input pattern")

    # Demo 2: Batch augmentation
    print(f"\n📋 Demo 2: Batch Augmentation")
    print("-" * 40)

    print(f"Input batch shape: {demo_patterns.shape}")

    # Augment all demo patterns
    augmented_batch, metadata = augmenter.augment_batch(
        synth_patterns=demo_patterns,
        samples_per_pattern=3,
        base_seed=123,
        progress_bar=True
    )

    print(f"Output batch shape: {augmented_batch.shape}")  # Should be [9, 1, 1000]
    print(f"Generated {augmented_batch.shape[0]} patterns from {demo_patterns.shape[0]} inputs")
    print(f"Metadata entries: {len(metadata)}")

    # Demo 3: Custom augmentation configuration
    print(f"\n📋 Demo 3: Custom Augmentation Parameters")
    print("-" * 40)

    # Create custom augmentation config
    custom_config = {
        'broadening': {
            'enabled': True,
            'factor_range': (1.0, 3.0),    # More aggressive broadening
            'profile': 'gaussian'
        },
        'intensity_variation': {
            'enabled': True,
            'variation_range': (0.1, 0.3),  # Higher intensity variation
        },
        'background_noise': {
            'enabled': True,
            'noise_level_range': (0.05, 0.15),  # More noise
            'noise_type': 'gaussian'
        },
        'peak_shifting': {
            'enabled': True,
            'max_shift_range': (2, 8)       # Larger shifts
        },
        'baseline_drift': {
            'enabled': True,
            'amplitude_range': (0.02, 0.08)  # More baseline variation
        },
        'preferred_orientation': {
            'enabled': True,
            'factor_range': (0.1, 0.5)      # Stronger orientation effects
        }
    }

    augmented_custom = augmenter.augment_pattern(
        synth_pattern=single_pattern,
        num_samples=4,
        base_seed=456,
        augmentation_config=custom_config
    )

    print(f"Custom augmentation shape: {augmented_custom.shape}")
    print(f"Applied custom parameter ranges for more aggressive augmentation")

    # Demo 4: Analysis and visualization
    print(f"\n📊 Demo 4: Analysis and Visualization")
    print("-" * 40)

    # Compare original vs augmented statistics
    original_mean = demo_patterns.mean().item()
    original_std = demo_patterns.std().item()
    augmented_mean = augmented_batch.mean().item()
    augmented_std = augmented_batch.std().item()

    print(f"Statistical Comparison:")
    print(f"   Original patterns:")
    print(f"     Mean intensity: {original_mean:.4f}")
    print(f"     Std intensity:  {original_std:.4f}")
    print(f"   Augmented patterns:")
    print(f"     Mean intensity: {augmented_mean:.4f}")
    print(f"     Std intensity:  {augmented_std:.4f}")
    print(f"   Intensity change: {((augmented_mean - original_mean) / original_mean * 100):+.2f}%")
    print(f"   Variability change: {((augmented_std - original_std) / original_std * 100):+.2f}%")

    # Create comprehensive visualization
    output_dir = "./demo_no_training_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save demo data
    torch.save({
        'original_patterns': demo_patterns,
        'augmented_basic': augmented_basic,
        'augmented_batch': augmented_batch,
        'augmented_custom': augmented_custom,
        'metadata': metadata,
        'two_theta_axis': two_theta_axis
    }, os.path.join(output_dir, "demo_results.pt"))

    # Create visualizations
    print(f"\n📈 Creating Visualizations...")

    # Visualization 1: Original vs Augmented comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Single pattern augmentation
    axes[0,0].plot(two_theta_axis, single_pattern.numpy().flatten(),
                  'k-', linewidth=2, label='Original', alpha=0.9)

    for i, aug_pattern in enumerate(augmented_basic):
        alpha = max(0.4, 1.0 - i * 0.15)
        axes[0,0].plot(two_theta_axis, aug_pattern.numpy().flatten(),
                      alpha=alpha, linewidth=1, label=f'Aug {i+1}' if i < 3 else '')

    axes[0,0].set_title('Single Pattern Augmentation')
    axes[0,0].set_xlabel('2θ (degrees)')
    axes[0,0].set_ylabel('Intensity (a.u.)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Statistical comparison
    original_stats = []
    augmented_stats = []

    for i in range(demo_patterns.shape[0]):
        start_idx = i * 3
        end_idx = start_idx + 3
        orig_pattern = demo_patterns[i]
        aug_patterns = augmented_batch[start_idx:end_idx]

        original_stats.append([orig_pattern.mean().item(), orig_pattern.std().item()])
        augmented_stats.append([aug_patterns.mean().item(), aug_patterns.std().item()])

    original_stats = np.array(original_stats)
    augmented_stats = np.array(augmented_stats)

    axes[0,1].scatter(original_stats[:, 0], original_stats[:, 1],
                     c='blue', alpha=0.7, s=100, label='Original', marker='o')
    axes[0,1].scatter(augmented_stats[:, 0], augmented_stats[:, 1],
                     c='red', alpha=0.7, s=100, label='Augmented', marker='^')

    axes[0,1].set_xlabel('Mean Intensity')
    axes[0,1].set_ylabel('Std Intensity')
    axes[0,1].set_title('Statistical Scatter Plot')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Custom vs Basic augmentation
    axes[1,0].plot(two_theta_axis, single_pattern.numpy().flatten(),
                  'k-', linewidth=2, label='Original', alpha=0.9)
    axes[1,0].plot(two_theta_axis, augmented_basic[0].numpy().flatten(),
                  'b-', alpha=0.7, linewidth=1, label='Basic Aug')
    axes[1,0].plot(two_theta_axis, augmented_custom[0].numpy().flatten(),
                  'r-', alpha=0.7, linewidth=1, label='Custom Aug')

    axes[1,0].set_title('Basic vs Custom Augmentation')
    axes[1,0].set_xlabel('2θ (degrees)')
    axes[1,0].set_ylabel('Intensity (a.u.)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Intensity distribution comparison
    orig_intensities = demo_patterns.numpy().flatten()
    aug_intensities = augmented_batch.numpy().flatten()

    axes[1,1].hist(orig_intensities, bins=50, alpha=0.6, label='Original', density=True, color='blue')
    axes[1,1].hist(aug_intensities, bins=50, alpha=0.6, label='Augmented', density=True, color='red')

    axes[1,1].set_xlabel('Intensity (a.u.)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Intensity Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization
    vis_path = os.path.join(output_dir, "demo_comparison.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"   Saved comparison plot: {vis_path}")

    # Visualization 2: Individual pattern showcase
    augmenter.visualize_augmentation(
        original_pattern=single_pattern,
        augmented_patterns=augmented_basic,
        save_path=os.path.join(output_dir, "demo_individual_showcase.png"),
        show_plot=False
    )

    plt.show()

    # Demo 5: Command line usage examples
    print(f"\n💡 Command Line Usage Examples:")
    print("-" * 40)
    print(f"Basic usage (no training required):")
    print(f"  python xrd_pattern_augmenter.py \\")
    print(f"    --input_file data/synthetic_patterns.pt \\")
    print(f"    --no_training \\")
    print(f"    --samples_per_pattern 5")
    print()
    print(f"Advanced usage:")
    print(f"  python classical_xrd_augmenter.py \\")
    print(f"    --input_file data/patterns.pt \\")
    print(f"    --samples_per_pattern 10 \\")
    print(f"    --visualize \\")
    print(f"    --output_dir ./my_augmented_data")

    print(f"\n✅ Demo Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Key Benefits of No-Training Augmentation:")
    print(f"   • ⚡ Instant usage - no model training required")
    print(f"   • 🎯 Physically realistic XRD augmentations")
    print(f"   • 🔧 Highly configurable parameters")
    print(f"   • 📈 Excellent for data expansion and robustness testing")
    print(f"   • 🚀 Fast processing - CPU friendly")

    return {
        'original_patterns': demo_patterns,
        'augmented_basic': augmented_basic,
        'augmented_batch': augmented_batch,
        'augmented_custom': augmented_custom,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    results = demonstrate_no_training_augmentation()