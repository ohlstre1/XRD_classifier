#!/usr/bin/env python3
"""
Example script demonstrating XRD Pattern Augmentation
====================================================

This script shows how to use the XRD Pattern Augmenter to generate
realistic experimental-like patterns from synthetic XRD data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from xrd_pattern_augmenter import XRDPatternAugmenter, load_synthetic_patterns
import os

def create_demo_patterns():
    """Create some demo synthetic XRD patterns if no data file is available."""
    print("📝 Creating demo synthetic XRD patterns...")

    # Create realistic-looking XRD patterns
    n_patterns = 10
    n_points = 1000
    two_theta = np.linspace(10, 80, n_points)

    patterns = []
    for i in range(n_patterns):
        # Create pattern with multiple peaks at different positions
        pattern = np.zeros(n_points)

        # Add some characteristic peaks with random positions and intensities
        peak_positions = [20 + i, 30 + i*0.5, 45 + i*0.3, 60 + i*0.2]
        peak_intensities = [100, 80, 60, 40]
        peak_widths = [1.0, 1.2, 0.8, 1.5]

        for pos, intensity, width in zip(peak_positions, peak_intensities, peak_widths):
            # Gaussian peaks
            pattern += intensity * np.exp(-((two_theta - pos) / width) ** 2)

        # Add some background
        background = 5 + 2 * np.random.random()
        pattern += background

        # Add small amount of noise
        pattern += np.random.normal(0, 2, n_points)

        # Ensure non-negative
        pattern = np.maximum(pattern, 0)

        patterns.append(pattern)

    patterns = torch.tensor(patterns, dtype=torch.float32)
    print(f"   Created {n_patterns} demo patterns with {n_points} points each")

    return patterns

def run_augmentation_example():
    """Run a comprehensive augmentation example."""
    print("🔬 XRD Pattern Augmentation Example")
    print("=" * 50)

    # Configuration
    data_file = "data/xrd_dataset_labeled_dtw_window.pt"
    model_path = "./models/xrd_diffusion/improved_diffusion_model_best.pth"
    output_dir = "./augmentation_example"

    # Load or create synthetic patterns
    if os.path.exists(data_file):
        print(f"📂 Loading synthetic patterns from: {data_file}")
        try:
            synth_patterns = load_synthetic_patterns(data_file, max_patterns=5)
            print(f"   Loaded {synth_patterns.shape[0]} patterns")
        except Exception as e:
            print(f"   Error loading data: {e}")
            print("   Using demo patterns instead")
            synth_patterns = create_demo_patterns()[:5]
    else:
        print(f"   Data file not found: {data_file}")
        synth_patterns = create_demo_patterns()[:5]

    # Initialize augmenter
    print(f"\n🤖 Initializing XRD Pattern Augmenter...")
    augmenter = XRDPatternAugmenter(
        model_path=model_path,
        device='auto',
        verbose=True
    )

    # Example 1: Basic augmentation
    print(f"\n📋 Example 1: Basic Pattern Augmentation")
    print("-" * 40)

    pattern_idx = 0
    single_pattern = synth_patterns[pattern_idx:pattern_idx+1]

    augmented_basic, metadata_basic = augmenter.augment_pattern(
        synth_pattern=single_pattern,
        num_samples=5,
        temp_range=(0.1, 1.0),
        temp_mode='random',
        base_seed=42,
        return_metadata=True
    )

    print(f"   Input shape: {single_pattern.shape}")
    print(f"   Output shape: {augmented_basic.shape}")
    print(f"   Temperature range used: {metadata_basic['temperatures'].min():.3f} - {metadata_basic['temperatures'].max():.3f}")

    # Example 2: Different temperature modes
    print(f"\n📋 Example 2: Different Temperature Modes")
    print("-" * 40)

    temp_modes = ['random', 'linear', 'exponential', 1.5]  # Last one is fixed temperature
    temp_results = {}

    for mode in temp_modes:
        augmented, metadata = augmenter.augment_pattern(
            synth_pattern=single_pattern,
            num_samples=3,
            temp_range=(0.2, 2.0),
            temp_mode=mode,
            base_seed=123,
            return_metadata=True
        )
        temp_results[str(mode)] = (augmented, metadata)

        temps = metadata['temperatures'].flatten()
        print(f"   {str(mode):12}: temps = [{temps.min():.3f}, {temps.max():.3f}]")

    # Example 3: Batch augmentation
    print(f"\n📋 Example 3: Batch Augmentation")
    print("-" * 40)

    batch_augmented, batch_metadata = augmenter.augment_batch(
        synth_patterns=synth_patterns,
        samples_per_pattern=3,
        temp_range=(0.1, 1.5),
        temp_mode='random',
        base_seed=999,
        batch_size=2,
        progress_bar=True
    )

    print(f"   Input batch shape: {synth_patterns.shape}")
    print(f"   Output batch shape: {batch_augmented.shape}")
    print(f"   Metadata entries: {len(batch_metadata)}")

    # Example 4: Different noise levels
    print(f"\n📋 Example 4: Different Noise Levels")
    print("-" * 40)

    noise_levels = [(0, 10), (10, 30), (30, 60)]
    noise_results = {}

    for noise_range in noise_levels:
        augmented, metadata = augmenter.augment_pattern(
            synth_pattern=single_pattern,
            num_samples=2,
            noise_timestep_range=noise_range,
            base_seed=456,
            return_metadata=True
        )
        noise_results[str(noise_range)] = (augmented, metadata)

        timesteps = metadata['timesteps']
        noise_lvls = metadata['noise_levels']
        print(f"   Range {noise_range}: timesteps = {timesteps}, noise = {[f'{n:.3f}' for n in noise_lvls]}")

    # Save results
    print(f"\n💾 Saving Results...")
    print("-" * 40)

    os.makedirs(output_dir, exist_ok=True)

    # Save batch results
    augmenter.save_results(
        augmented_patterns=batch_augmented,
        metadata_list=batch_metadata,
        original_patterns=synth_patterns,
        output_dir=output_dir,
        prefix="example_batch"
    )

    # Create comprehensive visualization
    print(f"\n📊 Creating Visualizations...")
    print("-" * 40)

    # Visualization 1: Basic augmentation
    vis_path_1 = os.path.join(output_dir, "example_1_basic_augmentation.png")
    augmenter.visualize_augmentation(
        original_pattern=single_pattern,
        augmented_patterns=augmented_basic,
        metadata=metadata_basic,
        save_path=vis_path_1,
        show_plot=False
    )

    # Visualization 2: Temperature mode comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    x_axis = np.linspace(0, 90, single_pattern.shape[-1])

    for i, (mode_name, (patterns, metadata)) in enumerate(temp_results.items()):
        if i >= 4:
            break

        axes[i].plot(x_axis, single_pattern.cpu().numpy().flatten(),
                    'k-', linewidth=2, label='Original', alpha=0.8)

        for j, pattern in enumerate(patterns):
            temp_val = metadata['temperatures'][j, 0]
            axes[i].plot(x_axis, pattern.cpu().numpy().flatten(),
                        alpha=0.7, linewidth=1,
                        label=f'T={temp_val:.2f}')

        axes[i].set_title(f'Temperature Mode: {mode_name}')
        axes[i].set_xlabel('2θ (degrees)')
        axes[i].set_ylabel('Intensity')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    vis_path_2 = os.path.join(output_dir, "example_2_temperature_comparison.png")
    plt.savefig(vis_path_2, dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Noise level comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (noise_range, (patterns, metadata)) in enumerate(noise_results.items()):
        axes[i].plot(x_axis, single_pattern.cpu().numpy().flatten(),
                    'k-', linewidth=2, label='Original', alpha=0.8)

        for j, pattern in enumerate(patterns):
            noise_level = metadata['noise_levels'][j]
            timestep = metadata['timesteps'][j]
            axes[i].plot(x_axis, pattern.cpu().numpy().flatten(),
                        alpha=0.7, linewidth=1,
                        label=f't={timestep}, n={noise_level:.3f}')

        axes[i].set_title(f'Noise Range: {noise_range}')
        axes[i].set_xlabel('2θ (degrees)')
        axes[i].set_ylabel('Intensity')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    vis_path_3 = os.path.join(output_dir, "example_3_noise_comparison.png")
    plt.savefig(vis_path_3, dpi=300, bbox_inches='tight')
    plt.close()

    # Summary statistics
    print(f"\n📈 Summary Statistics")
    print("-" * 40)

    original_mean = synth_patterns.mean().item()
    original_std = synth_patterns.std().item()
    augmented_mean = batch_augmented.mean().item()
    augmented_std = batch_augmented.std().item()

    print(f"   Original patterns:")
    print(f"     Mean intensity: {original_mean:.4f}")
    print(f"     Std intensity:  {original_std:.4f}")
    print(f"   Augmented patterns:")
    print(f"     Mean intensity: {augmented_mean:.4f}")
    print(f"     Std intensity:  {augmented_std:.4f}")
    print(f"   Intensity change: {((augmented_mean - original_mean) / original_mean * 100):.2f}%")
    print(f"   Variability change: {((augmented_std - original_std) / original_std * 100):.2f}%")

    print(f"\n✅ Augmentation Example Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Visualizations:")
    print(f"   • {vis_path_1}")
    print(f"   • {vis_path_2}")
    print(f"   • {vis_path_3}")

    return {
        'original_patterns': synth_patterns,
        'augmented_patterns': batch_augmented,
        'metadata': batch_metadata,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    results = run_augmentation_example()