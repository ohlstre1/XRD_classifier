"""
Plotting utilities for XRD diffusion validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import torch


def setup_plotting_style():
    """Setup consistent plotting style for validation plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_stochasticity_analysis(det_outputs: np.ndarray, sto_outputs: np.ndarray,
                               sample_index: int = 0) -> plt.Figure:
    """
    Plot stochasticity analysis results.

    Args:
        det_outputs: Deterministic mode outputs [n_runs, batch, channels, length]
        sto_outputs: Stochastic mode outputs [n_runs, batch, channels, length]
        sample_index: Sample index to plot

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot deterministic outputs
    axes[0, 0].plot(det_outputs[0, 0, 0, :], 'b-', linewidth=2, label='Run 1')
    for i in range(1, min(5, len(det_outputs))):
        axes[0, 0].plot(det_outputs[i, 0, 0, :], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Deterministic Mode (Multiple Runs)')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].legend()

    # Plot stochastic outputs
    axes[0, 1].plot(sto_outputs[0, 0, 0, :], 'r-', linewidth=2, label='Run 1')
    for i in range(1, min(5, len(sto_outputs))):
        axes[0, 1].plot(sto_outputs[i, 0, 0, :], 'r-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Stochastic Mode (Multiple Runs)')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].legend()

    # Plot variability comparison
    det_std_per_pos = np.std(det_outputs, axis=0)[0, 0, :]
    sto_std_per_pos = np.std(sto_outputs, axis=0)[0, 0, :]

    axes[1, 0].plot(det_std_per_pos, 'b-', label='Deterministic', linewidth=2)
    axes[1, 0].plot(sto_std_per_pos, 'r-', label='Stochastic', linewidth=2)
    axes[1, 0].set_title('Standard Deviation Across Runs')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')

    # Plot histogram of differences
    det_diffs = np.abs(det_outputs - det_outputs[0]).flatten()
    sto_diffs = np.abs(sto_outputs - sto_outputs[0]).flatten()

    axes[1, 1].hist(det_diffs, bins=50, alpha=0.7, label='Deterministic', density=True)
    axes[1, 1].hist(sto_diffs, bins=50, alpha=0.7, label='Stochastic', density=True)
    axes[1, 1].set_title('Distribution of Absolute Differences')
    axes[1, 1].set_xlabel('Absolute Difference')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    return fig


def plot_stochastic_variations(original: torch.Tensor, real: torch.Tensor,
                              variations: torch.Tensor, dtw_value: float,
                              sample_index: int = 0) -> plt.Figure:
    """
    Plot stochastic variations of a single sample.

    Args:
        original: Original synthetic pattern [channels, length]
        real: Real pattern [channels, length]
        variations: Stochastic variations [n_variations, channels, length]
        dtw_value: DTW distance value
        sample_index: Sample index for title

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original synthetic
    axes[0].plot(original[0].cpu(), 'k-', linewidth=2, label='Synthetic')
    axes[0].set_title(f'Sample {sample_index}: Original Synthetic')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()

    # Plot all variations
    for i, var in enumerate(variations):
        alpha = 0.7 if i == 0 else 0.4
        axes[1].plot(var[0], alpha=alpha, linewidth=1)
    axes[1].set_title(f'Sample {sample_index}: Stochastic Variations')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Intensity')

    # Plot comparison with real
    axes[2].plot(real[0].cpu(), 'g-', linewidth=2, label='Real', alpha=0.8)
    # Plot mean and std of variations
    var_mean = variations.mean(dim=0)[0]
    var_std = variations.std(dim=0)[0]
    axes[2].plot(var_mean, 'r-', linewidth=2, label='Mean Variation')
    axes[2].fill_between(range(len(var_mean)),
                        var_mean - var_std, var_mean + var_std,
                        alpha=0.3, color='red', label='±1 Std')
    axes[2].set_title(f'Sample {sample_index}: Comparison with Real (DTW: {dtw_value:.3f})')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Intensity')
    axes[2].legend()

    plt.tight_layout()
    return fig


def plot_timestep_analysis(timestep_results: Dict[int, Dict], original: torch.Tensor,
                          sample_index: int = 0) -> plt.Figure:
    """
    Plot timestep analysis results.

    Args:
        timestep_results: Results from analyze_timestep_effects
        original: Original pattern [channels, length]
        sample_index: Sample index for titles

    Returns:
        Figure object
    """
    timesteps = sorted(timestep_results.keys())
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Plot 1: Original vs noisy at different timesteps
    for i, (ax, t) in enumerate(zip(axes[0], [0, 200, 800])):
        if t in timestep_results:
            ax.plot(original[0].cpu(), 'k-', linewidth=2, label='Original', alpha=0.8)
            ax.plot(timestep_results[t]['noisy'][0], 'r-', linewidth=1.5,
                   label=f'Noisy (t={t})', alpha=0.7)
            ax.set_title(f'Timestep {t}: Forward Diffusion')
            ax.set_xlabel('Position')
            ax.set_ylabel('Intensity')
            ax.legend()

    # Plot 2: Noise prediction quality
    for i, (ax, t) in enumerate(zip(axes[1], [0, 200, 800])):
        if t in timestep_results:
            ax.plot(timestep_results[t]['noise_true'][0], 'b-', linewidth=2,
                   label='True Noise', alpha=0.8)
            ax.plot(timestep_results[t]['noise_pred'][0], 'orange', linewidth=1.5,
                   label='Predicted Noise', alpha=0.7)
            ax.set_title(f'Timestep {t}: Noise Prediction (MSE: {timestep_results[t]["noise_mse"]:.4f})')
            ax.set_xlabel('Position')
            ax.set_ylabel('Noise Amplitude')
            ax.legend()

    # Plot 3: Reconstruction quality
    for i, (ax, t) in enumerate(zip(axes[2], [0, 200, 800])):
        if t in timestep_results:
            ax.plot(original[0].cpu(), 'k-', linewidth=2, label='Original', alpha=0.8)
            ax.plot(timestep_results[t]['x0_pred'][0], 'g-', linewidth=1.5,
                   label='Reconstructed', alpha=0.7)
            ax.set_title(f'Timestep {t}: Reconstruction (MSE: {timestep_results[t]["reconstruction_mse"]:.4f})')
            ax.set_xlabel('Position')
            ax.set_ylabel('Intensity')
            ax.legend()

    plt.suptitle(f'Timestep Analysis - Sample {sample_index}', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


def plot_timestep_metrics(timestep_results: Dict[int, Dict]) -> plt.Figure:
    """
    Plot timestep metrics (MSE vs timestep).

    Args:
        timestep_results: Results from analyze_timestep_effects

    Returns:
        Figure object
    """
    timesteps = sorted(timestep_results.keys())
    noise_mses = [timestep_results[t]['noise_mse'] for t in timesteps]
    recon_mses = [timestep_results[t]['reconstruction_mse'] for t in timesteps]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(timesteps, noise_mses, 'bo-', linewidth=2, markersize=6)
    axes[0].set_title('Noise Prediction Quality vs Timestep')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Noise MSE')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, recon_mses, 'ro-', linewidth=2, markersize=6)
    axes[1].set_title('Reconstruction Quality vs Timestep')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Reconstruction MSE')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_progressive_augmentation(original: torch.Tensor, aug_patterns: Dict[int, torch.Tensor]) -> plt.Figure:
    """
    Plot progressive augmentation effects.

    Args:
        original: Original pattern [channels, length]
        aug_patterns: Dictionary mapping timestep to augmented pattern

    Returns:
        Figure object
    """
    timesteps = sorted(aug_patterns.keys())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, t in enumerate(timesteps):
        ax = axes[i//4, i%4]
        ax.plot(original[0].cpu(), 'k-', linewidth=2, label='Original', alpha=0.8)
        ax.plot(aug_patterns[t][0], 'r-', linewidth=1.5, label=f'Augmented (t={t})', alpha=0.7)
        ax.set_title(f'Progressive Augmentation: t={t}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Progressive Augmentation Effects', fontsize=16)
    plt.tight_layout()
    return fig


def plot_dtw_analysis(dtw_results: Dict[float, Dict], original: torch.Tensor,
                     real: torch.Tensor, original_dtw: float, sample_index: int = 0) -> plt.Figure:
    """
    Plot DTW conditioning analysis.

    Args:
        dtw_results: Results from analyze_dtw_conditioning
        original: Original synthetic pattern [channels, length]
        real: Real pattern [channels, length]
        original_dtw: Original DTW value
        sample_index: Sample index for titles

    Returns:
        Figure object
    """
    dtw_values = sorted(dtw_results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Show a few key DTW values
    key_dtw_values = [0.0, 0.3, 0.6, 1.0]

    for i, dtw_val in enumerate(key_dtw_values[:3]):
        if dtw_val in dtw_results:
            ax = axes[0, i]
            ax.plot(original[0].cpu(), 'k-', linewidth=2, label='Synthetic', alpha=0.8)
            ax.plot(real[0].cpu(), 'g-', linewidth=2, label='Real', alpha=0.6)
            ax.plot(dtw_results[dtw_val]['output'][0], 'r-', linewidth=1.5,
                   label=f'Transformed (DTW={dtw_val:.1f})', alpha=0.7)
            ax.set_title(f'DTW Conditioning: {dtw_val:.1f}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Show original DTW value
    ax = axes[0, 2]
    nearest_dtw = min(dtw_values, key=lambda x: abs(x - original_dtw))
    ax.plot(original[0].cpu(), 'k-', linewidth=2, label='Synthetic', alpha=0.8)
    ax.plot(real[0].cpu(), 'g-', linewidth=2, label='Real', alpha=0.6)
    ax.plot(dtw_results[nearest_dtw]['output'][0], 'r-', linewidth=1.5,
           label=f'Transformed (DTW={nearest_dtw:.1f})', alpha=0.7)
    ax.set_title(f'Original DTW: {original_dtw:.3f} (≈{nearest_dtw:.1f})')
    ax.set_xlabel('Position')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot heatmap of transformations
    transformation_matrix = np.array([dtw_results[dtw_val]['output'][0].numpy() for dtw_val in dtw_values])

    im = axes[1, 0].imshow(transformation_matrix, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Transformation Heatmap')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('DTW Value Index')
    axes[1, 0].set_yticks(range(0, len(dtw_values), 2))
    axes[1, 0].set_yticklabels([f'{dtw_values[i]:.1f}' for i in range(0, len(dtw_values), 2)])
    plt.colorbar(im, ax=axes[1, 0])

    # Plot MSE vs DTW value
    mse_values = [dtw_results[dtw_val]['mse_vs_input'] for dtw_val in dtw_values]
    axes[1, 1].plot(dtw_values, mse_values, 'bo-', linewidth=2, markersize=6)
    axes[1, 1].axvline(x=original_dtw, color='r', linestyle='--', alpha=0.7,
                      label=f'Original DTW: {original_dtw:.3f}')
    axes[1, 1].set_title('Transformation Strength vs DTW')
    axes[1, 1].set_xlabel('DTW Conditioning Value')
    axes[1, 1].set_ylabel('MSE vs Input')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot difference from original DTW
    if nearest_dtw in dtw_results:
        reference_output = dtw_results[nearest_dtw]['output'][0]
        differences = []
        for dtw_val in dtw_values:
            diff = np.mean(np.abs(dtw_results[dtw_val]['output'][0].numpy() - reference_output.numpy()))
            differences.append(diff)

        axes[1, 2].plot(dtw_values, differences, 'ro-', linewidth=2, markersize=6)
        axes[1, 2].axvline(x=original_dtw, color='k', linestyle='--', alpha=0.7,
                          label=f'Original DTW: {original_dtw:.3f}')
        axes[1, 2].set_title(f'Difference from DTW={nearest_dtw:.1f}')
        axes[1, 2].set_xlabel('DTW Conditioning Value')
        axes[1, 2].set_ylabel('Mean Absolute Difference')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'DTW Conditioning Analysis - Sample {sample_index}', fontsize=16, y=0.95)
    plt.tight_layout()
    return fig


def plot_test_performance(test_results: Dict[str, np.ndarray], test_dtw: Optional[torch.Tensor] = None) -> plt.Figure:
    """
    Plot comprehensive test performance analysis.

    Args:
        test_results: Results from evaluate_test_set_performance
        test_dtw: Optional DTW values for relationship plots

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Loss distributions
    axes[0, 0].hist(test_results['diffusion_losses'], bins=50, alpha=0.7, label='Diffusion Loss')
    axes[0, 0].hist(test_results['reconstruction_losses'], bins=50, alpha=0.7, label='Reconstruction Loss')
    axes[0, 0].hist(test_results['direct_transform_losses'], bins=50, alpha=0.7, label='Transform Loss')
    axes[0, 0].set_title('Loss Distributions')
    axes[0, 0].set_xlabel('Loss Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # 2. Real similarity distribution
    axes[0, 1].hist(test_results['real_similarities'], bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(test_results['real_similarities']), color='red',
                      linestyle='--', label=f'Mean: {np.mean(test_results["real_similarities"]):.3f}')
    axes[0, 1].set_title('Similarity to Real Patterns')
    axes[0, 1].set_xlabel('Correlation with Real')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # 3. Peak correlation distribution
    axes[0, 2].hist(test_results['peak_correlations'], bins=30, alpha=0.7, color='orange')
    axes[0, 2].axvline(np.mean(test_results['peak_correlations']), color='red',
                      linestyle='--', label=f'Mean: {np.mean(test_results["peak_correlations"]):.3f}')
    axes[0, 2].set_title('Peak Position Correlation')
    axes[0, 2].set_xlabel('Peak Correlation')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    # 4-6. DTW relationship plots (if DTW values provided)
    if test_dtw is not None:
        # Sort by DTW values for plotting
        dtw_sorted_idx = np.argsort(test_dtw.cpu().numpy())
        sorted_dtw = test_dtw.cpu().numpy()[dtw_sorted_idx]
        sorted_transform_loss = test_results['direct_transform_losses'][dtw_sorted_idx]

        # Bin and average for cleaner plot
        n_bins = 20
        bin_edges = np.linspace(sorted_dtw.min(), sorted_dtw.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_losses = []

        for i in range(n_bins):
            mask = (sorted_dtw >= bin_edges[i]) & (sorted_dtw < bin_edges[i + 1])
            if mask.sum() > 0:
                binned_losses.append(sorted_transform_loss[mask].mean())
            else:
                binned_losses.append(np.nan)

        axes[1, 0].plot(bin_centers, binned_losses, 'bo-', markersize=6)
        axes[1, 0].set_title('Transform Loss vs DTW Distance')
        axes[1, 0].set_xlabel('DTW Distance')
        axes[1, 0].set_ylabel('Average Transform Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Similarity vs DTW relationship
        sorted_similarity = test_results['real_similarities'][dtw_sorted_idx]
        binned_similarities = []

        for i in range(n_bins):
            mask = (sorted_dtw >= bin_edges[i]) & (sorted_dtw < bin_edges[i + 1])
            if mask.sum() > 0:
                binned_similarities.append(sorted_similarity[mask].mean())
            else:
                binned_similarities.append(np.nan)

        axes[1, 1].plot(bin_centers, binned_similarities, 'go-', markersize=6)
        axes[1, 1].set_title('Real Similarity vs DTW Distance')
        axes[1, 1].set_xlabel('DTW Distance')
        axes[1, 1].set_ylabel('Average Similarity to Real')
        axes[1, 1].grid(True, alpha=0.3)

    # Performance summary statistics
    stats_text = f"""
Test Set Performance Summary:

Diffusion Loss: {np.mean(test_results['diffusion_losses']):.4f} ± {np.std(test_results['diffusion_losses']):.4f}
Reconstruction Loss: {np.mean(test_results['reconstruction_losses']):.4f} ± {np.std(test_results['reconstruction_losses']):.4f}
Transform Loss: {np.mean(test_results['direct_transform_losses']):.4f} ± {np.std(test_results['direct_transform_losses']):.4f}

Real Similarity: {np.mean(test_results['real_similarities']):.3f} ± {np.std(test_results['real_similarities']):.3f}
Peak Correlation: {np.mean(test_results['peak_correlations']):.3f} ± {np.std(test_results['peak_correlations']):.3f}

High similarity (>0.8): {np.sum(test_results['real_similarities'] > 0.8) / len(test_results['real_similarities']) * 100:.1f}%
Low loss (<0.01): {np.sum(test_results['direct_transform_losses'] < 0.01) / len(test_results['direct_transform_losses']) * 100:.1f}%
    """.strip()

    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Performance Summary')

    plt.tight_layout()
    return fig