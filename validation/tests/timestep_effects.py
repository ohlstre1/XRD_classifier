"""
Timestep effect testing functionality for XRD diffusion validation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import find_peaks


def analyze_timestep_effects(model, diffusion, sample_synth: torch.Tensor,
                            sample_dtw: torch.Tensor, timesteps: Optional[List[int]] = None) -> Dict[int, Dict]:
    """
    Analyze how different timesteps affect the model output.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        sample_synth: Synthetic sample [L] or [1, L]
        sample_dtw: DTW value scalar or [1]
        timesteps: List of timesteps to test

    Returns:
        Dictionary mapping timestep to results
    """
    if timesteps is None:
        timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

    model.set_stochastic_mode(False)  # Deterministic for consistent comparison
    model.eval()

    # Ensure correct tensor shapes
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)
    if sample_dtw.dim() == 0:
        sample_dtw = sample_dtw.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth.unsqueeze(0).to(device)  # [1, 1, L]
    dtw = sample_dtw.unsqueeze(0).to(device)   # [1, 1]

    print(f"Analyzing timesteps - Input shape: {x.shape}, DTW shape: {dtw.shape}")

    results = {}

    with torch.no_grad():
        for t_val in tqdm(timesteps, desc="Testing timesteps"):
            t = torch.tensor([t_val], device=device, dtype=torch.long)

            # Get noisy version (forward diffusion)
            x_noisy, noise_true = diffusion.forward_diffusion(x, t)

            # Get model prediction
            noise_pred = model(x_noisy, t, dtw)

            # Estimate x0 from the noisy version
            alpha_bar_t = diffusion.alpha_bars[t].view(1, 1, 1)
            x0_pred = (x_noisy - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            results[t_val] = {
                'noisy': x_noisy[0].cpu(),
                'noise_true': noise_true[0].cpu(),
                'noise_pred': noise_pred[0].cpu(),
                'x0_pred': x0_pred[0].cpu(),
                'noise_mse': nn.MSELoss()(noise_pred, noise_true).item(),
                'reconstruction_mse': nn.MSELoss()(x0_pred, x).item()
            }

    return results


def test_progressive_augmentation(diffusion, sample_pattern: torch.Tensor,
                                 timesteps: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """
    Test how the diffusion process's augmentation changes with timestep.

    Args:
        diffusion: The diffusion process
        sample_pattern: Pattern to augment [L] or [1, L]
        timesteps: List of timesteps to test

    Returns:
        Dictionary mapping timestep to augmented pattern
    """
    if timesteps is None:
        timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

    # Ensure correct tensor shape
    if sample_pattern.dim() == 1:
        sample_pattern = sample_pattern.unsqueeze(0)

    x = sample_pattern.unsqueeze(0).to(diffusion.device)  # [1, 1, L]
    augmented_patterns = {}

    print(f"Testing progressive augmentation - Input shape: {x.shape}")

    for t_val in timesteps:
        t = torch.tensor([t_val], device=diffusion.device, dtype=torch.long)

        # Apply only the augmentation (without noise)
        x_aug = diffusion.augment(x, t)
        augmented_patterns[t_val] = x_aug[0].cpu()

    return augmented_patterns


def quantify_augmentation_strength(original: torch.Tensor,
                                  augmented_patterns: Dict[int, torch.Tensor]) -> Dict[int, Dict]:
    """
    Quantify augmentation strength across different timesteps.

    Args:
        original: Original pattern [1, L]
        augmented_patterns: Dictionary mapping timestep to augmented pattern

    Returns:
        Dictionary with augmentation metrics for each timestep
    """
    aug_metrics = {}
    original_np = original[0].numpy()

    for t, aug_pattern in augmented_patterns.items():
        augmented_np = aug_pattern[0].numpy()

        # Calculate various metrics
        mse = mean_squared_error(original_np, augmented_np)
        mae = mean_absolute_error(original_np, augmented_np)
        correlation = np.corrcoef(original_np, augmented_np)[0, 1]

        # Peak analysis
        try:
            orig_peaks, _ = find_peaks(original_np, height=0.1)
            aug_peaks, _ = find_peaks(augmented_np, height=0.1)

            peak_shift = len(aug_peaks) - len(orig_peaks)
        except:
            orig_peaks, aug_peaks = [], []
            peak_shift = 0

        aug_metrics[t] = {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'peak_count_orig': len(orig_peaks),
            'peak_count_aug': len(aug_peaks),
            'peak_shift': peak_shift
        }

    return aug_metrics


def validate_timestep_effects(timestep_results: Dict[int, Dict]) -> Dict:
    """
    Validate that timestep effects follow expected patterns.

    Args:
        timestep_results: Results from analyze_timestep_effects

    Returns:
        Validation metrics
    """
    timesteps = sorted(timestep_results.keys())
    noise_mses = [timestep_results[t]['noise_mse'] for t in timesteps]
    recon_mses = [timestep_results[t]['reconstruction_mse'] for t in timesteps]

    # Validate that higher timesteps generally have higher reconstruction error
    timestep_correlation = stats.spearmanr(timesteps, recon_mses)[0]

    # Check if noise prediction quality varies with timestep
    noise_mse_std = np.std(noise_mses)

    return {
        'timestep_recon_correlation': timestep_correlation,
        'noise_mse_variability': noise_mse_std,
        'test_pass': timestep_correlation > 0.5,  # Should be positive correlation
        'noise_mses': noise_mses,
        'recon_mses': recon_mses,
        'timesteps': timesteps
    }


def analyze_dtw_timestep_interaction(model, sample_synth: torch.Tensor,
                                   dtw_range: Tuple[float, float] = (0.0, 1.0),
                                   timestep_range: Tuple[int, int] = (0, 500)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze how DTW conditioning interacts with different timesteps.

    Args:
        model: The diffusion model
        sample_synth: Synthetic sample [L] or [1, L]
        dtw_range: Range of DTW values to test
        timestep_range: Range of timesteps to test

    Returns:
        Tuple of (results_matrix, dtw_values, timesteps)
    """
    model.set_stochastic_mode(False)
    model.eval()

    dtw_values = np.linspace(dtw_range[0], dtw_range[1], 6)  # 6 DTW values
    timesteps = np.linspace(timestep_range[0], timestep_range[1], 6, dtype=int)  # 6 timesteps

    # Ensure correct tensor shape
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth.unsqueeze(0).to(device)  # [1, 1, L]

    results = np.zeros((len(dtw_values), len(timesteps)))  # MSE matrix

    with torch.no_grad():
        for i, dtw_val in enumerate(tqdm(dtw_values, desc="DTW values")):
            dtw = torch.tensor([[dtw_val]], device=device, dtype=torch.float32)

            for j, t_val in enumerate(timesteps):
                t = torch.tensor([t_val], device=device, dtype=torch.long)

                # Get model output
                output = model(x, t, dtw)
                mse = nn.MSELoss()(output, x).item()
                results[i, j] = mse

    return results, dtw_values, timesteps


def batch_timestep_analysis(model, diffusion, test_synth: torch.Tensor,
                           test_dtw: torch.Tensor, n_samples: int = 3) -> List[Dict]:
    """
    Run timestep analysis on multiple samples.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        test_synth: Test synthetic patterns [N, ...]
        test_dtw: Test DTW values [N]
        n_samples: Number of samples to analyze

    Returns:
        List of timestep analysis results
    """
    results = []

    for i in tqdm(range(min(n_samples, len(test_synth))), desc="Analyzing timestep effects"):
        sample_results = analyze_timestep_effects(
            model, diffusion, test_synth[i], test_dtw[i]
        )

        validation_metrics = validate_timestep_effects(sample_results)
        validation_metrics['sample_index'] = i
        validation_metrics['timestep_results'] = sample_results

        results.append(validation_metrics)

    return results