"""
DTW conditioning testing functionality for XRD diffusion validation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from scipy import stats


def analyze_dtw_conditioning(model, sample_synth: torch.Tensor,
                           dtw_values: Optional[np.ndarray] = None,
                           fixed_timestep: int = 0) -> Dict[float, Dict]:
    """
    Analyze how different DTW conditioning values affect model output.

    Args:
        model: The diffusion model
        sample_synth: Synthetic sample [L] or [1, L]
        dtw_values: Array of DTW values to test
        fixed_timestep: Timestep to use for testing

    Returns:
        Dictionary mapping DTW value to results
    """
    if dtw_values is None:
        dtw_values = np.linspace(0.0, 1.0, 11)  # 0.0 to 1.0 in steps of 0.1

    model.set_stochastic_mode(False)
    model.eval()

    # Ensure correct tensor shape
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth.unsqueeze(0).to(device)  # [1, 1, L]
    t = torch.tensor([fixed_timestep], device=device, dtype=torch.long)

    results = {}

    with torch.no_grad():
        for dtw_val in tqdm(dtw_values, desc="Testing DTW values"):
            dtw = torch.tensor([[dtw_val]], device=device, dtype=torch.float32)

            # Get model output
            output = model(x, t, dtw)

            results[dtw_val] = {
                'output': output[0].cpu(),
                'mse_vs_input': nn.MSELoss()(output, x).item()
            }

    return results


def validate_dtw_conditioning(dtw_results: Dict[float, Dict],
                             original_dtw: float) -> Dict:
    """
    Validate DTW conditioning effectiveness.

    Args:
        dtw_results: Results from analyze_dtw_conditioning
        original_dtw: Original DTW value for the sample

    Returns:
        Validation metrics
    """
    dtw_values = sorted(dtw_results.keys())
    transform_mses = [dtw_results[d]['mse_vs_input'] for d in dtw_values]

    # Check if DTW conditioning affects output (should have variation)
    dtw_effect_std = np.std(transform_mses)
    dtw_effect_range = max(transform_mses) - min(transform_mses)

    # Check if there's a relationship between DTW value and transformation strength
    dtw_mse_correlation = stats.spearmanr(dtw_values, transform_mses)[0]

    # Find nearest tested DTW to original
    nearest_dtw = min(dtw_values, key=lambda x: abs(x - original_dtw))

    return {
        'effect_std': dtw_effect_std,
        'effect_range': dtw_effect_range,
        'dtw_mse_correlation': dtw_mse_correlation,
        'test_pass': dtw_effect_std > 1e-4,  # Should have noticeable effect
        'transform_mses': transform_mses,
        'dtw_values': dtw_values,
        'original_dtw': original_dtw,
        'nearest_tested_dtw': nearest_dtw
    }


def test_dtw_range_sensitivity(model, sample_synth: torch.Tensor,
                              n_points: int = 21) -> Dict[str, np.ndarray]:
    """
    Test model sensitivity across full DTW range.

    Args:
        model: The diffusion model
        sample_synth: Synthetic sample [L] or [1, L]
        n_points: Number of DTW points to test

    Returns:
        Dictionary with DTW values and corresponding outputs
    """
    model.set_stochastic_mode(False)
    model.eval()

    dtw_values = np.linspace(0.0, 1.0, n_points)

    # Ensure correct tensor shape
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth.unsqueeze(0).to(device)  # [1, 1, L]
    t = torch.zeros(1, dtype=torch.long, device=device)

    outputs = []
    mse_values = []

    with torch.no_grad():
        for dtw_val in tqdm(dtw_values, desc="Testing DTW sensitivity"):
            dtw = torch.tensor([[dtw_val]], device=device, dtype=torch.float32)

            output = model(x, t, dtw)
            mse = nn.MSELoss()(output, x).item()

            outputs.append(output[0, 0].cpu().numpy())
            mse_values.append(mse)

    return {
        'dtw_values': dtw_values,
        'outputs': np.array(outputs),
        'mse_values': np.array(mse_values)
    }


def analyze_dtw_feature_importance(model, sample_synth: torch.Tensor,
                                  base_dtw: float = 0.5,
                                  perturbation_size: float = 0.1) -> Dict:
    """
    Analyze how sensitive the model is to DTW value perturbations.

    Args:
        model: The diffusion model
        sample_synth: Synthetic sample [L] or [1, L]
        base_dtw: Base DTW value
        perturbation_size: Size of perturbation to test

    Returns:
        Feature importance metrics
    """
    model.set_stochastic_mode(False)
    model.eval()

    # Ensure correct tensor shape
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth.unsqueeze(0).to(device)  # [1, 1, L]
    t = torch.zeros(1, dtype=torch.long, device=device)

    # Test DTW values around the base
    test_dtws = [
        max(0.0, base_dtw - perturbation_size),
        base_dtw,
        min(1.0, base_dtw + perturbation_size)
    ]

    outputs = []

    with torch.no_grad():
        for dtw_val in test_dtws:
            dtw = torch.tensor([[dtw_val]], device=device, dtype=torch.float32)
            output = model(x, t, dtw)
            outputs.append(output[0, 0].cpu().numpy())

    # Calculate sensitivity metrics
    baseline_output = outputs[1]  # Middle value
    neg_perturbation_diff = np.mean(np.abs(outputs[0] - baseline_output))
    pos_perturbation_diff = np.mean(np.abs(outputs[2] - baseline_output))

    average_sensitivity = (neg_perturbation_diff + pos_perturbation_diff) / 2
    sensitivity_asymmetry = abs(neg_perturbation_diff - pos_perturbation_diff)

    return {
        'base_dtw': base_dtw,
        'perturbation_size': perturbation_size,
        'test_dtws': test_dtws,
        'outputs': outputs,
        'negative_perturbation_diff': neg_perturbation_diff,
        'positive_perturbation_diff': pos_perturbation_diff,
        'average_sensitivity': average_sensitivity,
        'sensitivity_asymmetry': sensitivity_asymmetry
    }


def compare_dtw_vs_real_similarity(model, sample_synth: torch.Tensor,
                                  sample_real: torch.Tensor,
                                  original_dtw: float,
                                  test_dtws: Optional[List[float]] = None) -> Dict:
    """
    Compare how different DTW values affect similarity to real pattern.

    Args:
        model: The diffusion model
        sample_synth: Synthetic sample [L] or [1, L]
        sample_real: Real sample [L] or [1, L]
        original_dtw: Original DTW distance between synth and real
        test_dtws: DTW values to test

    Returns:
        Comparison metrics
    """
    if test_dtws is None:
        test_dtws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, original_dtw]
        test_dtws = sorted(list(set(test_dtws)))  # Remove duplicates and sort

    model.set_stochastic_mode(False)
    model.eval()

    # Ensure correct tensor shapes
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)
    if sample_real.dim() == 1:
        sample_real = sample_real.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth.unsqueeze(0).to(device)  # [1, 1, L]
    t = torch.zeros(1, dtype=torch.long, device=device)

    real_pattern = sample_real[0].numpy()
    results = {}

    with torch.no_grad():
        for dtw_val in test_dtws:
            dtw = torch.tensor([[dtw_val]], device=device, dtype=torch.float32)

            output = model(x, t, dtw)
            transformed_pattern = output[0, 0].cpu().numpy()

            # Calculate similarity to real pattern
            correlation = np.corrcoef(transformed_pattern, real_pattern)[0, 1]
            mse_to_real = np.mean((transformed_pattern - real_pattern) ** 2)

            results[dtw_val] = {
                'correlation_to_real': correlation,
                'mse_to_real': mse_to_real,
                'transformed_pattern': transformed_pattern
            }

    return {
        'original_dtw': original_dtw,
        'test_dtws': test_dtws,
        'results': results,
        'real_pattern': real_pattern
    }


def batch_dtw_analysis(model, test_synth: torch.Tensor, test_real: torch.Tensor,
                      test_dtw: torch.Tensor, n_samples: int = 3) -> List[Dict]:
    """
    Run DTW conditioning analysis on multiple samples.

    Args:
        model: The diffusion model
        test_synth: Test synthetic patterns [N, ...]
        test_real: Test real patterns [N, ...]
        test_dtw: Test DTW values [N]
        n_samples: Number of samples to analyze

    Returns:
        List of DTW analysis results
    """
    results = []

    for i in tqdm(range(min(n_samples, len(test_synth))), desc="Analyzing DTW conditioning"):
        # Basic DTW conditioning test
        dtw_results = analyze_dtw_conditioning(model, test_synth[i])
        validation_metrics = validate_dtw_conditioning(dtw_results, test_dtw[i].item())

        # Sensitivity analysis
        sensitivity_metrics = analyze_dtw_feature_importance(
            model, test_synth[i], base_dtw=test_dtw[i].item()
        )

        # Real similarity comparison
        similarity_comparison = compare_dtw_vs_real_similarity(
            model, test_synth[i], test_real[i], test_dtw[i].item()
        )

        combined_results = {
            'sample_index': i,
            'validation_metrics': validation_metrics,
            'sensitivity_metrics': sensitivity_metrics,
            'similarity_comparison': similarity_comparison,
            'dtw_results': dtw_results
        }

        results.append(combined_results)

    return results