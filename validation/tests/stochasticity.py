"""
Stochasticity testing functionality for XRD diffusion validation.
"""

import torch
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


def test_model_stochasticity(model, sample_synth: torch.Tensor, sample_dtw: torch.Tensor,
                           n_runs: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test if the model produces different outputs in stochastic mode
    and consistent outputs in deterministic mode.

    Args:
        model: The diffusion model
        sample_synth: Synthetic XRD sample [N, L] or [L]
        sample_dtw: DTW distance value [N] or scalar
        n_runs: Number of test runs

    Returns:
        Tuple of (deterministic_outputs, stochastic_outputs)
    """
    model.eval()

    # Ensure correct tensor shapes
    if sample_synth.dim() == 1:
        sample_synth = sample_synth.unsqueeze(0)
    if sample_dtw.dim() == 0:
        sample_dtw = sample_dtw.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = sample_synth[:1].unsqueeze(1).to(device)  # [1, 1, L]
    dtw = sample_dtw[:1].unsqueeze(1).to(device)   # [1, 1]
    t = torch.zeros(1, dtype=torch.long, device=device)  # [1]

    print("Testing model stochasticity...")
    print(f"Input shape: {x.shape}, DTW shape: {dtw.shape}, t shape: {t.shape}")

    # Test deterministic mode
    model.set_stochastic_mode(False)
    deterministic_outputs = []

    for i in range(n_runs):
        with torch.no_grad():
            output = model(x, t, dtw)
            deterministic_outputs.append(output.cpu().numpy())

    # Test stochastic mode
    model.set_stochastic_mode(True)
    stochastic_outputs = []

    for i in range(n_runs):
        with torch.no_grad():
            output = model(x, t, dtw)
            stochastic_outputs.append(output.cpu().numpy())

    # Analyze variability
    det_outputs = np.array(deterministic_outputs)
    sto_outputs = np.array(stochastic_outputs)

    det_std = np.std(det_outputs, axis=0).mean()
    sto_std = np.std(sto_outputs, axis=0).mean()

    print(f"\nDeterministic mode:")
    print(f"  Average std across runs: {det_std:.8f}")
    print(f"  Max absolute difference: {np.max(np.abs(det_outputs - det_outputs[0])):.8f}")

    print(f"\nStochastic mode:")
    print(f"  Average std across runs: {sto_std:.8f}")
    print(f"  Max absolute difference: {np.max(np.abs(sto_outputs - sto_outputs[0])):.8f}")

    print(f"\nStochasticity ratio: {sto_std / max(det_std, 1e-10):.2f}")

    return det_outputs, sto_outputs


def generate_stochastic_variations(model, synth_pattern: torch.Tensor, dtw_value: torch.Tensor,
                                  n_variations: int = 10) -> torch.Tensor:
    """
    Generate multiple stochastic variations of the same input pattern.

    Args:
        model: The diffusion model
        synth_pattern: Synthetic pattern [L] or [1, L]
        dtw_value: DTW distance value scalar or [1]
        n_variations: Number of variations to generate

    Returns:
        Tensor of variations [n_variations, 1, L]
    """
    model.set_stochastic_mode(True)
    model.train()  # Enable dropout and stochastic depth

    # Ensure correct tensor shapes
    if synth_pattern.dim() == 1:
        synth_pattern = synth_pattern.unsqueeze(0)
    if dtw_value.dim() == 0:
        dtw_value = dtw_value.unsqueeze(0)

    # Get device from model parameters
    device = next(model.parameters()).device

    x = synth_pattern.unsqueeze(0).to(device)  # [1, 1, L]
    dtw = dtw_value.unsqueeze(0).to(device)    # [1, 1]
    t = torch.zeros(1, dtype=torch.long, device=device)  # Direct transformation

    print(f"Generating variations - Input shape: {x.shape}, DTW shape: {dtw.shape}")

    variations = []
    with torch.no_grad():
        for i in range(n_variations):
            output = model(x, t, dtw)
            variations.append(output[0].cpu())

    model.eval()
    return torch.stack(variations)


def analyze_stochasticity_metrics(deterministic_outputs: np.ndarray,
                                 stochastic_outputs: np.ndarray) -> dict:
    """
    Analyze stochasticity metrics from test outputs.

    Args:
        deterministic_outputs: Outputs from deterministic mode [n_runs, batch, channels, length]
        stochastic_outputs: Outputs from stochastic mode [n_runs, batch, channels, length]

    Returns:
        Dictionary with stochasticity metrics
    """
    # Calculate variability metrics
    det_std = np.std(deterministic_outputs, axis=0).mean()
    sto_std = np.std(stochastic_outputs, axis=0).mean()

    det_max_diff = np.max(np.abs(deterministic_outputs - deterministic_outputs[0]))
    sto_max_diff = np.max(np.abs(stochastic_outputs - stochastic_outputs[0]))

    stochasticity_ratio = sto_std / max(det_std, 1e-10)

    # Test if stochastic mode produces meaningfully different outputs
    stochastic_test_pass = stochasticity_ratio > 10  # At least 10x more variable

    return {
        'deterministic_std': det_std,
        'stochastic_std': sto_std,
        'deterministic_max_diff': det_max_diff,
        'stochastic_max_diff': sto_max_diff,
        'stochasticity_ratio': stochasticity_ratio,
        'test_pass': stochastic_test_pass
    }


def batch_stochasticity_test(model, test_synth: torch.Tensor, test_dtw: torch.Tensor,
                           n_samples: int = 5, n_runs: int = 5) -> List[dict]:
    """
    Run stochasticity tests on multiple samples.

    Args:
        model: The diffusion model
        test_synth: Test synthetic patterns [N, ...]
        test_dtw: Test DTW values [N]
        n_samples: Number of samples to test
        n_runs: Number of runs per sample

    Returns:
        List of stochasticity metrics for each sample
    """
    results = []

    for i in tqdm(range(min(n_samples, len(test_synth))), desc="Testing stochasticity"):
        det_outputs, sto_outputs = test_model_stochasticity(
            model, test_synth[i], test_dtw[i], n_runs=n_runs
        )

        metrics = analyze_stochasticity_metrics(det_outputs, sto_outputs)
        metrics['sample_index'] = i
        results.append(metrics)

    return results