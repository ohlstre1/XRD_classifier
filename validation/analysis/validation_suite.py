"""
Comprehensive validation suite for XRD diffusion models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy import stats
import pickle
from datetime import datetime
from pathlib import Path

# Import validation modules
from ..tests.stochasticity import (
    test_model_stochasticity, analyze_stochasticity_metrics
)
from ..tests.timestep_effects import (
    analyze_timestep_effects, validate_timestep_effects
)
from ..tests.dtw_conditioning import (
    analyze_dtw_conditioning, validate_dtw_conditioning
)


def evaluate_test_set_performance(model, diffusion, test_synth: torch.Tensor,
                                 test_real: torch.Tensor, test_dtw: torch.Tensor,
                                 batch_size: int = 32) -> Dict[str, np.ndarray]:
    """
    Comprehensive evaluation on the test set.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        test_synth: Test synthetic patterns [N, ...]
        test_real: Test real patterns [N, ...]
        test_dtw: Test DTW values [N]
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.set_stochastic_mode(False)  # Deterministic evaluation

    # Get device from model parameters
    device = next(model.parameters()).device

    n_samples = len(test_synth)
    n_batches = (n_samples + batch_size - 1) // batch_size

    results = {
        'diffusion_losses': [],
        'reconstruction_losses': [],
        'direct_transform_losses': [],
        'real_similarities': [],
        'peak_correlations': []
    }

    loss_fn = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Evaluating test set"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            # Get batch
            synth_batch = test_synth[start_idx:end_idx].unsqueeze(1).to(device)
            real_batch = test_real[start_idx:end_idx].unsqueeze(1).to(device)
            dtw_batch = test_dtw[start_idx:end_idx].unsqueeze(1).to(device)

            current_batch_size = synth_batch.shape[0]

            # 1. Diffusion loss (random timesteps)
            t = torch.randint(0, diffusion.num_timesteps, (current_batch_size,), device=device)
            x_t, noise = diffusion.forward_diffusion(synth_batch, t)
            noise_pred = model(x_t, t, dtw_batch)
            diff_loss = loss_fn(noise_pred, noise).mean(dim=[1, 2])
            results['diffusion_losses'].extend(diff_loss.cpu().numpy())

            # 2. Direct transformation (t=0)
            t_zero = torch.zeros(current_batch_size, dtype=torch.long, device=device)
            transformed = model(synth_batch, t_zero, dtw_batch)
            transform_loss = loss_fn(transformed, real_batch).mean(dim=[1, 2])
            results['direct_transform_losses'].extend(transform_loss.cpu().numpy())

            # 3. Reconstruction test (add noise, then denoise)
            t_recon = torch.full((current_batch_size,), 100, device=device)  # Moderate noise
            x_noisy, noise_true = diffusion.forward_diffusion(synth_batch, t_recon)
            noise_pred_recon = model(x_noisy, t_recon, dtw_batch)

            # Estimate x0
            alpha_bar_t = diffusion.alpha_bars[t_recon].view(-1, 1, 1)
            x0_pred = (x_noisy - torch.sqrt(1 - alpha_bar_t) * noise_pred_recon) / torch.sqrt(alpha_bar_t)
            recon_loss = loss_fn(x0_pred, synth_batch).mean(dim=[1, 2])
            results['reconstruction_losses'].extend(recon_loss.cpu().numpy())

            # 4. Similarity to real patterns
            for i in range(current_batch_size):
                synth_np = synth_batch[i, 0].cpu().numpy()
                real_np = real_batch[i, 0].cpu().numpy()
                transformed_np = transformed[i, 0].cpu().numpy()

                # Correlations
                real_similarity = np.corrcoef(transformed_np, real_np)[0, 1]
                results['real_similarities'].append(real_similarity)

                # Peak analysis
                try:
                    synth_peaks, _ = find_peaks(synth_np, height=0.1)
                    real_peaks, _ = find_peaks(real_np, height=0.1)

                    if len(synth_peaks) > 0 and len(real_peaks) > 0:
                        # Simple peak position correlation
                        peak_corr = len(set(synth_peaks) & set(real_peaks)) / max(len(synth_peaks), len(real_peaks))
                        results['peak_correlations'].append(peak_corr)
                except:
                    results['peak_correlations'].append(0.0)

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def test_model_stability(model) -> Dict:
    """
    Test model stability with extreme inputs.

    Args:
        model: The diffusion model

    Returns:
        Stability test results
    """
    model.eval()

    # Get device from model parameters
    device = next(model.parameters()).device

    # Test pattern length from a known working sample
    sample_length = 4500  # Typical XRD pattern length

    with torch.no_grad():
        # Zero input
        zero_input = torch.zeros(1, 1, sample_length).to(device)
        dtw_test = torch.tensor([[0.5]], device=device)
        t_test = torch.tensor([0], device=device)

        try:
            zero_output = model(zero_input, t_test, dtw_test)
            zero_stability = (torch.isfinite(zero_output).all() and
                            not torch.isnan(zero_output).any())
        except Exception as e:
            zero_stability = False
            print(f"Zero input test failed: {e}")

        # Random input
        try:
            random_input = torch.randn(1, 1, sample_length).to(device)
            random_output = model(random_input, t_test, dtw_test)
            random_stability = (torch.isfinite(random_output).all() and
                              not torch.isnan(random_output).any())
        except Exception as e:
            random_stability = False
            print(f"Random input test failed: {e}")

        # Check for NaN or inf
        stability_pass = zero_stability and random_stability

    return {
        'zero_input_stable': zero_stability,
        'random_input_stable': random_stability,
        'overall_stable': stability_pass,
        'pass': stability_pass
    }


def comprehensive_validation_suite(model, diffusion, test_synth: torch.Tensor,
                                  test_real: torch.Tensor, test_dtw: torch.Tensor,
                                  subset_size: int = 50) -> Dict:
    """
    Run a comprehensive validation suite covering all aspects.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        test_synth: Test synthetic patterns
        test_real: Test real patterns
        test_dtw: Test DTW values
        subset_size: Size of subset for quick testing

    Returns:
        Complete validation results
    """
    print("Running Comprehensive Validation Suite...")
    print("="*60)

    validation_results = {}

    # 1. Stochasticity Validation
    print("\n1. Testing Model Stochasticity...")
    det_outputs, sto_outputs = test_model_stochasticity(
        model, test_synth[:5], test_dtw[:5], n_runs=5
    )

    stochasticity_metrics = analyze_stochasticity_metrics(det_outputs, sto_outputs)
    validation_results['stochasticity'] = stochasticity_metrics

    print(f"   ✓ Deterministic std: {stochasticity_metrics['deterministic_std']:.8f}")
    print(f"   ✓ Stochastic std: {stochasticity_metrics['stochastic_std']:.8f}")
    print(f"   ✓ Ratio: {stochasticity_metrics['stochasticity_ratio']:.2f} "
          f"{'(PASS)' if stochasticity_metrics['test_pass'] else '(FAIL)'}")

    # 2. Timestep Effect Validation
    print("\n2. Testing Timestep Effects...")
    sample_idx = 0
    timestep_results = analyze_timestep_effects(
        model, diffusion, test_synth[sample_idx], test_dtw[sample_idx]
    )

    timestep_validation = validate_timestep_effects(timestep_results)
    validation_results['timestep_effects'] = timestep_validation

    print(f"   ✓ Timestep-reconstruction error correlation: "
          f"{timestep_validation['timestep_recon_correlation']:.3f} "
          f"{'(PASS)' if timestep_validation['test_pass'] else '(FAIL)'}")

    # 3. DTW Conditioning Validation
    print("\n3. Testing DTW Conditioning...")
    dtw_results = analyze_dtw_conditioning(model, test_synth[sample_idx])
    dtw_validation = validate_dtw_conditioning(dtw_results, test_dtw[sample_idx].item())
    validation_results['dtw_conditioning'] = dtw_validation

    print(f"   ✓ DTW conditioning effect std: {dtw_validation['effect_std']:.6f} "
          f"{'(PASS)' if dtw_validation['test_pass'] else '(FAIL)'}")
    print(f"   ✓ DTW conditioning range: {dtw_validation['effect_range']:.6f}")

    # 4. Real-world Similarity Validation
    print("\n4. Testing Real-world Similarity...")
    subset_results = evaluate_test_set_performance(
        model, diffusion,
        test_synth[:subset_size],
        test_real[:subset_size],
        test_dtw[:subset_size],
        batch_size=16
    )

    mean_similarity = np.mean(subset_results['real_similarities'])
    high_similarity_fraction = np.sum(subset_results['real_similarities'] > 0.6) / len(subset_results['real_similarities'])

    similarity_validation = {
        'mean_similarity': mean_similarity,
        'high_similarity_fraction': high_similarity_fraction,
        'pass': mean_similarity > 0.3 and high_similarity_fraction > 0.2
    }
    validation_results['real_world_similarity'] = similarity_validation

    print(f"   ✓ Mean similarity to real: {mean_similarity:.3f} "
          f"{'(PASS)' if mean_similarity > 0.3 else '(FAIL)'}")
    print(f"   ✓ High similarity fraction: {high_similarity_fraction:.3f} "
          f"{'(PASS)' if high_similarity_fraction > 0.2 else '(FAIL)'}")

    # 5. Model Stability Validation
    print("\n5. Testing Model Stability...")
    stability_results = test_model_stability(model)
    validation_results['stability'] = stability_results

    print(f"   ✓ Model stability: {'PASS' if stability_results['pass'] else 'FAIL'}")

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    # Count tests that have a 'pass' key (filter out non-test entries)
    # Safe way to handle validation results with different structures
    test_results = {}
    for k, v in validation_results.items():
        if isinstance(v, dict) and 'pass' in v:
            test_results[k] = v
        elif isinstance(v, dict) and 'test_pass' in v:
            # Handle alternative structure
            test_results[k] = {'pass': v['test_pass']}

    total_tests = len(test_results)
    if total_tests > 0:
        passed_tests = sum(1 for test in test_results.values() if test.get('pass', False))
    else:
        passed_tests = 0

    print(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")

    for test_name, result in test_results.items():
        status = "✓ PASS" if result.get('pass', False) else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")

    overall_pass = passed_tests == total_tests
    print(f"\nOVERALL RESULT: {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}")

    # Store test set results for detailed analysis
    validation_results['test_set_results'] = subset_results
    validation_results['overall_pass'] = overall_pass
    validation_results['passed_tests'] = passed_tests
    validation_results['total_tests'] = total_tests

    return validation_results


def save_validation_results(validation_results: Dict, test_results: Optional[Dict] = None,
                           model_info: Optional[Dict] = None, dataset_info: Optional[Dict] = None,
                           output_path: str = "validation_results.pkl") -> str:
    """
    Save comprehensive validation results.

    Args:
        validation_results: Results from comprehensive validation
        test_results: Full test set results (optional)
        model_info: Model information (optional)
        dataset_info: Dataset information (optional)
        output_path: Path to save results

    Returns:
        Path where results were saved
    """
    complete_results = {
        'timestamp': datetime.now().isoformat(),
        'validation_results': validation_results
    }

    if model_info is not None:
        complete_results['model_info'] = model_info

    if test_results is not None:
        complete_results['test_results'] = test_results

    if dataset_info is not None:
        complete_results['dataset_info'] = dataset_info

    output_path = Path(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(complete_results, f)

    print(f"\n💾 Validation results saved to: {output_path}")
    return str(output_path)


def load_validation_results(results_path: str) -> Dict:
    """
    Load previously saved validation results.

    Args:
        results_path: Path to saved results

    Returns:
        Loaded results dictionary
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print(f"📂 Loaded validation results from: {results_path}")
    print(f"   Timestamp: {results.get('timestamp', 'Unknown')}")

    return results