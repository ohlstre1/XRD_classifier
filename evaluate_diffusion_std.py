#!/usr/bin/env python3
"""
Evaluation script for XRD Diffusion Model with Standard Deviation Metrics
Uses components from diffusion_model_0.1.5.py to calculate comprehensive performance metrics
with standard deviation across multiple runs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
import json
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import components from the original diffusion model
# Note: Import from the actual filename
import importlib.util
import sys

# Load the diffusion model module
spec = importlib.util.spec_from_file_location("diffusion_model", "./diffusion_model_0.1.5.py")
diffusion_model = importlib.util.module_from_spec(spec)
sys.modules["diffusion_model"] = diffusion_model
spec.loader.exec_module(diffusion_model)

# Import the required classes
ImprovedDiffusionDenoiser = diffusion_model.ImprovedDiffusionDenoiser
DiffusionProcess = diffusion_model.DiffusionProcess
XRDTransformDataset = diffusion_model.XRDTransformDataset

class PerformanceEvaluator:
    """
    Comprehensive performance evaluator for XRD diffusion model with standard deviation calculations.
    """

    def __init__(self, model, diffusion, device='cpu'):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.loss_fn = nn.MSELoss()

    def calculate_peak_metrics(self, predicted, target, threshold=0.1):
        """
        Calculate peak-specific metrics for XRD patterns.

        Args:
            predicted: Predicted XRD pattern [N, 1, L]
            target: Target XRD pattern [N, 1, L]
            threshold: Minimum intensity to consider as a peak

        Returns:
            dict: Peak-specific metrics
        """
        pred_np = predicted.cpu().numpy().reshape(-1)
        target_np = target.cpu().numpy().reshape(-1)

        # Find peaks (simple local maxima above threshold)
        pred_peaks = []
        target_peaks = []

        for i in range(1, len(pred_np) - 1):
            if (pred_np[i] > pred_np[i-1] and pred_np[i] > pred_np[i+1] and
                pred_np[i] > threshold):
                pred_peaks.append((i, pred_np[i]))

            if (target_np[i] > target_np[i-1] and target_np[i] > target_np[i+1] and
                target_np[i] > threshold):
                target_peaks.append((i, target_np[i]))

        # Calculate peak position accuracy
        position_errors = []
        intensity_errors = []

        for pred_pos, pred_int in pred_peaks:
            # Find closest target peak
            min_dist = float('inf')
            closest_target = None

            for target_pos, target_int in target_peaks:
                dist = abs(pred_pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_target = (target_pos, target_int)

            if closest_target is not None and min_dist <= 5:  # Within 5 indices
                position_errors.append(min_dist)
                intensity_errors.append(abs(pred_int - closest_target[1]))

        return {
            'num_pred_peaks': len(pred_peaks),
            'num_target_peaks': len(target_peaks),
            'avg_position_error': np.mean(position_errors) if position_errors else 0,
            'avg_intensity_error': np.mean(intensity_errors) if intensity_errors else 0,
            'peak_detection_rate': len(position_errors) / max(len(target_peaks), 1)
        }

    def evaluate_single_run(self, dataloader, seed=None):
        """
        Perform a single evaluation run.

        Args:
            dataloader: DataLoader for evaluation data
            seed: Random seed for reproducibility

        Returns:
            dict: Evaluation metrics for this run
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model.eval()

        # Metrics storage
        metrics = {
            'total_loss': [],
            'diffusion_loss': [],
            'reconstruction_loss': [],
            'mse_scores': [],
            'mae_scores': [],
            'peak_metrics': {
                'position_errors': [],
                'intensity_errors': [],
                'detection_rates': []
            },
            'correlation_scores': [],
            'snr_improvements': []
        }

        with torch.no_grad():
            for synth, real, temp in tqdm(dataloader, desc="Evaluating"):
                synth = synth.to(self.device)
                real = real.to(self.device)
                temp = temp.to(self.device)
                batch_size = synth.shape[0]

                # 1. Diffusion branch evaluation
                t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                x_t, noise = self.diffusion.forward_diffusion(synth, t)
                noise_pred = self.model(x_t, t, temp)
                diffusion_loss = self.loss_fn(noise_pred, noise)

                # 2. Reconstruction branch evaluation
                t_zero = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                noise_pred_real = self.model(real, t_zero, temp)

                # Denoise real pattern
                alpha_bar_t = self.diffusion.alpha_bars[t_zero].view(-1, 1, 1)
                denoised_real = (real - torch.sqrt(1 - alpha_bar_t) * noise_pred_real) / torch.sqrt(alpha_bar_t)
                reconstruction_loss = self.loss_fn(denoised_real, synth)

                # Combined loss
                total_loss = 0.5 * diffusion_loss + 0.5 * reconstruction_loss

                # Store basic losses
                metrics['total_loss'].append(total_loss.item())
                metrics['diffusion_loss'].append(diffusion_loss.item())
                metrics['reconstruction_loss'].append(reconstruction_loss.item())

                # Calculate additional metrics for each sample in batch
                for i in range(batch_size):
                    pred_sample = denoised_real[i:i+1]
                    target_sample = synth[i:i+1]

                    # MSE and MAE
                    mse = mean_squared_error(
                        target_sample.cpu().numpy().flatten(),
                        pred_sample.cpu().numpy().flatten()
                    )
                    mae = mean_absolute_error(
                        target_sample.cpu().numpy().flatten(),
                        pred_sample.cpu().numpy().flatten()
                    )

                    metrics['mse_scores'].append(mse)
                    metrics['mae_scores'].append(mae)

                    # Correlation
                    corr, _ = stats.pearsonr(
                        target_sample.cpu().numpy().flatten(),
                        pred_sample.cpu().numpy().flatten()
                    )
                    metrics['correlation_scores'].append(corr if not np.isnan(corr) else 0)

                    # Peak-specific metrics
                    peak_metrics = self.calculate_peak_metrics(pred_sample, target_sample)
                    metrics['peak_metrics']['position_errors'].append(peak_metrics['avg_position_error'])
                    metrics['peak_metrics']['intensity_errors'].append(peak_metrics['avg_intensity_error'])
                    metrics['peak_metrics']['detection_rates'].append(peak_metrics['peak_detection_rate'])

                    # SNR improvement
                    # Calculate SNR of original real pattern vs denoised pattern
                    real_sample = real[i:i+1].cpu().numpy().flatten()
                    pred_sample_np = pred_sample.cpu().numpy().flatten()
                    target_sample_np = target_sample.cpu().numpy().flatten()

                    # SNR = signal_power / noise_power
                    signal_power = np.mean(target_sample_np**2)
                    noise_power_real = np.mean((real_sample - target_sample_np)**2)
                    noise_power_pred = np.mean((pred_sample_np - target_sample_np)**2)

                    snr_real = 10 * np.log10(signal_power / (noise_power_real + 1e-10))
                    snr_pred = 10 * np.log10(signal_power / (noise_power_pred + 1e-10))
                    snr_improvement = snr_pred - snr_real

                    metrics['snr_improvements'].append(snr_improvement)

        # Convert lists to arrays and calculate means
        results = {}
        for key, values in metrics.items():
            if key == 'peak_metrics':
                results[key] = {}
                for sub_key, sub_values in values.items():
                    results[key][sub_key] = np.mean(sub_values)
            else:
                results[key] = np.mean(values)

        return results

    def evaluate_with_std(self, dataloader, num_runs=10, base_seed=42):
        """
        Perform multiple evaluation runs to calculate standard deviation.

        Args:
            dataloader: DataLoader for evaluation data
            num_runs: Number of evaluation runs
            base_seed: Base seed for reproducibility

        Returns:
            dict: Mean and standard deviation of all metrics
        """
        print(f"Running {num_runs} evaluation runs for standard deviation calculation...")

        all_results = []

        for run in range(num_runs):
            seed = base_seed + run
            print(f"Run {run + 1}/{num_runs} (seed={seed})")

            results = self.evaluate_single_run(dataloader, seed=seed)
            all_results.append(results)

        # Calculate mean and std for each metric
        final_results = {}

        # Get all metric keys from first run
        sample_result = all_results[0]

        for key in sample_result.keys():
            if key == 'peak_metrics':
                final_results[key] = {}
                for sub_key in sample_result[key].keys():
                    values = [result[key][sub_key] for result in all_results]
                    final_results[key][sub_key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            else:
                values = [result[key] for result in all_results]
                final_results[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        return final_results, all_results

class ResultsVisualizer:
    """
    Visualization utilities for evaluation results.
    """

    @staticmethod
    def plot_metrics_with_std(results, save_path=None):
        """
        Plot metrics with error bars showing standard deviation.
        """
        # Extract metrics for plotting
        metrics_to_plot = [
            ('total_loss', 'Total Loss'),
            ('diffusion_loss', 'Diffusion Loss'),
            ('reconstruction_loss', 'Reconstruction Loss'),
            ('mse_scores', 'MSE Score'),
            ('mae_scores', 'MAE Score'),
            ('correlation_scores', 'Correlation Score'),
            ('snr_improvements', 'SNR Improvement (dB)')
        ]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, (key, title) in enumerate(metrics_to_plot):
            if i < len(axes):
                mean_val = results[key]['mean']
                std_val = results[key]['std']

                axes[i].bar(['Mean'], [mean_val], yerr=[std_val], capsize=5,
                           color='skyblue', alpha=0.7, error_kw={'color': 'red', 'linewidth': 2})
                axes[i].set_title(title, fontsize=12)
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)

                # Add text annotation
                axes[i].text(0, mean_val + std_val + 0.1 * mean_val,
                           f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                           ha='center', va='bottom', fontsize=10)

        # Plot peak metrics
        if len(axes) > len(metrics_to_plot):
            peak_metrics = ['position_errors', 'intensity_errors', 'detection_rates']
            peak_titles = ['Peak Position Error', 'Peak Intensity Error', 'Peak Detection Rate']

            for j, (peak_key, peak_title) in enumerate(zip(peak_metrics, peak_titles)):
                if len(metrics_to_plot) + j < len(axes):
                    idx = len(metrics_to_plot) + j
                    mean_val = results['peak_metrics'][peak_key]['mean']
                    std_val = results['peak_metrics'][peak_key]['std']

                    axes[idx].bar(['Mean'], [mean_val], yerr=[std_val], capsize=5,
                                 color='lightcoral', alpha=0.7, error_kw={'color': 'darkred', 'linewidth': 2})
                    axes[idx].set_title(peak_title, fontsize=12)
                    axes[idx].set_ylabel('Value')
                    axes[idx].grid(True, alpha=0.3)

                    axes[idx].text(0, mean_val + std_val + 0.1 * abs(mean_val),
                                 f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                                 ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_distribution_analysis(all_results, metric_key, save_path=None):
        """
        Plot distribution analysis for a specific metric across runs.
        """
        if metric_key == 'peak_metrics':
            # Handle nested peak metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            peak_keys = ['position_errors', 'intensity_errors', 'detection_rates']

            for i, peak_key in enumerate(peak_keys):
                values = [result[metric_key][peak_key] for result in all_results]

                axes[i].hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
                axes[i].set_title(f'Peak {peak_key.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        else:
            # Handle regular metrics
            values = [result[metric_key] for result in all_results]

            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
            plt.axvline(np.mean(values) + np.std(values), color='orange', linestyle=':', linewidth=2, label=f'Mean + Std: {np.mean(values) + np.std(values):.4f}')
            plt.axvline(np.mean(values) - np.std(values), color='orange', linestyle=':', linewidth=2, label=f'Mean - Std: {np.mean(values) - np.std(values):.4f}')
            plt.title(f'Distribution of {metric_key.replace("_", " ").title()} Across Runs')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")

        plt.show()

def load_model_and_data(model_path, data_path, device='cpu'):
    """
    Load the trained model and dataset.

    Args:
        model_path: Path to the trained model checkpoint
        data_path: Path to the dataset
        device: Device to load on

    Returns:
        tuple: (model, diffusion, test_dataloader)
    """
    print("Loading dataset...")
    dataset_dict = torch.load(data_path, map_location=device)

    synth_xrd = dataset_dict["synth_xrd"]
    real_xrd = dataset_dict["real_xrd"]
    global_temperature = dataset_dict["fast_dtw_distance"]

    # Limit dataset size for faster evaluation
    sample_limit = 200
    synth_xrd = synth_xrd[:sample_limit]
    real_xrd = real_xrd[:sample_limit]
    global_temperature = global_temperature[:sample_limit]

    print(f"Loaded dataset with {len(synth_xrd)} samples (limited to {sample_limit})")

    dataset = XRDTransformDataset(synth_xrd, real_xrd, global_temperature)

    # Create test dataloader
    test_dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print("Loading trained model...")

    # Initialize diffusion process
    diffusion = DiffusionProcess(num_timesteps=1000, schedule_type='cosine', device=device)

    # Initialize model
    model = ImprovedDiffusionDenoiser(
        in_channels=1,
        hidden_channels=16,
        time_embedding_dim=256,
        num_res_blocks=2,
        attention_levels=[1, 2],
        num_levels=2,
        temperature_condition=True
    ).to(device)

    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        print("Using randomly initialized model for demonstration")

    return model, diffusion, test_dataloader

def main():
    """
    Main function to run comprehensive evaluation with standard deviation metrics.
    """
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_path = "./models/xrd_diffusion/improved_diffusion_model_best.pth"
    data_path = "data/xrd_dataset_labeled_dtw_window.pt"
    results_dir = "./evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Load model and data
    model, diffusion, test_dataloader = load_model_and_data(model_path, data_path, device)

    # Initialize evaluator
    evaluator = PerformanceEvaluator(model, diffusion, device)

    # Run evaluation with standard deviation calculation
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION WITH STANDARD DEVIATION METRICS")
    print("="*60)

    num_runs = 10000  # Number of evaluation runs for std calculation
    start_time = time.time()

    results, all_results = evaluator.evaluate_with_std(
        test_dataloader,
        num_runs=num_runs,
        base_seed=42
    )

    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")

    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)

    print("\n🔍 CORE METRICS:")
    core_metrics = ['total_loss', 'diffusion_loss', 'reconstruction_loss']
    for metric in core_metrics:
        r = results[metric]
        print(f"  {metric.replace('_', ' ').title():.<25} μ={r['mean']:.6f} ± σ={r['std']:.6f} [{r['min']:.6f}, {r['max']:.6f}]")

    print("\n🎯 ACCURACY METRICS:")
    accuracy_metrics = ['mse_scores', 'mae_scores', 'correlation_scores']
    for metric in accuracy_metrics:
        r = results[metric]
        print(f"  {metric.replace('_', ' ').title():.<25} μ={r['mean']:.6f} ± σ={r['std']:.6f} [{r['min']:.6f}, {r['max']:.6f}]")

    print("\n🏔️ PEAK-SPECIFIC METRICS:")
    peak_metrics = results['peak_metrics']
    for metric_name, r in peak_metrics.items():
        print(f"  {metric_name.replace('_', ' ').title():.<25} μ={r['mean']:.6f} ± σ={r['std']:.6f} [{r['min']:.6f}, {r['max']:.6f}]")

    print("\n📶 SIGNAL QUALITY:")
    r = results['snr_improvements']
    print(f"  {'SNR Improvement (dB)':.<25} μ={r['mean']:.6f} ± σ={r['std']:.6f} [{r['min']:.6f}, {r['max']:.6f}]")

    # Save results
    results_file = os.path.join(results_dir, "evaluation_results_with_std.json")

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    serializable_results = convert_numpy_types(results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n📁 Results saved to: {results_file}")

    # Create visualizations
    print("\n📊 Creating visualizations...")

    visualizer = ResultsVisualizer()

    # Plot overall metrics with std
    metrics_plot_path = os.path.join(results_dir, "metrics_with_std.png")
    visualizer.plot_metrics_with_std(results, metrics_plot_path)

    # Plot distribution analysis for key metrics
    key_metrics = ['total_loss', 'mse_scores', 'correlation_scores', 'peak_metrics']

    for metric in key_metrics:
        dist_plot_path = os.path.join(results_dir, f"distribution_{metric}.png")
        visualizer.plot_distribution_analysis(all_results, metric, dist_plot_path)

    print("\n✅ Evaluation complete! Check the results directory for detailed outputs.")
    print(f"📂 Results directory: {results_dir}")

if __name__ == "__main__":
    main()