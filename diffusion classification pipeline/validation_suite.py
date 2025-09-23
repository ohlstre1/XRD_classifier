import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from robust_diffusion_model import (
    RobustDiffusionDenoiser,
    CleanDiffusionProcess,
    XRDDataset,
    train_robust_diffusion
)

class ValidationSuite:
    """
    Comprehensive validation suite for the clean diffusion model.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}

    def generate_synthetic_xrd(self, n_samples=1000, pattern_length=512, n_peaks_range=(5, 15)):
        """
        Generate synthetic XRD patterns with known characteristics.
        """
        print("Generating synthetic XRD patterns...")

        patterns = []
        conditions = []
        peak_info = []

        for i in range(n_samples):
            # Random number of peaks
            n_peaks = np.random.randint(*n_peaks_range)

            # Create base pattern
            x = np.linspace(10, 80, pattern_length)  # 2θ range
            pattern = np.zeros_like(x)

            # Add peaks with random positions, heights, and widths
            peak_positions = np.random.uniform(15, 75, n_peaks)
            peak_heights = np.random.exponential(1.0, n_peaks)
            peak_widths = np.random.uniform(0.5, 2.0, n_peaks)

            current_peaks = []
            for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
                peak = height * np.exp(-0.5 * ((x - pos) / width) ** 2)
                pattern += peak
                current_peaks.append((pos, height, width))

            # Add background
            background_level = np.random.uniform(0.01, 0.1)
            pattern += background_level

            # Add slight noise
            noise_level = np.random.uniform(0.01, 0.05)
            pattern += np.random.normal(0, noise_level, pattern_length)

            # Normalize to [0, 1]
            pattern = np.maximum(pattern, 0)
            if pattern.max() > 0:
                pattern = pattern / pattern.max()

            patterns.append(pattern)

            # Condition could be temperature, crystallite size, etc.
            condition = np.random.uniform(0.1, 1.0)
            conditions.append(condition)
            peak_info.append(current_peaks)

        patterns = np.array(patterns)
        conditions = np.array(conditions)

        print(f"Generated {n_samples} synthetic XRD patterns")
        print(f"Pattern shape: {patterns.shape}")
        print(f"Peak count range: {n_peaks_range}")

        return patterns, conditions, peak_info

    def add_realistic_noise(self, clean_patterns, noise_types=['gaussian', 'shot', 'drift']):
        """
        Add realistic experimental noise to clean patterns.
        """
        noisy_patterns = []

        for pattern in clean_patterns:
            noisy = pattern.copy()

            if 'gaussian' in noise_types:
                # Electronic noise
                noise_level = np.random.uniform(0.02, 0.08)
                noisy += np.random.normal(0, noise_level, len(pattern))

            if 'shot' in noise_types:
                # Poisson shot noise
                # Convert to "counts" then back to normalize
                max_counts = np.random.uniform(1000, 10000)
                counts = (pattern * max_counts).astype(int)
                noisy_counts = np.random.poisson(counts)
                noisy = noisy_counts / max_counts

            if 'drift' in noise_types:
                # Baseline drift
                drift_amplitude = np.random.uniform(0.01, 0.05)
                drift_frequency = np.random.uniform(0.1, 2.0)
                x = np.linspace(0, drift_frequency * np.pi, len(pattern))
                drift = drift_amplitude * np.sin(x + np.random.uniform(0, 2*np.pi))
                noisy += drift

            # Ensure non-negative
            noisy = np.maximum(noisy, 0)

            # Renormalize
            if noisy.max() > 0:
                noisy = noisy / noisy.max()

            noisy_patterns.append(noisy)

        return np.array(noisy_patterns)

    def test_model_components(self):
        """
        Unit tests for individual model components.
        """
        print("\n" + "="*50)
        print("TESTING MODEL COMPONENTS")
        print("="*50)

        batch_size, channels, length = 4, 1, 128
        hidden_channels = 64  # Use 64 to work well with GroupNorm (divisible by 8)
        time_embedding_dim = 64

        # Test data
        x = torch.randn(batch_size, channels, length).to(self.device)
        t = torch.randint(0, 1000, (batch_size,)).to(self.device)
        condition = torch.randn(batch_size, 1).to(self.device)

        # Test 1: Model initialization and forward pass
        try:
            model = RobustDiffusionDenoiser(
                in_channels=channels,
                hidden_channels=hidden_channels,
                time_embedding_dim=time_embedding_dim,
                condition_dim=1
            ).to(self.device)

            output = model(x, t, condition)

            assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
            assert not torch.isnan(output).any(), "Model output contains NaN values"
            assert torch.isfinite(output).all(), "Model output contains infinite values"

            print("✓ Model forward pass test passed")

        except Exception as e:
            print(f"✗ Model forward pass test failed: {e}")
            return False

        # Test 2: Gradient flow
        try:
            loss = nn.MSELoss()(output, x)
            loss.backward()

            has_grad = False
            for param in model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break

            assert has_grad, "No gradients computed"
            print("✓ Gradient flow test passed")

        except Exception as e:
            print(f"✗ Gradient flow test failed: {e}")
            return False

        # Test 3: Different input sizes
        try:
            for test_length in [64, 256, 512]:
                x_test = torch.randn(2, channels, test_length).to(self.device)
                t_test = torch.randint(0, 1000, (2,)).to(self.device)
                cond_test = torch.randn(2, 1).to(self.device)

                output_test = model(x_test, t_test, cond_test)
                assert output_test.shape == x_test.shape

            print("✓ Variable input size test passed")

        except Exception as e:
            print(f"✗ Variable input size test failed: {e}")
            return False

        self.results['component_tests'] = True
        return True

    def test_diffusion_process(self):
        """
        Test the diffusion process mathematics.
        """
        print("\n" + "="*50)
        print("TESTING DIFFUSION PROCESS")
        print("="*50)

        # Test 1: Forward process properties
        try:
            diffusion = CleanDiffusionProcess(num_timesteps=1000, device=self.device)

            # Check schedule properties
            assert len(diffusion.betas) == 1000
            assert (diffusion.betas > 0).all() and (diffusion.betas < 1).all()
            assert len(diffusion.alphas) == 1000
            assert len(diffusion.alpha_bars) == 1000
            assert diffusion.alpha_bars.is_monotonic_decreasing()

            print("✓ Diffusion schedule test passed")

        except Exception as e:
            print(f"✗ Diffusion schedule test failed: {e}")
            return False

        # Test 2: Forward diffusion properties
        try:
            x0 = torch.randn(8, 1, 128).to(self.device)

            # Test different timesteps
            for t_val in [0, 100, 500, 999]:
                t = torch.full((8,), t_val, device=self.device)
                x_t, noise = diffusion.forward_diffusion(x0, t)

                assert x_t.shape == x0.shape
                assert noise.shape == x0.shape
                assert not torch.isnan(x_t).any()
                assert not torch.isnan(noise).any()

                # At t=0, should be very close to original
                if t_val == 0:
                    assert torch.allclose(x_t, x0, atol=1e-3)

                # At high t, should be mostly noise
                if t_val == 999:
                    correlation = torch.corrcoef(torch.stack([x0.flatten(), x_t.flatten()]))[0, 1]
                    assert abs(correlation) < 0.1, f"High timestep still too correlated: {correlation}"

            print("✓ Forward diffusion test passed")

        except Exception as e:
            print(f"✗ Forward diffusion test failed: {e}")
            return False

        self.results['diffusion_tests'] = True
        return True

    def test_training_loop(self, max_epochs=5):
        """
        Test that training loop works and loss decreases.
        """
        print("\n" + "="*50)
        print("TESTING TRAINING LOOP")
        print("="*50)

        try:
            # Generate small test dataset
            patterns, conditions, _ = self.generate_synthetic_xrd(n_samples=200, pattern_length=128)

            dataset = XRDDataset(patterns, conditions)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

            # Small model for fast testing
            model = RobustDiffusionDenoiser(
                in_channels=1,
                hidden_channels=32,  # Works with LayerNorm (no divisibility constraints)
                time_embedding_dim=64,
                num_res_blocks=1,
                attention_levels=[],
                num_levels=2,
                condition_dim=1
            ).to(self.device)

            diffusion = CleanDiffusionProcess(num_timesteps=100, device=self.device)

            print(f"Training small model for {max_epochs} epochs...")

            history, trained_model = train_robust_diffusion(
                model=model,
                diffusion=diffusion,
                dataloader=train_loader,
                val_dataloader=val_loader,
                num_epochs=max_epochs,
                lr=1e-3,
                device=self.device,
                save_path='./temp_models'
            )

            # Check that loss decreased
            initial_loss = history['train_loss'][0]
            final_loss = history['train_loss'][-1]

            assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
            assert len(history['train_loss']) == max_epochs

            print(f"✓ Training test passed - Loss: {initial_loss:.4f} -> {final_loss:.4f}")

            self.results['training_test'] = {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement': initial_loss - final_loss
            }

            return trained_model, diffusion, patterns

        except Exception as e:
            print(f"✗ Training test failed: {e}")
            return None, None, None

    def test_sampling_quality(self, model, diffusion, reference_patterns, n_samples=20):
        """
        Test sampling quality and consistency.
        """
        print("\n" + "="*50)
        print("TESTING SAMPLING QUALITY")
        print("="*50)

        if model is None:
            print("✗ No model provided for sampling test")
            return False

        try:
            model.eval()

            # Test unconditional sampling
            with torch.no_grad():
                batch_size = n_samples
                length = reference_patterns.shape[-1]

                # Sample from noise
                x_T = torch.randn(batch_size, 1, length).to(self.device)

                print("Sampling from diffusion model...")
                samples = diffusion.sample(model, x_T)

                assert samples.shape == (batch_size, 1, length)
                assert not torch.isnan(samples).any()
                assert torch.isfinite(samples).all()

                samples_np = samples.cpu().numpy()

                # Basic quality checks
                # 1. Non-negative (XRD patterns should be non-negative)
                negative_fraction = (samples_np < 0).mean()
                print(f"Fraction of negative values: {negative_fraction:.4f}")

                # 2. Dynamic range
                dynamic_ranges = samples_np.max(axis=-1) - samples_np.min(axis=-1)
                mean_dynamic_range = dynamic_ranges.mean()
                print(f"Mean dynamic range: {mean_dynamic_range:.4f}")

                # 3. Compare with reference statistics
                ref_mean = reference_patterns.mean()
                ref_std = reference_patterns.std()
                sample_mean = samples_np.mean()
                sample_std = samples_np.std()

                print(f"Reference mean: {ref_mean:.4f}, std: {ref_std:.4f}")
                print(f"Sample mean: {sample_mean:.4f}, std: {sample_std:.4f}")

                # Rough statistical similarity
                mean_diff = abs(sample_mean - ref_mean)
                std_ratio = sample_std / ref_std if ref_std > 0 else 1

                assert mean_diff < 0.5, f"Mean too different: {mean_diff}"
                assert 0.3 < std_ratio < 3.0, f"Std ratio out of range: {std_ratio}"

                print("✓ Basic sampling quality test passed")

                self.results['sampling_test'] = {
                    'negative_fraction': negative_fraction,
                    'mean_dynamic_range': mean_dynamic_range,
                    'mean_difference': mean_diff,
                    'std_ratio': std_ratio
                }

                return samples_np

        except Exception as e:
            print(f"✗ Sampling quality test failed: {e}")
            return None

    def create_visualizations(self, patterns, samples=None, save_path='./validation_plots'):
        """
        Create comprehensive visualizations for validation.
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)

        import os
        os.makedirs(save_path, exist_ok=True)

        # Plot 1: Original patterns
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i in range(6):
            ax = axes[i//3, i%3]
            ax.plot(patterns[i])
            ax.set_title(f'Original Pattern {i+1}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Intensity')
        plt.tight_layout()
        plt.savefig(f'{save_path}/original_patterns.png', dpi=150)
        plt.close()

        # Plot 2: Add noise comparison
        noisy_patterns = self.add_realistic_noise(patterns[:6])

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i in range(6):
            ax = axes[i//3, i%3]
            ax.plot(patterns[i], label='Clean', alpha=0.7)
            ax.plot(noisy_patterns[i], label='Noisy', alpha=0.7)
            ax.set_title(f'Noise Comparison {i+1}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Intensity')
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/noise_comparison.png', dpi=150)
        plt.close()

        # Plot 3: Generated samples (if available)
        if samples is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for i in range(6):
                ax = axes[i//3, i%3]
                ax.plot(samples[i, 0])
                ax.set_title(f'Generated Sample {i+1}')
                ax.set_xlabel('Position')
                ax.set_ylabel('Intensity')
            plt.tight_layout()
            plt.savefig(f'{save_path}/generated_samples.png', dpi=150)
            plt.close()

            # Plot 4: Distribution comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Intensity distribution
            axes[0].hist(patterns.flatten(), bins=50, alpha=0.7, label='Original', density=True)
            axes[0].hist(samples.flatten(), bins=50, alpha=0.7, label='Generated', density=True)
            axes[0].set_title('Intensity Distribution')
            axes[0].set_xlabel('Intensity')
            axes[0].set_ylabel('Density')
            axes[0].legend()

            # Peak detection comparison
            orig_peaks = []
            gen_peaks = []
            for i in range(min(50, len(patterns))):
                peaks_orig, _ = find_peaks(patterns[i], height=0.1)
                peaks_gen, _ = find_peaks(samples[i, 0], height=0.1)
                orig_peaks.append(len(peaks_orig))
                gen_peaks.append(len(peaks_gen))

            axes[1].hist(orig_peaks, bins=10, alpha=0.7, label='Original', density=True)
            axes[1].hist(gen_peaks, bins=10, alpha=0.7, label='Generated', density=True)
            axes[1].set_title('Peak Count Distribution')
            axes[1].set_xlabel('Number of Peaks')
            axes[1].set_ylabel('Density')
            axes[1].legend()

            # Dynamic range comparison
            orig_ranges = patterns.max(axis=1) - patterns.min(axis=1)
            gen_ranges = samples.max(axis=2) - samples.min(axis=2)

            axes[2].hist(orig_ranges, bins=20, alpha=0.7, label='Original', density=True)
            axes[2].hist(gen_ranges.flatten(), bins=20, alpha=0.7, label='Generated', density=True)
            axes[2].set_title('Dynamic Range Distribution')
            axes[2].set_xlabel('Max - Min Intensity')
            axes[2].set_ylabel('Density')
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(f'{save_path}/distribution_comparison.png', dpi=150)
            plt.close()

        print(f"✓ Visualizations saved to {save_path}")

    def run_full_validation(self):
        """
        Run complete validation suite.
        """
        print("STARTING COMPREHENSIVE VALIDATION SUITE")
        print("="*60)

        start_time = time.time()

        # Step 1: Test model components
        if not self.test_model_components():
            print("❌ Component tests failed - stopping validation")
            return False

        # Step 2: Test diffusion process
        if not self.test_diffusion_process():
            print("❌ Diffusion process tests failed - stopping validation")
            return False

        # Step 3: Generate test data
        patterns, conditions, peak_info = self.generate_synthetic_xrd(n_samples=500, pattern_length=256)

        # Step 4: Test training
        model, diffusion, _ = self.test_training_loop(max_epochs=10)

        # Step 5: Test sampling quality
        samples = None
        if model is not None:
            samples = self.test_sampling_quality(model, diffusion, patterns, n_samples=50)

        # Step 6: Create visualizations
        self.create_visualizations(patterns, samples)

        # Summary
        total_time = time.time() - start_time

        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        for test_name, result in self.results.items():
            if isinstance(result, bool):
                status = "✓ PASSED" if result else "❌ FAILED"
                print(f"{test_name}: {status}")
            else:
                print(f"{test_name}: ✓ PASSED")
                for key, value in result.items():
                    print(f"  {key}: {value}")

        print(f"\nTotal validation time: {total_time:.2f} seconds")

        all_passed = all(self.results.values())
        print(f"\nOverall result: {'✓ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

        return all_passed

def main():
    """
    Run validation suite.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running validation on device: {device}")

    validator = ValidationSuite(device=device)
    success = validator.run_full_validation()

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)