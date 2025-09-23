#!/usr/bin/env python3
"""
XRD Pattern Augmenter: Generate Realistic Experimental-like Patterns
=====================================================================

This script uses the trained XRD diffusion model as an augmenter to transform
clean synthetic XRD patterns into realistic experimental-like patterns with
controlled noise, peak broadening, and measurement artifacts.

Features:
- Generate multiple augmented samples per input pattern
- Configurable temperature and noise conditions
- Stochastic seed variation for diverse outputs
- Batch processing for efficiency
- Comprehensive output saving and visualization

Usage:
    python xrd_pattern_augmenter.py --input_file data.pt --output_dir ./augmented --samples_per_pattern 5
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

# Import components from the original diffusion model
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

class XRDPatternAugmenter:
    """
    High-performance XRD pattern augmenter using trained diffusion model.
    """

    def __init__(self, model_path=None, device='auto', verbose=True, use_classical=False):
        """
        Initialize the XRD Pattern Augmenter.

        Args:
            model_path: Path to trained diffusion model checkpoint (None for classical mode)
            device: Device to run on ('auto', 'cpu', 'cuda')
            verbose: Enable verbose output
            use_classical: Use classical augmentation instead of model-based
        """
        self.verbose = verbose
        self.use_classical = use_classical or model_path is None

        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.verbose:
            print(f"🖥️  Using device: {self.device}")
            if self.device == 'cuda':
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        if self.use_classical:
            if self.verbose:
                print("🔧 Using classical augmentation mode (no training required)")
            self.model = None
            self.diffusion = None
            # Import classical augmenter
            from classical_xrd_augmenter import ClassicalXRDAugmenter
            self.classical_augmenter = ClassicalXRDAugmenter(verbose=False)
        else:
            # Load model
            self._load_model(model_path)

            # Initialize diffusion process
            self.diffusion = DiffusionProcess(
                num_timesteps=1000,
                schedule_type='cosine',
                device=self.device
            )

        if self.verbose:
            mode = "classical" if self.use_classical else "model-based"
            print(f"✅ XRD Pattern Augmenter initialized successfully in {mode} mode!")

    def _load_model(self, model_path):
        """Load the trained diffusion model."""
        if self.verbose:
            print(f"📂 Loading model from: {model_path}")

        # Initialize model architecture
        self.model = ImprovedDiffusionDenoiser(
            in_channels=1,
            hidden_channels=16,
            time_embedding_dim=256,
            num_res_blocks=2,
            attention_levels=[1, 2],
            num_levels=2,
            temperature_condition=True
        ).to(self.device)

        # Load checkpoint if exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self.verbose:
                epoch = checkpoint.get('epoch', 'Unknown')
                loss = checkpoint.get('loss', 'Unknown')
                print(f"   ✅ Loaded checkpoint from epoch {epoch} with loss {loss}")
        else:
            if self.verbose:
                print(f"   ⚠️  Checkpoint not found. Using randomly initialized model.")

        self.model.eval()

    def generate_temperature_conditions(self, num_samples, temp_range=(0.1, 2.0), temp_mode='random'):
        """
        Generate temperature conditioning values.

        Args:
            num_samples: Number of temperature values to generate
            temp_range: (min, max) temperature range
            temp_mode: 'random', 'linear', 'exponential', or specific value

        Returns:
            torch.Tensor: Temperature values [num_samples, 1]
        """
        if isinstance(temp_mode, (int, float)):
            # Fixed temperature
            temps = torch.full((num_samples, 1), temp_mode, dtype=torch.float32)
        elif temp_mode == 'random':
            # Random uniform distribution
            temps = torch.rand(num_samples, 1) * (temp_range[1] - temp_range[0]) + temp_range[0]
        elif temp_mode == 'linear':
            # Linear spacing
            temps = torch.linspace(temp_range[0], temp_range[1], num_samples).unsqueeze(1)
        elif temp_mode == 'exponential':
            # Exponential distribution (more low values)
            temps = torch.exponential(torch.ones(num_samples, 1)) * 0.3 + temp_range[0]
            temps = torch.clamp(temps, temp_range[0], temp_range[1])
        else:
            raise ValueError(f"Unknown temp_mode: {temp_mode}")

        return temps.to(self.device)

    def augment_pattern(self, synth_pattern, num_samples=5, temp_range=(0.1, 2.0),
                       temp_mode='random', noise_timestep_range=(0, 50),
                       base_seed=None, return_metadata=False, **kwargs):
        """
        Augment a single synthetic XRD pattern to create realistic experimental-like patterns.

        Args:
            synth_pattern: Input synthetic pattern [1, L] or [1, 1, L]
            num_samples: Number of augmented samples to generate
            temp_range: Temperature conditioning range (ignored in classical mode)
            temp_mode: Temperature sampling mode (ignored in classical mode)
            noise_timestep_range: Range of diffusion timesteps for noise addition (ignored in classical mode)
            base_seed: Base random seed (None for random)
            return_metadata: Whether to return augmentation metadata
            **kwargs: Additional arguments for classical mode

        Returns:
            torch.Tensor: Augmented patterns [num_samples, 1, L]
            dict: Metadata (if return_metadata=True)
        """
        if self.use_classical:
            # Use classical augmentation
            augmented, metadata = self.classical_augmenter.augment_pattern(
                pattern=synth_pattern,
                num_samples=num_samples,
                base_seed=base_seed,
                **kwargs
            )

            # Ensure output format matches [num_samples, 1, L]
            if augmented.dim() == 2:
                augmented = augmented.unsqueeze(1)

            if return_metadata:
                return augmented, metadata
            else:
                return augmented
        # Ensure proper input shape
        if synth_pattern.dim() == 2:
            synth_pattern = synth_pattern.unsqueeze(1)  # [1, 1, L]
        elif synth_pattern.dim() == 1:
            synth_pattern = synth_pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, L]

        synth_pattern = synth_pattern.to(self.device)

        # Generate conditions
        temperatures = self.generate_temperature_conditions(num_samples, temp_range, temp_mode)

        # Generate random seeds if base_seed provided
        if base_seed is not None:
            seeds = [base_seed + i for i in range(num_samples)]
        else:
            seeds = [None] * num_samples

        # Storage for results
        augmented_patterns = []
        metadata = {
            'temperatures': temperatures.cpu().numpy(),
            'seeds': seeds,
            'timesteps': [],
            'noise_levels': []
        }

        self.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                # Set seed for this sample
                if seeds[i] is not None:
                    torch.manual_seed(seeds[i])
                    np.random.seed(seeds[i])

                # Random timestep for noise addition
                timestep = torch.randint(
                    noise_timestep_range[0],
                    noise_timestep_range[1] + 1,
                    (1,),
                    device=self.device
                )

                # Get temperature for this sample
                temp = temperatures[i:i+1]

                # Apply augmentation process
                # Method 1: Direct noise prediction and application
                noise_pred = self.model(synth_pattern, timestep, temp)

                # Add predicted noise to create realistic experimental pattern
                noise_scale = 0.1 + (timestep.float() / 1000.0) * 0.3  # Scale noise by timestep
                augmented = synth_pattern + noise_pred * noise_scale

                # Clamp to reasonable values
                augmented = torch.clamp(augmented, 0, None)

                augmented_patterns.append(augmented)

                # Store metadata
                metadata['timesteps'].append(timestep.item())
                metadata['noise_levels'].append(noise_scale.item())

        # Combine results
        augmented_patterns = torch.cat(augmented_patterns, dim=0)

        if return_metadata:
            return augmented_patterns, metadata
        else:
            return augmented_patterns

    def augment_batch(self, synth_patterns, samples_per_pattern=5,
                     temp_range=(0.1, 2.0), temp_mode='random',
                     noise_timestep_range=(0, 50), base_seed=42,
                     batch_size=8, progress_bar=True, **kwargs):
        """
        Augment a batch of synthetic XRD patterns.

        Args:
            synth_patterns: Input patterns [N, L] or [N, 1, L]
            samples_per_pattern: Number of augmented samples per input
            temp_range: Temperature conditioning range (ignored in classical mode)
            temp_mode: Temperature sampling mode (ignored in classical mode)
            noise_timestep_range: Range of diffusion timesteps (ignored in classical mode)
            base_seed: Base random seed
            batch_size: Processing batch size (ignored in classical mode)
            progress_bar: Show progress bar
            **kwargs: Additional arguments for classical mode

        Returns:
            torch.Tensor: All augmented patterns [N*samples_per_pattern, 1, L]
            list: Metadata for each input pattern
        """
        if self.use_classical:
            # Use classical batch augmentation
            augmented, metadata = self.classical_augmenter.augment_batch(
                patterns=synth_patterns,
                samples_per_pattern=samples_per_pattern,
                base_seed=base_seed,
                progress_bar=progress_bar,
                **kwargs
            )

            # Ensure output format matches [N*samples_per_pattern, 1, L]
            if augmented.dim() == 2:
                augmented = augmented.unsqueeze(1)

            return augmented, metadata
        # Ensure proper shape
        if synth_patterns.dim() == 2:
            synth_patterns = synth_patterns.unsqueeze(1)  # [N, 1, L]

        num_patterns = synth_patterns.shape[0]
        all_augmented = []
        all_metadata = []

        iterator = range(0, num_patterns, batch_size)
        if progress_bar:
            iterator = tqdm(iterator, desc="Augmenting patterns")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, num_patterns)
            batch_patterns = synth_patterns[start_idx:end_idx]

            for i, pattern in enumerate(batch_patterns):
                pattern_seed = base_seed + (start_idx + i) * 1000 if base_seed else None

                augmented, metadata = self.augment_pattern(
                    pattern.unsqueeze(0),  # Add batch dimension
                    num_samples=samples_per_pattern,
                    temp_range=temp_range,
                    temp_mode=temp_mode,
                    noise_timestep_range=noise_timestep_range,
                    base_seed=pattern_seed,
                    return_metadata=True
                )

                all_augmented.append(augmented)
                all_metadata.append(metadata)

        # Combine all results
        all_augmented = torch.cat(all_augmented, dim=0)
        return all_augmented, all_metadata

    def save_results(self, augmented_patterns, metadata_list, original_patterns,
                    output_dir, prefix="augmented"):
        """
        Save augmented patterns and metadata.

        Args:
            augmented_patterns: Generated patterns [N, 1, L]
            metadata_list: List of metadata dicts
            original_patterns: Original synthetic patterns [M, 1, L]
            output_dir: Output directory
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save patterns
        patterns_file = os.path.join(output_dir, f"{prefix}_patterns_{timestamp}.pt")
        torch.save({
            'augmented_patterns': augmented_patterns.cpu(),
            'original_patterns': original_patterns.cpu(),
            'metadata': metadata_list,
            'generation_info': {
                'timestamp': timestamp,
                'num_original': original_patterns.shape[0],
                'num_augmented': augmented_patterns.shape[0],
                'samples_per_pattern': len(metadata_list[0]['temperatures']) if metadata_list else 0
            }
        }, patterns_file)

        if self.verbose:
            print(f"💾 Saved patterns to: {patterns_file}")

        # Save metadata as JSON
        metadata_file = os.path.join(output_dir, f"{prefix}_metadata_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        json_metadata = []
        for meta in metadata_list:
            json_meta = {}
            for key, value in meta.items():
                if isinstance(value, np.ndarray):
                    json_meta[key] = value.tolist()
                else:
                    json_meta[key] = value
            json_metadata.append(json_meta)

        with open(metadata_file, 'w') as f:
            json.dump({
                'metadata': json_metadata,
                'generation_info': {
                    'timestamp': timestamp,
                    'num_patterns': len(json_metadata),
                    'device_used': self.device
                }
            }, f, indent=2)

        if self.verbose:
            print(f"📄 Saved metadata to: {metadata_file}")

    def visualize_augmentation(self, original_pattern, augmented_patterns,
                              metadata=None, save_path=None, show_plot=True):
        """
        Visualize original pattern and its augmented versions.

        Args:
            original_pattern: Original synthetic pattern [1, L] or [L]
            augmented_patterns: Augmented patterns [N, 1, L]
            metadata: Augmentation metadata
            save_path: Path to save visualization
            show_plot: Whether to display plot
        """
        # Prepare data
        if original_pattern.dim() > 1:
            orig = original_pattern.cpu().numpy().flatten()
        else:
            orig = original_pattern.cpu().numpy()

        aug_data = augmented_patterns.cpu().numpy()
        if aug_data.ndim == 3:
            aug_data = aug_data.squeeze(1)

        # Create x-axis (assume 2theta range 0-90 degrees)
        x_axis = np.linspace(0, 90, len(orig))

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

        # Plot 4: Metadata visualization
        if metadata is not None:
            temps = metadata['temperatures'].flatten()
            timesteps = metadata['timesteps']
            noise_levels = metadata['noise_levels']

            axes[1,1].scatter(temps, timesteps, c=noise_levels, cmap='viridis', s=100, alpha=0.7)
            axes[1,1].set_xlabel('Temperature Conditioning')
            axes[1,1].set_ylabel('Diffusion Timestep')
            axes[1,1].set_title('Augmentation Parameters')
            cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
            cbar.set_label('Noise Level')
            axes[1,1].grid(True, alpha=0.3)
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
    """
    Load synthetic XRD patterns from file.

    Args:
        file_path: Path to data file (.pt, .npy, etc.)
        max_patterns: Maximum number of patterns to load

    Returns:
        torch.Tensor: Synthetic patterns
    """
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
    parser = argparse.ArgumentParser(description='XRD Pattern Augmenter')

    # Input/Output
    parser.add_argument('--input_file', required=True, help='Input synthetic patterns file')
    parser.add_argument('--output_dir', default='./augmented_patterns', help='Output directory')
    parser.add_argument('--model_path', default='./models/xrd_diffusion/improved_diffusion_model_best.pth',
                       help='Path to trained model (ignored if --no_training used)')

    # Mode selection
    parser.add_argument('--no_training', action='store_true',
                       help='Use classical augmentation without trained model')

    # Augmentation parameters
    parser.add_argument('--samples_per_pattern', type=int, default=5,
                       help='Number of augmented samples per input pattern')
    parser.add_argument('--temp_range', nargs=2, type=float, default=[0.1, 2.0],
                       help='Temperature conditioning range')
    parser.add_argument('--temp_mode', default='random',
                       choices=['random', 'linear', 'exponential'],
                       help='Temperature sampling mode')
    parser.add_argument('--noise_timestep_range', nargs=2, type=int, default=[0, 50],
                       help='Diffusion timestep range for noise addition')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed')

    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Processing batch size')
    parser.add_argument('--max_patterns', type=int, help='Maximum patterns to process')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    # Output options
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--vis_samples', type=int, default=3,
                       help='Number of patterns to visualize')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Initialize augmenter
    mode_str = "Classical (No Training)" if args.no_training else "Model-Based"
    print(f"🔬 XRD Pattern Augmenter - {mode_str}")
    print("=" * 50)

    if args.no_training:
        augmenter = XRDPatternAugmenter(
            model_path=None,
            device=args.device,
            verbose=not args.quiet,
            use_classical=True
        )
    else:
        augmenter = XRDPatternAugmenter(
            model_path=args.model_path,
            device=args.device,
            verbose=not args.quiet
        )

    # Load synthetic patterns
    print(f"\n📂 Loading synthetic patterns from: {args.input_file}")
    synth_patterns = load_synthetic_patterns(args.input_file, args.max_patterns)
    print(f"   Loaded {synth_patterns.shape[0]} patterns with {synth_patterns.shape[-1]} points each")

    # Run augmentation
    print(f"\n🎯 Starting augmentation...")
    print(f"   Samples per pattern: {args.samples_per_pattern}")
    print(f"   Temperature range: {args.temp_range}")
    print(f"   Noise timestep range: {args.noise_timestep_range}")

    start_time = time.time()

    augmented_patterns, metadata_list = augmenter.augment_batch(
        synth_patterns=synth_patterns,
        samples_per_pattern=args.samples_per_pattern,
        temp_range=tuple(args.temp_range),
        temp_mode=args.temp_mode,
        noise_timestep_range=tuple(args.noise_timestep_range),
        base_seed=args.base_seed,
        batch_size=args.batch_size,
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

            vis_path = os.path.join(vis_dir, f"augmentation_example_{i+1}.png")
            augmenter.visualize_augmentation(
                original_pattern=synth_patterns[i],
                augmented_patterns=augmented_patterns[start_idx:end_idx],
                metadata=metadata_list[i],
                save_path=vis_path,
                show_plot=False
            )

        print(f"   Saved {args.vis_samples} visualization(s) to: {vis_dir}")

    print(f"\n✅ XRD Pattern Augmentation Complete!")
    print(f"📁 Check output directory: {args.output_dir}")

if __name__ == "__main__":
    main()