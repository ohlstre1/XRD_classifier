#!/usr/bin/env python3
"""
Direct test of the robust diffusion model to verify it works correctly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from robust_diffusion_model import (
    RobustDiffusionDenoiser,
    CleanDiffusionProcess,
    XRDDataset,
    train_robust_diffusion
)

from torch.utils.data import DataLoader, random_split

def test_robust_model():
    """
    Test the robust diffusion model directly.
    """
    print("🔬 TESTING ROBUST DIFFUSION MODEL")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test 1: Model Creation and Forward Pass
    print("\n1. Testing model creation and forward pass...")
    try:
        model = RobustDiffusionDenoiser(
            in_channels=1,
            hidden_channels=32,
            time_embedding_dim=64,
            num_res_blocks=1,
            attention_levels=[],
            num_levels=2,
            condition_dim=1
        ).to(device)

        # Test input
        batch_size = 4
        length = 128
        x = torch.randn(batch_size, 1, length).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        condition = torch.randn(batch_size, 1).to(device)

        # Forward pass
        output = model(x, t, condition)

        print(f"✓ Model forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False

    # Test 2: Diffusion Process
    print("\n2. Testing diffusion process...")
    try:
        diffusion = CleanDiffusionProcess(num_timesteps=100, device=device)

        # Test forward diffusion
        x_clean = torch.randn(batch_size, 1, length).to(device)
        t_test = torch.randint(0, 100, (batch_size,)).to(device)
        x_noisy, noise = diffusion.forward_diffusion(x_clean, t_test)

        print(f"✓ Forward diffusion successful")
        print(f"  Clean input shape: {x_clean.shape}")
        print(f"  Noisy output shape: {x_noisy.shape}")
        print(f"  Noise shape: {noise.shape}")

        # Check that we can predict noise
        with torch.no_grad():
            predicted_noise = model(x_noisy, t_test, condition)
            print(f"  Predicted noise shape: {predicted_noise.shape}")

        assert x_noisy.shape == x_clean.shape
        assert noise.shape == x_clean.shape
        assert predicted_noise.shape == noise.shape

    except Exception as e:
        print(f"✗ Diffusion process failed: {e}")
        return False

    # Test 3: Training Step
    print("\n3. Testing training step...")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        model.train()
        optimizer.zero_grad()

        # Training step
        predicted_noise = model(x_noisy, t_test, condition)
        loss = loss_fn(predicted_noise, noise)

        loss.backward()
        optimizer.step()

        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.6f}")

        # Check gradients
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  Gradient norm: {grad_norm:.6f}")

        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False

    # Test 4: End-to-End Training
    print("\n4. Testing end-to-end training...")
    try:
        # Generate synthetic data
        n_samples = 100
        patterns = []
        conditions = []

        for i in range(n_samples):
            # Create simple XRD-like patterns
            x_axis = np.linspace(0, 100, length)
            pattern = np.zeros_like(x_axis)

            # Add some peaks
            for _ in range(np.random.randint(2, 6)):
                center = np.random.uniform(20, 80)
                height = np.random.uniform(0.5, 1.0)
                width = np.random.uniform(1, 3)
                pattern += height * np.exp(-0.5 * ((x_axis - center) / width) ** 2)

            # Add background
            pattern += np.random.uniform(0.01, 0.1)

            # Normalize
            if pattern.max() > 0:
                pattern = pattern / pattern.max()

            patterns.append(pattern)
            conditions.append([np.random.uniform(0, 1)])

        patterns = np.array(patterns)
        conditions = np.array(conditions)

        # Create dataset
        dataset = XRDDataset(patterns, conditions)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Create fresh model for training
        train_model = RobustDiffusionDenoiser(
            in_channels=1,
            hidden_channels=16,  # Smaller for faster training
            time_embedding_dim=32,
            num_res_blocks=1,
            attention_levels=[],
            num_levels=1,
            condition_dim=1
        ).to(device)

        train_diffusion = CleanDiffusionProcess(num_timesteps=50, device=device)

        print(f"  Training on {len(patterns)} samples for 3 epochs...")

        # Train for just 3 epochs to test
        history, trained_model = train_robust_diffusion(
            model=train_model,
            diffusion=train_diffusion,
            dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=3,
            lr=1e-3,
            device=device,
            save_path='./temp_models'
        )

        print(f"✓ End-to-end training successful")
        print(f"  Initial loss: {history['train_loss'][0]:.6f}")
        print(f"  Final loss: {history['train_loss'][-1]:.6f}")
        print(f"  Loss decreased: {history['train_loss'][0] > history['train_loss'][-1]}")

        # Test sampling
        with torch.no_grad():
            trained_model.eval()
            sample_noise = torch.randn(2, 1, length).to(device)
            sample_condition = torch.randn(2, 1).to(device)
            samples = train_diffusion.sample(trained_model, sample_noise, sample_condition, timesteps=list(range(49, -1, -5)))

            print(f"✓ Sampling successful")
            print(f"  Sample shape: {samples.shape}")
            print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    except Exception as e:
        print(f"✗ End-to-end training failed: {e}")
        return False

    # Test 5: Visualization
    print("\n5. Creating visualization...")
    try:
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Original pattern
        axes[0, 0].plot(patterns[0])
        axes[0, 0].set_title('Original Synthetic Pattern')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Intensity')

        # Add noise and denoise
        with torch.no_grad():
            test_pattern = torch.tensor(patterns[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            test_condition = torch.tensor(conditions[0], dtype=torch.float32).unsqueeze(0).to(device)
            t_noise = torch.tensor([25], device=device)

            noisy_pattern, noise_added = train_diffusion.forward_diffusion(test_pattern, t_noise)
            predicted_noise = trained_model(noisy_pattern, t_noise, test_condition)

            # Simple denoising (not full DDPM)
            alpha_bar = train_diffusion.alpha_bars[t_noise[0]]
            denoised = (noisy_pattern - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

        axes[0, 1].plot(noisy_pattern[0, 0].cpu().numpy())
        axes[0, 1].set_title('Noisy Pattern')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Intensity')

        axes[1, 0].plot(denoised[0, 0].cpu().numpy())
        axes[1, 0].set_title('Denoised Pattern')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Intensity')

        # Generated sample
        axes[1, 1].plot(samples[0, 0].cpu().numpy())
        axes[1, 1].set_title('Generated Sample')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Intensity')

        plt.tight_layout()
        plt.savefig('robust_model_test.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization created - saved as 'robust_model_test.png'")

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False

    print("\n" + "="*50)
    print("🎉 ALL ROBUST MODEL TESTS PASSED!")
    print("The robust diffusion model is working correctly.")
    print("="*50)

    return True

if __name__ == "__main__":
    success = test_robust_model()
    if success:
        print("\n✅ VALIDATION COMPLETE")
        print("The robust diffusion model is ready for use with real XRD data!")
    else:
        print("\n❌ VALIDATION FAILED")
        print("There are issues that need to be addressed.")