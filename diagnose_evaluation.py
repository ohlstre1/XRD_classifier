#!/usr/bin/env python3
"""
Diagnostic script to investigate the zero standard deviation issue
and provide insights into model behavior.
"""

import torch
import numpy as np
from evaluate_diffusion_std import load_model_and_data, PerformanceEvaluator
import matplotlib.pyplot as plt
import os

def diagnose_model_variability():
    """
    Investigate why standard deviations are near zero.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and data
    model_path = "./models/xrd_diffusion/improved_diffusion_model_best.pth"
    data_path = "data/xrd_dataset_labeled_dtw_window.pt"

    model, diffusion, test_dataloader = load_model_and_data(model_path, data_path, device)

    print("\n🔍 DIAGNOSING MODEL VARIABILITY")
    print("=" * 50)

    # Get a single batch for testing
    synth_batch, real_batch, temp_batch = next(iter(test_dataloader))
    synth_batch = synth_batch.to(device)
    real_batch = real_batch.to(device)
    temp_batch = temp_batch.to(device)

    # Take first sample
    synth_sample = synth_batch[0:1]
    real_sample = real_batch[0:1]
    temp_sample = temp_batch[0:1]

    print(f"Sample shapes: synth={synth_sample.shape}, real={real_sample.shape}")

    # Test 1: Multiple forward passes with different seeds
    print("\n📋 Test 1: Multiple forward passes with different random seeds")
    model.eval()
    outputs = []

    for i in range(10):
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)

        with torch.no_grad():
            t_zero = torch.zeros(1, dtype=torch.long, device=device)
            noise_pred = model(real_sample, t_zero, temp_sample)

            # Denoise
            alpha_bar_t = diffusion.alpha_bars[t_zero].view(-1, 1, 1)
            denoised = (real_sample - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            outputs.append(denoised.cpu().numpy())

    outputs = np.array(outputs)
    output_std = np.std(outputs, axis=0)
    output_mean = np.mean(outputs, axis=0)

    print(f"   Output mean: {np.mean(output_mean):.8f}")
    print(f"   Output std:  {np.mean(output_std):.8f}")
    print(f"   Max std:     {np.max(output_std):.8f}")
    print(f"   Min std:     {np.min(output_std):.8f}")

    if np.mean(output_std) < 1e-6:
        print("   ⚠️  ISSUE: Outputs are nearly identical across seeds!")
        print("   This suggests the model is deterministic or seeds aren't affecting inference")

    # Test 2: Dropout behavior
    print("\n📋 Test 2: Dropout behavior analysis")
    model.train()  # Enable dropout
    train_outputs = []

    for i in range(10):
        with torch.no_grad():
            t_zero = torch.zeros(1, dtype=torch.long, device=device)
            noise_pred = model(real_sample, t_zero, temp_sample)
            train_outputs.append(noise_pred.cpu().numpy())

    model.eval()  # Back to eval mode

    train_outputs = np.array(train_outputs)
    train_std = np.std(train_outputs, axis=0)

    print(f"   Train mode std: {np.mean(train_std):.8f}")

    if np.mean(train_std) > np.mean(output_std):
        print("   ✅ Dropout is working - more variation in train mode")
    else:
        print("   ⚠️  Model may not have dropout layers or they're not effective")

    # Test 3: Different timesteps
    print("\n📋 Test 3: Different timestep behavior")
    timestep_outputs = {}
    timesteps_to_test = [0, 100, 500, 900]

    model.eval()
    for t_val in timesteps_to_test:
        t = torch.tensor([t_val], device=device, dtype=torch.long)

        with torch.no_grad():
            noise_pred = model(real_sample, t, temp_sample)
            timestep_outputs[t_val] = noise_pred.cpu().numpy()

    # Check if different timesteps produce different outputs
    t0_output = timestep_outputs[0]
    for t_val in timesteps_to_test[1:]:
        diff = np.mean(np.abs(timestep_outputs[t_val] - t0_output))
        print(f"   t={t_val} vs t=0 difference: {diff:.8f}")

    # Test 4: Temperature conditioning effect
    print("\n📋 Test 4: Temperature conditioning effect")
    temp_outputs = {}
    temp_values = [0.1, 0.5, 1.0, 2.0]

    for temp_val in temp_values:
        temp_test = torch.tensor([[temp_val]], device=device, dtype=torch.float32)

        with torch.no_grad():
            t_zero = torch.zeros(1, dtype=torch.long, device=device)
            noise_pred = model(real_sample, t_zero, temp_test)
            temp_outputs[temp_val] = noise_pred.cpu().numpy()

    # Check temperature effect
    base_output = temp_outputs[0.1]
    for temp_val in temp_values[1:]:
        diff = np.mean(np.abs(temp_outputs[temp_val] - base_output))
        print(f"   temp={temp_val} vs temp=0.1 difference: {diff:.8f}")

    # Test 5: Input sensitivity
    print("\n📋 Test 5: Input sensitivity analysis")

    # Add small noise to input
    noise_levels = [0.0, 0.001, 0.01, 0.1]
    sensitivity_outputs = {}

    for noise_level in noise_levels:
        noisy_input = real_sample + torch.randn_like(real_sample) * noise_level

        with torch.no_grad():
            t_zero = torch.zeros(1, dtype=torch.long, device=device)
            noise_pred = model(noisy_input, t_zero, temp_sample)
            sensitivity_outputs[noise_level] = noise_pred.cpu().numpy()

    base_sens_output = sensitivity_outputs[0.0]
    for noise_level in noise_levels[1:]:
        diff = np.mean(np.abs(sensitivity_outputs[noise_level] - base_sens_output))
        print(f"   Input noise={noise_level} output difference: {diff:.8f}")

    # Create visualization
    print("\n📊 Creating diagnostic visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Output variability across seeds
    axes[0,0].plot(outputs.reshape(10, -1).T, alpha=0.5)
    axes[0,0].set_title('Output Variability Across Seeds')
    axes[0,0].set_ylabel('Output Value')
    axes[0,0].grid(True)

    # Plot 2: Standard deviation heatmap
    axes[0,1].imshow(output_std.reshape(1, -1), aspect='auto', cmap='viridis')
    axes[0,1].set_title('Output Standard Deviation')
    axes[0,1].set_ylabel('Batch')

    # Plot 3: Timestep comparison
    x_axis = range(real_sample.shape[-1])
    for i, t_val in enumerate(timesteps_to_test):
        axes[0,2].plot(x_axis, timestep_outputs[t_val].flatten(),
                      label=f't={t_val}', alpha=0.7)
    axes[0,2].set_title('Timestep Effect on Output')
    axes[0,2].legend()
    axes[0,2].grid(True)

    # Plot 4: Temperature effect
    for temp_val in temp_values:
        axes[1,0].plot(x_axis, temp_outputs[temp_val].flatten(),
                      label=f'temp={temp_val}', alpha=0.7)
    axes[1,0].set_title('Temperature Effect on Output')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Plot 5: Input sensitivity
    for noise_level in noise_levels:
        axes[1,1].plot(x_axis, sensitivity_outputs[noise_level].flatten(),
                      label=f'noise={noise_level}', alpha=0.7)
    axes[1,1].set_title('Input Sensitivity Analysis')
    axes[1,1].legend()
    axes[1,1].grid(True)

    # Plot 6: Original vs denoised comparison
    axes[1,2].plot(x_axis, real_sample.cpu().numpy().flatten(),
                  label='Original (noisy)', alpha=0.7)
    axes[1,2].plot(x_axis, outputs[0].flatten(),
                  label='Denoised', alpha=0.7)
    axes[1,2].plot(x_axis, synth_sample.cpu().numpy().flatten(),
                  label='Target (synthetic)', alpha=0.7)
    axes[1,2].set_title('Denoising Performance')
    axes[1,2].legend()
    axes[1,2].grid(True)

    plt.tight_layout()

    # Save diagnostic plot
    os.makedirs("./evaluation_results", exist_ok=True)
    plt.savefig("./evaluation_results/diagnostic_analysis.png", dpi=300, bbox_inches='tight')
    print("   Saved diagnostic plot to: ./evaluation_results/diagnostic_analysis.png")

    plt.show()

    # Summary and recommendations
    print("\n💡 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 50)

    if np.mean(output_std) < 1e-6:
        print("🚨 ISSUE IDENTIFIED: Model outputs are deterministic")
        print("   Possible causes:")
        print("   • Model weights are frozen or not properly loaded")
        print("   • All dropout/stochastic layers are disabled")
        print("   • Model has converged to a fixed point")
        print("   • Random seed is being set identically for each run")

        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Verify model is properly loaded and not frozen")
        print("   2. Check if evaluation script properly varies random seeds")
        print("   3. Consider adding noise injection during evaluation")
        print("   4. Use Monte Carlo dropout (model.train() during inference)")

    else:
        print("✅ Model shows appropriate variability")

    # Check SNR issue
    snr_improvement = -0.970076858997345  # From your results
    if snr_improvement < 0:
        print(f"\n🚨 SNR ISSUE: Model reduces signal quality by {abs(snr_improvement):.2f} dB")
        print("   This suggests the model is adding noise rather than removing it")
        print("   Possible causes:")
        print("   • Model not properly trained")
        print("   • Incorrect loss function or training objective")
        print("   • Model architecture issues")
        print("   • Training data quality problems")

if __name__ == "__main__":
    diagnose_model_variability()