"""
Progress visualization for XRD diffusion model training.

Extracted from diffusion_model_0.1.5.py visualize_progress function.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_progress(model, diffusion, dataloader, epoch, device, save_path, num_timesteps):
    """
    Visualize the current model's performance on validation data.
    """
    model.eval()

    # Get a batch from the dataloader
    for synth_batch, real_batch, temp_batch in dataloader:
        synth_batch = synth_batch.to(device)
        real_batch = real_batch.to(device)
        temp_batch = temp_batch.to(device)
        break  # Just use the first batch

    with torch.no_grad():
        # Get sample index
        sample_idx = 0

        # Denoise real pattern with temperature conditioning
        t_zero = torch.zeros(1, dtype=torch.long, device=device)
        # Get noise prediction
        noise_pred_real = model(real_batch[sample_idx:sample_idx+1], t_zero, temp_batch[sample_idx:sample_idx+1])

        # Calculate the denoised signal
        alpha_bar_t = diffusion.alpha_bars[t_zero].view(-1, 1, 1)
        denoised_real = (real_batch[sample_idx:sample_idx+1] - torch.sqrt(1 - alpha_bar_t) * noise_pred_real) / torch.sqrt(alpha_bar_t)

        # Create different noise levels for the synthetic pattern
        t_low = torch.tensor([num_timesteps // 10], device=device)
        t_mid = torch.tensor([num_timesteps // 2], device=device)
        t_high = torch.tensor([num_timesteps * 9 // 10], device=device)

        # Add noise to synthetic pattern
        noisy_low, _ = diffusion.forward_diffusion(synth_batch[sample_idx:sample_idx+1], t_low)
        noisy_mid, _ = diffusion.forward_diffusion(synth_batch[sample_idx:sample_idx+1], t_mid)
        noisy_high, _ = diffusion.forward_diffusion(synth_batch[sample_idx:sample_idx+1], t_high)

        # Denoise at different noise levels
        # Get noise predictions
        noise_pred_low = model(noisy_low, t_low, temp_batch[sample_idx:sample_idx+1])
        noise_pred_mid = model(noisy_mid, t_mid, temp_batch[sample_idx:sample_idx+1])
        noise_pred_high = model(noisy_high, t_high, temp_batch[sample_idx:sample_idx+1])

        # Calculate the denoised signals
        alpha_bar_low = diffusion.alpha_bars[t_low].view(-1, 1, 1)
        alpha_bar_mid = diffusion.alpha_bars[t_mid].view(-1, 1, 1)
        alpha_bar_high = diffusion.alpha_bars[t_high].view(-1, 1, 1)

        denoised_low = (noisy_low - torch.sqrt(1 - alpha_bar_low) * noise_pred_low) / torch.sqrt(alpha_bar_low)
        denoised_mid = (noisy_mid - torch.sqrt(1 - alpha_bar_mid) * noise_pred_mid) / torch.sqrt(alpha_bar_mid)
        denoised_high = (noisy_high - torch.sqrt(1 - alpha_bar_high) * noise_pred_high) / torch.sqrt(alpha_bar_high)

    # Create visualization
    fig, axs = plt.subplots(3, 1, figsize=(12, 28), constrained_layout=True)

    L_vis = synth_batch.shape[-1]  # Get length from one of the patterns
    two_theta = np.linspace(0, 90, L_vis)

    # Plot 1: Real vs Denoised Real vs Synthetic (Ground Truth)
    axs[0].plot(two_theta, synth_batch[sample_idx, 0].cpu().numpy(), label='Synthetic (Ground Truth)', color='grey', linewidth=6)
    axs[0].plot(two_theta, real_batch[sample_idx, 0].cpu().numpy(), label='Real (Noisy)', color='blue', linestyle="--", alpha=0.7, linewidth=4)
    axs[0].plot(two_theta, denoised_real[0, 0].cpu().numpy(), label='Denoised Real', color='green', linestyle="-.", linewidth=2)
    axs[0].set_title(f'Real Data Denoising', fontsize=18)
    axs[0].legend(fontsize=16)
    axs[0].set_xlabel('Position (2θ)', fontsize=18)
    axs[0].set_ylabel('Intensity', fontsize=18)
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Noise Level Analysis
    axs[1].plot(two_theta, noisy_mid[0, 0].cpu().numpy(), label=f'Noisy (t={t_mid.item()})', color='gray', alpha=0.5)
    axs[1].plot(two_theta, denoised_mid[0, 0].cpu().numpy(), label='Denoised', color='red', linewidth=0.5)
    axs[1].set_title('Noise Level Analysis', fontsize=18)
    axs[1].legend(fontsize=16)
    axs[1].set_xlabel('Position (2θ)', fontsize=18)
    axs[1].set_ylabel('Intensity', fontsize=18)
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Progressive Denoising at Different Noise Levels
    axs[2].plot(two_theta, denoised_high[0, 0].cpu().numpy(), label=f'High Noise (t={t_high.item()})', color='red', alpha=0.7)
    axs[2].plot(two_theta, denoised_mid[0, 0].cpu().numpy(), label=f'Mid Noise (t={t_mid.item()})', color='orange', alpha=0.7)
    axs[2].plot(two_theta, denoised_low[0, 0].cpu().numpy(), label=f'Low Noise (t={t_low.item()})', color='green', alpha=0.7)
    axs[2].plot(two_theta, synth_batch[sample_idx, 0].cpu().numpy(), label='Ground Truth', color='black', linewidth=1.5)
    axs[2].set_title('Progressive Denoising', fontsize=18)
    axs[2].legend(fontsize=16)
    axs[2].set_xlabel('Position (2θ)', fontsize=18)
    axs[2].set_ylabel('Intensity', fontsize=18)
    axs[2].grid(True, alpha=0.3)

    # Save figure
    plt.savefig(f"{save_path}/progress_epoch_{epoch+1}.png", dpi=300)
    plt.close()