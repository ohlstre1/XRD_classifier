"""
Plotting utilities for XRD diffusion model visualization.

Extracted from diffusion_model_0.1.5.py plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_overlay_sample(model, diffusion, synth_pattern, real_pattern, temp,
                        t_choice, two_theta_axis=None, save_path=None, title_suffix=""):
    """
    Plot (1) clean synthetic pattern, (2) noisy version at timestep t_choice,
    and (3) denoised output at that same t_choice. Optionally includes the real (measured)
    pattern for reference.

    Arguments:
        model         : your trained ImprovedDiffusionDenoiser (in eval mode)
        diffusion     : your DiffusionProcess instance
        synth_pattern : a single-sample tensor of shape [1, 1, L] (synthetic ground truth)
        real_pattern  : a single-sample tensor of shape [1, 1, L] (experimental/noisy measurement)
        temp          : a single-sample tensor of shape [1, 1] (temperature/conditioning scalar)
        t_choice      : integer timestep at which to add noise & denoise
        two_theta_axis: optional 1-D numpy array of length L for x-axis (°2θ). If None, use indices.
        save_path     : optional string path to save the figure (PNG). If None, does not save.
        title_suffix  : optional string to append to plot title (e.g. epoch number).
    """
    model.eval()
    device = next(model.parameters()).device
    L = synth_pattern.shape[-1]

    # Move all to device
    x0 = synth_pattern.to(device)          # [1,1,L]
    real = real_pattern.to(device)         # [1,1,L]
    t = torch.tensor([t_choice], device=device, dtype=torch.long)  # [1]
    temp_in = temp.to(device)              # [1,1]

    # 1) Create noisy synthetic at timestep t_choice
    with torch.no_grad():
        noisy_x, noise = diffusion.forward_diffusion(x0, t)
        # 2) Denoise that noisy_x
        noise_pred = model(noisy_x, t, temp_in)                        # predict noise
        alpha_bar_t = diffusion.alpha_bars[t].view(1, 1, 1)            # [1,1,1]
        x0_pred = (noisy_x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        denoised = x0_pred.clamp(0, None)  # clamp if intensities must be ≥ 0

    # Convert to NumPy for plotting
    clean_np    = x0.cpu().numpy().reshape(-1)        # length L
    noisy_np    = noisy_x.cpu().numpy().reshape(-1)   # length L
    denoised_np = denoised.cpu().numpy().reshape(-1)  # length L
    real_np     = real.cpu().numpy().reshape(-1)      # length L

    # Prepare x-axis
    if two_theta_axis is None:
        x_axis = np.arange(L)
    else:
        x_axis = two_theta_axis

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, clean_np,    label="Synthetic (clean)",     color="black", linewidth=1.5)
    plt.plot(x_axis, noisy_np,    label=f"Noisy (t={t_choice})",   color="tab:gray", alpha=0.6, linewidth=1)
    plt.plot(x_axis, denoised_np, label="Denoised at t",          color="tab:red",  alpha=0.8, linewidth=1)
    # Optionally overlay real (measured) pattern:
    plt.plot(x_axis, real_np,     label="Real (experimental)",    color="tab:blue", alpha=0.4, linewidth=1)

    plt.title(f"Overlay: clean vs noisy vs denoised {title_suffix}".strip())
    plt.xlabel("Position (°2θ)" if two_theta_axis is not None else "Index")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_training_history(history, save_path):
    """
    Plot the training history metrics.
    """
    plt.figure(figsize=(12, 12))

    # Plot 1: Overall Loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('MSE Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    # Plot 2: Loss Components
    plt.subplot(2, 1, 2)
    plt.plot(history['diff_loss'], label='Diffusion Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.title('Loss Components', fontsize=16)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('MSE Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    plt.savefig(f"{save_path}/training_history.png", dpi=300)
    plt.close()


def plot_overlay_sample(model, diffusion, synth_pattern, real_pattern, temp,
                        t_choice, two_theta_axis=None, save_path=None, title_suffix=""):
    """
    Plot (1) clean synthetic pattern, (2) noisy version at timestep t_choice,
    and (3) denoised output at that same t_choice. Optionally includes the real (measured)
    pattern for reference.

    Arguments:
        model         : your trained ImprovedDiffusionDenoiser (in eval mode)
        diffusion     : your DiffusionProcess instance
        synth_pattern : a single-sample tensor of shape [1, 1, L] (synthetic ground truth)
        real_pattern  : a single-sample tensor of shape [1, 1, L] (experimental/noisy measurement)
        temp          : a single-sample tensor of shape [1, 1] (temperature/conditioning scalar)
        t_choice      : integer timestep at which to add noise & denoise
        two_theta_axis: optional 1-D numpy array of length L for x-axis (°2θ). If None, use indices.
        save_path     : optional string path to save the figure (PNG). If None, does not save.
        title_suffix  : optional string to append to plot title (e.g. epoch number).
    """
    model.eval()
    device = next(model.parameters()).device
    L = synth_pattern.shape[-1]

    # Move all to device
    x0 = synth_pattern.to(device)          # [1,1,L]
    real = real_pattern.to(device)         # [1,1,L]
    t = torch.tensor([t_choice], device=device, dtype=torch.long)  # [1]
    temp_in = temp.to(device)              # [1,1]

    # 1) Create noisy synthetic at timestep t_choice
    with torch.no_grad():
        noisy_x, noise = diffusion.forward_diffusion(x0, t)
        # 2) Denoise that noisy_x
        noise_pred = model(noisy_x, t, temp_in)                        # predict noise
        alpha_bar_t = diffusion.alpha_bars[t].view(1, 1, 1)            # [1,1,1]
        x0_pred = (noisy_x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        denoised = x0_pred.clamp(0, None)  # clamp if intensities must be ≥ 0

    # Convert to NumPy for plotting
    clean_np    = x0.cpu().numpy().reshape(-1)        # length L
    noisy_np    = noisy_x.cpu().numpy().reshape(-1)   # length L
    denoised_np = denoised.cpu().numpy().reshape(-1)  # length L
    real_np     = real.cpu().numpy().reshape(-1)      # length L

    # Prepare x-axis
    if two_theta_axis is None:
        x_axis = np.arange(L)
    else:
        x_axis = two_theta_axis

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, clean_np,    label="Synthetic (clean)",     color="black", linewidth=1.5)
    plt.plot(x_axis, noisy_np,    label=f"Noisy (t={t_choice})",   color="tab:gray", alpha=0.6, linewidth=1)
    plt.plot(x_axis, denoised_np, label="Denoised at t",          color="tab:red",  alpha=0.8, linewidth=1)
    # Optionally overlay real (measured) pattern:
    plt.plot(x_axis, real_np,     label="Real (experimental)",    color="tab:blue", alpha=0.4, linewidth=1)

    plt.title(f"Overlay: clean vs noisy vs denoised {title_suffix}".strip())
    plt.xlabel("Position (°2θ)" if two_theta_axis is not None else "Index")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()