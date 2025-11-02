"""
Interactive visualization utilities for XRD diffusion validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Tuple, Any
import ipywidgets as widgets
from IPython.display import display, clear_output

# Try to import widgets, handle case where they're not available
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Interactive widgets not available. Install ipywidgets for interactive features.")


def create_interactive_explorer(model, diffusion, test_synth: torch.Tensor,
                               test_real: torch.Tensor, test_dtw: torch.Tensor) -> Optional[widgets.Widget]:
    """
    Create interactive widgets for exploring model behavior.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        test_synth: Test synthetic patterns
        test_real: Test real patterns
        test_dtw: Test DTW values

    Returns:
        Interactive widget or None if widgets not available
    """
    if not WIDGETS_AVAILABLE:
        print("Interactive widgets not available. Cannot create explorer.")
        return None

    # Sample selection
    sample_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(test_synth) - 1,
        step=1,
        description='Sample:',
        continuous_update=False
    )

    # Timestep selection
    timestep_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=diffusion.num_timesteps - 1,
        step=10,
        description='Timestep:',
        continuous_update=False
    )

    # DTW conditioning
    dtw_slider = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        description='DTW Value:',
        continuous_update=False
    )

    # Stochastic mode
    stochastic_checkbox = widgets.Checkbox(
        value=False,
        description='Enable Stochastic Mode'
    )

    # Number of variations
    n_variations_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Variations:',
        continuous_update=False
    )

    # Output widget
    output = widgets.Output()

    def update_plot(sample_idx, timestep, dtw_value, use_stochastic, n_variations):
        with output:
            clear_output(wait=True)

            # Get sample data
            synth_sample = test_synth[sample_idx]
            real_sample = test_real[sample_idx]
            original_dtw = test_dtw[sample_idx].item()

            # Set model mode
            model.set_stochastic_mode(use_stochastic)
            if use_stochastic:
                model.train()
            else:
                model.eval()

            # Prepare inputs
            if synth_sample.dim() == 1:
                synth_sample = synth_sample.unsqueeze(0)

            # Get device from model parameters
            device = next(model.parameters()).device
            x = synth_sample.unsqueeze(0).to(device)
            t = torch.tensor([timestep], device=device, dtype=torch.long)
            dtw = torch.tensor([[dtw_value]], device=device, dtype=torch.float32)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot 1: Original data
            axes[0, 0].plot(synth_sample[0].cpu(), 'k-', linewidth=2, label='Synthetic', alpha=0.8)
            axes[0, 0].plot(real_sample[0].cpu(), 'g-', linewidth=2, label='Real', alpha=0.6)
            axes[0, 0].set_title(f'Sample {sample_idx} (Original DTW: {original_dtw:.3f})')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Intensity')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Model outputs
            with torch.no_grad():
                if use_stochastic and n_variations > 1:
                    # Multiple stochastic variations
                    for i in range(n_variations):
                        output_sample = model(x, t, dtw)
                        alpha = 0.8 if i == 0 else 0.4
                        label = 'Model Output' if i == 0 else None
                        axes[0, 1].plot(output_sample[0, 0].cpu(), 'r-', alpha=alpha, linewidth=1.5, label=label)
                else:
                    # Single output
                    output_sample = model(x, t, dtw)
                    axes[0, 1].plot(output_sample[0, 0].cpu(), 'r-', linewidth=2, label='Model Output')

                axes[0, 1].plot(real_sample[0].cpu(), 'g-', linewidth=2, label='Real Target', alpha=0.6)

            mode_str = "Stochastic" if use_stochastic else "Deterministic"
            axes[0, 1].set_title(f'{mode_str} Output (t={timestep}, DTW={dtw_value:.2f})')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Intensity')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Diffusion process visualization (if timestep > 0)
            if timestep > 0:
                with torch.no_grad():
                    x_noisy, noise = diffusion.forward_diffusion(x, t)
                    noise_pred = model(x_noisy, t, dtw)

                    axes[1, 0].plot(synth_sample[0].cpu(), 'k-', linewidth=2, label='Original', alpha=0.8)
                    axes[1, 0].plot(x_noisy[0, 0].cpu(), 'orange', linewidth=1.5, label='Noisy', alpha=0.7)

                    # Reconstruct x0
                    alpha_bar_t = diffusion.alpha_bars[t].view(1, 1, 1)
                    x0_pred = (x_noisy - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                    axes[1, 0].plot(x0_pred[0, 0].cpu(), 'r--', linewidth=1.5, label='Reconstructed', alpha=0.7)

                axes[1, 0].set_title(f'Diffusion Process (t={timestep})')
                axes[1, 0].set_xlabel('Position')
                axes[1, 0].set_ylabel('Intensity')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No diffusion at t=0\n(Direct transformation)',
                               ha='center', va='center', transform=axes[1, 0].transAxes,
                               fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
                axes[1, 0].set_title('Diffusion Process')

            # Plot 4: Parameter info
            info_text = f"""
Current Settings:
Sample Index: {sample_idx}
Timestep: {timestep}
DTW Value: {dtw_value:.3f}
Original DTW: {original_dtw:.3f}
Mode: {mode_str}
Variations: {n_variations if use_stochastic else 1}

Sample Info:
Pattern Length: {len(synth_sample[0])}
Synth Range: [{synth_sample[0].min():.3f}, {synth_sample[0].max():.3f}]
Real Range: [{real_sample[0].min():.3f}, {real_sample[0].max():.3f}]
            """.strip()

            axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Parameters & Info')

            plt.tight_layout()
            plt.show()

    # Create interactive widget
    interactive_widget = widgets.interactive(
        update_plot,
        sample_idx=sample_slider,
        timestep=timestep_slider,
        dtw_value=dtw_slider,
        use_stochastic=stochastic_checkbox,
        n_variations=n_variations_slider
    )

    # Layout
    controls = widgets.VBox([
        widgets.HBox([sample_slider, timestep_slider]),
        widgets.HBox([dtw_slider, stochastic_checkbox]),
        n_variations_slider
    ])

    full_widget = widgets.VBox([controls, output])

    # Initial plot
    update_plot(0, 0, 0.5, False, 5)

    return full_widget


def create_dtw_comparison_widget(model, test_synth: torch.Tensor,
                                test_real: torch.Tensor, test_dtw: torch.Tensor) -> Optional[widgets.Widget]:
    """
    Create widget for comparing DTW effects.

    Args:
        model: The diffusion model
        test_synth: Test synthetic patterns
        test_real: Test real patterns
        test_dtw: Test DTW values

    Returns:
        Interactive widget or None if widgets not available
    """
    if not WIDGETS_AVAILABLE:
        print("Interactive widgets not available. Cannot create DTW comparison.")
        return None

    sample_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(test_synth) - 1,
        step=1,
        description='Sample:',
        continuous_update=False
    )

    dtw_range_slider = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0,
        max=1.0,
        step=0.1,
        description='DTW Range:',
        continuous_update=False
    )

    n_points_slider = widgets.IntSlider(
        value=11,
        min=5,
        max=21,
        step=2,
        description='DTW Points:',
        continuous_update=False
    )

    output = widgets.Output()

    def update_dtw_plot(sample_idx, dtw_range, n_points):
        with output:
            clear_output(wait=True)

            model.eval()
            model.set_stochastic_mode(False)

            # Get sample
            synth_sample = test_synth[sample_idx]
            real_sample = test_real[sample_idx]
            original_dtw = test_dtw[sample_idx].item()

            if synth_sample.dim() == 1:
                synth_sample = synth_sample.unsqueeze(0)

            # Get device from model parameters
            device = next(model.parameters()).device
            x = synth_sample.unsqueeze(0).to(device)
            t = torch.zeros(1, dtype=torch.long, device=device)

            # Test DTW values
            dtw_values = np.linspace(dtw_range[0], dtw_range[1], n_points)
            outputs = []
            mse_values = []

            with torch.no_grad():
                for dtw_val in dtw_values:
                    dtw = torch.tensor([[dtw_val]], device=device, dtype=torch.float32)
                    output_tensor = model(x, t, dtw)
                    outputs.append(output_tensor[0, 0].cpu().numpy())
                    mse = torch.nn.MSELoss()(output_tensor, x).item()
                    mse_values.append(mse)

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot 1: Original vs Real
            axes[0, 0].plot(synth_sample[0].cpu(), 'k-', linewidth=2, label='Synthetic')
            axes[0, 0].plot(real_sample[0].cpu(), 'g-', linewidth=2, label='Real')
            axes[0, 0].axvline(x=len(synth_sample[0])//4, color='r', linestyle='--',
                              alpha=0.5, label=f'DTW: {original_dtw:.3f}')
            axes[0, 0].set_title(f'Sample {sample_idx}: Original Data')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Intensity')
            axes[0, 0].legend()

            # Plot 2: DTW transformations
            for i, (dtw_val, output) in enumerate(zip(dtw_values, outputs)):
                alpha = 1.0 if abs(dtw_val - original_dtw) < 0.1 else 0.6
                linewidth = 2 if abs(dtw_val - original_dtw) < 0.1 else 1
                axes[0, 1].plot(output, alpha=alpha, linewidth=linewidth,
                               label=f'DTW={dtw_val:.2f}' if i % 3 == 0 else None)

            axes[0, 1].plot(real_sample[0].cpu(), 'g-', linewidth=3, alpha=0.7, label='Real Target')
            axes[0, 1].set_title('DTW Transformations')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Intensity')
            axes[0, 1].legend()

            # Plot 3: MSE vs DTW
            axes[1, 0].plot(dtw_values, mse_values, 'bo-', markersize=6)
            axes[1, 0].axvline(x=original_dtw, color='r', linestyle='--',
                              label=f'Original DTW: {original_dtw:.3f}')
            axes[1, 0].set_title('Transformation Strength vs DTW')
            axes[1, 0].set_xlabel('DTW Value')
            axes[1, 0].set_ylabel('MSE vs Input')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Similarity to real
            similarities = []
            for output in outputs:
                corr = np.corrcoef(output, real_sample[0].cpu().numpy())[0, 1]
                similarities.append(corr)

            axes[1, 1].plot(dtw_values, similarities, 'ro-', markersize=6)
            axes[1, 1].axvline(x=original_dtw, color='k', linestyle='--',
                              label=f'Original DTW: {original_dtw:.3f}')
            axes[1, 1].set_title('Similarity to Real vs DTW')
            axes[1, 1].set_xlabel('DTW Value')
            axes[1, 1].set_ylabel('Correlation with Real')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    # Create interactive widget
    interactive_widget = widgets.interactive(
        update_dtw_plot,
        sample_idx=sample_slider,
        dtw_range=dtw_range_slider,
        n_points=n_points_slider
    )

    controls = widgets.VBox([sample_slider, dtw_range_slider, n_points_slider])
    full_widget = widgets.VBox([controls, output])

    # Initial plot
    update_dtw_plot(0, [0.0, 1.0], 11)

    return full_widget


def create_timestep_explorer(model, diffusion, test_synth: torch.Tensor,
                            test_dtw: torch.Tensor) -> Optional[widgets.Widget]:
    """
    Create widget for exploring timestep effects.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        test_synth: Test synthetic patterns
        test_dtw: Test DTW values

    Returns:
        Interactive widget or None if widgets not available
    """
    if not WIDGETS_AVAILABLE:
        print("Interactive widgets not available. Cannot create timestep explorer.")
        return None

    sample_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(test_synth) - 1,
        step=1,
        description='Sample:',
        continuous_update=False
    )

    timestep_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=diffusion.num_timesteps - 1,
        step=50,
        description='Timestep:',
        continuous_update=False
    )

    show_noise_checkbox = widgets.Checkbox(
        value=True,
        description='Show Noise Components'
    )

    output = widgets.Output()

    def update_timestep_plot(sample_idx, timestep, show_noise):
        with output:
            clear_output(wait=True)

            model.eval()
            model.set_stochastic_mode(False)

            # Get sample
            synth_sample = test_synth[sample_idx]
            dtw_value = test_dtw[sample_idx]

            if synth_sample.dim() == 1:
                synth_sample = synth_sample.unsqueeze(0)

            # Get device from model parameters
            device = next(model.parameters()).device
            x = synth_sample.unsqueeze(0).to(device)
            dtw = dtw_value.unsqueeze(0).to(device)
            t = torch.tensor([timestep], device=device, dtype=torch.long)

            with torch.no_grad():
                if timestep > 0:
                    # Forward diffusion
                    x_noisy, noise_true = diffusion.forward_diffusion(x, t)
                    noise_pred = model(x_noisy, t, dtw)

                    # Reconstruction
                    alpha_bar_t = diffusion.alpha_bars[t].view(1, 1, 1)
                    x0_pred = (x_noisy - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                else:
                    # Direct transformation
                    x_noisy = x
                    noise_true = torch.zeros_like(x)
                    noise_pred = model(x, t, dtw)
                    x0_pred = noise_pred  # At t=0, output is direct transformation

            # Create plots
            n_plots = 4 if show_noise and timestep > 0 else 2
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))

            if n_plots == 2:
                axes = [axes[0], axes[1]]
            else:
                axes = list(axes)

            # Plot 1: Original vs noisy
            axes[0].plot(synth_sample[0].cpu(), 'k-', linewidth=2, label='Original')
            if timestep > 0:
                axes[0].plot(x_noisy[0, 0].cpu(), 'orange', linewidth=1.5, label=f'Noisy (t={timestep})')
            axes[0].set_title(f'Sample {sample_idx}: Forward Process')
            axes[0].set_xlabel('Position')
            axes[0].set_ylabel('Intensity')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Reconstruction/transformation
            axes[1].plot(synth_sample[0].cpu(), 'k-', linewidth=2, label='Original')
            axes[1].plot(x0_pred[0, 0].cpu(), 'r-', linewidth=1.5,
                        label='Reconstructed' if timestep > 0 else 'Transformed')
            axes[1].set_title(f'{"Reconstruction" if timestep > 0 else "Transformation"} (t={timestep})')
            axes[1].set_xlabel('Position')
            axes[1].set_ylabel('Intensity')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            if show_noise and timestep > 0:
                # Plot 3: True noise
                axes[2].plot(noise_true[0, 0].cpu(), 'b-', linewidth=2)
                axes[2].set_title('True Noise')
                axes[2].set_xlabel('Position')
                axes[2].set_ylabel('Noise Amplitude')
                axes[2].grid(True, alpha=0.3)

                # Plot 4: Predicted noise
                axes[3].plot(noise_pred[0, 0].cpu(), 'orange', linewidth=2)
                axes[3].set_title('Predicted Noise')
                axes[3].set_xlabel('Position')
                axes[3].set_ylabel('Noise Amplitude')
                axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    # Create interactive widget
    interactive_widget = widgets.interactive(
        update_timestep_plot,
        sample_idx=sample_slider,
        timestep=timestep_slider,
        show_noise=show_noise_checkbox
    )

    controls = widgets.VBox([sample_slider, timestep_slider, show_noise_checkbox])
    full_widget = widgets.VBox([controls, output])

    # Initial plot
    update_timestep_plot(0, 0, True)

    return full_widget