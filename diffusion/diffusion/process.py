"""
Diffusion Process for XRD Pattern Transformation

Extracted from diffusion_model_0.1.5.py DiffusionProcess class.
Includes Scherrer equation physics and augmentation methods.
"""

import torch
import torch.nn.functional as F


class DiffusionProcess:
    def __init__(self, num_timesteps=1000, schedule_type='cosine', beta_start=1e-4, beta_end=10,
                 device='cpu', wavelength=1.54056, min_crystallite_size=5, max_crystallite_size=100):
        self.num_timesteps = num_timesteps
        self.device = device

        # Scherrer equation parameters
        self.wavelength = wavelength  # X-ray wavelength in Angstroms (Cu Kα radiation by default)
        self.K = 0.9  # Scherrer constant (shape factor)
        self.min_crystallite_size = min_crystallite_size  # in nm
        self.max_crystallite_size = max_crystallite_size  # in nm

        # Choose noise schedule based on type
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        elif schedule_type == 'cosine':
            self.betas = self.cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Create a beta schedule that follows a cosine curve.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def calculate_broadening_from_scherrer(self, t_fraction, two_theta_rad):
        """
        Calculate peak broadening (FWHM) using Scherrer's equation.

        Parameters:
        - t_fraction: Fraction of total timesteps (0 to 1)
        - two_theta_rad: 2θ angle in radians

        Returns:
        - FWHM in radians
        """
        # Map time fraction to crystallite size (smaller as t increases)
        # At t=0 (clean data), use max size (minimum broadening)
        # At t=1 (fully noisy), use min size (maximum broadening)
        crystallite_size = self.max_crystallite_size - t_fraction * (self.max_crystallite_size - self.min_crystallite_size)

        # Convert nm to meters
        crystallite_size_m = crystallite_size * 1e-9

        # Scherrer equation: β = K·λ / (L·cos(θ))
        # where β is FWHM in radians, K is shape factor, λ is wavelength,
        # L is crystallite size, and θ is Bragg angle

        # Calculate Bragg angle (θ) from 2θ
        theta = two_theta_rad / 2

        # Calculate FWHM using Scherrer equation (in radians)
        wavelength_m = self.wavelength * 1e-10  # Convert Å to meters
        # Add small epsilon to avoid division by zero
        fwhm = self.K * wavelength_m / (crystallite_size_m * torch.cos(theta) + 1e-10)
        return fwhm

    def create_scherrer_kernel(self, t_step, L):
        """
        Create a Gaussian convolution kernel based on Scherrer broadening.

        Parameters:
        - t_step: timestep index (integer)
        - L: length of the pattern

        Returns:
        - Convolution kernel
        """
        # Convert timestep to fraction (0 to 1)
        t_fraction = t_step / self.num_timesteps

        # Calculate broadening based on timestep
        # As t_fraction increases, we increase the broadening amount
        # Start with very small broadening at t_fraction=0
        # End with significant broadening at t_fraction=1

        # Base sigma calculation that's proportional to timestep
        # Use a more straightforward approach that doesn't depend on the 2theta values
        base_sigma = 0.02 + t_fraction * 0.98  # Scale from 0.02 to 1.0

        # Scale sigma based on pattern length to get proper pixel-space broadening
        sigma_pixels = base_sigma * (L / 100)  # Scale to pixel space

        # Determine appropriate kernel size based on sigma (odd number, at least 3)
        kernel_size = max(3, int(6 * sigma_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Ensure kernel size is reasonable compared to pattern length
        kernel_size = min(kernel_size, L//4)

        # Create Gaussian kernel
        kernel = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=self.device)
        kernel = torch.exp(-0.5 * (kernel / sigma_pixels) ** 2)
        kernel = kernel / kernel.sum()  # normalize

        return kernel.view(1, 1, -1)

    def apply_progressive_broadening(self, x, t):
        """
        Apply peak broadening that progressively increases with diffusion timestep.
        """
        batch, _, L = x.shape
        x_broadened = x.clone()

        for i in range(batch):
            # Create kernel based on current timestep
            kernel = self.create_scherrer_kernel(t[i].item(), L)

            # Apply convolution
            x_temp = x_broadened[i:i+1]
            pad_size = kernel.shape[2] // 2
            x_padded = F.pad(x_temp, (pad_size, pad_size), mode='reflect')
            x_broadened[i:i+1] = F.conv1d(x_padded, kernel)

        return x_broadened

    def augment(self, x0, t, p_shift=0.001, p_var=0.001, p_remove=0.001,
                max_shift=5, variation_range=(0.9, 1.1), threshold=0.01):
        """
        Augments the input x0 with additional operations that vary with diffusion timestep:
        - Peak shifting: randomly shift the spectrum by a small offset.
        - Peak variations: randomly scale intensities at peaks.
        - Removing peaks: randomly remove some peaks.
        - Peak broadening: apply broadening based on crystallite size simulation.
        """
        x_aug = x0.clone()
        batch, _, L = x0.shape

        # Progressive parameter scaling based on timestep
        # Higher t values mean more augmentation
        t_fractions = t.float() / self.num_timesteps

        for i in range(batch):
            t_frac = t_fractions[i].item()

            # Scale probabilities based on timestep
            p_shift_scaled = p_shift * (1 + 2 * t_frac)
            p_var_scaled = p_var * (1 + 2 * t_frac)
            p_remove_scaled = p_remove * (1 + 2 * t_frac)

            # Peak Shifting
            if torch.rand(1).item() < p_shift_scaled:
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                x_aug[i, 0, :] = torch.roll(x_aug[i, 0, :], shifts=shift)

            # Peak Variations
            if torch.rand(1).item() < p_var_scaled:
                peak_mask = (x_aug[i, 0, :] > threshold)
                # Wider variation range for higher t
                var_range = (variation_range[0] - 0.1 * t_frac,
                            variation_range[1] + 0.1 * t_frac)
                random_factors = torch.empty_like(x_aug[i, 0, :]).uniform_(*var_range)
                x_aug[i, 0, :][peak_mask] *= random_factors[peak_mask]

            # Removing Peaks
            if torch.rand(1).item() < p_remove_scaled:
                peak_mask = (x_aug[i, 0, :] > threshold)
                # Higher removal probability for higher t
                removal_probability = 0.2 * (1 + t_frac)
                removal_probability = min(removal_probability, 0.5)  # Cap at 0.5
                removal_mask = (torch.rand(peak_mask.shape, device=x_aug.device) < removal_probability) & peak_mask
                x_aug[i, 0, :][removal_mask] = 0.0

        # Apply broadening that increases with timestep
        x_aug = self.apply_progressive_broadening(x_aug, t)

        return x_aug

    def forward_diffusion(self, x0, t, noise=None):
        """
        Adds noise to an augmented version of the clean input x0 at time step t.
        """
        # Apply data augmentation with timestep-dependent broadening
        x0_aug = self.augment(x0, t)

        if noise is None:
            noise = torch.randn_like(x0_aug)

        t = t.to(self.betas.device)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1)
        x_t = sqrt_alpha_bar * x0_aug + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def sample(self, model, x_t, temperature=None, timesteps=None, stochastic=True, noise_scale=0.1):
        """
        Sample from the diffusion model in reverse.
        """
        model.eval()
        batch_size = x_t.shape[0]

        if timesteps is None:
            timesteps = list(range(self.num_timesteps))[::-1]

        x = x_t.clone()

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

                # Predict noise
                noise_pred = model(x, t_batch, temperature)

                # Compute x_{t-1}
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=x.device)

                # One-step denoising
                coef1 = torch.sqrt(alpha_bar_prev) / torch.sqrt(alpha_bar)
                coef2 = torch.sqrt(1 - alpha_bar_prev - noise_scale**2) / torch.sqrt(1 - alpha_bar)

                pred_x0 = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

                # Add noise if stochastic sampling
                noise = torch.zeros_like(x)
                if stochastic and i < len(timesteps) - 1:
                    noise = torch.randn_like(x) * noise_scale

                x = coef1 * pred_x0 + coef2 * noise_pred + noise

        model.train()
        return x