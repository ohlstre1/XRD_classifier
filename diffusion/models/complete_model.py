"""
Complete XRD Diffusion Model - All Components in One File

Extracted from diffusion_model_0.1.5.py with all neural network components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List


# -------------------------------
# Supporting Modules
# -------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class StochasticResidualBlock(nn.Module):
    """
    Stochastic residual block with time conditioning, dropout, and stochastic depth.
    """
    def __init__(self, in_channels, out_channels, time_channels, groups=8,
                 dropout_p=0.1, stochastic_depth_p=0.1):
        super().__init__()
        self.stochastic_depth_p = stochastic_depth_p

        # Dynamic GroupNorm for robustness
        groups1 = min(groups, in_channels)
        while in_channels % groups1 != 0 and groups1 > 1:
            groups1 -= 1

        groups2 = min(groups, out_channels)
        while out_channels % groups2 != 0 and groups2 > 1:
            groups2 -= 1

        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(dropout_p)

        # Time projection
        self.time_mlp = nn.Linear(time_channels, out_channels)

        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(dropout_p)

        # Residual connection handling different channel sizes
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()


    def forward(self, x, time_emb, use_stochastic=True):
        residual = x

        # First block
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = self.dropout1(h) if use_stochastic else h

        # Time conditioning
        time_emb = self.time_mlp(time_emb)[:, :, None]  # [batch, channels, 1]
        h = h + time_emb

        # Second block
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        h = self.dropout2(h) if use_stochastic else h

        # Stochastic depth - randomly skip this block during training
        if use_stochastic and self.training and torch.rand(1) < self.stochastic_depth_p:
            return self.residual_conv(residual)

        # Normal residual connection
        return h + self.residual_conv(residual)


class Attention(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        # Dynamic GroupNorm for robustness
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.to_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, l = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Get query, key, value
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=1)

        # Compute attention
        # Reshape for matrix multiplication
        q = q.permute(0, 2, 1)  # [batch, length, channels]
        k = k.permute(0, 2, 1)  # [batch, length, channels]
        v = v.permute(0, 2, 1)  # [batch, length, channels]

        # Scale dot-product attention - fix scaling bug
        scale = (c // 3) ** -0.5  # Use per-head channels (c is total, divided by 3 for q,k,v)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = (attn @ v).permute(0, 2, 1)  # [batch, channels, length]
        out = self.to_out(out)

        # Residual connection
        return out + x


class DownBlock(nn.Module):
    """
    Downsampling block for the encoder path.
    """
    def __init__(self, in_channels, out_channels, time_channels, num_res_blocks=2, attention=False,
                 dropout_p=0.1, stochastic_depth_p=0.1):
        super().__init__()

        # Residual blocks with time conditioning
        self.res_blocks = nn.ModuleList()

        # Manage channel dimensions correctly through multiple blocks
        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(StochasticResidualBlock(channels, out_channels, time_channels,
                                                          dropout_p=dropout_p, stochastic_depth_p=stochastic_depth_p))

        # Attention blocks
        self.attentions = nn.ModuleList([
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        # Downsampling
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb, use_stochastic=True):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb, use_stochastic)
            x = attn(x)

        return self.downsample(x)


class MiddleBlock(nn.Module):
    """
    Middle block (bottleneck) with attention.
    """
    def __init__(self, channels, time_channels, attention=True):
        super().__init__()

        self.res_block1 = StochasticResidualBlock(channels, channels, time_channels)
        self.attention = Attention(channels) if attention else nn.Identity()
        self.res_block2 = StochasticResidualBlock(channels, channels, time_channels)

    def forward(self, x, time_emb, use_stochastic=True):
        x = self.res_block1(x, time_emb, use_stochastic)
        x = self.attention(x)
        x = self.res_block2(x, time_emb, use_stochastic)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block for the decoder path.
    """
    def __init__(self, in_channels, out_channels, time_channels, num_res_blocks=2, attention=False,
                 dropout_p=0.1, stochastic_depth_p=0.1):
        super().__init__()

        # Residual blocks with time conditioning
        self.res_blocks = nn.ModuleList()

        # Manage channel dimensions correctly
        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(StochasticResidualBlock(channels, out_channels, time_channels,
                                                          dropout_p=dropout_p, stochastic_depth_p=stochastic_depth_p))

        # Attention blocks
        self.attentions = nn.ModuleList([
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        # Upsampling - keep output_padding=0 for exact size matching
        self.upsample = nn.ConvTranspose1d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0  # Keep at 0 to prevent size mismatch
        )

    def forward(self, x, time_emb, use_stochastic=True):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb, use_stochastic)
            x = attn(x)

        # Upsample
        x = self.upsample(x)
        return x


# -------------------------------
# Main Diffusion Model
# -------------------------------
class DiffusionAugmentor(nn.Module):
    """
    A 1D diffusion model with integrated stochastic augmentation for XRD pattern transformation.
    Features stochastic depth, dropout, DTW noise injection, and multi-scale augmentation.
    """
    def __init__(self, in_channels=1, hidden_channels=32, time_embedding_dim=64,
                 num_res_blocks=2, attention_levels=[2], num_levels=3, temperature_condition=True,
                 enable_stochastic=True, dropout_p=0.1, stochastic_depth_p=0.1,
                 use_gradient_checkpointing=False):
        super().__init__()
        self.temperature_condition = temperature_condition
        self.enable_stochastic = enable_stochastic
        self.dropout_p = dropout_p
        self.stochastic_depth_p = stochastic_depth_p
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Temperature embedding (optional) - Enhanced capacity for DTW conditioning
        if temperature_condition:
            self.temp_embed = nn.Sequential(
                nn.Linear(1, hidden_channels),  # Full capacity instead of half
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels)  # Added extra layer for better DTW representation
            )

        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Encoder (downsampling)
        self.downs = nn.ModuleList()

        ch = hidden_channels
        input_channels = [hidden_channels]

        # Calculate channels at each level
        for i in range(num_levels):
            out_ch = hidden_channels * (2**(i+1))
            input_channels.append(out_ch)

        # Build encoder blocks
        for i in range(num_levels):
            in_ch = input_channels[i]
            out_ch = input_channels[i+1]
            is_attention = i in attention_levels

            self.downs.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_channels=hidden_channels,
                    num_res_blocks=num_res_blocks,
                    attention=is_attention,
                    dropout_p=dropout_p,
                    stochastic_depth_p=stochastic_depth_p
                )
            )

        # Middle block (bottleneck with attention)
        self.middle = MiddleBlock(
            channels=input_channels[-1],
            time_channels=hidden_channels,
            attention=True
        )

        # Decoder (upsampling)
        self.ups = nn.ModuleList()

        # Build decoder blocks (reversed)
        for i in reversed(range(num_levels)):
            in_ch = input_channels[i+1] * 2  # *2 for skip connections
            out_ch = input_channels[i]
            is_attention = i in attention_levels

            self.ups.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_channels=hidden_channels,
                    num_res_blocks=num_res_blocks,
                    attention=is_attention,
                    dropout_p=dropout_p,
                    stochastic_depth_p=stochastic_depth_p
                )
            )

        # Final layers with dynamic GroupNorm
        final_groups = min(8, hidden_channels)
        while hidden_channels % final_groups != 0 and final_groups > 1:
            final_groups -= 1
        self.norm_out = nn.GroupNorm(final_groups, hidden_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, temperature=None):
        """
        Forward pass through the denoiser network.
        Uses stochastic features if enabled and model is in training mode.
        """
        use_stochastic = self.enable_stochastic and self.training
        return self.forward_with_stochastic(x, t, temperature, use_stochastic)

    def forward_with_stochastic(self, x, t, temperature=None, use_stochastic=True):
        """
        Forward pass with explicit stochastic control for augmentation.
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Add temperature conditioning if enabled and provided
        if self.temperature_condition and temperature is not None:
            temp_emb = self.temp_embed(temperature)
            t_emb = t_emb + temp_emb

        # Initial convolution
        h = self.conv_in(x)

        # Store skip connections
        skips = [h]

        # Encoder (downsampling)
        for down_block in self.downs:
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(down_block, h, t_emb, use_stochastic)
            else:
                h = down_block(h, t_emb, use_stochastic)
            skips.append(h)

        # Middle block
        if self.use_gradient_checkpointing and self.training:
            h = checkpoint(self.middle, h, t_emb, use_stochastic)
        else:
            h = self.middle(h, t_emb, use_stochastic)

        # Decoder (upsampling) with skip connections
        for up_block in self.ups:
            # Use skip connection (take from end of list)
            skip = skips.pop()

            # Fix for dimension mismatch: Resize h to match skip if needed
            if h.shape[2] != skip.shape[2]:
                # Adjust h size to match skip size
                h = F.interpolate(h, size=skip.shape[2], mode='linear', align_corners=False)

            h = torch.cat([h, skip], dim=1)
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(up_block, h, t_emb, use_stochastic)
            else:
                h = up_block(h, t_emb, use_stochastic)

        # Final output
        h = self.norm_out(h)
        h = self.act_out(h)
        output = self.conv_out(h)

        # Ensure XRD physical constraints (non-negative intensities)
        # Use softplus for smoother constraint enforcement to prevent gradient issues
        output = F.softplus(output, beta=2.0)  # More aggressive than clamp, smoother gradients

        return output

    def generate_augmented_batch(self, synth_patterns: torch.Tensor, dtw_values: torch.Tensor,
                               num_variations: int = 5, noise_scales: Optional[List[float]] = None,
                               use_timestep_variation: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple stochastic variations from synthetic patterns.

        Args:
            synth_patterns: Input synthetic XRD patterns [batch, channels, length]
            dtw_values: DTW conditioning values [batch, 1]
            num_variations: Number of variations to generate per pattern
            noise_scales: List of noise scales for DTW conditioning
            use_timestep_variation: Whether to vary timesteps

        Returns:
            Tuple of (augmented_patterns, original_patterns, conditioning_values)
        """
        if noise_scales is None:
            noise_scales = [0.01, 0.03, 0.05, 0.07, 0.1]

        device = synth_patterns.device
        batch_size = synth_patterns.shape[0]

        # Expand patterns and conditioning for multiple variations
        expanded_patterns = synth_patterns.repeat(num_variations, 1, 1)
        expanded_dtw = dtw_values.repeat(num_variations, 1)

        # Add DTW noise for diversity
        noise_scale = torch.tensor(noise_scales, device=device)[torch.randint(0, len(noise_scales), (num_variations * batch_size,))]
        dtw_noise = torch.randn_like(expanded_dtw) * noise_scale.unsqueeze(1)
        noisy_dtw = torch.clamp(expanded_dtw + dtw_noise, 0.0, 1.0)

        # Variable timestep sampling for different augmentation strengths
        if use_timestep_variation:
            timesteps = torch.randint(0, 200, (num_variations * batch_size,), device=device)
        else:
            timesteps = torch.zeros(num_variations * batch_size, dtype=torch.long, device=device)

        # Generate augmented patterns using stochastic forward pass
        self.train()  # Enable stochastic behavior
        with torch.no_grad():
            augmented = self.forward_with_stochastic(
                expanded_patterns, timesteps, noisy_dtw, use_stochastic=True
            )

        return augmented, expanded_patterns, noisy_dtw

    def augment_with_dtw_noise(self, dtw_values: torch.Tensor, noise_scale: float = 0.05) -> torch.Tensor:
        """
        Add noise to DTW conditioning values for augmentation.

        Args:
            dtw_values: Original DTW values [batch, 1]
            noise_scale: Scale of noise to add

        Returns:
            Noisy DTW values clamped to valid range [0, 1]
        """
        noise = torch.randn_like(dtw_values) * noise_scale
        noisy_dtw = dtw_values + noise
        return torch.clamp(noisy_dtw, 0.0, 1.0)

    def set_stochastic_mode(self, enable: bool = True):
        """
        Enable or disable stochastic features globally.
        """
        self.enable_stochastic = enable
        if enable:
            self.train()  # Enable dropout and stochastic depth
        else:
            self.eval()  # Disable stochastic behavior


# Backward compatibility alias
ImprovedDiffusionDenoiser = DiffusionAugmentor