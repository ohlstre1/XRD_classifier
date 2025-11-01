"""
Complete XRD Diffusion Model - All Components in One File

Extracted from diffusion_model_0.1.5.py with all neural network components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    """
    def __init__(self, in_channels, out_channels, time_channels, groups=8):
        super().__init__()

        # Use correct channel count for GroupNorm
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time projection
        self.time_mlp = nn.Linear(time_channels, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection handling different channel sizes
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        residual = x

        # First block
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Time conditioning
        time_emb = self.time_mlp(time_emb)[:, :, None]  # [batch, channels, 1]
        h = h + time_emb

        # Second block
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual_conv(residual)


class Attention(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
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

        # Scale dot-product attention
        scale = (c) ** -0.5
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
    def __init__(self, in_channels, out_channels, time_channels, num_res_blocks=2, attention=False):
        super().__init__()

        # Residual blocks with time conditioning
        self.res_blocks = nn.ModuleList()

        # Manage channel dimensions correctly through multiple blocks
        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(channels, out_channels, time_channels))

        # Attention blocks
        self.attentions = nn.ModuleList([
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        # Downsampling
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb)
            x = attn(x)

        return self.downsample(x)


class MiddleBlock(nn.Module):
    """
    Middle block (bottleneck) with attention.
    """
    def __init__(self, channels, time_channels, attention=True):
        super().__init__()

        self.res_block1 = ResidualBlock(channels, channels, time_channels)
        self.attention = Attention(channels) if attention else nn.Identity()
        self.res_block2 = ResidualBlock(channels, channels, time_channels)

    def forward(self, x, time_emb):
        x = self.res_block1(x, time_emb)
        x = self.attention(x)
        x = self.res_block2(x, time_emb)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block for the decoder path.
    """
    def __init__(self, in_channels, out_channels, time_channels, num_res_blocks=2, attention=False):
        super().__init__()

        # Residual blocks with time conditioning
        self.res_blocks = nn.ModuleList()

        # Manage channel dimensions correctly
        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(channels, out_channels, time_channels))

        # Attention blocks
        self.attentions = nn.ModuleList([
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        # Upsampling - use output_padding=1 to ensure proper size matching with skip connections
        self.upsample = nn.ConvTranspose1d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0  # Use 0 as default, but may need to be 1 in some cases
        )

    def forward(self, x, time_emb):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb)
            x = attn(x)

        # Upsample
        x = self.upsample(x)
        return x


# -------------------------------
# Main Diffusion Model
# -------------------------------
class ImprovedDiffusionDenoiser(nn.Module):
    """
    An improved 1D denoiser model with U-Net architecture for XRD pattern denoising.
    """
    def __init__(self, in_channels=1, hidden_channels=32, time_embedding_dim=64,
                 num_res_blocks=2, attention_levels=[2], num_levels=3, temperature_condition=True):
        super().__init__()
        self.temperature_condition = temperature_condition

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Temperature embedding (optional)
        if temperature_condition:
            self.temp_embed = nn.Sequential(
                nn.Linear(1, hidden_channels//2),
                nn.SiLU(),
                nn.Linear(hidden_channels//2, hidden_channels)
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
                    attention=is_attention
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
                    attention=is_attention
                )
            )

        # Final layers
        self.norm_out = nn.GroupNorm(8, hidden_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, temperature=None):
        """
        Forward pass through the denoiser network.
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
            h = down_block(h, t_emb)
            skips.append(h)

        # Middle block
        h = self.middle(h, t_emb)

        # Decoder (upsampling) with skip connections
        for up_block in self.ups:
            # Use skip connection (take from end of list)
            skip = skips.pop()

            # Fix for dimension mismatch: Resize h to match skip if needed
            if h.shape[2] != skip.shape[2]:
                # Adjust h size to match skip size
                h = F.interpolate(h, size=skip.shape[2], mode='linear', align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)

        # Final output
        h = self.norm_out(h)
        h = self.act_out(h)
        output = self.conv_out(h)

        return output