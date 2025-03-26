import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
import math
from matplotlib.gridspec import GridSpec

# -------------------------------
# Supporting Modules (copy from original script)
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
        
        # Upsampling
        self.upsample = nn.ConvTranspose1d(
            out_channels, 
            out_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1,
            output_padding=0
        )
        
    def forward(self, x, time_emb):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb)
            x = attn(x)
        
        # Upsample
        x = self.upsample(x)
        return x


# -------------------------------
# Diffusion Model
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


# -------------------------------
# Diffusion Process
# -------------------------------
class DiffusionProcess:
    def __init__(self, num_timesteps=1000, schedule_type='cosine', beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
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
        
    def estimate_noise_level(self, xrd_pattern, base_level=0.01):
        """
        Estimates a noise level based on XRD pattern characteristics.
        Returns a value between 0.1 and 0.5 (normalized timestep)
        """
        # Calculate simple statistics about the pattern to estimate noise
        # Higher peaks to background ratio = cleaner pattern = lower noise level
        
        # Get the background level (approximated as lower percentile)
        background = torch.quantile(xrd_pattern, base_level)
        
        # Get max peak height above background
        max_peak = torch.max(xrd_pattern) - background
        
        if max_peak <= 0:
            return 0.5  # Very noisy pattern
        
        # Calculate signal-to-noise ratio proxy
        peak_to_background = max_peak / (background + 1e-6)
        
        # Map this to a noise level between 0.1 and 0.5
        # Higher peak_to_background = lower noise level
        noise_level = 0.5 - min(0.4, 0.4 * (peak_to_background / 10))
        
        return max(0.001, noise_level)

    def denoise_xrd(self, model, xrd_pattern, temperature=None, noise_level=None, stochastic=False):
        """
        Denoise an XRD pattern using the diffusion model.
        
        Args:
            model: The trained diffusion model
            xrd_pattern: The XRD pattern to denoise [batch, 1, length]
            temperature: Optional condition value (e.g., cleaning difficulty) [batch, 1]
            noise_level: Optional manual noise level (0.0-1.0), if None will be estimated
            stochastic: Whether to use stochastic sampling
            
        Returns:
            Denoised XRD pattern
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Make sure the input is on the right device and has the right shape
        if not torch.is_tensor(xrd_pattern):
            xrd_pattern = torch.tensor(xrd_pattern, dtype=torch.float32)
        
        if len(xrd_pattern.shape) == 1:
            # Add batch and channel dimensions if not present
            xrd_pattern = xrd_pattern.unsqueeze(0).unsqueeze(0)
        elif len(xrd_pattern.shape) == 2:
            # Add channel dimension if not present
            xrd_pattern = xrd_pattern.unsqueeze(1)
            
        xrd_pattern = xrd_pattern.to(device)
        
        # Default temperature if not provided
        if temperature is None:
            # Use a default moderate temperature
            temperature = torch.tensor([[0.3]], dtype=torch.float32, device=device)
        else:
            if not torch.is_tensor(temperature):
                temperature = torch.tensor([[temperature]], dtype=torch.float32, device=device)
            temperature = temperature.to(device)
        
        # Estimate or use provided noise level
        if noise_level is None:
            noise_level = self.estimate_noise_level(xrd_pattern)
        
        # Convert noise level to timestep
        t = int(noise_level * self.num_timesteps)
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Get noise prediction
            noise_pred = model(xrd_pattern, t_tensor, temperature)
            
            # Calculate the denoised signal from the noise prediction
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
            denoised = (xrd_pattern - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
        return denoised


# Main function to denoise XRD patterns
def denoise_xrd_dataset(model_path, dataset_path, output_dir, num_examples=10, device='cpu'):
    """
    Load the trained model and denoise XRD patterns from the dataset.
    
    Args:
        model_path: Path to the trained model checkpoint
        dataset_path: Path to the XRD dataset
        output_dir: Directory to save results
        num_examples: Number of examples to denoise and visualize
        device: Device to run the model on ('cpu' or 'cuda')
    """
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    
    # Initialize model
    # These parameters should match those used during training
    num_timesteps = 500
    hidden_channels = 16
    time_embedding_dim = 128
    num_res_blocks = 1
    attention_levels = [1, 2]
    num_levels = 1
    
    # Create model
    model = ImprovedDiffusionDenoiser(
        in_channels=1,
        hidden_channels=hidden_channels,
        time_embedding_dim=time_embedding_dim,
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_levels=num_levels,
        temperature_condition=True
    ).to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully. Trained for {checkpoint['epoch'] + 1} epochs.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If we can't load with the expected structure, try direct loading
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("Model loaded with alternative loading method.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(num_timesteps=num_timesteps, schedule_type='cosine', device=device)
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        data = torch.load(dataset_path, map_location=device)
        xrd_patterns = data["patterns"]
        print(f"Dataset loaded successfully. Found {len(xrd_patterns)} patterns.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Select random patterns for denoising
    if num_examples > len(xrd_patterns):
        num_examples = len(xrd_patterns)
        print(f"Reducing examples to {num_examples} to match dataset size.")
    
    # Select random indices
    indices = np.random.choice(len(xrd_patterns), num_examples, replace=False)
    
    # Process each selected pattern
    print(f"Denoising {num_examples} XRD patterns...")
    
    # Set up the figure for visualization
    rows = int(np.ceil(num_examples / 2))
    fig = plt.figure(figsize=(15, 4 * rows))
    gs = GridSpec(rows, 2, figure=fig)
    
    for i, idx in enumerate(indices):
        # Get the pattern
        pattern = xrd_patterns[idx]
        
        # Convert to tensor if needed
        if not torch.is_tensor(pattern):
            pattern = torch.tensor(pattern, dtype=torch.float32)
        
        # Add batch and channel dimensions if needed
        if len(pattern.shape) == 1:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        
        # Move to device
        pattern = pattern.to(device)
        
        # Estimate noise level
        estimated_noise = diffusion.estimate_noise_level(pattern)
        # Use temperature proportional to noise level for conditioning
        temperature = torch.tensor([[estimated_noise * 2]], dtype=torch.float32, device=device)
        
        print(f"Processing pattern {i+1}/{num_examples} (index {idx}), estimated noise level: {estimated_noise:.4f}")
        
        # Denoise the pattern
        denoised = diffusion.denoise_xrd(model, pattern, temperature)
        
        # Reshape for plotting
        original = pattern.squeeze().cpu().numpy()
        denoised = denoised.squeeze().cpu().numpy()
        
        # Create subplot
        ax = fig.add_subplot(gs[i // 2, i % 2])
        
        # Plot
        ax.plot(original, label='Original', alpha=0.7)
        ax.plot(denoised, label='Denoised', linewidth=1.0)
        ax.set_title(f"Pattern {idx} (Noise: {estimated_noise:.3f})")
        ax.legend()
        ax.set_xlabel('Position (2θ)')
        ax.set_ylabel('Intensity')
        
        # Save individual pattern
        plt.figure(figsize=(10, 6))
        plt.plot(original, label='Original', alpha=0.7, linewidth=1.5)
        plt.plot(denoised, label='Denoised', linewidth=1.0)
        plt.title(f"XRD Pattern {idx} - Denoising Result")
        plt.xlabel('Position (2θ)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pattern_{idx}_denoised.png", dpi=300)
        plt.close()
        
        # Save the denoised pattern data
        torch.save({
            'original': pattern.cpu(),
            'denoised': torch.tensor(denoised).unsqueeze(0).unsqueeze(0),
            'noise_level': estimated_noise,
        }, f"{output_dir}/pattern_{idx}_denoised.pt")
    
    # Save the overview plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/denoising_overview.png", dpi=300)
    plt.close()
    
    print(f"Denoising complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Configuration
    model_path = "./models/xrd_diffusion/improved_diffusion_model_best.pth"  # Path to your trained model
    dataset_path = "xrd_RRUFF_dataset_normalized.pt"  # Path to the dataset
    output_dir = "./denoised_results"  # Directory to save results
    num_examples = 12  # Number of examples to process
    
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run the denoising
    denoise_xrd_dataset(model_path, dataset_path, output_dir, num_examples, device)
