import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import time
import math
import numpy as np
from tqdm import tqdm
import os
import json

# -------------------------------
# Dataset for XRD Data Transformation
# -------------------------------
class XRDTransformDataset(Dataset):
    """
    Dataset that returns (synth_xrd, real_xrd, temp) triplets.
    Both synth_xrd and real_xrd are expected to be [N, L] arrays/tensors.
    They are reshaped to (N, 1, L). The temp is a scalar tensor for each pair.
    """
    def __init__(self, synth_xrd, real_xrd, temperature):
        assert len(synth_xrd) == len(real_xrd) == len(temperature), "Mismatched data lengths!"
        
        if torch.is_tensor(synth_xrd):
            self.synth_xrd = synth_xrd.clone().detach().float().unsqueeze(1)
        else:
            self.synth_xrd = torch.tensor(synth_xrd, dtype=torch.float32).unsqueeze(1)
            
        if torch.is_tensor(real_xrd):
            self.real_xrd = real_xrd.clone().detach().float().unsqueeze(1)
        else:
            self.real_xrd = torch.tensor(real_xrd, dtype=torch.float32).unsqueeze(1)
            
        if torch.is_tensor(temperature):
            self.temperature = temperature.clone().detach().float().unsqueeze(1)
        else:
            self.temperature = torch.tensor(temperature, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.synth_xrd)

    def __getitem__(self, idx):
        return self.synth_xrd[idx], self.real_xrd[idx], self.temperature[idx]


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
    def __init__(self, num_timesteps=1000, schedule_type='cosine', beta_start=1e-4, beta_end=20, device='cpu'):
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

    def augment(self, x0, p_shift=0.001, p_var=0.001, p_remove=0.001, p_broaden=0.001,
                max_shift=5, variation_range=(0.9, 1.1), threshold=0.01):
        """
        Augments the input x0 with additional operations:
          - Peak shifting: randomly shift the spectrum by a small offset.
          - Peak variations: randomly scale intensities at peaks.
          - Removing peaks: randomly remove some peaks.
          - Peak broadening: apply Gaussian convolution to broaden peaks.
        """
        x_aug = x0.clone()
        batch, _, L = x0.shape
        
        for i in range(batch):
            # Peak Shifting
            if torch.rand(1).item() < p_shift:
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                x_aug[i, 0, :] = torch.roll(x_aug[i, 0, :], shifts=shift)
            
            # Peak Variations
            if torch.rand(1).item() < p_var:
                peak_mask = (x_aug[i, 0, :] > threshold)
                random_factors = torch.empty_like(x_aug[i, 0, :]).uniform_(*variation_range)
                x_aug[i, 0, :][peak_mask] *= random_factors[peak_mask]
            
            # Removing Peaks
            if torch.rand(1).item() < p_remove:
                peak_mask = (x_aug[i, 0, :] > threshold)
                removal_probability = 0.2
                removal_mask = (torch.rand(peak_mask.shape, device=x_aug.device) < removal_probability) & peak_mask
                x_aug[i, 0, :][removal_mask] = 0.0
                
            # Peak Broadening
            if torch.rand(1).item() < p_broaden:
                sigma = torch.rand(1).item() * 0.68 + 0.02  # random between 0.02 and 0.2
                kernel_size = int(6 * sigma) + 1
                # ensure kernel_size is odd and at least 3
                kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                kernel_size = max(3, kernel_size)
                
                # Create Gaussian kernel
                kernel = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=x_aug.device)
                kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
                kernel = kernel / kernel.sum()  # normalize
                
                # Convolve
                x_temp = x_aug[i, 0, :].view(1, 1, -1)
                pad_size = kernel_size // 2
                x_padded = F.pad(x_temp, (pad_size, pad_size), mode='reflect')
                kernel = kernel.view(1, 1, -1)
                x_aug[i, 0, :] = F.conv1d(x_padded, kernel)[0, 0, :]
        
        return x_aug

    def forward_diffusion(self, x0, t, noise=None):
        """
        Adds noise to an augmented version of the clean input x0 at time step t.
        """
        # Apply data augmentation
        x0_aug = self.augment(x0)
        
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


# -------------------------------
# Training Function
# -------------------------------
def train_model(model, diffusion, train_dataloader, val_dataloader, 
               num_epochs=50, lr=1e-4, weight_decay=1e-5, device='cpu',
               save_path='./models', num_timesteps=1000):
    """
    Train the diffusion model on XRD data with progressive phases.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
    loss_fn = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'diff_loss': [],
        'recon_loss': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Determine phase (gradually increase the difficulty)
        phase = min(epoch // (num_epochs // 3) + 1, 3)
        
        # Adjust phase-specific parameters
        if phase == 1:
            # Phase 1: Focus more on standard diffusion denoising
            diffusion_weight = 0.8
            reconstruction_weight = 0.2
            max_timestep = num_timesteps // 2
        elif phase == 2:
            # Phase 2: Balance both objectives
            diffusion_weight = 0.5
            reconstruction_weight = 0.5
            max_timestep = int(num_timesteps * 0.75)
        else:
            # Phase 3: Focus more on real data reconstruction
            diffusion_weight = 0.3
            reconstruction_weight = 0.7
            max_timestep = num_timesteps
        
        print(f"Epoch {epoch+1}/{num_epochs} (Phase {phase}): " +
              f"Diffusion weight: {diffusion_weight}, Reconstruction weight: {reconstruction_weight}")
        
        # Training
        model.train()
        train_loss = 0.0
        diff_loss_sum = 0.0
        recon_loss_sum = 0.0
        
        for synth, real, temp in tqdm(train_dataloader, desc=f"Training"):
            synth = synth.to(device)
            real = real.to(device)
            temp = temp.to(device)
            batch_size = synth.shape[0]
            
            # 1. Diffusion denoising branch
            t = torch.randint(0, max_timestep, (batch_size,), device=device)
            x_t, noise = diffusion.forward_diffusion(synth, t)
            noise_pred = model(x_t, t, temp)
            loss_diffusion = loss_fn(noise_pred, noise)
            
            # 2. Real data reconstruction branch - temperature-guided approach
            # Higher temperature = more noisy = needs higher timestep
            noise_level = torch.clamp(temp * 0.5, 0.1, 0.4)  # Scale to reasonable noise level
            t_real = (noise_level * max_timestep).long().squeeze(-1)
            noise_pred_real = model(real, t_real, temp)
            alpha_bar_t = diffusion.alpha_bars[t_real].view(-1, 1, 1)
            denoised_real = (real - torch.sqrt(1 - alpha_bar_t) * noise_pred_real) / torch.sqrt(alpha_bar_t)
            loss_reconstruction = loss_fn(denoised_real, synth)
            
            # Combined weighted loss
            loss = (diffusion_weight * loss_diffusion) + (reconstruction_weight * loss_reconstruction)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            diff_loss_sum += loss_diffusion.item()
            recon_loss_sum += loss_reconstruction.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_diff_loss = diff_loss_sum / len(train_dataloader)
        avg_recon_loss = recon_loss_sum / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for synth, real, temp in tqdm(val_dataloader, desc=f"Validation"):
                synth = synth.to(device)
                real = real.to(device)
                temp = temp.to(device)
                batch_size = synth.shape[0]
                
                # 1. Diffusion denoising validation
                t = torch.randint(0, max_timestep, (batch_size,), device=device)
                x_t, noise = diffusion.forward_diffusion(synth, t)
                noise_pred = model(x_t, t, temp)
                loss_diffusion = loss_fn(noise_pred, noise)
                
                # 2. Real data reconstruction validation
                noise_level = torch.clamp(temp * 0.5, 0.1, 0.4)
                t_real = (noise_level * max_timestep).long().squeeze(-1)
                # Get noise prediction
                noise_pred_real = model(real, t_real, temp)

                # Calculate the denoised signal from the noise prediction
                alpha_bar_t = diffusion.alpha_bars[t_real].view(-1, 1, 1)
                denoised_real = (real - torch.sqrt(1 - alpha_bar_t) * noise_pred_real) / torch.sqrt(alpha_bar_t)

                # Compare the denoised real with synthetic target
                loss_reconstruction = loss_fn(denoised_real, synth)
                
                # Combined weighted loss
                loss = (diffusion_weight * loss_diffusion) + (reconstruction_weight * loss_reconstruction)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print results
        print(f"Epoch {epoch+1}/{num_epochs} - " +
              f"Train Loss: {avg_train_loss:.6f} (Diff: {avg_diff_loss:.6f}, Recon: {avg_recon_loss:.6f}), " +
              f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['diff_loss'].append(avg_diff_loss)
        history['recon_loss'].append(avg_recon_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'history': history
            }, f"{save_path}/improved_diffusion_model_best.pth")
            print(f"✓ Saved best model with validation loss: {best_val_loss:.6f}")
        
        # Visualize progress occasionally
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            visualize_progress(model, diffusion, val_dataloader, epoch, device, save_path, num_timesteps)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Final visualization
    plot_training_history(history, save_path)
    
    return history, model


# Main execution function
def main():
    """
    Main execution function to train the XRD diffusion model.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_timesteps = 1000
    hidden_channels = 16  # Reduced from 64 to avoid memory issues
    time_embedding_dim = 128
    num_res_blocks = 1
    attention_levels = [1,2]  # Reduced from [1,2] to simplify model
    num_levels = 1
    batch_size = 8  # Reduced from 32 to fit in memory
    num_epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    save_path = "./models/xrd_diffusion"
    os.makedirs(save_path, exist_ok=True)
    

    print("Loading dataset...")

    dataset_dict = torch.load("xrd_dataset_labeled_dtw_window.pt", map_location=device)

    synth_xrd = dataset_dict["synth_xrd"]
    real_xrd = dataset_dict["real_xrd"]
    global_temperature = dataset_dict["fast_dtw_distance"]
    print(f"Loaded dataset with {len(synth_xrd)} samples")
   
    sample_limit = 1000
    synth_xrd = synth_xrd[:sample_limit]
    real_xrd = real_xrd[:sample_limit]
    global_temperature = global_temperature[:sample_limit]

    print(f"Loaded dataset with {len(synth_xrd)} samples (limited to {sample_limit})")
    

    dataset = XRDTransformDataset(synth_xrd, real_xrd, global_temperature)
    
    # Split dataset
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create dataloaders with correct pin_memory setting
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
        
    # Initialize diffusion process
    print("Initializing diffusion process with cosine schedule...")
    diffusion = DiffusionProcess(num_timesteps=num_timesteps, schedule_type='cosine', device=device)
    
    # Initialize model
    print("Building improved diffusion model...")
    model = ImprovedDiffusionDenoiser(
        in_channels=1,
        hidden_channels=hidden_channels,
        time_embedding_dim=time_embedding_dim,
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_levels=num_levels,
        temperature_condition=True
    ).to(device)
    
    # Print model parameter count
    model_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model has {model_parameters:,} parameters")
    
    # Train model
    print("\nStarting model training...")
    history, trained_model = train_model(
        model=model,
        diffusion=diffusion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        save_path=save_path,
        num_timesteps=num_timesteps
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_diff_loss = 0.0
    test_recon_loss = 0.0
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for synth, real, temp in tqdm(test_dataloader, desc="Testing"):
            synth = synth.to(device)
            real = real.to(device)
            temp = temp.to(device)
            batch_size = synth.shape[0]
            
            # Diffusion branch
            t = torch.randint(0, num_timesteps, (batch_size,), device=device)
            x_t, noise = diffusion.forward_diffusion(synth, t)
            noise_pred = model(x_t, t, temp)
            loss_diffusion = loss_fn(noise_pred, noise)
            
            # Reconstruction branch
            t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
            # Get noise prediction
            noise_pred_real = model(real, t_zero, temp)

            # Calculate the denoised signal 
            alpha_bar_t = diffusion.alpha_bars[t_zero].view(-1, 1, 1)
            denoised_real = (real - torch.sqrt(1 - alpha_bar_t) * noise_pred_real) / torch.sqrt(alpha_bar_t)

            # Compare the denoised real with synthetic target
            loss_reconstruction = loss_fn(denoised_real, synth)
                        
            # Combined loss
            loss = 0.5 * loss_diffusion + 0.5 * loss_reconstruction
            
            test_loss += loss.item()
            test_diff_loss += loss_diffusion.item()
            test_recon_loss += loss_reconstruction.item()
    
    avg_test_loss = test_loss / len(test_dataloader)
    avg_test_diff_loss = test_diff_loss / len(test_dataloader)
    avg_test_recon_loss = test_recon_loss / len(test_dataloader)
    
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Diffusion Loss: {avg_test_diff_loss:.6f}")
    print(f"Test Reconstruction Loss: {avg_test_recon_loss:.6f}")
    
    # Create final visualizations
    print("\nCreating final visualizations...")
    visualize_progress(model, diffusion, test_dataloader, num_epochs, device, save_path, num_timesteps)
    
    print("\nTraining and evaluation complete!")


def plot_training_history(history, save_path):
    """
    Plot the training history metrics.
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Overall Loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components
    plt.subplot(2, 1, 2)
    plt.plot(history['diff_loss'], label='Diffusion Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_history.png", dpi=300)
    plt.close()


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
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Real vs Denoised Real vs Synthetic (Ground Truth)
    axs[0, 0].plot(synth_batch[sample_idx, 0].cpu().numpy(), label='Synthetic (Ground Truth)', color='black', linewidth=1.5)
    axs[0, 0].plot(real_batch[sample_idx, 0].cpu().numpy(), label='Real (Noisy)', color='blue', alpha=0.7, linewidth=1)
    axs[0, 0].plot(denoised_real[0, 0].cpu().numpy(), label='Denoised Real', color='red', linewidth=0.5)
    axs[0, 0].set_title(f'Real Data Denoising - Temperature: {temp_batch[sample_idx, 0].item():.4f}')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Position (2θ)')
    axs[0, 0].set_ylabel('Intensity')
    
    # Plot 2: Noise Level Analysis
    axs[0, 1].plot(synth_batch[sample_idx, 0].cpu().numpy(), label='Original Synthetic', color='black', linewidth=1.5)
    axs[0, 1].plot(noisy_mid[0, 0].cpu().numpy(), label=f'Noisy (t={t_mid.item()})', color='gray', alpha=0.5)
    axs[0, 1].plot(denoised_mid[0, 0].cpu().numpy(), label='Denoised', color='red', linewidth=0.5)
    axs[0, 1].set_title('Noise Level Analysis')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Position (2θ)')
    axs[0, 1].set_ylabel('Intensity')
    
    # Plot 3: Error Analysis for Real Data Denoising
    error = np.abs(denoised_real[0, 0].cpu().numpy() - synth_batch[sample_idx, 0].cpu().numpy())
    axs[1, 0].plot(error, color='red', label='|Denoised Real - Synthetic|')
    axs[1, 0].set_title('Absolute Error')
    axs[1, 0].set_xlabel('Position (2θ)')
    axs[1, 0].set_ylabel('Error Magnitude')
    axs[1, 0].legend()
    
    # Plot 4: Progressive Denoising at Different Noise Levels
    axs[1, 1].plot(synth_batch[sample_idx, 0].cpu().numpy(), label='Ground Truth', color='black', linewidth=1.5)
    axs[1, 1].plot(denoised_low[0, 0].cpu().numpy(), label=f'Low Noise (t={t_low.item()})', color='green', alpha=0.7)
    axs[1, 1].plot(denoised_mid[0, 0].cpu().numpy(), label=f'Mid Noise (t={t_mid.item()})', color='orange', alpha=0.7)
    axs[1, 1].plot(denoised_high[0, 0].cpu().numpy(), label=f'High Noise (t={t_high.item()})', color='red', alpha=0.7)
    axs[1, 1].set_title('Progressive Denoising')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Position (2θ)')
    axs[1, 1].set_ylabel('Intensity')
    
    plt.suptitle(f'XRD Diffusion Model - Epoch {epoch+1}', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{save_path}/progress_epoch_{epoch+1}.png", dpi=300)
    plt.close()

main()