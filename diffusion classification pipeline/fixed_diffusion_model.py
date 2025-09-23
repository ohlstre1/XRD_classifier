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
# Dataset for XRD Data
# -------------------------------
class XRDDataset(Dataset):
    """
    Simple dataset for XRD patterns with optional conditioning.
    """
    def __init__(self, xrd_patterns, conditions=None):
        if torch.is_tensor(xrd_patterns):
            self.xrd_patterns = xrd_patterns.clone().detach().float()
        else:
            self.xrd_patterns = torch.tensor(xrd_patterns, dtype=torch.float32)

        # Ensure correct shape [N, 1, L]
        if len(self.xrd_patterns.shape) == 2:
            self.xrd_patterns = self.xrd_patterns.unsqueeze(1)

        self.conditions = None
        if conditions is not None:
            if torch.is_tensor(conditions):
                self.conditions = conditions.clone().detach().float()
            else:
                self.conditions = torch.tensor(conditions, dtype=torch.float32)
            if len(self.conditions.shape) == 1:
                self.conditions = self.conditions.unsqueeze(1)

    def __len__(self):
        return len(self.xrd_patterns)

    def __getitem__(self, idx):
        if self.conditions is not None:
            return self.xrd_patterns[idx], self.conditions[idx]
        return self.xrd_patterns[idx]

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

        # Ensure groups divides channels evenly
        groups1 = min(groups, in_channels)
        while in_channels % groups1 != 0 and groups1 > 1:
            groups1 -= 1
        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_channels, out_channels)

        groups2 = min(groups, out_channels)
        while out_channels % groups2 != 0 and groups2 > 1:
            groups2 -= 1
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        residual = x

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(time_emb)[:, :, None]
        h = h + time_emb

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.residual_conv(residual)

class Attention(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.channels = channels
        # Ensure groups divides channels evenly
        norm_groups = min(groups, channels)
        while channels % norm_groups != 0 and norm_groups > 1:
            norm_groups -= 1
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.to_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, l = x.shape

        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=1)

        # Properly scaled attention
        q = q.permute(0, 2, 1) / math.sqrt(c)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        attn = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).permute(0, 2, 1)
        out = self.to_out(out)

        return out + x

class DownBlock(nn.Module):
    """
    Downsampling block for the encoder path.
    """
    def __init__(self, in_channels, out_channels, time_channels, num_res_blocks=2, attention=False):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(channels, out_channels, time_channels))

        self.attentions = nn.ModuleList([
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb)
            x = attn(x)
        return self.downsample(x)

class UpBlock(nn.Module):
    """
    Upsampling block for the decoder path.
    """
    def __init__(self, in_channels, out_channels, time_channels, num_res_blocks=2, attention=False):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            channels = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(channels, out_channels, time_channels))

        self.attentions = nn.ModuleList([
            Attention(out_channels) if attention else nn.Identity()
            for _ in range(num_res_blocks)
        ])

        self.upsample = nn.ConvTranspose1d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, time_emb):
        for res_block, attn in zip(self.res_blocks, self.attentions):
            x = res_block(x, time_emb)
            x = attn(x)
        return self.upsample(x)

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

# -------------------------------
# Clean Diffusion Model
# -------------------------------
class CleanDiffusionDenoiser(nn.Module):
    """
    A clean 1D U-Net denoiser model for XRD pattern denoising.
    """
    def __init__(self, in_channels=1, hidden_channels=64, time_embedding_dim=128,
                 num_res_blocks=2, attention_levels=[2], num_levels=3, condition_dim=0):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Condition embedding (optional)
        self.condition_embed = None
        if condition_dim > 0:
            self.condition_embed = nn.Sequential(
                nn.Linear(condition_dim, hidden_channels//2),
                nn.SiLU(),
                nn.Linear(hidden_channels//2, hidden_channels)
            )

        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Calculate channel dimensions properly
        channels = [hidden_channels]
        for i in range(num_levels):
            channels.append(hidden_channels * (2**(i+1)))

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(num_levels):
            in_ch = channels[i]
            out_ch = channels[i+1]
            is_attention = i in attention_levels

            self.downs.append(DownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                time_channels=hidden_channels,
                num_res_blocks=num_res_blocks,
                attention=is_attention
            ))

        # Middle
        self.middle = MiddleBlock(
            channels=channels[-1],
            time_channels=hidden_channels,
            attention=True
        )

        # Decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(num_levels)):
            in_ch = channels[i+1] + channels[i]  # Skip connection
            out_ch = channels[i]
            is_attention = i in attention_levels

            self.ups.append(UpBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                time_channels=hidden_channels,
                num_res_blocks=num_res_blocks,
                attention=is_attention
            ))

        # Ensure groups divides channels evenly
        out_groups = min(8, hidden_channels)
        while hidden_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1
        self.norm_out = nn.GroupNorm(out_groups, hidden_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, condition=None):
        """
        Forward pass through the denoiser network.
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Add condition if provided
        if condition is not None and self.condition_embed is not None:
            cond_emb = self.condition_embed(condition)
            t_emb = t_emb + cond_emb

        h = self.conv_in(x)

        # Store skip connections
        skips = [h]

        # Encoder
        for down_block in self.downs:
            h = down_block(h, t_emb)
            skips.append(h)

        # Middle
        h = self.middle(h, t_emb)

        # Decoder with proper skip connections
        for up_block in self.ups:
            skip = skips.pop()

            # Ensure dimensions match exactly
            if h.shape[2] != skip.shape[2]:
                h = F.interpolate(h, size=skip.shape[2], mode='linear', align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)

        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        output = self.conv_out(h)

        return output

# -------------------------------
# Clean Diffusion Process
# -------------------------------
class CleanDiffusionProcess:
    """
    Clean diffusion process with proper DDPM implementation.
    """
    def __init__(self, num_timesteps=1000, schedule_type='cosine', beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # Create noise schedule
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

    def forward_diffusion(self, x0, t, noise=None):
        """
        Clean forward diffusion - just add Gaussian noise.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1)

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def sample(self, model, x_t, condition=None, timesteps=None):
        """
        Clean DDPM sampling.
        """
        model.eval()
        batch_size = x_t.shape[0]

        if timesteps is None:
            timesteps = list(range(self.num_timesteps))[::-1]

        x = x_t.clone()

        with torch.no_grad():
            for t in timesteps:
                t_batch = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

                # Predict noise
                noise_pred = model(x, t_batch, condition)

                # DDPM update
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=x.device)

                # Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

                # Compute x_{t-1}
                if t > 0:
                    beta = self.betas[t]
                    noise = torch.randn_like(x)
                    x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * noise
                else:
                    x = pred_x0

        model.train()
        return x

# -------------------------------
# Clean Training Function
# -------------------------------
def train_clean_diffusion(model, diffusion, dataloader, val_dataloader=None,
                         num_epochs=50, lr=1e-4, weight_decay=1e-5, device='cpu', save_path='./models'):
    """
    Clean training function with single objective.
    """
    os.makedirs(save_path, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
    loss_fn = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if len(batch) == 2:
                x0, condition = batch
                x0 = x0.to(device)
                condition = condition.to(device)
            else:
                x0 = batch.to(device)
                condition = None

            batch_size = x0.shape[0]

            # Sample timesteps
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)

            # Forward diffusion
            x_t, noise = diffusion.forward_diffusion(x0, t)

            # Predict noise
            noise_pred = model(x_t, t, condition)

            # Loss
            loss = loss_fn(noise_pred, noise)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(dataloader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        avg_val_loss = 0.0
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    if len(batch) == 2:
                        x0, condition = batch
                        x0 = x0.to(device)
                        condition = condition.to(device)
                    else:
                        x0 = batch.to(device)
                        condition = None

                    batch_size = x0.shape[0]
                    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
                    x_t, noise = diffusion.forward_diffusion(x0, t)
                    noise_pred = model(x_t, t, condition)
                    loss = loss_fn(noise_pred, noise)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            history['val_loss'].append(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'history': history
                }, f"{save_path}/clean_diffusion_model_best.pth")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    return history, model

# -------------------------------
# Example Usage
# -------------------------------
def main():
    """
    Example usage of the clean diffusion model.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model parameters
    num_timesteps = 1000
    hidden_channels = 64
    time_embedding_dim = 128
    num_res_blocks = 2
    attention_levels = [2]
    num_levels = 3
    condition_dim = 1  # Set to 0 for unconditional

    # Initialize model
    model = CleanDiffusionDenoiser(
        in_channels=1,
        hidden_channels=hidden_channels,
        time_embedding_dim=time_embedding_dim,
        num_res_blocks=num_res_blocks,
        attention_levels=attention_levels,
        num_levels=num_levels,
        condition_dim=condition_dim
    ).to(device)

    # Initialize diffusion process
    diffusion = CleanDiffusionProcess(
        num_timesteps=num_timesteps,
        schedule_type='cosine',
        device=device
    )

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Example with dummy data
    # Replace this with your actual data loading
    dummy_data = torch.randn(1000, 1, 512)  # 1000 samples, 1 channel, 512 length
    dummy_conditions = torch.randn(1000, 1)  # Optional conditions

    dataset = XRDDataset(dummy_data, dummy_conditions if condition_dim > 0 else None)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train model
    history, trained_model = train_clean_diffusion(
        model=model,
        diffusion=diffusion,
        dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=10,
        lr=1e-4,
        device=device,
        save_path='./models/clean_diffusion'
    )

    print("Training completed successfully!")

if __name__ == "__main__":
    main()