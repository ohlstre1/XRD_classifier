"""
Training module for XRD diffusion model.

Extracted from diffusion_model_0.1.5.py training function.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import visualization functions
try:
    from visualization.plotting import plot_training_history, plot_overlay_sample
except ImportError:
    # Fallback for when running from diffusion directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from visualization.plotting import plot_training_history, plot_overlay_sample


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

            # 2. Direct transformation branch - synth→real with DTW conditioning
            # Use t=0 for clean transformation (no denoising, just conversion)
            t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
            synth_to_real = model(synth, t_zero, temp)  # Transform synth to real using DTW
            loss_reconstruction = loss_fn(synth_to_real, real)

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

                # 2. Direct transformation validation - synth→real with DTW conditioning
                t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
                synth_to_real = model(synth, t_zero, temp)  # Transform synth to real using DTW
                loss_reconstruction = loss_fn(synth_to_real, real)

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

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Final visualization
    plot_training_history(history, save_path)

    return history, model