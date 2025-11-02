"""
Training module for XRD diffusion model with enhanced checkpointing and W&B integration.

Extracted from diffusion_model_0.1.5.py training function.
"""

import os
import time
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Import visualization functions
try:
    from visualization.plotting import plot_training_history, plot_overlay_sample
except ImportError:
    # Fallback for when running from diffusion directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from visualization.plotting import plot_training_history, plot_overlay_sample

# Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class ModelCheckpoint:
    """Enhanced checkpointing system with top-k model tracking."""

    def __init__(self, save_path: str, keep_top_k: int = 3, save_every_n_epochs: int = 10):
        self.save_path = save_path
        self.keep_top_k = keep_top_k
        self.save_every_n_epochs = save_every_n_epochs
        self.best_models = []  # List of (val_loss, epoch, filepath)
        os.makedirs(save_path, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, val_loss: float,
                       train_loss: float, history: Dict, is_best: bool = False,
                       is_regular: bool = False, wandb_run_id: str = None,
                       global_step: int = None) -> str:
        """Save model checkpoint with comprehensive state."""

        # Determine filename
        if is_best:
            filename = "best_model.pth"
        elif is_regular:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        else:
            filename = f"model_epoch_{epoch:04d}_val_{val_loss:.6f}.pth"

        filepath = os.path.join(self.save_path, filename)

        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'history': history,
            'timestamp': datetime.now().isoformat(),
            'wandb_run_id': wandb_run_id,
            'global_step': global_step,
            'model_config': {
                'model_class': model.__class__.__name__,
                'model_params': sum(p.numel() for p in model.parameters())
            }
        }

        # Save checkpoint
        torch.save(checkpoint, filepath)

        # Update best models tracking
        if not is_regular:  # Don't track regular checkpoints in top-k
            self.best_models.append((val_loss, epoch, filepath))
            self.best_models.sort(key=lambda x: x[0])  # Sort by validation loss

            # Remove excess models
            if len(self.best_models) > self.keep_top_k:
                _, _, old_filepath = self.best_models.pop()
                if os.path.exists(old_filepath) and not old_filepath.endswith("best_model.pth"):
                    os.remove(old_filepath)

        return filepath

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint for resuming."""
        checkpoints = glob.glob(os.path.join(self.save_path, "checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
        return checkpoints[-1] if checkpoints else None


def load_checkpoint(filepath: str, model, optimizer=None, scheduler=None, device='cpu') -> Dict:
    """Load checkpoint and restore training state."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def init_wandb(config, model, resume_id=None):
    """Initialize Weights & Biases logging."""
    if not WANDB_AVAILABLE or not getattr(config, 'use_wandb', False):
        return None

    wandb_config = {
        'hidden_channels': getattr(config, 'hidden_channels', 16),
        'time_embedding_dim': getattr(config, 'time_embedding_dim', 256),
        'num_res_blocks': getattr(config, 'num_res_blocks', 2),
        'attention_levels': getattr(config, 'attention_levels', [1, 2]),
        'num_levels': getattr(config, 'num_levels', 2),
        'num_timesteps': getattr(config, 'num_timesteps', 1000),
        'batch_size': getattr(config, 'batch_size', 8),
        'num_epochs': getattr(config, 'num_epochs', 200),
        'lr': getattr(config, 'lr', 1e-4),
        'weight_decay': getattr(config, 'weight_decay', 1e-5),
        'model_parameters': sum(p.numel() for p in model.parameters()),
    }

    run = wandb.init(
        project=getattr(config, 'wandb_project', 'xrd-diffusion'),
        entity=getattr(config, 'wandb_entity', None),
        name=getattr(config, 'wandb_run_name', None),
        config=wandb_config,
        resume="allow" if resume_id else None,
        id=resume_id
    )

    wandb.watch(model, log="all", log_freq=100)
    return run


def train_model(model, diffusion, train_dataloader, val_dataloader,
               num_epochs=50, lr=1e-4, weight_decay=1e-5, device='cpu',
               save_path='./models', num_timesteps=1000, config=None):
    """
    Enhanced training function with comprehensive checkpointing and W&B integration.
    """
    # Use config if provided, otherwise fall back to parameters
    if config is not None:
        num_epochs = getattr(config, 'num_epochs', num_epochs)
        lr = getattr(config, 'lr', lr)
        weight_decay = getattr(config, 'weight_decay', weight_decay)
        save_path = getattr(config, 'save_path', save_path)
        num_timesteps = getattr(config, 'num_timesteps', num_timesteps)

    # Initialize enhanced checkpointing
    checkpoint_manager = ModelCheckpoint(
        save_path=save_path,
        keep_top_k=getattr(config, 'keep_top_k_models', 3),
        save_every_n_epochs=getattr(config, 'save_every_n_epochs', 10)
    )

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
    loss_fn = nn.MSELoss()

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0  # Track total training steps across all epochs
    history = {
        'train_loss': [],
        'val_loss': [],
        'diff_loss': [],
        'recon_loss': [],
        'learning_rate': []
    }

    # Resume from checkpoint if available
    wandb_run_id = None
    if getattr(config, 'auto_resume', True):
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint_data = load_checkpoint(latest_checkpoint, model, optimizer, scheduler, device)
            start_epoch = checkpoint_data['epoch'] + 1
            best_val_loss = checkpoint_data.get('val_loss', float('inf'))
            history = checkpoint_data.get('history', history)
            wandb_run_id = checkpoint_data.get('wandb_run_id', None)
            global_step = checkpoint_data.get('global_step', 0)  # Restore global step counter
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}, global step: {global_step}")

    # Initialize W&B
    wandb_run = init_wandb(config, model, wandb_run_id)
    if wandb_run:
        wandb_run_id = wandb_run.id

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        # Determine training phase
        phase = min(epoch // (num_epochs // 3) + 1, 3)

        # Adjust phase-specific parameters
        if phase == 1:
            diffusion_weight = 0.8
            reconstruction_weight = 0.2
            max_timestep = num_timesteps // 2
        elif phase == 2:
            diffusion_weight = 0.5
            reconstruction_weight = 0.5
            max_timestep = int(num_timesteps * 0.75)
        else:
            diffusion_weight = 0.1
            reconstruction_weight = 0.9
            max_timestep = num_timesteps

        print(f"Epoch {epoch+1}/{num_epochs} (Phase {phase}): " +
              f"Diffusion weight: {diffusion_weight}, Reconstruction weight: {reconstruction_weight}")

        # Training phase
        model.train()
        train_loss = 0.0
        diff_loss_sum = 0.0
        recon_loss_sum = 0.0

        train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (synth, real, temp) in enumerate(train_pbar):
            synth = synth.to(device)
            real = real.to(device)
            temp = temp.to(device)
            batch_size = synth.shape[0]

            # Diffusion denoising branch
            t = torch.randint(0, max_timestep, (batch_size,), device=device)
            x_t, noise = diffusion.forward_diffusion(synth, t)
            noise_pred = model(x_t, t, temp)
            loss_diffusion = loss_fn(noise_pred, noise)

            # Direct transformation branch
            t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
            synth_to_real = model(synth, t_zero, temp)
            loss_reconstruction = loss_fn(synth_to_real, real)

            # Combined loss
            loss = (diffusion_weight * loss_diffusion) + (reconstruction_weight * loss_reconstruction)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            diff_loss_sum += loss_diffusion.item()
            recon_loss_sum += loss_reconstruction.item()

            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Diff': f"{loss_diffusion.item():.4f}",
                'Recon': f"{loss_reconstruction.item():.4f}"
            })

            # Increment global step counter
            global_step += 1

            # Log to W&B (every 100 steps)
            if wandb_run and global_step % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_diff_loss': loss_diffusion.item(),
                    'batch_recon_loss': loss_reconstruction.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'phase': phase
                }, step=global_step)

        avg_train_loss = train_loss / len(train_dataloader)
        avg_diff_loss = diff_loss_sum / len(train_dataloader)
        avg_recon_loss = recon_loss_sum / len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        val_pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")
        with torch.no_grad():
            for synth, real, temp in val_pbar:
                synth = synth.to(device)
                real = real.to(device)
                temp = temp.to(device)
                batch_size = synth.shape[0]

                # Validation losses
                t = torch.randint(0, max_timestep, (batch_size,), device=device)
                x_t, noise = diffusion.forward_diffusion(synth, t)
                noise_pred = model(x_t, t, temp)
                loss_diffusion = loss_fn(noise_pred, noise)

                t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
                synth_to_real = model(synth, t_zero, temp)
                loss_reconstruction = loss_fn(synth_to_real, real)

                loss = (diffusion_weight * loss_diffusion) + (reconstruction_weight * loss_reconstruction)
                val_loss += loss.item()

                val_pbar.set_postfix({'Val Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_dataloader)
        current_lr = scheduler.get_last_lr()[0]

        # Update learning rate
        scheduler.step()

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['diff_loss'].append(avg_diff_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['learning_rate'].append(current_lr)

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - " +
              f"Train Loss: {avg_train_loss:.6f} (Diff: {avg_diff_loss:.6f}, Recon: {avg_recon_loss:.6f}), " +
              f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

        # Log to W&B (epoch-level metrics)
        if wandb_run:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'diff_loss': avg_diff_loss,
                'recon_loss': avg_recon_loss,
                'learning_rate': current_lr,
                'phase': phase,
                'diffusion_weight': diffusion_weight,
                'reconstruction_weight': reconstruction_weight
            }, step=global_step)

        # Save best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss, avg_train_loss,
                history, is_best=True, wandb_run_id=wandb_run_id, global_step=global_step
            )
            print(f"✓ Saved best model with validation loss: {best_val_loss:.6f}")

        # Save regular checkpoint
        if (epoch + 1) % checkpoint_manager.save_every_n_epochs == 0:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss, avg_train_loss, history,
                is_regular=True, wandb_run_id=wandb_run_id, global_step=global_step
            )
            print(f"✓ Saved regular checkpoint: {os.path.basename(checkpoint_path)}")

        # Save top-k model
        if not is_best:  # Don't duplicate if already saved as best
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss, avg_train_loss, history,
                wandb_run_id=wandb_run_id, global_step=global_step
            )

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Final visualization
    plot_training_history(history, save_path)

    # Get final learning rate (handle case where no training occurred)
    final_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr

    # Save final training summary
    summary = {
        'total_epochs': num_epochs,
        'best_val_loss': best_val_loss,
        'total_training_time': total_time,
        'final_lr': final_lr,
        'model_parameters': sum(p.numel() for p in model.parameters()),
    }

    with open(os.path.join(save_path, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if wandb_run:
        wandb.log(summary)
        wandb.finish()

    return history, model