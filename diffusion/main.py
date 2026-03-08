"""
Main execution script for XRD Diffusion Model Training

Refactored from diffusion_model_0.1.5.py main() function.
Uses modular components for clean separation of concerns.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import os

# Import refactored modules
from datasets.xrd_dataset import XRDTransformDataset
from models.complete_model import ImprovedDiffusionDenoiser
from diffusion.process import DiffusionProcess
from training.trainer import train_model
from training.config import TrainingConfig
from visualization.plotting import plot_training_history, plot_overlay_sample


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

    # Load configuration
    config = TrainingConfig()
    print(f"Training configuration loaded")

    # Create save directory
    os.makedirs(config.save_path, exist_ok=True)

    print("Loading dataset...")
    dataset_dict = torch.load("../data/xrd_dataset_labeled_dtw_window.pt", map_location=device)

    synth_xrd = dataset_dict["synth_xrd"]
    real_xrd = dataset_dict["real_xrd"]
    global_temperature = dataset_dict["fast_dtw_distance"]
    print(f"Loaded dataset with {len(synth_xrd)} samples")

    # Optional: Limit dataset size for testing
    #sample_limit = 250
    #synth_xrd = synth_xrd[:sample_limit]
    #real_xrd = real_xrd[:sample_limit]
    #global_temperature = global_temperature[:sample_limit]
    #print(f"Limited dataset to {sample_limit} samples")

    # Create dataset
    dataset = XRDTransformDataset(synth_xrd, real_xrd, global_temperature)

    # Split dataset
    train_size = int(config.train_ratio * len(dataset))
    val_size = int(config.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Get indices for each split
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # Combine train and validation indices
    train_val_indices = np.concatenate([train_indices, val_indices])

    # Save combined train+val dataset
    train_val_dataset_dict = {
        "synth_xrd": synth_xrd[train_val_indices],
        "real_xrd": real_xrd[train_val_indices],
        "fast_dtw_distance": global_temperature[train_val_indices],
        "indices": train_val_indices,
        "original_dataset_size": len(synth_xrd),
        "split": "train_val_combined"
    }
    torch.save(train_val_dataset_dict, "../data/xrd_train_val_dataset.pt")
    print(f"Combined train+val dataset saved with {len(train_val_indices)} samples")

    # Save test dataset separately
    test_dataset_dict = {
        "synth_xrd": synth_xrd[test_indices],
        "real_xrd": real_xrd[test_indices],
        "fast_dtw_distance": global_temperature[test_indices],
        "indices": test_indices,
        "original_dataset_size": len(synth_xrd),
        "split": "test"
    }
    torch.save(test_dataset_dict, "../data/xrd_test_dataset.pt")
    print(f"Test dataset saved with {len(test_indices)} samples")

    # Save split summary
    split_summary = {
        "train_val_indices": train_val_indices.tolist() if hasattr(train_val_indices, 'tolist') else list(train_val_indices),
        "test_indices": list(test_indices),
        "train_val_size": len(train_val_indices),
        "test_size": len(test_indices),
        "total_size": len(synth_xrd),
        "random_seed": 42,
        "original_train_size": len(train_indices),
        "original_val_size": len(val_indices)
    }
    import json
    with open("../data/dataset_split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)
    print(f"Split summary saved to ../data/dataset_split_summary.json")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize diffusion process
    print("Initializing diffusion process with cosine schedule...")
    diffusion = DiffusionProcess(
        num_timesteps=config.num_timesteps,
        schedule_type='cosine',
        device=device
    )

    # Initialize model
    print("Building improved diffusion model...")
    model = ImprovedDiffusionDenoiser(
        in_channels=1,
        hidden_channels=config.hidden_channels,
        time_embedding_dim=config.time_embedding_dim,
        num_res_blocks=config.num_res_blocks,
        attention_levels=config.attention_levels,
        num_levels=config.num_levels,
        temperature_condition=True
    ).to(device)

    # Print model parameter count
    model_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model has {model_parameters:,} parameters")

    # Train model with enhanced features
    print("\nStarting model training with enhanced checkpointing and W&B...")
    history, trained_model = train_model(
        model=model,
        diffusion=diffusion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        config=config  # Pass config object for enhanced features
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_diff_loss = 0.0
    test_recon_loss = 0.0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        from tqdm import tqdm
        for synth, real, temp in tqdm(test_dataloader, desc="Testing"):
            synth = synth.to(device)
            real = real.to(device)
            temp = temp.to(device)
            batch_size = synth.shape[0]

            # Diffusion branch
            t = torch.randint(0, config.num_timesteps, (batch_size,), device=device)
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
    plot_training_history(history, config.save_path)

    # Create sample overlay plot with test data
    if len(test_dataloader) > 0:
        with torch.no_grad():
            for synth, real, temp in test_dataloader:
                synth, real, temp = synth.to(device), real.to(device), temp.to(device)
                plot_overlay_sample(model, diffusion, synth[0:1], real[0:1], temp[0:1],
                                  t_choice=100, save_path=f"{config.save_path}/final_sample.png",
                                  title_suffix=f"Epoch {config.num_epochs}")
                break  # Just plot one sample

    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()