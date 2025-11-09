#!/usr/bin/env python3
"""
Full Dataset XRD Training Script
=================================

Train the XRD prototypical classifier on the complete dataset.
Includes proper data augmentation, validation, and comprehensive evaluation.

Usage:
    python train_full_dataset.py --epochs 100 --batch_size 256
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import yaml
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import warnings

# Add parent directories to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import XRDPrototypicalClassifier
from utils.data_loader import create_data_loaders, create_test_loader
from utils.augmentation import DualXRDAugmenter

# Import helper functions from train_500_samples
from train_500_samples import (
    normalize_patterns,
    compute_classification_accuracy,
    update_prototype_bank
)

warnings.filterwarnings('ignore')


def load_full_dataset(dataset_path: str):
    """Load the complete XRD dataset."""
    print(f"Loading full dataset from {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)

    n_samples = data['synth_xrd'].shape[0]
    print(f"✅ Loaded full dataset: {n_samples} samples")

    return data


def create_compound_mapping(data: dict):
    """Create compound mapping for the full dataset."""
    print("Creating compound mapping...")

    synth_normalized = normalize_patterns(data['synth_xrd'])
    real_normalized = normalize_patterns(data['real_xrd'])

    compound_mapping = {}
    n_compounds = len(synth_normalized)

    for i in tqdm(range(n_compounds), desc="Processing compounds"):
        compound_id = f"compound_{i:05d}"
        compound_mapping[compound_id] = {
            'index': i,
            'synth_pattern': synth_normalized[i].numpy().tolist(),
            'real_pattern': real_normalized[i].numpy().tolist(),
            'file_info': str(data['file_info'][i]) if i < len(data['file_info']) else f"compound_{i}",
            'dtw_distance': float(data['fast_dtw_distance'][i]) if i < len(data['fast_dtw_distance']) else 0.0
        }

    print(f"✅ Created mapping for {len(compound_mapping)} compounds")
    return compound_mapping


def create_train_val_split(compound_mapping: dict, train_ratio: float = 0.8, seed: int = 42):
    """Create train/validation split."""
    np.random.seed(seed)
    compound_ids = list(compound_mapping.keys())
    np.random.shuffle(compound_ids)

    n_train = int(len(compound_ids) * train_ratio)
    train_ids = compound_ids[:n_train]
    val_ids = compound_ids[n_train:]

    split_info = {
        'train': train_ids,
        'val': val_ids
    }

    print(f"✅ Split created: {len(train_ids)} train, {len(val_ids)} val compounds")
    return split_info


def train_epoch(model, train_loader, optimizer, device, epoch, scaler=None):
    """Train for one epoch with mixed precision support."""
    model.train()

    total_loss = 0
    total_proto_loss = 0
    total_triplet_loss = 0
    total_batch_accuracy = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (xrd_patterns, labels, compound_ids) in enumerate(pbar):
        xrd_patterns = xrd_patterns.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                embeddings, loss, metrics = model(xrd_patterns, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            embeddings, loss, metrics = model(xrd_patterns, labels)
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_proto_loss += metrics.get('proto_loss_component', loss).item()
        total_triplet_loss += metrics.get('triplet_loss_component', torch.tensor(0.0)).item()
        total_batch_accuracy += metrics.get('proto_accuracy', 0.0).item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'batch_acc': f'{metrics.get("proto_accuracy", 0.0).item():.3f}'
        })

        # Update model training state
        model.update_training_state()

    avg_loss = total_loss / len(train_loader)
    avg_proto_loss = total_proto_loss / len(train_loader)
    avg_triplet_loss = total_triplet_loss / len(train_loader)
    avg_batch_accuracy = total_batch_accuracy / len(train_loader)

    return avg_loss, avg_proto_loss, avg_triplet_loss, avg_batch_accuracy


def validate_epoch(model, val_loader, device, val_ids, compound_mapping):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0
    total_batch_accuracy = 0

    with torch.no_grad():
        for xrd_patterns, labels, compound_ids in tqdm(val_loader, desc='Validation'):
            xrd_patterns = xrd_patterns.to(device)
            labels = labels.to(device)

            embeddings, loss, metrics = model(xrd_patterns, labels)

            total_loss += loss.item()
            total_batch_accuracy += metrics.get('proto_accuracy', 0.0).item()

    avg_loss = total_loss / len(val_loader)
    avg_batch_accuracy = total_batch_accuracy / len(val_loader)

    # Compute real classification accuracy
    val_acc_metrics = compute_classification_accuracy(
        model, val_loader, compound_mapping, val_ids, device, k_values=[1, 5, 10]
    )

    return avg_loss, avg_batch_accuracy, val_acc_metrics


def main():
    """Main training pipeline for full dataset."""
    parser = argparse.ArgumentParser(description='Train XRD classifier on full dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_augmentations', type=int, default=10, help='Augmentations per compound')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    args = parser.parse_args()

    print("=" * 80)
    print("XRD Prototypical Classification - Full Dataset Training")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Augmentations: {args.n_augmentations} per compound")
    print(f"  Mixed precision: {args.mixed_precision}")
    print()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load full dataset
    dataset_path = '../../data/xrd_dataset_labeled_dtw_window.pt'
    data = load_full_dataset(dataset_path)

    # Create compound mapping
    compound_mapping = create_compound_mapping(data)

    # Create train/val split
    split_info = create_train_val_split(compound_mapping, train_ratio=0.8)
    train_ids = split_info['train']
    val_ids = split_info['val']

    # Save split info
    os.makedirs('../data/processed', exist_ok=True)
    with open('../data/processed/full_train_val_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✅ Split saved to ../data/processed/full_train_val_split.json")

    # Configuration for data loaders
    config = {
        'training': {
            'batch_size': args.batch_size,
            'num_workers': 8,  # More workers for full dataset
            'pin_memory': True
        },
        'augmentation': {
            'n_augmentations': args.n_augmentations,
            'classical': {
                'enabled': True,
                'samples_ratio': 0.5
            },
            'diffusion': {
                'enabled': True,
                'samples_ratio': 0.5
            }
        }
    }

    # Initialize augmenter
    try:
        augmenter = DualXRDAugmenter(config, verbose=True)
        print(f"✅ Augmenter initialized with methods: {augmenter.get_available_methods()}")
    except Exception as e:
        print(f"⚠️ Full augmenter failed, using fallback: {e}")
        augmenter = DualXRDAugmenter(config, verbose=False)

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        train_ids, val_ids, compound_mapping, config, augmenter
    )

    print(f"Data loaders created:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Initialize model
    model = XRDPrototypicalClassifier(
        embedding_dim=args.embedding_dim,
        loss_type='prototypical_triplet',
        temperature=0.1,
        proto_weight=1.0,
        triplet_weight=0.5,
        triplet_margin=0.2
    ).to(device)

    print(f"✅ Model initialized: {model.get_model_info()}")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Training loop
    print(f"\n{'='*50}")
    print("TRAINING")
    print(f"{'='*50}")

    best_val_accuracy = 0
    best_epoch = 1
    patience_counter = 0
    training_history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 30)

        # Train
        train_loss, train_proto, train_triplet, train_batch_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, scaler
        )

        # Validate
        val_loss, val_batch_acc, val_metrics = validate_epoch(
            model, val_loader, device, val_ids, compound_mapping
        )

        val_top1_acc = val_metrics['top1_accuracy']
        val_top5_acc = val_metrics['top5_accuracy']

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Batch Acc: {train_batch_acc:.3f}")
        print(f"Val Loss: {val_loss:.4f} | Batch Acc: {val_batch_acc:.3f}")
        print(f"Val Classification - Top-1: {val_top1_acc:.3f} | Top-5: {val_top5_acc:.3f}")
        print(f"LR: {current_lr:.6f}")

        # Track best model
        if val_top1_acc > best_val_accuracy:
            best_val_accuracy = val_top1_acc
            best_epoch = epoch
            patience_counter = 0
            print(f"✅ New best validation accuracy: {val_top1_acc:.3f}")

            # Save best model
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            model.save_checkpoint(
                checkpoint_path, epoch,
                optimizer.state_dict(),
                scheduler.state_dict(),
                {'val_top1': val_top1_acc, 'val_top5': val_top5_acc}
            )
            print(f"✅ Best model saved to {checkpoint_path}")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            model.save_checkpoint(
                checkpoint_path, epoch,
                optimizer.state_dict(),
                scheduler.state_dict(),
                {'val_top1': val_top1_acc, 'val_top5': val_top5_acc}
            )
            print(f"✅ Checkpoint saved to {checkpoint_path}")

        # Store training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_batch_accuracy': train_batch_acc,
            'val_batch_accuracy': val_batch_acc,
            'val_top1_accuracy': val_top1_acc,
            'val_top5_accuracy': val_top5_acc,
            'learning_rate': current_lr
        })

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_accuracy:.3f} at epoch {best_epoch}")

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print(f"{'='*50}")

    # Compute final prototypes
    print("Computing final prototypes...")
    prototypes = update_prototype_bank(model, val_loader, device)

    # Save prototypes
    prototype_path = '../data/prototypes/final_prototypes.pt'
    os.makedirs(os.path.dirname(prototype_path), exist_ok=True)
    torch.save(prototypes, prototype_path)
    print(f"✅ Prototypes saved to {prototype_path}")

    # Generate final results
    final_results = {
        'configuration': {
            'n_samples': len(compound_mapping),
            'n_train': len(train_ids),
            'n_val': len(val_ids),
            'epochs_trained': epoch,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'embedding_dim': args.embedding_dim,
            'n_augmentations': args.n_augmentations
        },
        'training': {
            'best_val_top1_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'training_time_minutes': training_time / 60,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss
        },
        'model_info': model.get_model_info(),
        'training_history': training_history
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/full_training_results_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")
    print(f"{'='*80}")

    # Print summary
    print("\nTRAINING SUMMARY")
    print(f"Dataset: {len(compound_mapping)} compounds ({len(train_ids)} train, {len(val_ids)} val)")
    print(f"Training: {epoch} epochs, best val Top-1: {best_val_accuracy:.3f}")
    print(f"Time: {training_time/60:.1f} minutes")


if __name__ == "__main__":
    main()