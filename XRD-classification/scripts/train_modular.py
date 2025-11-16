#!/usr/bin/env python3
"""
Modular XRD Pipeline Training
==============================

This script runs the XRD prototypical classification pipeline
using modular components for better maintainability and flexibility.

Usage:
    python train_modular.py [--epochs 10] [--batch_size 32] [--n_samples 500]
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
import time
from datetime import datetime
from pathlib import Path
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import XRDPrototypicalClassifier
from utils.augmentation import DualXRDAugmenter
from utils.data_loading import (
    load_subset_data,
    duplicate_patterns,
    create_subset_mapping,
    create_subset_split,
    load_synthetic_data,
    load_real_val_data,
    load_real_test_data,
    create_synthetic_real_split,
    create_combined_mapping
)
from utils.datasets import XRDDuplicatedDataset, XRDSyntheticTrainDataset, XRDRealEvalDataset
from utils.training import train_epoch, validate_epoch, TrainingTracker
from utils.evaluation import evaluate_on_real_patterns, evaluate_cross_set
from utils.prototypes import compute_prototypes, PrototypeBank

from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, install with 'pip install wandb'")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str, config_dir: str) -> str:
    """Resolve relative paths from config directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(config_dir, path)


def setup_wandb(config: dict, args: argparse.Namespace, val_samples: int = None, test_samples: int = None) -> bool:
    """
    Setup wandb logging if available and enabled.

    Args:
        config: Configuration dictionary
        args: Command line arguments
        val_samples: Number of validation samples
        test_samples: Number of test samples

    Returns:
        True if wandb is initialized, False otherwise
    """
    wandb_config = config.get('wandb', {})
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb and wandb_config.get('enabled', True)

    if use_wandb:
        if args.wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_name = f"xrd_{args.n_samples}samples_{timestamp}"

        project_name = args.wandb_project if args.wandb_project != 'xrd-classification' \
            else wandb_config.get('project', 'xrd-classification')

        # Use provided samples or default to n_samples
        val_samples = val_samples if val_samples is not None else args.n_samples
        test_samples = test_samples if test_samples is not None else args.n_samples

        wandb.init(
            project=project_name,
            name=args.wandb_name,
            entity=wandb_config.get('entity'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config={
                'n_samples': args.n_samples,
                'val_samples': val_samples,
                'test_samples': test_samples,
                'epochs': config['training']['epochs'],
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'embedding_dim': config['model']['embedding_dim'],
                'temperature': config['model']['temperature'],
                'proto_weight': config['model']['proto_weight'],
                'triplet_weight': config['model']['triplet_weight'],
                'triplet_margin': config['model']['triplet_margin'],
                'augmentations_per_sample': config['augmentation']['n_augmentations'],
                'classical_enabled': config['augmentation']['classical']['enabled'],
                'diffusion_enabled': config['augmentation']['diffusion']['enabled']
            }
        )
        print(f"✅ Wandb initialized: project={project_name}, run={args.wandb_name}")
        return True

    print("⚠️ Wandb logging disabled")
    return False


def create_data_loaders(config: dict,
                         train_data: dict,
                         val_data: dict,
                         train_ids: list,
                         val_ids: list,
                         augmenter: object) -> tuple:
    """
    Create training and validation data loaders.

    Args:
        config: Configuration dictionary
        train_data: Synthetic training data
        val_data: Real validation data
        train_ids: List of training compound IDs
        val_ids: List of validation compound IDs
        augmenter: Augmenter object (only used for training)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset: synthetic patterns with augmentation
    train_dataset = XRDSyntheticTrainDataset(
        train_data, train_ids, augmenter,
        samples_per_pattern=config['augmentation']['n_augmentations']
    )

    # Validation dataset: real patterns without augmentation (NEVER augmented)
    val_dataset = XRDRealEvalDataset(
        val_data, val_ids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )

    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_dataset)} augmented samples from {len(train_ids)} synthetic compounds")
    print(f"   Val: {len(val_dataset)} real samples from {len(val_ids)} compounds (no augmentation)")

    return train_loader, val_loader


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: dict,
                device: torch.device,
                train_ids: list,
                val_ids: list,
                compound_mapping: dict,
                use_wandb: bool) -> TrainingTracker:
    """
    Train the model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        train_ids: Training compound IDs
        val_ids: Validation compound IDs
        compound_mapping: Compound mapping
        use_wandb: Whether to log to wandb

    Returns:
        TrainingTracker with training history
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']
    )

    print(f"\n{'='*50}")
    print("TRAINING")
    print(f"{'='*50}")

    tracker = TrainingTracker()
    start_time = time.time()

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 30)

        train_loss, _, _, train_class_acc = train_epoch(
            model, train_loader, optimizer, device, epoch,
            train_ids=train_ids, compound_mapping=compound_mapping,
            compute_accuracy_every=5
        )

        val_loss, val_class_acc = validate_epoch(
            model, val_loader, device, val_ids, compound_mapping
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Batch accuracy is meaningless for prototypical learning with dynamic classes
        # Only report meaningful metrics
        print(f"Train Loss: {train_loss:.4f}", end="")
        if train_class_acc is not None:
            print(f" | Classification Acc: {train_class_acc:.3f}")
        else:
            print("")
        print(f"Val Loss: {val_loss:.4f} | Classification Acc: {val_class_acc:.3f}")
        print(f"LR: {current_lr:.6f}")

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_classification_accuracy': train_class_acc,
            'val_classification_accuracy': val_class_acc,
            'learning_rate': current_lr
        }

        if use_wandb:
            wandb.log(metrics)

        is_best = tracker.update(epoch, metrics)
        if is_best:
            print(f"✅ New best validation classification accuracy: {val_class_acc:.3f}")

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    print(f"Best validation accuracy: {tracker.best_val_accuracy:.3f} at epoch {tracker.best_epoch}")

    return tracker, training_time


def save_results(results: dict, output_dir: str = "../results") -> str:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_dir: Directory to save results

    Returns:
        Path to saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{results['configuration']['n_samples']}_sample_results_{timestamp}.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results_file


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train XRD classifier with modular components')
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--n_samples', type=int, default=-1, help='Number of compounds to use for debugging (-1 for all data)')
    parser.add_argument('--val_samples', type=int, help='Number of validation samples (deprecated, use ratios)')
    parser.add_argument('--test_samples', type=int, help='Number of test samples (deprecated, use ratios)')
    # Removed ratio arguments as we always use all available data per split
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension (overrides config)')
    parser.add_argument('--wandb_project', type=str, default='xrd-classification',
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, help='Wandb run name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    print(f"Loading config from: {config_path}")

    try:
        config = load_config(config_path)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.embedding_dim is not None:
        config['model']['embedding_dim'] = args.embedding_dim

    # n_samples controls how many unique compounds to use (for debugging/testing)
    # Each split uses ALL its available data, not subsets
    n_compounds_to_use = args.n_samples  # This limits unique compounds, not total samples

    # These variables are kept for compatibility but won't limit the actual data loaded
    # We'll load ALL available data from each dataset
    n_train = n_compounds_to_use  # Will load ALL synthetic data (13k+)
    val_samples = n_compounds_to_use  # Will load ALL validation data (~9k)
    test_samples = n_compounds_to_use  # Will load ALL test data (~3k)

    print("=" * 80)
    print(f"XRD Synthetic-to-Real Transfer Learning")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Config file: {config_path}")
    print(f"  Compound limit: {n_compounds_to_use} (for debugging, -1 for all)")
    print(f"  Data loading:")
    print(f"    Train: ALL synthetic data (~13k compounds)")
    print(f"    Val: ALL validation data (~9k compounds)")
    print(f"    Test: ALL test data (~3k compounds)")
    print(f"  Note: Same compounds appear in train (synthetic) and val/test (real)")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Embedding dim: {config['model']['embedding_dim']}")
    print(f"  Augmentations per sample: {config['augmentation']['n_augmentations']}")
    print()

    device_config = config['hardware']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")

    use_wandb = setup_wandb(config, args, val_samples, test_samples)

    # Load all three datasets
    synthetic_train_path = resolve_path(config['data']['synthetic_train_path'], config_dir)
    real_val_path = resolve_path(config['data']['real_val_path'], config_dir)
    real_test_path = resolve_path(config['data']['real_test_path'], config_dir)

    print(f"\n{'='*50}")
    print("LOADING DATASETS")
    print(f"{'='*50}")

    # Always load ALL available data from each dataset
    # n_compounds_to_use only limits the number of unique compounds for debugging

    if n_compounds_to_use > 0 and n_compounds_to_use < 13325:
        # Limited compounds for debugging/testing
        print(f"Debug mode: Limiting to {n_compounds_to_use} compounds")
        # Use first n_compounds_to_use compounds (same compounds across all splits)
        compound_indices = list(range(n_compounds_to_use))

        # Load subsets with same compound indices
        train_data = load_subset_data(synthetic_train_path, n_samples=n_compounds_to_use, indices=compound_indices)
        train_data = duplicate_patterns(train_data, config)
        val_data = load_real_val_data(real_val_path, indices=compound_indices)
        test_data = load_real_test_data(real_test_path, indices=compound_indices)

        print(f"  Loaded {n_compounds_to_use} compounds from each dataset")
    else:
        # Use ALL available data (normal mode)
        print(f"Loading ALL available data from each dataset")
        train_data = load_synthetic_data(synthetic_train_path)
        train_data = duplicate_patterns(train_data, config)
        val_data = load_real_val_data(real_val_path)
        test_data = load_real_test_data(real_test_path)

        print(f"  Train: {len(train_data['synth_xrd'])} synthetic patterns")
        print(f"  Val: {len(val_data['real_xrd'])} real patterns")
        print(f"  Test: {len(test_data['real_xrd'])} real patterns")

    # Create split information for synthetic-to-real transfer learning
    # use_common_compounds=True because we're testing if synthetic patterns
    # can identify real patterns of the SAME compounds
    if n_compounds_to_use > 0 and n_compounds_to_use < 13325:
        # Limited compounds - use the same indices for all
        split_info = create_synthetic_real_split(
            train_data, val_data, test_data,
            use_common_compounds=True,  # Same compounds, different domains
            train_indices=compound_indices,
            val_indices=compound_indices,
            test_indices=compound_indices
        )
    else:
        # Full data - compounds naturally overlap
        split_info = create_synthetic_real_split(
            train_data, val_data, test_data,
            use_common_compounds=True  # Same compounds, different domains
        )
    train_ids = split_info['train']
    val_ids = split_info['val']
    test_ids = split_info['test']

    # Create compound mapping for all datasets
    compound_mapping = create_combined_mapping(train_data, val_data, test_data, split_info)

    # Initialize augmenter for training only
    try:
        augmenter = DualXRDAugmenter(config, verbose=config.get('logging', {}).get('verbose', True))
        print(f"✅ Augmenter initialized for training with methods: {augmenter.get_available_methods()}")
    except Exception as e:
        print(f"⚠️ Augmenter failed, using no augmentation: {e}")
        augmenter = None
        config['augmentation']['n_augmentations'] = 0

    # Create data loaders (augmenter only used for training)
    train_loader, val_loader = create_data_loaders(
        config, train_data, val_data, train_ids, val_ids, augmenter
    )

    model = XRDPrototypicalClassifier(
        embedding_dim=config['model']['embedding_dim'],
        loss_type='prototypical_triplet',
        temperature=config['model']['temperature'],
        proto_weight=config['model']['proto_weight'],
        triplet_weight=config['model']['triplet_weight'],
        triplet_margin=config['model']['triplet_margin']
    ).to(device)

    print(f"✅ Model initialized: {model.get_model_info()}")

    tracker, training_time = train_model(
        model, train_loader, val_loader, config, device,
        train_ids, val_ids, compound_mapping, use_wandb
    )

    print(f"\n{'='*50}")
    print("PROTOTYPE COMPUTATION")
    print(f"{'='*50}")

    # Compute prototypes from TRAINING data (not validation)
    # Create a loader without augmentation for clean prototype computation
    from utils.datasets import XRDSyntheticTrainDataset
    proto_dataset = XRDSyntheticTrainDataset(
        train_data, train_ids, augmenter=None, samples_per_pattern=1
    )
    proto_loader = DataLoader(
        proto_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    prototypes = compute_prototypes(model, proto_loader, device)
    print(f"  Computed {len(prototypes)} prototypes from training data")

    print(f"\n{'='*50}")
    print("VALIDATION EVALUATION (REAL PATTERNS)")
    print(f"{'='*50}")

    # Use cross-set evaluation for validation against training prototypes
    val_eval_results = evaluate_cross_set(
        model, prototypes, val_loader, val_ids, train_ids, device,
        k_values=config.get('evaluation', {}).get('top_k_values', [1, 5, 10])
    )

    print(f"\n{'='*50}")
    print("TEST EVALUATION (REAL PATTERNS)")
    print(f"{'='*50}")

    # Create test data loader (real patterns, no augmentation)
    test_dataset = XRDRealEvalDataset(test_data, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )

    # Evaluate test set against TRAINING prototypes (not computing new prototypes)
    test_eval_results = evaluate_cross_set(
        model, prototypes, test_loader, test_ids, train_ids, device,
        k_values=config.get('evaluation', {}).get('top_k_values', [1, 5, 10])
    )

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    final_results = {
        'configuration': {
            'n_samples': args.n_samples,
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'embedding_dim': config['model']['embedding_dim'],
            'augmentations_per_sample': config['augmentation']['n_augmentations'],
            'classical_enabled': config['augmentation']['classical']['enabled'],
            'diffusion_enabled': config['augmentation']['diffusion']['enabled']
        },
        'data_split': {
            'train_compounds': len(train_ids),
            'val_compounds': len(val_ids),
            'test_compounds': len(test_ids),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_dataset)
        },
        'training': {
            'best_val_classification_accuracy': tracker.best_val_accuracy,
            'best_epoch': tracker.best_epoch,
            'training_time_seconds': training_time
        },
        'validation_evaluation': val_eval_results,
        'test_evaluation': test_eval_results,
        'model_info': model.get_model_info(),
        'training_history': tracker.training_history
    }

    print(f"Dataset:")
    print(f"  Train: {len(train_ids)} synthetic compounds ({len(train_loader.dataset)} augmented samples)")
    print(f"  Val: {len(val_ids)} real compounds")
    print(f"  Test: {len(test_ids)} real compounds")
    print(f"\nTraining: {config['training']['epochs']} epochs, best val acc: {tracker.best_val_accuracy:.3f}")
    print(f"\nValidation evaluation (real patterns):")
    print(f"  Top-1 accuracy: {val_eval_results['top1_accuracy']:.3f}")
    print(f"  Top-5 accuracy: {val_eval_results['top5_accuracy']:.3f}")
    print(f"\nTest evaluation (real patterns):")
    print(f"  Top-1 accuracy: {test_eval_results['top1_accuracy']:.3f}")
    print(f"  Top-5 accuracy: {test_eval_results['top5_accuracy']:.3f}")

    results_file = save_results(final_results)

    if use_wandb:
        wandb.log({
            'final_best_val_accuracy': tracker.best_val_accuracy,
            'final_val_top1_accuracy': val_eval_results['top1_accuracy'],
            'final_val_top5_accuracy': val_eval_results['top5_accuracy'],
            'final_val_eval_samples': val_eval_results['total_samples'],
            'final_test_top1_accuracy': test_eval_results['top1_accuracy'],
            'final_test_top5_accuracy': test_eval_results['top5_accuracy'],
            'final_test_eval_samples': test_eval_results['total_samples'],
            'training_time_seconds': training_time,
            'total_epochs': config['training']['epochs']
        })

        artifact = wandb.Artifact(f"xrd_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}", type="results")
        artifact.add_file(results_file)
        wandb.log_artifact(artifact)
        wandb.finish()
        print(f"✅ Results logged to wandb")

    print(f"\n✅ Results saved to: {results_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()