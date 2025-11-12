#!/usr/bin/env python3
"""
Complete XRD Pipeline Training on 500 Samples
=============================================

This script runs the complete XRD prototypical classification pipeline
on a subset of 500 samples for quick testing and validation.

Pipeline steps:
1. Load and prepare data (500 samples)
2. Create train/val split (400/100)
3. Initialize dual augmentation system
4. Create data loaders with augmentation
5. Initialize model and training components
6. Train for reduced epochs
7. Compute prototypes
8. Evaluate on real test patterns
9. Generate comprehensive results

Usage:
    python train_500_samples.py [--epochs 10] [--batch_size 32]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, install with 'pip install wandb'")

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
from typing import List
from torch.utils.data import Dataset

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from models import XRDPrototypicalClassifier
from utils.augmentation import DualXRDAugmenter

warnings.filterwarnings('ignore')


class XRDDuplicatedDataset(Dataset):
    """
    Custom dataset that handles pattern duplication correctly for prototypical learning.

    Duplicated patterns belong to the same compound but provide more training diversity.
    """

    def __init__(self, compound_ids: List[str], compound_mapping: dict, data: dict,
                 augmenter, samples_per_pattern: int = 5):
        self.compound_ids = compound_ids
        self.compound_mapping = compound_mapping
        self.data = data
        self.augmenter = augmenter
        self.samples_per_pattern = samples_per_pattern

        # Create label mapping
        unique_ids = sorted(set(compound_ids))
        self.id_to_label = {cid: idx for idx, cid in enumerate(unique_ids)}

        print(f"XRDDuplicatedDataset: {len(compound_ids)} compounds, {samples_per_pattern} augmentations each")

    def __len__(self):
        return len(self.compound_ids) * self.samples_per_pattern

    def __getitem__(self, idx):
        """Get augmented pattern from potentially duplicated base patterns."""
        # Which compound does this sample belong to?
        compound_idx = idx // self.samples_per_pattern
        compound_id = self.compound_ids[compound_idx]

        # Get pattern indices for this compound (includes duplicates)
        pattern_indices = self.compound_mapping[compound_id].get('pattern_indices', [compound_idx])

        # Randomly select from available patterns for this compound
        selected_pattern_idx = np.random.choice(pattern_indices)

        # Get the base synthetic pattern
        base_pattern = self.data['synth_xrd'][selected_pattern_idx]

        # Apply augmentation
        if self.augmenter is not None:
            augmented_pattern, _ = self.augmenter.augment_pattern_mixed(base_pattern, num_samples=1)
            pattern_tensor = augmented_pattern[0]  # Take first (and only) augmented sample
        else:
            pattern_tensor = base_pattern.unsqueeze(0)  # Add channel dimension

        # Get label
        label = self.id_to_label[compound_id]

        return pattern_tensor, label, compound_id


def load_subset_data(dataset_path: str, n_samples: int = 500) -> dict:
    """
    Load a subset of the XRD dataset.

    Args:
        dataset_path: Path to the full dataset
        n_samples: Number of samples to load

    Returns:
        Subset dataset dictionary
    """
    print(f"Loading subset of {n_samples} samples from {dataset_path}")

    # Load full dataset
    data = torch.load(dataset_path, weights_only=False)

    # Extract subset
    subset_data = {
        'synth_xrd': data['synth_xrd'][:n_samples],
        'real_xrd': data['real_xrd'][:n_samples],
        'file_info': data['file_info'][:n_samples],
        'fast_dtw_distance': data['fast_dtw_distance'][:n_samples]
    }

    print(f"✅ Loaded subset: {subset_data['synth_xrd'].shape[0]} samples")
    return subset_data


def normalize_patterns(patterns: torch.Tensor) -> torch.Tensor:
    """Normalize XRD patterns to [0, 1] range."""
    patterns_min = patterns.min(dim=1, keepdim=True)[0]
    patterns_max = patterns.max(dim=1, keepdim=True)[0]
    patterns_range = patterns_max - patterns_min
    patterns_range[patterns_range == 0] = 1.0
    return (patterns - patterns_min) / patterns_range


def duplicate_patterns(data: dict, config: dict) -> dict:
    """
    Duplicate synthetic patterns to increase base diversity before augmentation.

    IMPORTANT: Duplicates are treated as the SAME compound for prototypical learning.

    Args:
        data: Dataset dictionary containing synth_xrd and real_xrd
        config: Configuration dictionary with pattern_processing settings

    Returns:
        Modified dataset with duplicated synthetic patterns + compound_groups mapping
    """
    dup_config = config.get('pattern_processing', {}).get('pattern_duplication', {})

    if not dup_config.get('enabled', False):
        print("Pattern duplication disabled, using original patterns")
        # Add trivial compound groups (each pattern is its own group)
        data['compound_groups'] = {i: [i] for i in range(len(data['synth_xrd']))}
        return data

    duplication_factor = dup_config.get('duplication_factor', 2)
    noise_level = dup_config.get('duplication_noise', 0.005)

    print(f"Duplicating synthetic patterns: factor={duplication_factor}, noise={noise_level}")

    original_synth = data['synth_xrd']
    original_real = data['real_xrd']
    original_info = data['file_info']
    original_dtw = data['fast_dtw_distance']

    n_original = len(original_synth)

    # Create duplicates with small noise variations
    all_synth = [original_synth]  # Start with originals
    all_real = [original_real]
    all_info = [original_info]
    all_dtw = [original_dtw]

    # Track which indices belong to the same compound
    compound_groups = {}
    current_idx = 0

    # Initialize groups with original patterns
    for orig_idx in range(n_original):
        compound_groups[orig_idx] = [current_idx]
        current_idx += 1

    for dup_idx in range(duplication_factor):
        # Add small Gaussian noise to create variations
        noise = torch.randn_like(original_synth) * noise_level
        synth_variant = original_synth + noise

        # Duplicate corresponding real patterns and metadata
        all_synth.append(synth_variant)
        all_real.append(original_real.clone())
        all_info.append([f"{info}_dup{dup_idx+1}" for info in original_info])
        all_dtw.append(original_dtw.clone())

        # Add duplicate indices to their original compound groups
        for orig_idx in range(n_original):
            compound_groups[orig_idx].append(current_idx)
            current_idx += 1

    # Concatenate all patterns
    data_duplicated = {
        'synth_xrd': torch.cat(all_synth, dim=0),
        'real_xrd': torch.cat(all_real, dim=0),
        'file_info': [item for sublist in all_info for item in sublist],
        'fast_dtw_distance': torch.cat(all_dtw, dim=0),
        'compound_groups': compound_groups  # Track which patterns belong to same compound
    }

    n_final = len(data_duplicated['synth_xrd'])
    print(f"✅ Pattern duplication completed: {n_original} → {n_final} samples")
    print(f"   Multiplier: {n_final/n_original:.1f}x")
    print(f"   Compound groups: {len(compound_groups)} unique compounds")
    print(f"   Patterns per compound: {len(compound_groups[0])}")

    return data_duplicated


def create_subset_mapping(data: dict) -> dict:
    """Create compound mapping for subset data."""
    print("Creating compound mapping for subset...")

    synth_normalized = normalize_patterns(data['synth_xrd'])
    real_normalized = normalize_patterns(data['real_xrd'])

    compound_mapping = {}
    n_patterns = len(synth_normalized)

    # Get compound groups if they exist
    compound_groups = data.get('compound_groups', {})

    if compound_groups:
        # Use only original compounds (first pattern in each group)
        print(f"Using compound groups: {len(compound_groups)} unique compounds")
        for compound_idx in range(len(compound_groups)):
            compound_id = f"compound_{compound_idx:05d}"
            pattern_indices = compound_groups[compound_idx]

            # Use first pattern as representative
            main_idx = pattern_indices[0]

            compound_mapping[compound_id] = {
                'index': compound_idx,
                'synth_pattern': synth_normalized[main_idx].numpy().tolist(),
                'real_pattern': real_normalized[main_idx].numpy().tolist(),
                'file_info': str(data['file_info'][main_idx]),
                'dtw_distance': float(data['fast_dtw_distance'][main_idx]),
                'pattern_indices': pattern_indices  # Track all patterns for this compound
            }
    else:
        # Fallback: each pattern is its own compound
        for i in range(n_patterns):
            compound_id = f"compound_{i:05d}"
            compound_mapping[compound_id] = {
                'index': i,
                'synth_pattern': synth_normalized[i].numpy().tolist(),
                'real_pattern': real_normalized[i].numpy().tolist(),
                'file_info': str(data['file_info'][i]),
                'dtw_distance': float(data['fast_dtw_distance'][i])
            }

    print(f"✅ Created mapping for {len(compound_mapping)} compounds")
    return compound_mapping


def create_subset_split(compound_mapping: dict, train_ratio: float = 0.8) -> dict:
    """Create train/val split for subset data."""
    np.random.seed(42)
    compound_ids = list(compound_mapping.keys())
    np.random.shuffle(compound_ids)

    n_train = int(len(compound_ids) * train_ratio)
    train_ids = compound_ids[:n_train]
    val_ids = compound_ids[n_train:]

    split_info = {
        'train': train_ids,
        'val': val_ids
    }

    print(f"✅ Split created: {len(train_ids)} train, {len(val_ids)} val")
    return split_info


def compute_classification_accuracy(model, data_loader, compound_mapping, compound_ids, device, k_values=[1, 5]):
    """
    Compute classification accuracy using prototype-based matching.

    Args:
        model: Trained model
        data_loader: DataLoader with patterns
        compound_mapping: Dictionary with compound information
        compound_ids: List of compound IDs to evaluate
        device: Device to run computation on
        k_values: List of k values for top-k accuracy

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()

    # First pass: compute prototypes from synthetic patterns
    compound_embeddings = {}

    with torch.no_grad():
        for xrd_patterns, labels, batch_compound_ids in data_loader:
            xrd_patterns = xrd_patterns.to(device)
            embeddings = model.backbone(xrd_patterns).cpu().numpy()

            for i, compound_id in enumerate(batch_compound_ids):
                if compound_id not in compound_embeddings:
                    compound_embeddings[compound_id] = []
                compound_embeddings[compound_id].append(embeddings[i])

    # Compute prototypes (mean of embeddings for each compound)
    prototypes = {}
    for compound_id, embeddings_list in compound_embeddings.items():
        if compound_id in compound_ids:  # Only include requested compounds
            embeddings_array = np.stack(embeddings_list)
            prototype = embeddings_array.mean(axis=0)
            prototype = prototype / np.linalg.norm(prototype)  # Re-normalize
            prototypes[compound_id] = prototype

    if len(prototypes) == 0:
        return {f'top{k}_accuracy': 0.0 for k in k_values}

    # Second pass: evaluate real patterns against prototypes
    prototype_embeddings = np.stack(list(prototypes.values()))
    prototype_ids = list(prototypes.keys())

    correct_counts = {k: 0 for k in k_values}
    total_samples = 0

    with torch.no_grad():
        for compound_id in compound_ids:
            if compound_id not in compound_mapping or compound_id not in prototypes:
                continue

            # Load real pattern
            real_pattern = np.array(compound_mapping[compound_id]['real_pattern'], dtype=np.float32)
            real_tensor = torch.from_numpy(real_pattern).unsqueeze(0).unsqueeze(0).to(device)

            # Get embedding
            embedding = model.backbone(real_tensor).cpu().numpy()[0]

            # Compute similarities to all prototypes
            similarities = np.dot(prototype_embeddings, embedding)
            top_indices = np.argsort(similarities)[::-1]

            # Get true label index
            try:
                true_idx = prototype_ids.index(compound_id)
            except ValueError:
                continue

            # Check top-K accuracy
            for k in k_values:
                if true_idx in top_indices[:k]:
                    correct_counts[k] += 1

            total_samples += 1

    # Compute accuracy metrics
    accuracy_metrics = {}
    for k in k_values:
        accuracy_metrics[f'top{k}_accuracy'] = correct_counts[k] / total_samples if total_samples > 0 else 0.0

    accuracy_metrics['total_samples'] = total_samples
    accuracy_metrics['num_prototypes'] = len(prototypes)

    return accuracy_metrics


def update_prototype_bank(model, val_loader, device):
    """
    Compute current prototypes from validation data.

    Args:
        model: Current model
        val_loader: Validation data loader
        device: Device for computation

    Returns:
        Dictionary mapping compound_id -> prototype_embedding
    """
    model.eval()
    compound_embeddings = {}

    with torch.no_grad():
        for xrd_patterns, labels, compound_ids in val_loader:
            xrd_patterns = xrd_patterns.to(device)
            embeddings = model.backbone(xrd_patterns).cpu().numpy()

            for i, compound_id in enumerate(compound_ids):
                if compound_id not in compound_embeddings:
                    compound_embeddings[compound_id] = []
                compound_embeddings[compound_id].append(embeddings[i])

    # Compute prototypes
    prototypes = {}
    for compound_id, embeddings_list in compound_embeddings.items():
        embeddings_array = np.stack(embeddings_list)
        prototype = embeddings_array.mean(axis=0)
        prototype = prototype / np.linalg.norm(prototype)
        prototypes[compound_id] = prototype

    return prototypes


def train_epoch(model, train_loader, optimizer, device, epoch, train_ids=None, compound_mapping=None, compute_accuracy_every=5):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_proto_loss = 0
    total_triplet_loss = 0
    total_batch_accuracy = 0  # Batch-level proto accuracy
    classification_accuracy = None  # Real classification accuracy

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (xrd_patterns, _, compound_ids) in enumerate(pbar):
        xrd_patterns = xrd_patterns.to(device)
        # Use compound indices as labels for prototypical learning
        labels = torch.tensor([train_ids.index(cid) for cid in compound_ids], device=device)

        optimizer.zero_grad()

        # Forward pass
        embeddings, loss, metrics = model(xrd_patterns, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_proto_loss += metrics.get('proto_loss_component', loss).item()
        total_triplet_loss += metrics.get('triplet_loss_component', torch.tensor(0.0)).item()
        proto_accuracy = metrics.get('proto_accuracy', 0.0)
        total_batch_accuracy += proto_accuracy.item() if hasattr(proto_accuracy, 'item') else proto_accuracy

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'batch_acc': f'{proto_accuracy.item() if hasattr(proto_accuracy, "item") else proto_accuracy:.3f}'
        })

        # Update model training state
        model.update_training_state()

    avg_loss = total_loss / len(train_loader)
    avg_proto_loss = total_proto_loss / len(train_loader)
    avg_triplet_loss = total_triplet_loss / len(train_loader)
    avg_batch_accuracy = total_batch_accuracy / len(train_loader)

    # Compute real classification accuracy periodically
    if epoch % compute_accuracy_every == 0 and train_ids is not None and compound_mapping is not None:
        print("\n  Computing training classification accuracy...")
        model.eval()
        train_acc_metrics = compute_classification_accuracy(
            model, train_loader, compound_mapping, train_ids, device, k_values=[1, 5]
        )
        classification_accuracy = train_acc_metrics['top1_accuracy']
        model.train()
        print(f"  Training classification accuracy: {classification_accuracy:.3f}")

    return avg_loss, avg_proto_loss, avg_triplet_loss, avg_batch_accuracy, classification_accuracy


def validate_epoch(model, val_loader, device, val_ids, compound_mapping):
    """Validate for one epoch using proper classification accuracy."""
    model.eval()

    total_loss = 0
    total_batch_accuracy = 0

    with torch.no_grad():
        for xrd_patterns, _, compound_ids in tqdm(val_loader, desc='Validation'):
            xrd_patterns = xrd_patterns.to(device)
            # Use compound indices as labels for prototypical learning
            labels = torch.tensor([val_ids.index(cid) for cid in compound_ids], device=device)

            _, loss, metrics = model(xrd_patterns, labels)

            total_loss += loss.item()
            proto_accuracy = metrics.get('proto_accuracy', 0.0)
            total_batch_accuracy += proto_accuracy.item() if hasattr(proto_accuracy, 'item') else proto_accuracy

    avg_loss = total_loss / len(val_loader)
    avg_batch_accuracy = total_batch_accuracy / len(val_loader)

    # Compute real classification accuracy
    val_acc_metrics = compute_classification_accuracy(
        model, val_loader, compound_mapping, val_ids, device, k_values=[1, 5]
    )
    classification_accuracy = val_acc_metrics['top1_accuracy']

    return avg_loss, avg_batch_accuracy, classification_accuracy


def compute_prototypes(model, val_loader, device):
    """Compute prototype embeddings for validation compounds."""
    print("Computing validation prototypes...")
    model.eval()

    compound_embeddings = {}

    with torch.no_grad():
        for xrd_patterns, labels, compound_ids in tqdm(val_loader, desc='Computing embeddings'):
            xrd_patterns = xrd_patterns.to(device)
            embeddings = model.backbone(xrd_patterns)
            embeddings = embeddings.cpu().numpy()

            for i, compound_id in enumerate(compound_ids):
                if compound_id not in compound_embeddings:
                    compound_embeddings[compound_id] = []
                compound_embeddings[compound_id].append(embeddings[i])

    # Compute prototypes (mean of embeddings)
    prototypes = {}
    for compound_id, embeddings_list in compound_embeddings.items():
        embeddings_array = np.stack(embeddings_list)
        prototype = embeddings_array.mean(axis=0)
        prototype = prototype / np.linalg.norm(prototype)  # Re-normalize
        prototypes[compound_id] = prototype

    print(f"✅ Computed {len(prototypes)} prototypes")
    return prototypes


def evaluate_on_real_patterns(model, prototypes, compound_mapping, val_ids, device):
    """Evaluate model on real measured patterns."""
    print("Evaluating on real measured patterns...")

    model.eval()
    prototype_embeddings = np.stack(list(prototypes.values()))
    prototype_ids = list(prototypes.keys())

    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for compound_id in tqdm(val_ids, desc='Evaluating real patterns'):
            if compound_id not in compound_mapping:
                continue

            # Load real pattern
            real_pattern = np.array(compound_mapping[compound_id]['real_pattern'], dtype=np.float32)
            real_tensor = torch.from_numpy(real_pattern).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 4500]

            # Get embedding
            embedding = model.backbone(real_tensor).cpu().numpy()[0]  # [embedding_dim]

            # Compute similarities
            similarities = np.dot(prototype_embeddings, embedding)
            top_indices = np.argsort(similarities)[::-1]

            # Get true label index
            try:
                true_idx = prototype_ids.index(compound_id)
            except ValueError:
                continue  # Skip if compound not in prototypes

            # Check top-K accuracy
            if true_idx in top_indices[:1]:
                correct_top1 += 1
            if true_idx in top_indices[:5]:
                correct_top5 += 1

            total_samples += 1

    top1_accuracy = correct_top1 / total_samples if total_samples > 0 else 0
    top5_accuracy = correct_top5 / total_samples if total_samples > 0 else 0

    print(f"✅ Evaluation completed:")
    print(f"   Samples evaluated: {total_samples}")
    print(f"   Top-1 accuracy: {top1_accuracy:.3f} ({correct_top1}/{total_samples})")
    print(f"   Top-5 accuracy: {top5_accuracy:.3f} ({correct_top5}/{total_samples})")

    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': total_samples,
        'correct_top1': correct_top1,
        'correct_top5': correct_top5
    }


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


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train XRD classifier on 500 samples')
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples to use')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension (overrides config)')
    parser.add_argument('--wandb_project', type=str, default='xrd-classification', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, help='Wandb run name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()

    # Load configuration
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    print(f"Loading config from: {config_path}")

    try:
        config = load_config(config_path)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # Override config with command line arguments if provided
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.embedding_dim is not None:
        config['model']['embedding_dim'] = args.embedding_dim

    print("=" * 80)
    print("XRD Prototypical Classification - 500 Sample Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Config file: {config_path}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Embedding dim: {config['model']['embedding_dim']}")
    print(f"  Augmentations per sample: {config['augmentation']['n_augmentations']}")
    print()

    # Device setup
    device_config = config['hardware']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")

    # Initialize wandb
    wandb_config = config.get('wandb', {})
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb and wandb_config.get('enabled', True)
    if use_wandb:
        # Generate run name if not provided
        if args.wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_name = f"xrd_500samples_{timestamp}"

        # Use project from config if not specified in args
        project_name = args.wandb_project if args.wandb_project != 'xrd-classification' else wandb_config.get('project', 'xrd-classification')

        wandb.init(
            project=project_name,
            name=args.wandb_name,
            entity=wandb_config.get('entity'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config={
                'n_samples': args.n_samples,
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
                'diffusion_enabled': config['augmentation']['diffusion']['enabled'],
                'device': str(device)
            }
        )
        print(f"✅ Wandb initialized: project={args.wandb_project}, run={args.wandb_name}")
    else:
        print("⚠️ Wandb logging disabled")

    # 1. Load subset data
    dataset_path = resolve_path(config['data']['dataset_path'], config_dir)
    print(f"Loading data from: {dataset_path}")
    data = load_subset_data(dataset_path, n_samples=args.n_samples)

    # 1.5. Duplicate patterns for increased diversity
    data = duplicate_patterns(data, config)

    # 2. Create compound mapping
    compound_mapping = create_subset_mapping(data)

    # 3. Create train/val split
    split_info = create_subset_split(compound_mapping)
    train_ids = split_info['train']
    val_ids = split_info['val']

    # Note: Config is already loaded from YAML file above

    # 5. Initialize augmenter using config
    try:
        augmenter = DualXRDAugmenter(config, verbose=config.get('logging', {}).get('verbose', True))
        print(f"✅ Augmenter initialized with methods: {augmenter.get_available_methods()}")
        print(f"   Classical enabled: {config['augmentation']['classical']['enabled']}")
        print(f"   Diffusion enabled: {config['augmentation']['diffusion']['enabled']}")
        print(f"   Augmentations per sample: {config['augmentation']['n_augmentations']}")
    except Exception as e:
        print(f"⚠️ Augmenter failed, using no augmentation: {e}")
        augmenter = None
        config['augmentation']['n_augmentations'] = 0

    # 6. Create custom data loaders that handle pattern duplication
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = XRDDuplicatedDataset(
        train_ids, compound_mapping, data, augmenter,
        samples_per_pattern=config['augmentation']['n_augmentations']
    )

    val_dataset = XRDDuplicatedDataset(
        val_ids, compound_mapping, data, augmenter,
        samples_per_pattern=config['augmentation']['n_augmentations']
    )

    # Create data loaders
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

    print(f"✅ Custom data loaders created:")
    print(f"   Train: {len(train_dataset)} samples from {len(train_ids)} compounds")
    print(f"   Val: {len(val_dataset)} samples from {len(val_ids)} compounds")

    # 7. Initialize model using config
    model = XRDPrototypicalClassifier(
        embedding_dim=config['model']['embedding_dim'],
        loss_type='prototypical_triplet',
        temperature=config['model']['temperature'],
        proto_weight=config['model']['proto_weight'],
        triplet_weight=config['model']['triplet_weight'],
        triplet_margin=config['model']['triplet_margin']
    ).to(device)

    print(f"✅ Model initialized: {model.get_model_info()}")

    # 8. Setup optimizer and scheduler using config
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

    # 9. Training loop
    print(f"\n{'='*50}")
    print("TRAINING")
    print(f"{'='*50}")

    best_val_accuracy = 0
    best_epoch = 1
    training_history = []

    start_time = time.time()

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 30)

        # Train
        train_loss, _, _, train_batch_acc, train_class_acc = train_epoch(
            model, train_loader, optimizer, device, epoch,
            train_ids=train_ids, compound_mapping=compound_mapping, compute_accuracy_every=5
        )

        # Validate
        val_loss, val_batch_acc, val_class_acc = validate_epoch(
            model, val_loader, device, val_ids, compound_mapping
        )

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Batch Acc: {train_batch_acc:.3f}", end="")
        if train_class_acc is not None:
            print(f" | Class Acc: {train_class_acc:.3f}")
        else:
            print("")
        print(f"Val Loss: {val_loss:.4f} | Batch Acc: {val_batch_acc:.3f} | Class Acc: {val_class_acc:.3f}")
        print(f"LR: {current_lr:.6f}")

        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_batch_accuracy': train_batch_acc,
                'val_batch_accuracy': val_batch_acc,
                'val_classification_accuracy': val_class_acc,
                'learning_rate': current_lr
            }
            if train_class_acc is not None:
                log_dict['train_classification_accuracy'] = train_class_acc

            wandb.log(log_dict)

        # Track best model (use classification accuracy)
        if val_class_acc > best_val_accuracy:
            best_val_accuracy = val_class_acc
            best_epoch = epoch
            print(f"✅ New best validation classification accuracy: {val_class_acc:.3f}")

            # Store training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_batch_accuracy': train_batch_acc,
            'val_batch_accuracy': val_batch_acc,
            'train_classification_accuracy': train_class_acc,
            'val_classification_accuracy': val_class_acc,
            'learning_rate': current_lr
        })

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    print(f"Best validation accuracy: {best_val_accuracy:.3f} at epoch {best_epoch}")

    # 10. Compute prototypes
    print(f"\n{'='*50}")
    print("PROTOTYPE COMPUTATION")
    print(f"{'='*50}")

    prototypes = compute_prototypes(model, val_loader, device)

    # 11. Evaluate on real patterns
    print(f"\n{'='*50}")
    print("EVALUATION ON REAL PATTERNS")
    print(f"{'='*50}")

    eval_results = evaluate_on_real_patterns(
        model, prototypes, compound_mapping, val_ids, device
    )

    # 12. Generate final results
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
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset)
        },
        'training': {
            'best_val_classification_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'training_time_seconds': training_time,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_val_classification_accuracy': val_class_acc
        },
        'evaluation': eval_results,
        'model_info': model.get_model_info(),
        'training_history': training_history,
        'accuracy_explanation': {
            'batch_accuracy': 'Prototypical accuracy within batch (measures prototype assignment)',
            'classification_accuracy': 'Real classification accuracy (cosine similarity to prototypes)',
            'evaluation_top1_accuracy': 'Final evaluation accuracy using real vs synthetic patterns'
        }
    }

    # Print summary
    print(f"Dataset: {args.n_samples} samples ({len(train_ids)} train, {len(val_ids)} val)")
    print(f"Training: {config['training']['epochs']} epochs, best val classification acc: {best_val_accuracy:.3f}")
    print(f"Augmentation: {config['augmentation']['n_augmentations']} samples per original (classical: {config['augmentation']['classical']['enabled']}, diffusion: {config['augmentation']['diffusion']['enabled']})")
    print(f"Final validation classification accuracy: {val_class_acc:.3f}")
    print(f"Real pattern evaluation (final):")
    print(f"  Top-1 accuracy: {eval_results['top1_accuracy']:.3f}")
    print(f"  Top-5 accuracy: {eval_results['top5_accuracy']:.3f}")
    print(f"  Samples evaluated: {eval_results['total_samples']}")
    print(f"\nAccuracy Types:")
    print(f"  - Batch accuracy: Measures prototype assignment within training batches")
    print(f"  - Classification accuracy: Real classification using cosine similarity")
    print(f"  - Evaluation accuracy: Final test on real vs synthetic patterns")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/500_sample_results_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'final_best_val_accuracy': best_val_accuracy,
            'final_top1_accuracy': eval_results['top1_accuracy'],
            'final_top5_accuracy': eval_results['top5_accuracy'],
            'final_eval_samples': eval_results['total_samples'],
            'training_time_seconds': training_time,
            'total_epochs': config['training']['epochs']
        })

        # Log training history as a table
        wandb.log({"training_history": wandb.Table(
            columns=["epoch", "train_loss", "val_loss", "train_batch_acc", "val_batch_acc",
                    "train_class_acc", "val_class_acc", "learning_rate"],
            data=[[h['epoch'], h['train_loss'], h['val_loss'], h['train_batch_accuracy'],
                  h['val_batch_accuracy'], h.get('train_classification_accuracy'),
                  h['val_classification_accuracy'], h['learning_rate']] for h in training_history]
        )})

        # Save final results as wandb artifact
        artifact = wandb.Artifact(f"xrd_results_{timestamp}", type="results")
        artifact.add_file(results_file)
        wandb.log_artifact(artifact)

        wandb.finish()
        print(f"✅ Results logged to wandb")

    print(f"\n✅ Results saved to: {results_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()