#!/usr/bin/env python3
"""
Data Preparation Script for XRD Prototypical Classification
==========================================================

This script processes the existing XRD dataset to create compound mappings
and train/validation splits for prototypical learning.

Input: data/xrd_dataset_labeled_dtw_window.pt
Output:
- data/processed/compound_mapping.json
- data/processed/train_val_split.json

Key features:
- Uses existing synthetic and real XRD patterns
- Creates unique compound IDs from file information
- Generates stratified train/val split (80/20)
- Validates data integrity and consistency
- Normalizes XRD patterns to [0, 1] range
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import yaml
from tqdm import tqdm
import warnings

# Set warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_xrd_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load the XRD dataset.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Dictionary with dataset components
    """
    print(f"Loading XRD dataset from: {dataset_path}")

    try:
        data = torch.load(dataset_path, weights_only=False)
        print(f"✅ Dataset loaded successfully")

        # Print dataset info
        print(f"Dataset keys: {list(data.keys())}")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: length {len(value)}")
            else:
                print(f"  {key}: {type(value)}")

        return data

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def normalize_xrd_patterns(patterns: torch.Tensor) -> torch.Tensor:
    """
    Normalize XRD patterns to [0, 1] range.

    Args:
        patterns: XRD patterns [N, L]

    Returns:
        Normalized patterns [N, L]
    """
    print("Normalizing XRD patterns...")

    # Min-max normalization per pattern
    patterns_min = patterns.min(dim=1, keepdim=True)[0]
    patterns_max = patterns.max(dim=1, keepdim=True)[0]

    # Avoid division by zero
    patterns_range = patterns_max - patterns_min
    patterns_range[patterns_range == 0] = 1.0

    normalized = (patterns - patterns_min) / patterns_range

    print(f"✅ Patterns normalized. Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    return normalized


def create_compound_mapping(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Create compound mapping from dataset file information.

    Args:
        data: Dataset dictionary

    Returns:
        Compound mapping dictionary
    """
    print("Creating compound mapping...")

    synth_xrd = data['synth_xrd']
    real_xrd = data['real_xrd']
    file_info = data['file_info']
    dtw_distances = data['fast_dtw_distance']

    n_compounds = len(synth_xrd)
    print(f"Processing {n_compounds} compound pairs...")

    # Normalize patterns
    synth_normalized = normalize_xrd_patterns(synth_xrd)
    real_normalized = normalize_xrd_patterns(real_xrd)

    compound_mapping = {}

    for i in tqdm(range(n_compounds), desc="Creating compound mapping"):
        # Create compound ID
        compound_id = f"compound_{i:05d}"

        # Extract file information if available
        file_info_entry = file_info[i] if i < len(file_info) else f"unknown_{i}"

        # Create mapping entry
        compound_mapping[compound_id] = {
            'index': i,
            'synth_pattern': synth_normalized[i].numpy().tolist(),  # Store as list for JSON
            'real_pattern': real_normalized[i].numpy().tolist(),
            'file_info': str(file_info_entry),
            'dtw_distance': float(dtw_distances[i]) if i < len(dtw_distances) else 0.0,
            'synth_stats': {
                'mean': float(synth_normalized[i].mean()),
                'std': float(synth_normalized[i].std()),
                'min': float(synth_normalized[i].min()),
                'max': float(synth_normalized[i].max())
            },
            'real_stats': {
                'mean': float(real_normalized[i].mean()),
                'std': float(real_normalized[i].std()),
                'min': float(real_normalized[i].min()),
                'max': float(real_normalized[i].max())
            }
        }

    print(f"✅ Created mapping for {len(compound_mapping)} compounds")
    return compound_mapping


def analyze_dataset_statistics(compound_mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze dataset statistics for validation.

    Args:
        compound_mapping: Compound mapping dictionary

    Returns:
        Statistics dictionary
    """
    print("Analyzing dataset statistics...")

    dtw_distances = [entry['dtw_distance'] for entry in compound_mapping.values()]
    synth_means = [entry['synth_stats']['mean'] for entry in compound_mapping.values()]
    real_means = [entry['real_stats']['mean'] for entry in compound_mapping.values()]

    stats = {
        'total_compounds': len(compound_mapping),
        'dtw_distance_stats': {
            'mean': float(np.mean(dtw_distances)),
            'std': float(np.std(dtw_distances)),
            'min': float(np.min(dtw_distances)),
            'max': float(np.max(dtw_distances)),
            'median': float(np.median(dtw_distances))
        },
        'synth_pattern_stats': {
            'mean_intensity': float(np.mean(synth_means)),
            'std_intensity': float(np.std(synth_means))
        },
        'real_pattern_stats': {
            'mean_intensity': float(np.mean(real_means)),
            'std_intensity': float(np.std(real_means))
        }
    }

    print(f"✅ Dataset statistics computed")
    print(f"   Total compounds: {stats['total_compounds']}")
    print(f"   DTW distance range: {stats['dtw_distance_stats']['min']:.3f} - {stats['dtw_distance_stats']['max']:.3f}")
    print(f"   DTW distance mean: {stats['dtw_distance_stats']['mean']:.3f} ± {stats['dtw_distance_stats']['std']:.3f}")

    return stats


def create_train_val_split(compound_mapping: Dict[str, Dict[str, Any]],
                          train_ratio: float = 0.8,
                          random_seed: int = 42,
                          stratify_by_dtw: bool = False) -> Dict[str, List[str]]:
    """
    Create train/validation split.

    Args:
        compound_mapping: Compound mapping dictionary
        train_ratio: Ratio of compounds for training
        random_seed: Random seed for reproducibility
        stratify_by_dtw: Whether to stratify by DTW distance quartiles

    Returns:
        Split dictionary with train/val compound IDs
    """
    print(f"Creating train/val split (train_ratio={train_ratio})...")

    np.random.seed(random_seed)
    compound_ids = list(compound_mapping.keys())

    if stratify_by_dtw:
        print("Using DTW distance-based stratification...")

        # Get DTW distances
        dtw_distances = [compound_mapping[cid]['dtw_distance'] for cid in compound_ids]

        # Create quartile-based strata
        quartiles = np.percentile(dtw_distances, [25, 50, 75])

        strata = defaultdict(list)
        for cid in compound_ids:
            dtw_dist = compound_mapping[cid]['dtw_distance']
            if dtw_dist <= quartiles[0]:
                stratum = 'q1'
            elif dtw_dist <= quartiles[1]:
                stratum = 'q2'
            elif dtw_dist <= quartiles[2]:
                stratum = 'q3'
            else:
                stratum = 'q4'
            strata[stratum].append(cid)

        # Split each stratum
        train_ids = []
        val_ids = []

        for stratum, ids in strata.items():
            np.random.shuffle(ids)
            n_train = int(len(ids) * train_ratio)
            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train:])
            print(f"   {stratum}: {n_train} train, {len(ids) - n_train} val")

    else:
        print("Using random split...")
        np.random.shuffle(compound_ids)
        n_train = int(len(compound_ids) * train_ratio)
        train_ids = compound_ids[:n_train]
        val_ids = compound_ids[n_train:]

    split_info = {
        'train': train_ids,
        'val': val_ids,
        'split_config': {
            'train_ratio': train_ratio,
            'random_seed': random_seed,
            'stratify_by_dtw': stratify_by_dtw,
            'total_compounds': len(compound_ids)
        }
    }

    print(f"✅ Split created: {len(train_ids)} train, {len(val_ids)} val")
    return split_info


def validate_split(compound_mapping: Dict[str, Dict[str, Any]],
                  split_info: Dict[str, List[str]]) -> bool:
    """
    Validate the train/val split.

    Args:
        compound_mapping: Compound mapping dictionary
        split_info: Split information

    Returns:
        True if validation passes
    """
    print("Validating train/val split...")

    train_ids = set(split_info['train'])
    val_ids = set(split_info['val'])
    all_compound_ids = set(compound_mapping.keys())

    # Check for overlaps
    overlap = train_ids.intersection(val_ids)
    if overlap:
        print(f"❌ Overlap found between train and val: {len(overlap)} compounds")
        return False

    # Check coverage
    covered_ids = train_ids.union(val_ids)
    missing_ids = all_compound_ids - covered_ids
    extra_ids = covered_ids - all_compound_ids

    if missing_ids:
        print(f"❌ Missing compounds in split: {len(missing_ids)}")
        return False

    if extra_ids:
        print(f"❌ Extra compounds in split: {len(extra_ids)}")
        return False

    # Check DTW distance distribution
    train_dtw = [compound_mapping[cid]['dtw_distance'] for cid in train_ids]
    val_dtw = [compound_mapping[cid]['dtw_distance'] for cid in val_ids]

    train_mean_dtw = np.mean(train_dtw)
    val_mean_dtw = np.mean(val_dtw)

    print(f"✅ Split validation passed")
    print(f"   Train DTW mean: {train_mean_dtw:.3f}")
    print(f"   Val DTW mean: {val_mean_dtw:.3f}")
    print(f"   DTW difference: {abs(train_mean_dtw - val_mean_dtw):.3f}")

    return True


def save_results(compound_mapping: Dict[str, Dict[str, Any]],
                split_info: Dict[str, List[str]],
                dataset_stats: Dict[str, Any],
                output_dir: str):
    """
    Save all results to disk.

    Args:
        compound_mapping: Compound mapping dictionary
        split_info: Split information
        dataset_stats: Dataset statistics
        output_dir: Output directory
    """
    print(f"Saving results to {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save compound mapping
    mapping_path = os.path.join(output_dir, 'compound_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(compound_mapping, f, indent=2)
    print(f"✅ Compound mapping saved to {mapping_path}")

    # Save train/val split
    split_path = os.path.join(output_dir, 'train_val_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✅ Train/val split saved to {split_path}")

    # Save dataset statistics
    stats_path = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"✅ Dataset statistics saved to {stats_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare XRD data for prototypical classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--dataset_path', type=str, default='../data/xrd_dataset_labeled_dtw_window.pt',
                       help='Path to XRD dataset')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--stratify_by_dtw', action='store_true',
                       help='Stratify split by DTW distance quartiles')

    args = parser.parse_args()

    print("=" * 60)
    print("XRD Data Preparation for Prototypical Classification")
    print("=" * 60)

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"✅ Configuration loaded from {args.config}")
    else:
        print(f"⚠️ Configuration file not found: {args.config}")
        print("Using default parameters")
        config = {
            'data_split': {
                'train_ratio': 0.8,
                'random_seed': 42
            }
        }

    # Load dataset
    data = load_xrd_dataset(args.dataset_path)

    # Create compound mapping
    compound_mapping = create_compound_mapping(data)

    # Analyze dataset statistics
    dataset_stats = analyze_dataset_statistics(compound_mapping)

    # Create train/val split
    split_config = config.get('data_split', {})
    split_info = create_train_val_split(
        compound_mapping,
        train_ratio=split_config.get('train_ratio', 0.8),
        random_seed=split_config.get('random_seed', 42),
        stratify_by_dtw=args.stratify_by_dtw
    )

    # Validate split
    if not validate_split(compound_mapping, split_info):
        print("❌ Split validation failed!")
        return

    # Save results
    save_results(compound_mapping, split_info, dataset_stats, args.output_dir)

    print("\n" + "=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)
    print(f"Total compounds: {len(compound_mapping)}")
    print(f"Training compounds: {len(split_info['train'])}")
    print(f"Validation compounds: {len(split_info['val'])}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()