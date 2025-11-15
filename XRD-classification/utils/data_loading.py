"""
Data loading utilities for XRD classification
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


def load_subset_data(dataset_path: str, n_samples: int = 500, indices: List[int] = None) -> dict:
    """
    Load a subset of the XRD dataset.

    Args:
        dataset_path: Path to the full dataset
        n_samples: Number of samples to load
        indices: Specific indices to load (if None, uses first n_samples)

    Returns:
        Subset dataset dictionary
    """
    print(f"Loading subset of {n_samples} samples from {dataset_path}")

    data = torch.load(dataset_path, weights_only=False)

    if indices is not None:
        # Use specific indices
        subset_data = {
            'synth_xrd': data['synth_xrd'][indices],
            'real_xrd': data['real_xrd'][indices],
            'file_info': [data['file_info'][i] for i in indices],
            'fast_dtw_distance': data['fast_dtw_distance'][indices]
        }
    else:
        # Use first n_samples
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

    all_synth = [original_synth]
    all_real = [original_real]
    all_info = [original_info]
    all_dtw = [original_dtw]

    compound_groups = {}
    current_idx = 0

    for orig_idx in range(n_original):
        compound_groups[orig_idx] = [current_idx]
        current_idx += 1

    for dup_idx in range(duplication_factor):
        noise = torch.randn_like(original_synth) * noise_level
        synth_variant = original_synth + noise

        all_synth.append(synth_variant)
        all_real.append(original_real.clone())
        all_info.append([f"{info}_dup{dup_idx+1}" for info in original_info])
        all_dtw.append(original_dtw.clone())

        for orig_idx in range(n_original):
            compound_groups[orig_idx].append(current_idx)
            current_idx += 1

    data_duplicated = {
        'synth_xrd': torch.cat(all_synth, dim=0),
        'real_xrd': torch.cat(all_real, dim=0),
        'file_info': [item for sublist in all_info for item in sublist],
        'fast_dtw_distance': torch.cat(all_dtw, dim=0),
        'compound_groups': compound_groups
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

    compound_groups = data.get('compound_groups', {})

    if compound_groups:
        print(f"Using compound groups: {len(compound_groups)} unique compounds")
        for compound_idx in range(len(compound_groups)):
            compound_id = f"compound_{compound_idx:05d}"
            pattern_indices = compound_groups[compound_idx]

            main_idx = pattern_indices[0]

            compound_mapping[compound_id] = {
                'index': compound_idx,
                'synth_pattern': synth_normalized[main_idx].numpy().tolist(),
                'real_pattern': real_normalized[main_idx].numpy().tolist(),
                'file_info': str(data['file_info'][main_idx]),
                'dtw_distance': float(data['fast_dtw_distance'][main_idx]),
                'pattern_indices': pattern_indices
            }
    else:
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


def load_synthetic_data(dataset_path: str) -> dict:
    """
    Load synthetic XRD dataset for training.

    Args:
        dataset_path: Path to synthetic dataset (xrd_dataset_labeled_dtw_window.pt)

    Returns:
        Dataset dictionary with synthetic patterns
    """
    print(f"Loading synthetic training data from {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)

    print(f"✅ Loaded synthetic data: {data['synth_xrd'].shape[0]} samples")
    print(f"   Pattern shape: {data['synth_xrd'].shape}")

    return data


def load_real_val_data(dataset_path: str, n_samples: int = None, indices: List[int] = None) -> dict:
    """
    Load real XRD validation dataset.

    Args:
        dataset_path: Path to validation dataset (xrd_train_val_dataset.pt)
        n_samples: Optional number of samples to load (None = all samples)
        indices: Specific indices to load (if None and n_samples set, uses first n_samples)

    Returns:
        Dataset dictionary with real validation patterns
    """
    print(f"Loading real validation data from {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)

    # Limit samples if specified
    if indices is not None:
        print(f"  Using specific indices: {len(indices)} samples")
        subset_data = {
            'real_xrd': data['real_xrd'][indices],
            'fast_dtw_distance': data['fast_dtw_distance'][indices] if 'fast_dtw_distance' in data else torch.zeros(len(indices))
        }
        # Preserve any other fields that might exist
        for key in data:
            if key not in ['real_xrd', 'fast_dtw_distance']:
                if isinstance(data[key], list):
                    subset_data[key] = [data[key][i] for i in indices if i < len(data[key])]
                elif isinstance(data[key], torch.Tensor):
                    subset_data[key] = data[key][indices]
                else:
                    subset_data[key] = data[key]
        data = subset_data
    elif n_samples is not None and n_samples < len(data['real_xrd']):
        print(f"  Limiting to {n_samples} samples (from {len(data['real_xrd'])} total)")
        subset_data = {
            'real_xrd': data['real_xrd'][:n_samples],
            'fast_dtw_distance': data['fast_dtw_distance'][:n_samples] if 'fast_dtw_distance' in data else torch.zeros(n_samples)
        }
        # Preserve any other fields that might exist
        for key in data:
            if key not in ['real_xrd', 'fast_dtw_distance']:
                if isinstance(data[key], (torch.Tensor, list)):
                    subset_data[key] = data[key][:n_samples] if len(data[key]) >= n_samples else data[key]
                else:
                    subset_data[key] = data[key]
        data = subset_data

    print(f"✅ Loaded real validation data: {data['real_xrd'].shape[0]} samples")
    print(f"   Pattern shape: {data['real_xrd'].shape}")

    return data


def load_real_test_data(dataset_path: str, n_samples: int = None, indices: List[int] = None) -> dict:
    """
    Load real XRD test dataset.

    Args:
        dataset_path: Path to test dataset (xrd_test_dataset.pt)
        n_samples: Optional number of samples to load (None = all samples)
        indices: Specific indices to load (if None and n_samples set, uses first n_samples)

    Returns:
        Dataset dictionary with real test patterns
    """
    print(f"Loading real test data from {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)

    # Limit samples if specified
    if indices is not None:
        print(f"  Using specific indices: {len(indices)} samples")
        subset_data = {
            'real_xrd': data['real_xrd'][indices],
            'fast_dtw_distance': data['fast_dtw_distance'][indices] if 'fast_dtw_distance' in data else torch.zeros(len(indices))
        }
        # Preserve any other fields that might exist
        for key in data:
            if key not in ['real_xrd', 'fast_dtw_distance']:
                if isinstance(data[key], list):
                    subset_data[key] = [data[key][i] for i in indices if i < len(data[key])]
                elif isinstance(data[key], torch.Tensor):
                    subset_data[key] = data[key][indices]
                else:
                    subset_data[key] = data[key]
        data = subset_data
    elif n_samples is not None and n_samples < len(data['real_xrd']):
        print(f"  Limiting to {n_samples} samples (from {len(data['real_xrd'])} total)")
        subset_data = {
            'real_xrd': data['real_xrd'][:n_samples],
            'fast_dtw_distance': data['fast_dtw_distance'][:n_samples] if 'fast_dtw_distance' in data else torch.zeros(n_samples)
        }
        # Preserve any other fields that might exist
        for key in data:
            if key not in ['real_xrd', 'fast_dtw_distance']:
                if isinstance(data[key], (torch.Tensor, list)):
                    subset_data[key] = data[key][:n_samples] if len(data[key]) >= n_samples else data[key]
                else:
                    subset_data[key] = data[key]
        data = subset_data

    print(f"✅ Loaded real test data: {data['real_xrd'].shape[0]} samples")
    print(f"   Pattern shape: {data['real_xrd'].shape}")

    return data


def create_synthetic_real_split(train_data: dict, val_data: dict, test_data: dict,
                               use_common_compounds: bool = False) -> dict:
    """
    Create train/val/test split info using synthetic and real data.

    Args:
        train_data: Synthetic training data
        val_data: Real validation data
        test_data: Real test data
        use_common_compounds: If True, uses same compound IDs across datasets

    Returns:
        Split information dictionary
    """
    n_train = len(train_data['synth_xrd'])
    n_val = len(val_data['real_xrd'])
    n_test = len(test_data['real_xrd'])

    if use_common_compounds:
        # Use same compound IDs when datasets have the same compounds
        # This assumes the indices correspond to the same compounds
        base_compounds = min(n_train, n_val, n_test)

        # For training, handle duplicated patterns
        if n_train > base_compounds:
            # Training has duplicated patterns
            train_ids = []
            for i in range(n_train):
                compound_idx = i % base_compounds  # Map duplicates to same compound
                train_ids.append(f"compound_{compound_idx:05d}")
        else:
            train_ids = [f"compound_{i:05d}" for i in range(n_train)]

        val_ids = [f"compound_{i:05d}" for i in range(n_val)]
        test_ids = [f"compound_{i:05d}" for i in range(n_test)]
    else:
        # Create separate compound IDs for each dataset
        train_ids = [f"train_compound_{i:05d}" for i in range(n_train)]
        val_ids = [f"val_compound_{i:05d}" for i in range(n_val)]
        test_ids = [f"test_compound_{i:05d}" for i in range(n_test)]

    split_info = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
        'n_train_samples': n_train,
        'n_val_samples': n_val,
        'n_test_samples': n_test,
        'use_common_compounds': use_common_compounds
    }

    print(f"✅ Split created:")
    print(f"   Train: {n_train} synthetic samples")
    print(f"   Val: {n_val} real samples")
    print(f"   Test: {n_test} real samples")
    if use_common_compounds:
        print(f"   Using common compound IDs across datasets")

    return split_info


def create_combined_mapping(train_data: dict, val_data: dict, test_data: dict,
                           split_info: dict) -> dict:
    """
    Create compound mapping for combined synthetic/real datasets.

    Args:
        train_data: Synthetic training data
        val_data: Real validation data
        test_data: Real test data
        split_info: Split information with compound IDs

    Returns:
        Combined compound mapping dictionary
    """
    print("Creating combined compound mapping...")

    compound_mapping = {}

    # Normalize patterns
    train_synth_normalized = normalize_patterns(train_data['synth_xrd'])
    train_real_normalized = normalize_patterns(train_data['real_xrd'])
    val_real_normalized = normalize_patterns(val_data['real_xrd'])
    test_real_normalized = normalize_patterns(test_data['real_xrd'])

    # Map training compounds (synthetic)
    for i, compound_id in enumerate(split_info['train']):
        compound_mapping[compound_id] = {
            'index': i,
            'synth_pattern': train_synth_normalized[i].numpy().tolist(),
            'real_pattern': train_real_normalized[i].numpy().tolist(),
            'file_info': str(train_data['file_info'][i]) if 'file_info' in train_data else f"train_{i}",
            'dtw_distance': float(train_data['fast_dtw_distance'][i]) if 'fast_dtw_distance' in train_data else 0.0,
            'split': 'train'
        }

    # Map validation compounds (real)
    for i, compound_id in enumerate(split_info['val']):
        compound_mapping[compound_id] = {
            'index': i,
            'synth_pattern': None,  # No synthetic pattern for validation
            'real_pattern': val_real_normalized[i].cpu().numpy().tolist(),
            'file_info': f"val_{i}",
            'dtw_distance': float(val_data['fast_dtw_distance'][i].cpu()) if 'fast_dtw_distance' in val_data else 0.0,
            'split': 'val'
        }

    # Map test compounds (real)
    for i, compound_id in enumerate(split_info['test']):
        compound_mapping[compound_id] = {
            'index': i,
            'synth_pattern': None,  # No synthetic pattern for test
            'real_pattern': test_real_normalized[i].cpu().numpy().tolist(),
            'file_info': f"test_{i}",
            'dtw_distance': float(test_data['fast_dtw_distance'][i].cpu()) if 'fast_dtw_distance' in test_data else 0.0,
            'split': 'test'
        }

    print(f"✅ Created mapping for {len(compound_mapping)} compounds")
    print(f"   Train: {len(split_info['train'])} compounds")
    print(f"   Val: {len(split_info['val'])} compounds")
    print(f"   Test: {len(split_info['test'])} compounds")

    return compound_mapping