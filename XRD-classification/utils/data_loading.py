"""
Data loading utilities for XRD classification
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


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

    data = torch.load(dataset_path, weights_only=False)

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