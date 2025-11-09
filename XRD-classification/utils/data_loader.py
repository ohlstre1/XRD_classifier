#!/usr/bin/env python3
"""
Data Loaders for XRD Prototypical Classification
===============================================

Dataset classes and data loaders for XRD pattern classification using prototypical learning.
Supports both augmented training data and real test patterns.

Key features:
- XRDDataset: For augmented training patterns from compound mapping
- XRDRealDataset: For real measured test patterns
- Efficient data loading with proper label encoding
- Support for both synthetic and real pattern loading
"""

import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import warnings


class XRDDataset(Dataset):
    """
    Dataset for XRD patterns from compound mapping.

    Loads synthetic patterns from the compound mapping and converts them
    to the format expected by the prototypical classifier.
    """

    def __init__(self, compound_ids: List[str], compound_mapping: Dict[str, Any],
                 mode: str = 'train', pattern_type: str = 'synth'):
        """
        Initialize XRD Dataset.

        Args:
            compound_ids: List of compound IDs to include
            compound_mapping: Dictionary mapping compound IDs to pattern data
            mode: 'train' or 'val' (for future use)
            pattern_type: 'synth' for synthetic, 'real' for real patterns
        """
        self.compound_ids = compound_ids
        self.compound_mapping = compound_mapping
        self.mode = mode
        self.pattern_type = pattern_type

        # Create label encoding: compound_id -> integer label
        unique_ids = sorted(set(compound_ids))
        self.id_to_label = {cid: idx for idx, cid in enumerate(unique_ids)}
        self.label_to_id = {idx: cid for cid, idx in self.id_to_label.items()}

        print(f"XRDDataset initialized: {len(compound_ids)} compounds, {len(unique_ids)} unique classes")

    def __len__(self):
        return len(self.compound_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (xrd_tensor, label, compound_id)
        """
        compound_id = self.compound_ids[idx]

        # Load pattern from compound mapping
        compound_data = self.compound_mapping[compound_id]

        if self.pattern_type == 'synth':
            pattern = compound_data['synth_pattern']
        else:
            pattern = compound_data['real_pattern']

        # Convert to tensor and add channel dimension: (1, 4500)
        if isinstance(pattern, list):
            pattern = np.array(pattern, dtype=np.float32)

        xrd_tensor = torch.from_numpy(pattern).unsqueeze(0)  # (1, 4500)

        # Get integer label
        label = self.id_to_label[compound_id]

        return xrd_tensor, label, compound_id


class XRDAugmentedDataset(Dataset):
    """
    Dataset for augmented XRD patterns.

    Generates augmented patterns on-the-fly using the dual augmentation system.
    """

    def __init__(self, compound_ids: List[str], compound_mapping: Dict[str, Any],
                 augmenter, samples_per_pattern: int = 5, mode: str = 'train'):
        """
        Initialize augmented XRD dataset.

        Args:
            compound_ids: List of compound IDs
            compound_mapping: Compound mapping dictionary
            augmenter: DualXRDAugmenter instance
            samples_per_pattern: Number of augmented samples per compound
            mode: 'train' or 'val'
        """
        self.compound_ids = compound_ids
        self.compound_mapping = compound_mapping
        self.augmenter = augmenter
        self.samples_per_pattern = samples_per_pattern
        self.mode = mode

        # Create label encoding
        unique_ids = sorted(set(compound_ids))
        self.id_to_label = {cid: idx for idx, cid in enumerate(unique_ids)}
        self.label_to_id = {idx: cid for cid, idx in self.id_to_label.items()}

        # Pre-compute total samples
        self.total_samples = len(compound_ids) * samples_per_pattern

        print(f"XRDAugmentedDataset initialized: {len(compound_ids)} compounds, "
              f"{self.total_samples} total augmented samples")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get augmented item from dataset.

        Args:
            idx: Global index

        Returns:
            Tuple of (augmented_xrd_tensor, label, compound_id)
        """
        # Map global index to compound and augmentation index
        compound_idx = idx // self.samples_per_pattern
        aug_idx = idx % self.samples_per_pattern

        compound_id = self.compound_ids[compound_idx]

        # Load synthetic pattern
        compound_data = self.compound_mapping[compound_id]
        synth_pattern = np.array(compound_data['synth_pattern'], dtype=np.float32)
        synth_tensor = torch.from_numpy(synth_pattern)

        # Generate single augmented sample
        try:
            augmented, methods = self.augmenter.augment_pattern_mixed(synth_tensor, num_samples=1)
            augmented_tensor = augmented[0]  # Take first (and only) sample
        except Exception as e:
            # Fallback to original pattern if augmentation fails
            warnings.warn(f"Augmentation failed for {compound_id}: {e}")
            augmented_tensor = synth_tensor.unsqueeze(0)  # Add channel dimension

        # Get integer label
        label = self.id_to_label[compound_id]

        return augmented_tensor, label, compound_id


class XRDRealDataset(Dataset):
    """
    Dataset for real measured XRD patterns (test set).
    """

    def __init__(self, compound_ids: List[str], compound_mapping: Dict[str, Any]):
        """
        Initialize real XRD dataset.

        Args:
            compound_ids: List of compound IDs to include
            compound_mapping: Dictionary mapping compound_id -> file paths and data
        """
        self.compound_ids = compound_ids
        self.compound_mapping = compound_mapping

        print(f"XRDRealDataset initialized: {len(compound_ids)} real patterns")

    def __len__(self):
        return len(self.compound_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get real pattern from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (real_xrd_tensor, compound_id)
        """
        compound_id = self.compound_ids[idx]

        # Load real pattern from compound mapping
        compound_data = self.compound_mapping[compound_id]
        real_pattern = np.array(compound_data['real_pattern'], dtype=np.float32)

        # Convert to tensor: (1, 4500)
        xrd_tensor = torch.from_numpy(real_pattern).unsqueeze(0)

        return xrd_tensor, compound_id


def create_data_loaders(train_ids: List[str], val_ids: List[str],
                       compound_mapping: Dict[str, Any],
                       config: Dict[str, Any],
                       augmenter=None) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation data loaders.

    Args:
        train_ids: Training compound IDs
        val_ids: Validation compound IDs
        compound_mapping: Compound mapping dictionary
        config: Configuration dictionary
        augmenter: Optional augmenter for training data

    Returns:
        Tuple of (train_loader, val_loader, id_to_label_mapping)
    """

    # Determine if we should use augmentation
    use_augmentation = augmenter is not None and config.get('augmentation', {}).get('n_augmentations', 0) > 0

    if use_augmentation:
        # Create augmented training dataset
        train_dataset = XRDAugmentedDataset(
            compound_ids=train_ids,
            compound_mapping=compound_mapping,
            augmenter=augmenter,
            samples_per_pattern=config['augmentation']['n_augmentations'],
            mode='train'
        )
    else:
        # Create standard training dataset (synthetic patterns)
        train_dataset = XRDDataset(
            compound_ids=train_ids,
            compound_mapping=compound_mapping,
            mode='train',
            pattern_type='synth'
        )

    # Create validation dataset (synthetic patterns)
    val_dataset = XRDDataset(
        compound_ids=val_ids,
        compound_mapping=compound_mapping,
        mode='val',
        pattern_type='synth'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=config['training'].get('pin_memory', True),
        drop_last=True  # Ensure consistent batch sizes for prototypical loss
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=config['training'].get('pin_memory', True)
    )

    # Get label mapping from training dataset
    id_to_label = train_dataset.id_to_label

    print(f"Data loaders created:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Number of classes: {len(id_to_label)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    return train_loader, val_loader, id_to_label


def create_test_loader(test_ids: List[str], compound_mapping: Dict[str, Any],
                      batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """
    Create data loader for real test patterns.

    Args:
        test_ids: Test compound IDs
        compound_mapping: Compound mapping dictionary
        batch_size: Batch size for testing
        num_workers: Number of worker processes

    Returns:
        DataLoader for real test patterns
    """
    test_dataset = XRDRealDataset(test_ids, compound_mapping)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Test loader created: {len(test_dataset)} samples, {len(test_loader)} batches")

    return test_loader


def test_data_loaders():
    """Test function for data loaders."""
    print("Testing XRD data loaders...")

    # Create mock compound mapping
    mock_mapping = {}
    for i in range(10):
        compound_id = f"compound_{i:05d}"
        mock_mapping[compound_id] = {
            'synth_pattern': np.random.rand(4500).tolist(),
            'real_pattern': np.random.rand(4500).tolist(),
            'file_info': f"mock_file_{i}",
            'dtw_distance': np.random.rand()
        }

    # Test compound IDs
    train_ids = [f"compound_{i:05d}" for i in range(8)]
    val_ids = [f"compound_{i:05d}" for i in range(8, 10)]

    # Test XRDDataset
    print("\n--- Testing XRDDataset ---")
    dataset = XRDDataset(train_ids, mock_mapping, mode='train', pattern_type='synth')
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample types: {[type(x) for x in sample]}")
    print(f"Pattern shape: {sample[0].shape}")
    print(f"Label: {sample[1]}")
    print(f"Compound ID: {sample[2]}")

    # Test XRDRealDataset
    print("\n--- Testing XRDRealDataset ---")
    real_dataset = XRDRealDataset(val_ids, mock_mapping)
    real_sample = real_dataset[0]
    print(f"Real sample types: {[type(x) for x in real_sample]}")
    print(f"Real pattern shape: {real_sample[0].shape}")

    # Test DataLoader
    print("\n--- Testing DataLoader ---")
    mock_config = {
        'training': {
            'batch_size': 4,
            'num_workers': 0,  # Use 0 for testing
            'pin_memory': False
        },
        'augmentation': {
            'n_augmentations': 0  # No augmentation for testing
        }
    }

    train_loader, val_loader, id_to_label = create_data_loaders(
        train_ids, val_ids, mock_mapping, mock_config, augmenter=None
    )

    # Test batch loading
    for batch_idx, (patterns, labels, compound_ids) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Patterns shape: {patterns.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Compound IDs: {compound_ids}")
        if batch_idx >= 1:  # Only test first 2 batches
            break

    print("✅ Data loader tests passed!")


if __name__ == "__main__":
    test_data_loaders()