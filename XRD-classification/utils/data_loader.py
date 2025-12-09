"""
Data loading utilities for XRD classification
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
from torch.utils.data import Dataset


class XRDDataset(Dataset):
    """
    Dataset for XRD patterns from compound mapping.

    Loads synthetic patterns from the compound mapping and converts them
    to the format expected by the prototypical classifier.
    """

    def __init__(self, dataset_path, test_dataset_path, num_classes, mode='train'):
        """
        Initialize XRD dataset.

        Args:
            dataset_path: Path to training dataset (xrd_dataset_labeled_dtw_window.pt)
            test_dataset_path: Path to test dataset (xrd_test_dataset.pt)
            num_classes: Number of classes to use
            mode: 'train', 'val', or 'test'
        """
        self.mode = mode
        self.num_classes = num_classes

        # Load main dataset
        self.data = torch.load(dataset_path, weights_only=False)

        # Load test dataset
        test_data = torch.load(test_dataset_path, weights_only=False)
        self.test_patterns = test_data['real_xrd']

        # Load split information
        try:
            with open('data/dataset_split_summary.json', 'r') as f:
                split_info = json.load(f)
            self.test_indices = split_info.get("test_indices", [])
        except:
            self.test_indices = list(range(len(self.test_patterns)))

        # Create labels for each pattern
        n_total = len(self.data['synth_xrd'])

        # Use first num_classes patterns
        if num_classes < n_total:
            # Take first num_classes samples
            self.train_patterns = self.data['synth_xrd'][:num_classes]
            self.val_patterns = self.data['real_xrd'][:num_classes]
            self.pattern_labels = torch.arange(num_classes)
            self.compound_ids = [f"compound_{i:05d}" for i in range(num_classes)]
        else:
            # Use all available data
            self.train_patterns = self.data['synth_xrd']
            self.val_patterns = self.data['real_xrd']
            self.pattern_labels = torch.arange(n_total)
            self.compound_ids = [f"compound_{i:05d}" for i in range(n_total)]

        # Split train/val for synthetic data
        n_available = len(self.train_patterns)
        n_train = int(0.8 * n_available)

        indices = torch.randperm(n_available)
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:]

        print(f"XRDDataset ({mode}): {self._get_size()} samples")

    def _get_size(self):
        """Get size based on mode."""
        if self.mode == 'train':
            return len(self.train_indices)
        elif self.mode == 'val':
            return len(self.val_indices)
        elif self.mode == 'test':
            return min(len(self.test_patterns), self.num_classes)
        return 0

    def __len__(self):
        return self._get_size()

    def __getitem__(self, idx):
        """Get item based on mode."""
        if self.mode == 'train':
            # Return synthetic pattern for training
            actual_idx = self.train_indices[idx]
            pattern = self.train_patterns[actual_idx]
            label = self.pattern_labels[actual_idx]
            compound_id = self.compound_ids[actual_idx]

        elif self.mode == 'val':
            # Return synthetic pattern for validation (prototype building)
            actual_idx = self.val_indices[idx]
            pattern = self.train_patterns[actual_idx]  # Still synthetic for prototype building
            label = self.pattern_labels[actual_idx]
            compound_id = self.compound_ids[actual_idx]

        elif self.mode == 'test':
            # Return real pattern for testing
            if idx >= len(self.test_patterns):
                idx = idx % len(self.test_patterns)
            pattern = self.test_patterns[idx]
            label = torch.tensor(idx)  # Test labels
            compound_id = f"test_compound_{idx:05d}"

        return pattern, label, compound_id

    def get_train_data(self):
        """Get all training data."""
        patterns = self.train_patterns[self.train_indices]
        labels = self.pattern_labels[self.train_indices]
        compound_ids = [self.compound_ids[i] for i in self.train_indices]
        return patterns, labels, compound_ids

    def get_val_data(self):
        """Get all validation data."""
        patterns = self.train_patterns[self.val_indices]  # Synthetic for prototype building
        labels = self.pattern_labels[self.val_indices]
        compound_ids = [self.compound_ids[i] for i in self.val_indices]
        return patterns, labels, compound_ids

    def get_test_data(self):
        """Get all test data."""
        n_test = min(len(self.test_patterns), self.num_classes)
        patterns = self.test_patterns[:n_test]
        labels = torch.arange(n_test)
        compound_ids = [f"test_compound_{i:05d}" for i in range(n_test)]
        return patterns, labels, compound_ids


def create_xrd_datasets(dataset_path='data/xrd_dataset_labeled_dtw_window.pt',
                       test_dataset_path='data/xrd_test_dataset.pt',
                       num_classes=1000):
    """
    Create train, validation, and test datasets.

    Args:
        dataset_path: Path to main dataset
        test_dataset_path: Path to test dataset
        num_classes: Number of classes to use

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = XRDDataset(dataset_path, test_dataset_path, num_classes, mode='train')
    val_dataset = XRDDataset(dataset_path, test_dataset_path, num_classes, mode='val')
    test_dataset = XRDDataset(dataset_path, test_dataset_path, num_classes, mode='test')

    return train_dataset, val_dataset, test_dataset