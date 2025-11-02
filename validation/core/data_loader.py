"""
Core data loading functionality for XRD diffusion validation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class XRDDataLoader:
    """
    Handles loading and splitting of XRD dataset for validation.
    """

    def __init__(self, dataset_path: str, device: str = 'auto'):
        """
        Initialize data loader.

        Args:
            dataset_path: Path to the dataset file
            device: Device to load data to ('auto', 'cuda', 'cpu')
        """
        self.dataset_path = Path(dataset_path)
        self.device = self._get_device(device)
        self._data = None

    def _get_device(self, device: str) -> str:
        """Get the appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def load_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Load the XRD dataset.

        Returns:
            Dictionary containing synthetic XRD, real XRD, and DTW distances
        """
        print(f"Loading XRD dataset from {self.dataset_path}...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        dataset_dict = torch.load(self.dataset_path, map_location=self.device)

        synth_xrd = dataset_dict["synth_xrd"]
        real_xrd = dataset_dict["real_xrd"]
        dtw_distances = dataset_dict["fast_dtw_distance"]

        print(f"Dataset loaded:")
        print(f"  Synthetic XRD shape: {synth_xrd.shape}")
        print(f"  Real XRD shape: {real_xrd.shape}")
        print(f"  DTW distances shape: {dtw_distances.shape}")
        print(f"  DTW distance range: [{dtw_distances.min():.3f}, {dtw_distances.max():.3f}]")

        self._data = {
            'synth_xrd': synth_xrd,
            'real_xrd': real_xrd,
            'dtw_distances': dtw_distances
        }

        return self._data

    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Create train/validation/test splits.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation

        Returns:
            Dictionary with train, val, test splits
        """
        if self._data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        synth_xrd = self._data['synth_xrd']
        real_xrd = self._data['real_xrd']
        dtw_distances = self._data['dtw_distances']

        total_samples = len(synth_xrd)
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        print(f"\nDataset splits:")
        print(f"  Train: {train_size} samples")
        print(f"  Validation: {val_size} samples")
        print(f"  Test: {test_size} samples")

        # Create splits
        splits = {
            'train': {
                'synth': synth_xrd[:train_size],
                'real': real_xrd[:train_size],
                'dtw': dtw_distances[:train_size]
            },
            'val': {
                'synth': synth_xrd[train_size:train_size+val_size],
                'real': real_xrd[train_size:train_size+val_size],
                'dtw': dtw_distances[train_size:train_size+val_size]
            },
            'test': {
                'synth': synth_xrd[train_size+val_size:],
                'real': real_xrd[train_size+val_size:],
                'dtw': dtw_distances[train_size+val_size:]
            }
        }

        return splits

    def get_sample(self, split: str, index: int, splits: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a specific sample from a split.

        Args:
            split: Split name ('train', 'val', 'test')
            index: Sample index
            splits: Split data dictionary

        Returns:
            Tuple of (synthetic, real, dtw) tensors
        """
        if split not in splits:
            raise ValueError(f"Split '{split}' not found. Available: {list(splits.keys())}")

        split_data = splits[split]

        if index >= len(split_data['synth']):
            raise IndexError(f"Index {index} out of range for split '{split}' (size: {len(split_data['synth'])})")

        return (
            split_data['synth'][index],
            split_data['real'][index],
            split_data['dtw'][index]
        )


def load_xrd_data(dataset_path: str = "data/xrd_dataset_labeled_dtw_window.pt",
                  device: str = 'auto') -> Tuple[Dict[str, Dict[str, torch.Tensor]], XRDDataLoader]:
    """
    Convenience function to load and split XRD data.

    Args:
        dataset_path: Path to dataset file
        device: Device to use

    Returns:
        Tuple of (splits_dict, data_loader)
    """
    loader = XRDDataLoader(dataset_path, device)
    loader.load_dataset()
    splits = loader.create_splits()

    return splits, loader