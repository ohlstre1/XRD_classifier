"""
Custom dataset classes for XRD classification
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class XRDDuplicatedDataset(Dataset):
    """
    Custom dataset that handles pattern duplication correctly for prototypical learning.

    Duplicated patterns belong to the same compound but provide more training diversity.
    """

    def __init__(self,
                 compound_ids: List[str],
                 compound_mapping: dict,
                 data: dict,
                 augmenter: Optional[object] = None,
                 samples_per_pattern: int = 5):
        """
        Initialize XRD dataset with duplication support.

        Args:
            compound_ids: List of compound IDs to include in this dataset
            compound_mapping: Dictionary mapping compound IDs to their metadata
            data: Raw data dictionary containing patterns
            augmenter: Optional augmenter object for pattern augmentation
            samples_per_pattern: Number of augmented samples per base pattern
        """
        self.compound_ids = compound_ids
        self.compound_mapping = compound_mapping
        self.data = data
        self.augmenter = augmenter
        self.samples_per_pattern = samples_per_pattern

        unique_ids = sorted(set(compound_ids))
        self.id_to_label = {cid: idx for idx, cid in enumerate(unique_ids)}

        print(f"XRDDuplicatedDataset: {len(compound_ids)} compounds, {samples_per_pattern} augmentations each")

    def __len__(self):
        return len(self.compound_ids) * self.samples_per_pattern

    def __getitem__(self, idx):
        """Get augmented pattern from potentially duplicated base patterns."""
        compound_idx = idx // self.samples_per_pattern
        compound_id = self.compound_ids[compound_idx]

        pattern_indices = self.compound_mapping[compound_id].get('pattern_indices', [compound_idx])

        selected_pattern_idx = np.random.choice(pattern_indices)

        base_pattern = self.data['synth_xrd'][selected_pattern_idx]

        if self.augmenter is not None:
            augmented_pattern, _ = self.augmenter.augment_pattern_mixed(base_pattern, num_samples=1)
            pattern_tensor = augmented_pattern[0]
        else:
            pattern_tensor = base_pattern.unsqueeze(0)

        label = self.id_to_label[compound_id]

        return pattern_tensor, label, compound_id


class XRDSimpleDataset(Dataset):
    """
    Simple dataset for XRD patterns without duplication support.

    Useful for evaluation or when not using pattern duplication.
    """

    def __init__(self,
                 patterns: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 compound_ids: Optional[List[str]] = None,
                 augmenter: Optional[object] = None):
        """
        Initialize simple XRD dataset.

        Args:
            patterns: Tensor of XRD patterns
            labels: Optional tensor of labels
            compound_ids: Optional list of compound IDs
            augmenter: Optional augmenter object
        """
        self.patterns = patterns
        self.labels = labels if labels is not None else torch.arange(len(patterns))
        self.compound_ids = compound_ids if compound_ids is not None else [f"compound_{i:05d}" for i in range(len(patterns))]
        self.augmenter = augmenter

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]

        if self.augmenter is not None:
            augmented_pattern, _ = self.augmenter.augment_pattern_mixed(pattern, num_samples=1)
            pattern = augmented_pattern[0]
        else:
            pattern = pattern.unsqueeze(0)

        return pattern, self.labels[idx], self.compound_ids[idx]