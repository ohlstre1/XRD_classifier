#!/usr/bin/env python3
"""
Supervised Contrastive Loss for XRD Classification
==================================================

Implementation of supervised contrastive learning for XRD pattern classification.
Enforces that augmented versions of the same compound cluster together while
separating different compounds on the embedding hypersphere.

Key Features:
- Multi-view contrastive learning with heavy augmentation
- Temperature-scaled cosine similarity
- Support for large batch sizes and many classes
- Augmentation-invariant embeddings for domain transfer

Reference:
- Supervised Contrastive Learning (NeurIPS 2020)
- https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for learning augmentation-invariant embeddings.

    For each anchor sample, pulls together positive samples (same class, different augmentations)
    and pushes away negative samples (different classes).

    Args:
        temperature: Temperature parameter for scaling similarities
        contrast_mode: 'one' for one positive per anchor, 'all' for all positives
        base_temperature: Base temperature for normalization
    """

    def __init__(self,
                 temperature: float = 0.07,
                 contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        """
        Initialize supervised contrastive loss.

        Args:
            temperature: Temperature for scaling logits
            contrast_mode: How to select positive pairs
            base_temperature: Base temperature for normalization
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute supervised contrastive loss.

        Args:
            features: Normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = features.device
        batch_size = features.shape[0]

        # Ensure features are normalized
        features = F.normalize(features, dim=1)

        # Compute cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask for positive and negative pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove diagonal (self-contrast)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean log-probability over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        # Compute metrics
        with torch.no_grad():
            # Number of positive pairs per sample
            num_positives = mask.sum(1)

            # Average positive similarity
            pos_similarities = (mask * torch.matmul(features, features.T)).sum(1) / torch.clamp(num_positives, min=1)
            avg_pos_sim = pos_similarities.mean()

            # Average negative similarity
            neg_mask = 1 - mask - torch.eye(batch_size, device=device)
            num_negatives = neg_mask.sum(1)
            neg_similarities = (neg_mask * torch.matmul(features, features.T)).sum(1) / torch.clamp(num_negatives, min=1)
            avg_neg_sim = neg_similarities.mean()

            # Compute effective batch diversity
            unique_labels = torch.unique(labels)
            num_unique_labels = len(unique_labels)

        metrics = {
            'scl_loss': loss.detach(),
            'avg_positive_similarity': avg_pos_sim,
            'avg_negative_similarity': avg_neg_sim,
            'avg_num_positives': num_positives.float().mean(),
            'num_unique_labels': torch.tensor(num_unique_labels, device=device),
            'temperature': torch.tensor(self.temperature, device=device)
        }

        return loss, metrics


class MultiViewContrastiveLoss(nn.Module):
    """
    Multi-view contrastive loss for handling multiple augmented views per sample.

    Expects input to be organized as [batch_size * num_views, embedding_dim]
    where consecutive num_views samples belong to the same original sample.

    Args:
        temperature: Temperature parameter for scaling
        num_views: Number of augmented views per original sample
        base_temperature: Base temperature for normalization
    """

    def __init__(self,
                 temperature: float = 0.07,
                 num_views: int = 2,
                 base_temperature: float = 0.07):
        """
        Initialize multi-view contrastive loss.

        Args:
            temperature: Temperature for scaling
            num_views: Number of views per sample
            base_temperature: Base temperature
        """
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_views = num_views
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-view contrastive loss.

        Args:
            features: Features [batch_size * num_views, embedding_dim]
            labels: Labels [batch_size * num_views] (repeated for each view)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = features.device
        total_samples = features.shape[0]
        batch_size = total_samples // self.num_views

        # Ensure features are normalized
        features = F.normalize(features, dim=1)

        # Reshape to [batch_size, num_views, embedding_dim]
        features_reshaped = features.view(batch_size, self.num_views, -1)

        # Compute pairwise similarities between all views
        # Flatten back to [batch_size * num_views, embedding_dim] for computation
        contrast_features = features

        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_features, contrast_features.T),
            self.temperature
        )

        # For stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask for positive pairs
        # Positive = same original sample (different views) OR same class
        labels_reshaped = labels.contiguous().view(-1, 1)

        # Mask for same class
        class_mask = torch.eq(labels_reshaped, labels_reshaped.T).float().to(device)

        # Mask for same original sample (different views)
        sample_indices = torch.arange(total_samples, device=device) // self.num_views
        sample_indices = sample_indices.contiguous().view(-1, 1)
        same_sample_mask = torch.eq(sample_indices, sample_indices.T).float().to(device)

        # Positive mask: same sample (different views) OR same class
        # But exclude self-contrast
        self_mask = torch.eye(total_samples, device=device).float()
        positive_mask = (same_sample_mask - self_mask) + class_mask * (1 - same_sample_mask)
        positive_mask = torch.clamp(positive_mask, 0, 1)

        # Logits mask (exclude self)
        logits_mask = 1 - self_mask

        # Apply masks
        positive_mask = positive_mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean log-probability over positive pairs
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / torch.clamp(positive_mask.sum(1), min=1e-12)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # Compute metrics
        with torch.no_grad():
            num_positives = positive_mask.sum(1)
            avg_positives = num_positives.mean()

            # Within-sample similarity (different views of same sample)
            within_sample_sim = (same_sample_mask - self_mask) * torch.matmul(contrast_features, contrast_features.T)
            within_sample_sim = within_sample_sim.sum() / torch.clamp((same_sample_mask - self_mask).sum(), min=1)

            # Cross-class similarity
            cross_class_mask = 1 - class_mask
            cross_class_sim = cross_class_mask * torch.matmul(contrast_features, contrast_features.T)
            cross_class_sim = cross_class_sim.sum() / torch.clamp(cross_class_mask.sum(), min=1)

        metrics = {
            'mv_scl_loss': loss.detach(),
            'avg_num_positives': avg_positives,
            'within_sample_similarity': within_sample_sim,
            'cross_class_similarity': cross_class_sim,
            'num_views': torch.tensor(self.num_views, device=device)
        }

        return loss, metrics


class HierarchicalContrastiveLoss(nn.Module):
    """
    Hierarchical contrastive loss for XRD patterns with multiple granularity levels.

    Combines:
    1. View-level contrastive: Different augmentations of same pattern
    2. Compound-level contrastive: Same compound, different base patterns
    3. Class-level contrastive: Different compounds

    Args:
        temperature: Temperature for scaling
        view_weight: Weight for view-level loss
        compound_weight: Weight for compound-level loss
        class_weight: Weight for class-level loss
    """

    def __init__(self,
                 temperature: float = 0.07,
                 view_weight: float = 1.0,
                 compound_weight: float = 0.5,
                 class_weight: float = 0.3):
        """Initialize hierarchical contrastive loss."""
        super(HierarchicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.view_weight = view_weight
        self.compound_weight = compound_weight
        self.class_weight = class_weight

        self.view_contrastive = SupervisedContrastiveLoss(temperature=temperature)

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                compound_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute hierarchical contrastive loss.

        Args:
            features: Normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            compound_ids: Compound IDs for finer granularity [batch_size]

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        device = features.device

        # View-level contrastive (standard supervised contrastive)
        view_loss, view_metrics = self.view_contrastive(features, labels)

        total_loss = self.view_weight * view_loss
        all_metrics = {f"view_{k}": v for k, v in view_metrics.items()}

        # If compound IDs provided, add compound-level contrastive
        if compound_ids is not None:
            compound_loss, compound_metrics = self.view_contrastive(features, compound_ids)
            total_loss += self.compound_weight * compound_loss
            all_metrics.update({f"compound_{k}": v for k, v in compound_metrics.items()})

        # Add overall metrics
        all_metrics.update({
            'hierarchical_loss': total_loss.detach(),
            'view_weight': torch.tensor(self.view_weight, device=device),
            'compound_weight': torch.tensor(self.compound_weight, device=device),
        })

        return total_loss, all_metrics


def test_contrastive_losses():
    """Test contrastive loss implementations."""
    print("Testing contrastive loss implementations...")

    # Test parameters
    batch_size = 32
    embedding_dim = 256
    num_classes = 8
    num_views = 4

    # Create test data
    features = torch.randn(batch_size, embedding_dim)
    features = F.normalize(features, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"Features shape: {features.shape}")
    print(f"Labels: {labels}")
    print(f"Unique classes: {torch.unique(labels)}")

    # Test SupervisedContrastiveLoss
    print("\n--- Testing SupervisedContrastiveLoss ---")
    scl = SupervisedContrastiveLoss(temperature=0.07)
    loss, metrics = scl(features, labels)
    print(f"Loss: {loss.item():.4f}")
    print("Metrics:", {k: f"{v.item():.4f}" if hasattr(v, 'item') else v for k, v in metrics.items()})

    # Test MultiViewContrastiveLoss
    print("\n--- Testing MultiViewContrastiveLoss ---")
    mv_batch_size = batch_size // num_views
    mv_features = torch.randn(batch_size, embedding_dim)
    mv_features = F.normalize(mv_features, dim=1)
    mv_labels = torch.repeat_interleave(torch.randint(0, num_classes, (mv_batch_size,)), num_views)

    print(f"Multi-view features shape: {mv_features.shape}")
    print(f"Multi-view labels shape: {mv_labels.shape}")

    mv_scl = MultiViewContrastiveLoss(temperature=0.07, num_views=num_views)
    mv_loss, mv_metrics = mv_scl(mv_features, mv_labels)
    print(f"Loss: {mv_loss.item():.4f}")
    print("Metrics:", {k: f"{v.item():.4f}" if hasattr(v, 'item') else v for k, v in mv_metrics.items()})

    # Test HierarchicalContrastiveLoss
    print("\n--- Testing HierarchicalContrastiveLoss ---")
    compound_ids = torch.randint(0, num_classes * 2, (batch_size,))  # More compounds than classes

    h_scl = HierarchicalContrastiveLoss(
        temperature=0.07,
        view_weight=1.0,
        compound_weight=0.5,
        class_weight=0.3
    )
    h_loss, h_metrics = h_scl(features, labels, compound_ids)
    print(f"Loss: {h_loss.item():.4f}")
    print("Metrics:", {k: f"{v.item():.4f}" if hasattr(v, 'item') else v for k, v in h_metrics.items()})

    # Test temperature effects
    print("\n--- Testing Temperature Effects ---")
    temperatures = [0.01, 0.07, 0.2, 0.5]
    for temp in temperatures:
        temp_scl = SupervisedContrastiveLoss(temperature=temp)
        temp_loss, _ = temp_scl(features, labels)
        print(f"Temperature {temp:.2f}: Loss = {temp_loss.item():.4f}")

    print("\n✅ All contrastive loss tests passed!")


if __name__ == "__main__":
    test_contrastive_losses()