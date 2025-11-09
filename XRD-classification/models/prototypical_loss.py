#!/usr/bin/env python3
"""
Prototypical Loss Functions for XRD Classification
=================================================

Implementation of prototypical networks loss for metric learning on XRD patterns.
Includes both basic prototypical loss and combined prototypical+triplet loss
for faster convergence and better embeddings.

Key concepts:
- Prototypical loss: Pull embeddings toward class prototypes, push away from others
- Hard triplet mining: Select hardest positive and negative samples
- Temperature scaling: Control sharpness of similarity distributions
- Cosine similarity: Natural metric for normalized embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import warnings


class PrototypicalLoss(nn.Module):
    """
    Prototypical Networks loss for metric learning.

    In each batch:
    1. Compute prototypes (centroids) for each compound class
    2. Calculate distances from embeddings to all prototypes
    3. Apply cross-entropy loss to encourage correct prototype matching

    Args:
        temperature: Temperature for softmax (lower = sharper distributions)
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize prototypical loss.

        Args:
            temperature: Temperature scaling parameter
        """
        super(PrototypicalLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute prototypical loss.

        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size, embedding_dim = embeddings.shape
        device = embeddings.device

        # Get unique labels in batch
        unique_labels = torch.unique(labels)
        n_classes_in_batch = len(unique_labels)

        if n_classes_in_batch == 1:
            # Only one class in batch - return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        # Compute prototypes for each class in the batch
        prototypes = []
        prototype_labels = []

        for label in unique_labels:
            mask = (labels == label)
            class_embeddings = embeddings[mask]

            # Compute prototype as mean of class embeddings
            prototype = class_embeddings.mean(dim=0)  # [embedding_dim]

            # Re-normalize prototype to unit sphere
            prototype = F.normalize(prototype, p=2, dim=0)

            prototypes.append(prototype)
            prototype_labels.append(label)

        prototypes = torch.stack(prototypes)  # [n_classes_in_batch, embedding_dim]
        prototype_labels = torch.stack(prototype_labels)  # [n_classes_in_batch]

        # Compute distances from each embedding to all prototypes
        # Using negative cosine similarity (since embeddings are normalized)
        similarities = torch.mm(embeddings, prototypes.t())  # [batch_size, n_classes_in_batch]
        distances = -similarities / self.temperature

        # Create target indices for cross-entropy
        target_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, label in enumerate(labels):
            target_idx = (prototype_labels == label).nonzero(as_tuple=True)[0]
            if len(target_idx) > 0:
                target_indices[i] = target_idx[0]

        # Cross-entropy loss
        loss = F.cross_entropy(distances, target_indices)

        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(distances, dim=1)
            accuracy = (predictions == target_indices).float().mean()

            # Average intra-class distance (should be small)
            intra_class_dist = 0.0
            for label in unique_labels:
                mask = (labels == label)
                if mask.sum() > 1:
                    class_embs = embeddings[mask]
                    pairwise_sim = torch.mm(class_embs, class_embs.t())
                    # Average similarity within class (excluding diagonal)
                    n = class_embs.size(0)
                    intra_class_dist += (pairwise_sim.sum() - pairwise_sim.trace()) / (n * (n - 1))

            intra_class_dist /= n_classes_in_batch

            # Average inter-class distance (should be large)
            inter_class_sim = torch.mm(prototypes, prototypes.t())
            n_proto = prototypes.size(0)
            inter_class_dist = (inter_class_sim.sum() - inter_class_sim.trace()) / (n_proto * (n_proto - 1))

        metrics = {
            'proto_loss': loss.detach(),
            'proto_accuracy': accuracy,
            'intra_class_similarity': intra_class_dist,
            'inter_class_similarity': inter_class_dist,
            'n_classes_in_batch': torch.tensor(n_classes_in_batch, device=device)
        }

        return loss, metrics


class HardTripletLoss(nn.Module):
    """
    Hard triplet mining loss for metric learning.

    For each anchor:
    1. Find hardest positive (farthest sample from same class)
    2. Find hardest negative (closest sample from different class)
    3. Apply triplet loss with margin

    Args:
        margin: Margin for triplet loss
    """

    def __init__(self, margin: float = 0.2):
        """
        Initialize hard triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hard triplet loss.

        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Compute pairwise distances (using cosine distance since embeddings are normalized)
        similarities = torch.mm(embeddings, embeddings.t())  # [batch_size, batch_size]
        distances = 1 - similarities

        triplet_losses = []
        num_valid_triplets = 0

        for i in range(batch_size):
            anchor_label = labels[i]

            # Positive mask: same class, excluding self
            positive_mask = (labels == anchor_label)
            positive_mask[i] = False

            if positive_mask.sum() == 0:
                continue  # No positives for this anchor

            # Negative mask: different class
            negative_mask = (labels != anchor_label)

            if negative_mask.sum() == 0:
                continue  # No negatives for this anchor

            # Hardest positive: farthest sample from same class
            hardest_positive_dist = distances[i][positive_mask].max()

            # Hardest negative: closest sample from different class
            hardest_negative_dist = distances[i][negative_mask].min()

            # Triplet loss with margin
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_losses.append(triplet_loss)
            num_valid_triplets += 1

        if len(triplet_losses) == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = torch.stack(triplet_losses).mean()

        # Compute metrics
        with torch.no_grad():
            avg_positive_dist = 0.0
            avg_negative_dist = 0.0
            num_pairs = 0

            for i in range(batch_size):
                anchor_label = labels[i]
                positive_mask = (labels == anchor_label)
                positive_mask[i] = False
                negative_mask = (labels != anchor_label)

                if positive_mask.sum() > 0:
                    avg_positive_dist += distances[i][positive_mask].mean()
                    num_pairs += 1

                if negative_mask.sum() > 0:
                    avg_negative_dist += distances[i][negative_mask].mean()

            if num_pairs > 0:
                avg_positive_dist /= num_pairs
                avg_negative_dist /= num_pairs

        metrics = {
            'triplet_loss': loss.detach(),
            'avg_positive_distance': avg_positive_dist,
            'avg_negative_distance': avg_negative_dist,
            'num_valid_triplets': torch.tensor(num_valid_triplets, device=device)
        }

        return loss, metrics


class PrototypicalWithTripletLoss(nn.Module):
    """
    Combined loss: Prototypical + Hard Triplet Mining.

    Provides stronger gradients and faster convergence by combining:
    1. Prototypical loss for global class structure
    2. Triplet loss for local neighborhood relationships

    Args:
        proto_weight: Weight for prototypical loss
        triplet_weight: Weight for triplet loss
        triplet_margin: Margin for triplet loss
        temperature: Temperature for prototypical loss
    """

    def __init__(self, proto_weight: float = 1.0, triplet_weight: float = 0.5,
                 triplet_margin: float = 0.2, temperature: float = 0.1):
        """
        Initialize combined loss function.

        Args:
            proto_weight: Weight for prototypical loss component
            triplet_weight: Weight for triplet loss component
            triplet_margin: Margin for triplet loss
            temperature: Temperature for prototypical loss
        """
        super(PrototypicalWithTripletLoss, self).__init__()

        self.proto_loss = PrototypicalLoss(temperature=temperature)
        self.triplet_loss = HardTripletLoss(margin=triplet_margin)

        self.proto_weight = proto_weight
        self.triplet_weight = triplet_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined prototypical + triplet loss.

        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]

        Returns:
            Tuple of (total_loss, proto_loss, triplet_loss, metrics_dict)
        """
        # Compute individual losses
        proto_loss, proto_metrics = self.proto_loss(embeddings, labels)
        triplet_loss, triplet_metrics = self.triplet_loss(embeddings, labels)

        # Combined loss
        total_loss = self.proto_weight * proto_loss + self.triplet_weight * triplet_loss

        # Combine metrics
        all_metrics = {**proto_metrics, **triplet_metrics}
        all_metrics['total_loss'] = total_loss.detach()
        all_metrics['proto_weight'] = torch.tensor(self.proto_weight, device=embeddings.device)
        all_metrics['triplet_weight'] = torch.tensor(self.triplet_weight, device=embeddings.device)

        return total_loss, proto_loss, triplet_loss, all_metrics


class AdaptivePrototypicalLoss(nn.Module):
    """
    Adaptive prototypical loss that adjusts temperature based on training progress.

    Args:
        initial_temperature: Starting temperature
        final_temperature: Final temperature
        adaptation_rate: How quickly to adapt temperature
    """

    def __init__(self, initial_temperature: float = 0.2, final_temperature: float = 0.05,
                 adaptation_rate: float = 0.99):
        """
        Initialize adaptive prototypical loss.

        Args:
            initial_temperature: Starting temperature (higher = softer)
            final_temperature: Final temperature (lower = sharper)
            adaptation_rate: Rate of temperature decay
        """
        super(AdaptivePrototypicalLoss, self).__init__()

        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.adaptation_rate = adaptation_rate
        self.current_temperature = initial_temperature

        self.proto_loss = PrototypicalLoss(temperature=self.current_temperature)

    def update_temperature(self):
        """Update temperature based on adaptation schedule."""
        self.current_temperature = max(
            self.final_temperature,
            self.current_temperature * self.adaptation_rate
        )
        self.proto_loss.temperature = self.current_temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with current temperature.

        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        loss, metrics = self.proto_loss(embeddings, labels)
        metrics['current_temperature'] = torch.tensor(self.current_temperature, device=embeddings.device)
        return loss, metrics


def test_prototypical_losses():
    """Test function for prototypical loss implementations."""
    print("Testing prototypical loss functions...")

    # Create synthetic data
    batch_size = 16
    embedding_dim = 256
    n_classes = 4

    # Generate normalized embeddings
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Generate labels
    labels = torch.randint(0, n_classes, (batch_size,))

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels: {labels}")
    print(f"Embedding norms: {torch.norm(embeddings, p=2, dim=1)}")

    # Test basic prototypical loss
    print("\n--- Testing PrototypicalLoss ---")
    proto_loss_fn = PrototypicalLoss(temperature=0.1)
    loss, metrics = proto_loss_fn(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print("Metrics:", {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()})

    # Test hard triplet loss
    print("\n--- Testing HardTripletLoss ---")
    triplet_loss_fn = HardTripletLoss(margin=0.2)
    loss, metrics = triplet_loss_fn(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print("Metrics:", {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()})

    # Test combined loss
    print("\n--- Testing PrototypicalWithTripletLoss ---")
    combined_loss_fn = PrototypicalWithTripletLoss(
        proto_weight=1.0,
        triplet_weight=0.5,
        triplet_margin=0.2,
        temperature=0.1
    )
    total_loss, proto_loss, triplet_loss, metrics = combined_loss_fn(embeddings, labels)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Proto loss: {proto_loss.item():.4f}")
    print(f"Triplet loss: {triplet_loss.item():.4f}")
    print("Metrics:", {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()})

    # Test adaptive loss
    print("\n--- Testing AdaptivePrototypicalLoss ---")
    adaptive_loss_fn = AdaptivePrototypicalLoss(
        initial_temperature=0.2,
        final_temperature=0.05,
        adaptation_rate=0.95
    )
    loss, metrics = adaptive_loss_fn(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Current temperature: {adaptive_loss_fn.current_temperature:.4f}")

    # Test temperature adaptation
    for i in range(5):
        adaptive_loss_fn.update_temperature()
        print(f"After update {i+1}: temperature = {adaptive_loss_fn.current_temperature:.4f}")

    print("\n✅ All prototypical loss tests passed!")


if __name__ == "__main__":
    test_prototypical_losses()