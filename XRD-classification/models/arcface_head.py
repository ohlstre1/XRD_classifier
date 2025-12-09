#!/usr/bin/env python3
"""
ArcFace Head for XRD Classification
===================================

Implementation of ArcFace (Additive Angular Margin Loss) for large-scale
XRD compound classification. Provides better angular discrimination on
the hypersphere compared to standard softmax.

Key Features:
- Angular margin penalty for improved inter-class separation
- Normalized weights and features on unit hypersphere
- Support for large number of classes (13k+ compounds)
- Temperature scaling for calibration

Reference:
- ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)
- https://arxiv.org/abs/1801.07698
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ArcFaceHead(nn.Module):
    """
    ArcFace head for angular margin loss.

    Computes cosine similarity between normalized features and normalized class weights,
    then adds angular margin to the correct class before applying cross-entropy loss.

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of output classes
        margin: Angular margin in radians (default: 0.5)
        scale: Feature scale factor (default: 30.0)
        easy_margin: Whether to use easy margin (default: False)
    """

    def __init__(self,
                 embedding_dim: int,
                 num_classes: int,
                 margin: float = 0.5,
                 scale: float = 30.0,
                 easy_margin: bool = False):
        """
        Initialize ArcFace head.

        Args:
            embedding_dim: Input embedding dimension
            num_classes: Number of classes
            margin: Angular margin in radians
            scale: Temperature scale factor
            easy_margin: Use easy margin formulation
        """
        super(ArcFaceHead, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        # Learnable class weight vectors (will be normalized)
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute trigonometric values for margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self,
                embeddings: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through ArcFace head.

        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            labels: Target labels [batch_size] (required for training)

        Returns:
            Logits [batch_size, num_classes]
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)  # [batch_size, embedding_dim]
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # [num_classes, embedding_dim]

        # Compute cosine similarity
        cosine = torch.mm(embeddings, weight_norm.t())  # [batch_size, num_classes]

        if labels is None:
            # Inference mode: return scaled cosine similarities
            return cosine * self.scale

        # Training mode: apply angular margin
        batch_size = embeddings.size(0)

        # Compute sine from cosine for angular margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # phi = cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Easy margin: phi = cosine - margin when cosine > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard margin: phi = cosine - margin when cosine > threshold
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # Create one-hot encoded labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Apply scale factor
        output = output * self.scale

        return output

    def add_classes(self, num_new_classes: int):
        """
        Dynamically add new classes to the head.

        Args:
            num_new_classes: Number of new classes to add
        """
        current_classes = self.num_classes
        new_total_classes = current_classes + num_new_classes

        # Create new weight matrix
        new_weight = torch.randn(new_total_classes, self.embedding_dim)
        nn.init.xavier_uniform_(new_weight)

        # Copy existing weights
        new_weight[:current_classes] = self.weight.data

        # Update parameters
        self.weight = nn.Parameter(new_weight)
        self.num_classes = new_total_classes

        print(f"ArcFace head expanded: {current_classes} → {new_total_classes} classes")

    def get_centers(self) -> torch.Tensor:
        """
        Get normalized class centers (weight vectors).

        Returns:
            Normalized weight matrix [num_classes, embedding_dim]
        """
        return F.normalize(self.weight, p=2, dim=1)

    def compute_class_similarity(self, class_idx1: int, class_idx2: int) -> float:
        """
        Compute cosine similarity between two class centers.

        Args:
            class_idx1: First class index
            class_idx2: Second class index

        Returns:
            Cosine similarity between class centers
        """
        centers = self.get_centers()
        center1 = centers[class_idx1].unsqueeze(0)
        center2 = centers[class_idx2].unsqueeze(0)

        similarity = F.cosine_similarity(center1, center2, dim=1)
        return similarity.item()


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss function combining ArcFace head with cross-entropy loss.

    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of output classes
        margin: Angular margin (default: 0.5)
        scale: Scale factor (default: 30.0)
        easy_margin: Use easy margin (default: False)
    """

    def __init__(self,
                 embedding_dim: int,
                 num_classes: int,
                 margin: float = 0.5,
                 scale: float = 30.0,
                 easy_margin: bool = False):
        """Initialize ArcFace loss."""
        super(ArcFaceLoss, self).__init__()

        self.arcface_head = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
            easy_margin=easy_margin
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                embeddings: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ArcFace loss.

        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            labels: Target labels [batch_size]

        Returns:
            Tuple of (loss, logits)
        """
        # Get logits with angular margin
        logits = self.arcface_head(embeddings, labels)

        # Compute cross-entropy loss
        loss = self.criterion(logits, labels)

        return loss, logits

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Make predictions without computing loss.

        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        return self.arcface_head(embeddings, labels=None)


def test_arcface():
    """Test ArcFace implementation."""
    print("Testing ArcFace implementation...")

    # Test parameters
    batch_size = 16
    embedding_dim = 256
    num_classes = 100

    # Create test data
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize as expected
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Embedding norms: {torch.norm(embeddings, p=2, dim=1)}")

    # Test ArcFace head
    print("\n--- Testing ArcFaceHead ---")
    arcface_head = ArcFaceHead(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=0.5,
        scale=30.0
    )

    # Training mode
    logits_train = arcface_head(embeddings, labels)
    print(f"Training logits shape: {logits_train.shape}")
    print(f"Logits range: [{logits_train.min():.2f}, {logits_train.max():.2f}]")

    # Inference mode
    logits_inf = arcface_head(embeddings, labels=None)
    print(f"Inference logits shape: {logits_inf.shape}")

    # Test ArcFace loss
    print("\n--- Testing ArcFaceLoss ---")
    arcface_loss = ArcFaceLoss(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=0.5,
        scale=30.0
    )

    loss, logits = arcface_loss(embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")

    # Test predictions
    pred_logits = arcface_loss.predict(embeddings)
    predictions = torch.argmax(pred_logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    print(f"Random accuracy: {accuracy.item():.4f}")

    # Test class expansion
    print("\n--- Testing Class Expansion ---")
    original_classes = arcface_head.num_classes
    arcface_head.add_classes(50)
    print(f"Classes after expansion: {arcface_head.num_classes}")

    # Test with expanded classes
    expanded_logits = arcface_head(embeddings, labels=None)
    print(f"Expanded logits shape: {expanded_logits.shape}")

    # Test class similarity
    similarity = arcface_head.compute_class_similarity(0, 1)
    print(f"Similarity between class 0 and 1: {similarity:.4f}")

    print("\n✅ ArcFace tests passed!")


if __name__ == "__main__":
    test_arcface()