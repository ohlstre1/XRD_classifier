#!/usr/bin/env python3
"""
XRD Prototypical Classifier
===========================

Complete XRD classification model combining ResNet-18 backbone with prototypical learning.
Supports both training and inference modes with comprehensive metrics tracking.

Key features:
- ResNet-18 1D backbone for feature extraction
- Prototypical learning with optional triplet loss
- Temperature scaling for better calibration
- Inference mode for prototype-based classification
- Comprehensive metrics and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import warnings

# Import our custom modules
try:
    from .resnet1d import ResNet1D, create_resnet1d_18
    from .prototypical_loss import (
        PrototypicalLoss,
        PrototypicalWithTripletLoss,
        AdaptivePrototypicalLoss
    )
    from .arcface_head import ArcFaceHead
    from .contrastive_loss import SupervisedContrastiveLoss
except ImportError:
    # For standalone testing
    from resnet1d import ResNet1D, create_resnet1d_18
    from prototypical_loss import (
        PrototypicalLoss,
        PrototypicalWithTripletLoss,
        AdaptivePrototypicalLoss
    )
    from arcface_head import ArcFaceHead
    from contrastive_loss import SupervisedContrastiveLoss


class XRDPrototypicalClassifier(nn.Module):
    """
    Complete XRD classification model with prototypical learning.

    Architecture:
    - ResNet-18 1D backbone: [batch, 1, 4500] → [batch, embedding_dim]
    - Prototypical loss: Learns to cluster embeddings by compound class
    - Optional triplet loss: Improves local neighborhood structure

    Args:
        embedding_dim: Dimension of output embeddings
        loss_type: Type of loss function ('prototypical', 'prototypical_triplet', 'adaptive', 'arcface_supcon')
        temperature: Temperature for prototypical loss
        **loss_kwargs: Additional arguments for loss function
    """

    def __init__(self, embedding_dim: int = 256, loss_type: str = 'prototypical_triplet',
                 temperature: float = 0.1, **loss_kwargs):
        """
        Initialize XRD Prototypical Classifier.

        Args:
            embedding_dim: Output embedding dimension
            loss_type: Loss function type
            temperature: Temperature for softmax scaling
            **loss_kwargs: Additional loss function arguments
        """
        super(XRDPrototypicalClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.temperature = temperature

        # Initialize backbone
        self.backbone = create_resnet1d_18(embedding_dim=embedding_dim)

        # For ArcFace + SupCon, initialize additional components
        self.arcface_head = None
        if loss_type == 'arcface_supcon':
            # We'll need to know the number of classes, passed via loss_kwargs
            num_classes = loss_kwargs.get('num_classes', None)
            if num_classes is None:
                raise ValueError("num_classes must be provided for arcface_supcon loss type")

            self.arcface_head = ArcFaceHead(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                margin=loss_kwargs.get('arcface_margin', 0.5),
                scale=loss_kwargs.get('arcface_scale', 30.0),
                easy_margin=loss_kwargs.get('arcface_easy_margin', False)
            )

        # Initialize loss function
        self.criterion = self._create_loss_function(loss_type, temperature, **loss_kwargs)

        # Training state
        self.training_step = 0

    def _create_loss_function(self, loss_type: str, temperature: float, **kwargs) -> nn.Module:
        """
        Create loss function based on type.

        Args:
            loss_type: Type of loss function
            temperature: Temperature parameter
            **kwargs: Additional arguments

        Returns:
            Loss function module
        """
        if loss_type == 'prototypical':
            return PrototypicalLoss(temperature=temperature)

        elif loss_type == 'prototypical_triplet':
            proto_weight = kwargs.get('proto_weight', 1.0)
            triplet_weight = kwargs.get('triplet_weight', 0.5)
            triplet_margin = kwargs.get('triplet_margin', 0.2)

            return PrototypicalWithTripletLoss(
                proto_weight=proto_weight,
                triplet_weight=triplet_weight,
                triplet_margin=triplet_margin,
                temperature=temperature
            )

        elif loss_type == 'adaptive':
            initial_temp = kwargs.get('initial_temperature', 0.2)
            final_temp = kwargs.get('final_temperature', 0.05)
            adaptation_rate = kwargs.get('adaptation_rate', 0.99)

            return AdaptivePrototypicalLoss(
                initial_temperature=initial_temp,
                final_temperature=final_temp,
                adaptation_rate=adaptation_rate
            )

        elif loss_type == 'arcface_supcon':
            supcon_temperature = kwargs.get('supcon_temperature', 0.07)
            return SupervisedContrastiveLoss(temperature=supcon_temperature)

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through the model.

        Args:
            x: Input XRD patterns [batch, 1, 4500]
            labels: Class labels [batch] (required for training)

        Returns:
            Training mode: (embeddings, loss, metrics)
            Inference mode: embeddings
        """
        # Input validation
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected input shape [batch, 1, length], got {x.shape}")

        # Extract embeddings through backbone
        embeddings = self.backbone(x)  # [batch, embedding_dim]

        # Training mode: compute loss
        if labels is not None:
            loss, metrics = self._compute_loss(embeddings, labels)
            return embeddings, loss, metrics

        # Inference mode: return only embeddings
        return embeddings

    def _compute_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss and metrics.

        Args:
            embeddings: Model embeddings [batch, embedding_dim]
            labels: Ground truth labels [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if self.loss_type == 'prototypical_triplet':
            total_loss, proto_loss, triplet_loss, metrics = self.criterion(embeddings, labels)
            metrics['proto_loss_component'] = proto_loss.detach()
            metrics['triplet_loss_component'] = triplet_loss.detach()
            return total_loss, metrics

        elif self.loss_type == 'arcface_supcon':
            # For ArcFace + SupCon, we use contrastive loss during training
            # and can optionally use ArcFace head for inference/evaluation
            loss, metrics = self.criterion(embeddings, labels)
            return loss, metrics

        else:
            loss, metrics = self.criterion(embeddings, labels)
            return loss, metrics

    def update_training_state(self):
        """Update training state (for adaptive loss functions)."""
        self.training_step += 1

        # Update adaptive loss if applicable
        if hasattr(self.criterion, 'update_temperature'):
            self.criterion.update_temperature()

    def extract_features(self, x: torch.Tensor, layer_name: str = 'embedding') -> torch.Tensor:
        """
        Extract features from specific layer.

        Args:
            x: Input XRD patterns [batch, 1, 4500]
            layer_name: Name of layer to extract features from

        Returns:
            Features from specified layer
        """
        if layer_name == 'embedding':
            return self.backbone(x)
        else:
            feature_maps = self.backbone.get_feature_maps(x)
            if layer_name in feature_maps:
                return feature_maps[layer_name]
            else:
                raise ValueError(f"Unknown layer name: {layer_name}")

    def compute_similarity_matrix(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings [n1, embedding_dim]
            embeddings2: Second set of embeddings [n2, embedding_dim]

        Returns:
            Similarity matrix [n1, n2]
        """
        # Ensure embeddings are normalized
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarity
        similarity_matrix = torch.mm(embeddings1, embeddings2.t())
        return similarity_matrix

    def predict_top_k(self, query_embeddings: torch.Tensor, prototype_embeddings: torch.Tensor,
                      prototype_labels: list, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-K most similar prototypes for query embeddings.

        Args:
            query_embeddings: Query embeddings [n_queries, embedding_dim]
            prototype_embeddings: Prototype embeddings [n_prototypes, embedding_dim]
            prototype_labels: List of prototype labels
            k: Number of top predictions to return

        Returns:
            Tuple of (top_k_similarities, top_k_indices)
        """
        # Compute similarity matrix
        similarities = self.compute_similarity_matrix(query_embeddings, prototype_embeddings)

        # Get top-K for each query
        top_k_similarities, top_k_indices = torch.topk(similarities, k=k, dim=1)

        return top_k_similarities, top_k_indices

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_type': 'XRDPrototypicalClassifier',
            'backbone': 'ResNet1D-18',
            'embedding_dim': self.embedding_dim,
            'loss_type': self.loss_type,
            'temperature': self.temperature,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_step': self.training_step
        }

        # Add loss-specific information
        if hasattr(self.criterion, 'current_temperature'):
            info['current_temperature'] = self.criterion.current_temperature

        if hasattr(self.criterion, 'proto_weight'):
            info['proto_weight'] = self.criterion.proto_weight
            info['triplet_weight'] = self.criterion.triplet_weight
            info['triplet_margin'] = self.criterion.triplet_loss.margin

        if self.loss_type == 'arcface_supcon' and self.arcface_head is not None:
            info['arcface_num_classes'] = self.arcface_head.num_classes
            info['arcface_margin'] = self.arcface_head.margin
            info['arcface_scale'] = self.arcface_head.scale
            info['supcon_temperature'] = self.criterion.temperature

        return info

    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None, metrics: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'loss_type': self.loss_type,
                'temperature': self.temperature
            },
            'model_info': self.get_model_info(),
            'training_step': self.training_step
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, filepath)

    @classmethod
    def load_from_checkpoint(cls, filepath: str, device: str = 'cpu') -> 'XRDPrototypicalClassifier':
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Extract model configuration
        model_config = checkpoint.get('model_config', {})
        embedding_dim = model_config.get('embedding_dim', 256)
        loss_type = model_config.get('loss_type', 'prototypical_triplet')
        temperature = model_config.get('temperature', 0.1)

        # Create model
        model = cls(
            embedding_dim=embedding_dim,
            loss_type=loss_type,
            temperature=temperature
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training step
        if 'training_step' in checkpoint:
            model.training_step = checkpoint['training_step']

        return model


def test_xrd_classifier():
    """Test function for XRD classifier."""
    print("Testing XRD Prototypical Classifier...")

    # Create model
    model = XRDPrototypicalClassifier(
        embedding_dim=256,
        loss_type='prototypical_triplet',
        temperature=0.1,
        proto_weight=1.0,
        triplet_weight=0.5,
        triplet_margin=0.2
    )

    print(f"Model info: {model.get_model_info()}")

    # Test with random data
    batch_size = 8
    input_length = 4500
    n_classes = 4

    x = torch.randn(batch_size, 1, input_length)
    labels = torch.randint(0, n_classes, (batch_size,))

    print(f"Input shape: {x.shape}")
    print(f"Labels: {labels}")

    # Training mode
    model.train()
    embeddings, loss, metrics = model(x, labels)

    print(f"Training mode:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {list(metrics.keys())}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        test_embeddings = model(x)
        print(f"Inference mode:")
        print(f"  Embeddings shape: {test_embeddings.shape}")
        print(f"  Embedding norms: {torch.norm(test_embeddings, p=2, dim=1)}")

        # Test top-K prediction
        prototype_embeddings = torch.randn(10, 256)
        prototype_embeddings = F.normalize(prototype_embeddings, p=2, dim=1)
        prototype_labels = [f"compound_{i}" for i in range(10)]

        top_k_sim, top_k_idx = model.predict_top_k(
            test_embeddings[:2], prototype_embeddings, prototype_labels, k=3
        )
        print(f"Top-3 similarities shape: {top_k_sim.shape}")
        print(f"Top-3 indices shape: {top_k_idx.shape}")

    # Test checkpoint saving/loading
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name

    try:
        model.save_checkpoint(checkpoint_path, epoch=5, metrics={'test_acc': 0.85})
        loaded_model = XRDPrototypicalClassifier.load_from_checkpoint(checkpoint_path)
        print(f"Checkpoint test passed: {loaded_model.get_model_info()['training_step']}")
    finally:
        os.unlink(checkpoint_path)

    print("✅ XRD Classifier test passed!")


if __name__ == "__main__":
    test_xrd_classifier()