#!/usr/bin/env python3
"""
XRD Classification Models Package
================================

This package contains all the neural network models and loss functions
for XRD pattern classification using prototypical learning.

Key components:
- ResNet1D: 1D ResNet architecture adapted for XRD signals
- Prototypical Loss: Loss functions for metric learning
- XRD Classifier: Complete classification model wrapper
"""

from .resnet1d import ResNet1D, create_resnet1d_18, create_resnet1d_34, BasicBlock1D
from .prototypical_loss import (
    PrototypicalLoss,
    HardTripletLoss,
    PrototypicalWithTripletLoss,
    AdaptivePrototypicalLoss
)
from .arcface_head import ArcFaceHead, ArcFaceLoss
from .contrastive_loss import SupervisedContrastiveLoss, MultiViewContrastiveLoss, HierarchicalContrastiveLoss
from .xrd_classifier import XRDPrototypicalClassifier

__all__ = [
    # ResNet architectures
    'ResNet1D',
    'BasicBlock1D',
    'create_resnet1d_18',
    'create_resnet1d_34',

    # Loss functions
    'PrototypicalLoss',
    'HardTripletLoss',
    'PrototypicalWithTripletLoss',
    'AdaptivePrototypicalLoss',

    # ArcFace components
    'ArcFaceHead',
    'ArcFaceLoss',

    # Contrastive loss functions
    'SupervisedContrastiveLoss',
    'MultiViewContrastiveLoss',
    'HierarchicalContrastiveLoss',

    # Main classifier
    'XRDPrototypicalClassifier'
]

# Version info
__version__ = '1.0.0'
__author__ = 'XRD Classification Team'