"""
Utility functions for XRD diffusion validation.
"""

import torch
import numpy as np
import random
from typing import Optional


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> str:
    """
    Get appropriate device for computations.

    Args:
        device: Specific device or None for auto-detection

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def ensure_tensor_shape(tensor: torch.Tensor, expected_dims: int) -> torch.Tensor:
    """
    Ensure tensor has the expected number of dimensions by adding singleton dims.

    Args:
        tensor: Input tensor
        expected_dims: Expected number of dimensions

    Returns:
        Tensor with correct shape
    """
    while tensor.dim() < expected_dims:
        tensor = tensor.unsqueeze(0)
    return tensor


def calculate_similarity_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Calculate various similarity metrics between predictions and targets.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        Dictionary with similarity metrics
    """
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))

    # Correlation (handle edge cases)
    try:
        correlation = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0

    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'rmse': np.sqrt(mse)
    }


__all__ = [
    'set_random_seeds',
    'get_device',
    'ensure_tensor_shape',
    'calculate_similarity_metrics'
]