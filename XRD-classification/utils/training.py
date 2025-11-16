"""
Training utilities for XRD classification
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple


def train_epoch(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                train_ids: Optional[List[str]] = None,
                compound_mapping: Optional[Dict] = None,
                compute_accuracy_every: int = 5) -> Tuple[float, float, float, Optional[float]]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to run training on
        epoch: Current epoch number
        train_ids: List of training compound IDs
        compound_mapping: Mapping of compound IDs to metadata
        compute_accuracy_every: Compute classification accuracy every N epochs

    Returns:
        Tuple of (avg_loss, avg_proto_loss, avg_triplet_loss, classification_accuracy)
    """
    model.train()

    total_loss = 0
    total_proto_loss = 0
    total_triplet_loss = 0
    classification_accuracy = None

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (xrd_patterns, _, compound_ids) in enumerate(pbar):
        xrd_patterns = xrd_patterns.to(device)
        labels = torch.tensor([train_ids.index(cid) for cid in compound_ids], device=device)

        optimizer.zero_grad()

        embeddings, loss, metrics = model(xrd_patterns, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_proto_loss += metrics.get('proto_loss_component', loss).item()
        total_triplet_loss += metrics.get('triplet_loss_component', torch.tensor(0.0)).item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })

        model.update_training_state()

    avg_loss = total_loss / len(train_loader)
    avg_proto_loss = total_proto_loss / len(train_loader)
    avg_triplet_loss = total_triplet_loss / len(train_loader)

    if epoch % compute_accuracy_every == 0 and train_ids is not None and compound_mapping is not None:
        print("\n  Computing training classification accuracy...")
        model.eval()

        from .evaluation import compute_classification_accuracy
        train_acc_metrics = compute_classification_accuracy(
            model, train_loader, compound_mapping, train_ids, device, k_values=[1, 5]
        )
        classification_accuracy = train_acc_metrics['top1_accuracy']
        model.train()
        print(f"  Training classification accuracy: {classification_accuracy:.3f}")

    return avg_loss, avg_proto_loss, avg_triplet_loss, classification_accuracy


def validate_epoch(model: nn.Module,
                   val_loader: torch.utils.data.DataLoader,
                   device: torch.device,
                   val_ids: List[str],
                   compound_mapping: Dict) -> Tuple[float, float]:
    """
    Validate for one epoch using proper classification accuracy.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        val_ids: List of validation compound IDs
        compound_mapping: Mapping of compound IDs to metadata

    Returns:
        Tuple of (avg_loss, classification_accuracy)
    """
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for xrd_patterns, _, compound_ids in tqdm(val_loader, desc='Validation'):
            xrd_patterns = xrd_patterns.to(device)
            labels = torch.tensor([val_ids.index(cid) for cid in compound_ids], device=device)

            _, loss, metrics = model(xrd_patterns, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)

    from .evaluation import compute_classification_accuracy
    val_acc_metrics = compute_classification_accuracy(
        model, val_loader, compound_mapping, val_ids, device, k_values=[1, 5]
    )
    classification_accuracy = val_acc_metrics['top1_accuracy']

    return avg_loss, classification_accuracy


class TrainingTracker:
    """
    Helper class to track training metrics and history.
    """

    def __init__(self):
        self.best_val_accuracy = 0
        self.best_epoch = 1
        self.training_history = []

    def update(self, epoch: int, metrics: Dict):
        """
        Update tracking with new epoch results.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
        """
        self.training_history.append(metrics)

        val_acc = metrics.get('val_classification_accuracy', 0)
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch
            return True
        return False

    def get_summary(self) -> Dict:
        """Get summary of training results."""
        return {
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history
        }