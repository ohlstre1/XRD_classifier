"""
Evaluation utilities for XRD classification
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional


def compute_classification_accuracy(model: torch.nn.Module,
                                     data_loader: torch.utils.data.DataLoader,
                                     compound_mapping: Dict,
                                     compound_ids: List[str],
                                     device: torch.device,
                                     k_values: List[int] = [1, 5]) -> Dict:
    """
    Compute classification accuracy using prototype-based matching.

    Args:
        model: Trained model
        data_loader: DataLoader with patterns
        compound_mapping: Dictionary with compound information
        compound_ids: List of compound IDs to evaluate
        device: Device to run computation on
        k_values: List of k values for top-k accuracy

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()

    compound_embeddings = {}

    with torch.no_grad():
        for xrd_patterns, labels, batch_compound_ids in data_loader:
            xrd_patterns = xrd_patterns.to(device)
            embeddings = model.backbone(xrd_patterns).cpu().numpy()

            for i, compound_id in enumerate(batch_compound_ids):
                if compound_id not in compound_embeddings:
                    compound_embeddings[compound_id] = []
                compound_embeddings[compound_id].append(embeddings[i])

    prototypes = {}
    for compound_id, embeddings_list in compound_embeddings.items():
        if compound_id in compound_ids:
            embeddings_array = np.stack(embeddings_list)
            prototype = embeddings_array.mean(axis=0)
            prototype = prototype / np.linalg.norm(prototype)
            prototypes[compound_id] = prototype

    if len(prototypes) == 0:
        return {f'top{k}_accuracy': 0.0 for k in k_values}

    prototype_embeddings = np.stack(list(prototypes.values()))
    prototype_ids = list(prototypes.keys())

    correct_counts = {k: 0 for k in k_values}
    total_samples = 0

    with torch.no_grad():
        for compound_id in compound_ids:
            if compound_id not in compound_mapping or compound_id not in prototypes:
                continue

            real_pattern = np.array(compound_mapping[compound_id]['real_pattern'], dtype=np.float32)
            real_tensor = torch.from_numpy(real_pattern).unsqueeze(0).unsqueeze(0).to(device)

            embedding = model.backbone(real_tensor).cpu().numpy()[0]

            similarities = np.dot(prototype_embeddings, embedding)
            top_indices = np.argsort(similarities)[::-1]

            try:
                true_idx = prototype_ids.index(compound_id)
            except ValueError:
                continue

            for k in k_values:
                if true_idx in top_indices[:k]:
                    correct_counts[k] += 1

            total_samples += 1

    accuracy_metrics = {}
    for k in k_values:
        accuracy_metrics[f'top{k}_accuracy'] = correct_counts[k] / total_samples if total_samples > 0 else 0.0

    accuracy_metrics['total_samples'] = total_samples
    accuracy_metrics['num_prototypes'] = len(prototypes)

    return accuracy_metrics


def evaluate_on_real_patterns(model: torch.nn.Module,
                               prototypes: Dict[str, np.ndarray],
                               compound_mapping: Dict,
                               eval_ids: List[str],
                               device: torch.device) -> Dict:
    """
    Evaluate model on real measured patterns.

    Args:
        model: Trained model
        prototypes: Dictionary mapping compound_id to prototype embeddings
        compound_mapping: Dictionary with compound information
        eval_ids: List of compound IDs to evaluate
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating on real measured patterns...")

    model.eval()
    prototype_embeddings = np.stack(list(prototypes.values()))
    prototype_ids = list(prototypes.keys())

    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for compound_id in tqdm(eval_ids, desc='Evaluating real patterns'):
            if compound_id not in compound_mapping:
                continue

            real_pattern = np.array(compound_mapping[compound_id]['real_pattern'], dtype=np.float32)
            real_tensor = torch.from_numpy(real_pattern).unsqueeze(0).unsqueeze(0).to(device)

            embedding = model.backbone(real_tensor).cpu().numpy()[0]

            similarities = np.dot(prototype_embeddings, embedding)
            top_indices = np.argsort(similarities)[::-1]

            try:
                true_idx = prototype_ids.index(compound_id)
            except ValueError:
                continue

            if true_idx in top_indices[:1]:
                correct_top1 += 1
            if true_idx in top_indices[:5]:
                correct_top5 += 1

            total_samples += 1

    top1_accuracy = correct_top1 / total_samples if total_samples > 0 else 0
    top5_accuracy = correct_top5 / total_samples if total_samples > 0 else 0

    print(f"✅ Evaluation completed:")
    print(f"   Samples evaluated: {total_samples}")
    print(f"   Top-1 accuracy: {top1_accuracy:.3f} ({correct_top1}/{total_samples})")
    print(f"   Top-5 accuracy: {top5_accuracy:.3f} ({correct_top5}/{total_samples})")

    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': total_samples,
        'correct_top1': correct_top1,
        'correct_top5': correct_top5
    }


def compute_batch_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute batch-level accuracy.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Accuracy as a float
    """
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0