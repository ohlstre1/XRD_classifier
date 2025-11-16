"""
Evaluation utilities for XRD classification
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


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


# Note: batch_accuracy removed as it's meaningless for prototypical learning
# with dynamic class indices per batch in synthetic-to-real transfer


def evaluate_cross_set(model: torch.nn.Module,
                       prototypes: dict,
                       eval_loader: torch.utils.data.DataLoader,
                       eval_ids: List[str],
                       train_ids: List[str],
                       device: torch.device,
                       k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate real data against synthetic training prototypes (synthetic-to-real transfer).

    This function evaluates how well a model trained on synthetic XRD patterns
    can identify real measured XRD patterns of the same compounds.

    For each real pattern in val/test:
    1. Find the k-nearest synthetic prototypes
    2. Check if the correct compound's synthetic prototype is in top-k
    3. Compute top-k accuracy

    Args:
        model: Trained model
        prototypes: Dictionary of synthetic training prototypes
        eval_loader: DataLoader for real evaluation set (val or test)
        eval_ids: List of evaluation compound IDs (real data)
        train_ids: List of training compound IDs (synthetic data)
        device: Device to run evaluation on
        k_values: List of k values for top-k accuracy

    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating real patterns against synthetic prototypes...")
    model.eval()

    # Get synthetic training prototypes
    train_prototypes = {k: v for k, v in prototypes.items() if k in train_ids}
    if len(train_prototypes) == 0:
        print("Warning: No training prototypes found!")
        return {f'top{k}_accuracy': 0.0 for k in k_values}

    prototype_embeddings = np.stack(list(train_prototypes.values()))
    prototype_ids = list(train_prototypes.keys())

    # For synthetic-to-real evaluation
    correct_counts = {k: 0 for k in k_values}
    total_samples = 0
    per_compound_results = {}

    with torch.no_grad():
        for xrd_patterns, labels, batch_compound_ids in tqdm(eval_loader, desc='Evaluating'):
            xrd_patterns = xrd_patterns.to(device)
            embeddings = model.backbone(xrd_patterns).cpu().numpy()

            for embedding, label, compound_id in zip(embeddings, labels, batch_compound_ids):
                # Find nearest synthetic prototypes
                similarities = np.dot(prototype_embeddings, embedding)
                top_indices = np.argsort(similarities)[::-1]
                top_prototype_ids = [prototype_ids[i] for i in top_indices]

                # Check if correct synthetic prototype is in top-k
                # For same-compound evaluation: compound_00005 should match compound_00005
                for k in k_values:
                    top_k_ids = top_prototype_ids[:k]
                    # Exact compound ID matching
                    is_correct = compound_id in top_k_ids
                    if is_correct:
                        correct_counts[k] += 1

                # Store per-compound results
                if compound_id not in per_compound_results:
                    per_compound_results[compound_id] = []
                per_compound_results[compound_id].append({
                    'top_match': top_prototype_ids[0],
                    'similarity': float(similarities[top_indices[0]]),
                    'correct': compound_id == top_prototype_ids[0]
                })

                total_samples += 1

    # Compute accuracy metrics
    results = {}
    for k in k_values:
        accuracy = correct_counts[k] / total_samples if total_samples > 0 else 0.0
        results[f'top{k}_accuracy'] = accuracy

    # Compute per-compound accuracy
    compound_accuracies = []
    for compound_id, results_list in per_compound_results.items():
        compound_accuracy = sum(r['correct'] for r in results_list) / len(results_list)
        compound_accuracies.append(compound_accuracy)

    results['total_samples'] = total_samples
    results['num_prototypes'] = len(train_prototypes)
    results['num_eval_compounds'] = len(per_compound_results)
    results['per_compound_accuracy'] = np.mean(compound_accuracies) if compound_accuracies else 0.0
    results['per_compound_std'] = np.std(compound_accuracies) if compound_accuracies else 0.0

    print(f"✅ Synthetic-to-Real Evaluation completed:")
    print(f"   Samples evaluated: {total_samples}")
    print(f"   Unique compounds evaluated: {len(per_compound_results)}")
    print(f"   Synthetic prototypes used: {len(train_prototypes)}")
    for k in k_values:
        print(f"   Top-{k} accuracy: {results[f'top{k}_accuracy']:.3f}")
    print(f"   Per-compound accuracy: {results['per_compound_accuracy']:.3f} ± {results['per_compound_std']:.3f}")

    return results


def evaluate_test_on_val_prototypes(model: torch.nn.Module,
                                     prototypes: Dict[str, np.ndarray],
                                     compound_mapping: Dict,
                                     test_ids: List[str],
                                     val_ids: List[str],
                                     device: torch.device) -> Dict:
    """
    Evaluate test set using validation prototypes.
    This is for evaluating held-out test data against learned prototypes from validation.

    Args:
        model: Trained model
        prototypes: Dictionary mapping val compound_id to prototype embeddings
        compound_mapping: Dictionary with compound information
        test_ids: List of test compound IDs to evaluate
        val_ids: List of validation compound IDs (for mapping)
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating test set on validation prototypes...")

    model.eval()

    # Get validation prototypes only
    val_prototypes = {k: v for k, v in prototypes.items() if k in val_ids}
    prototype_embeddings = np.stack(list(val_prototypes.values()))

    # Map test patterns to closest validation prototypes
    test_to_val_mapping = {}
    total_test_samples = 0

    with torch.no_grad():
        for test_id in tqdm(test_ids, desc='Finding closest validation prototypes'):
            if test_id not in compound_mapping:
                continue

            real_pattern = np.array(compound_mapping[test_id]['real_pattern'], dtype=np.float32)
            real_tensor = torch.from_numpy(real_pattern).unsqueeze(0).unsqueeze(0).to(device)

            embedding = model.backbone(real_tensor).cpu().numpy()[0]

            # Find closest validation prototype
            similarities = np.dot(prototype_embeddings, embedding)
            best_val_idx = np.argmax(similarities)
            best_val_id = list(val_prototypes.keys())[best_val_idx]

            test_to_val_mapping[test_id] = {
                'closest_val_id': best_val_id,
                'similarity': float(similarities[best_val_idx]),
                'top5_val_ids': [list(val_prototypes.keys())[i] for i in np.argsort(similarities)[-5:][::-1]]
            }
            total_test_samples += 1

    print(f"✅ Test evaluation completed:")
    print(f"   Test samples evaluated: {total_test_samples}")
    print(f"   Using {len(val_prototypes)} validation prototypes")

    # Compute average similarity scores
    if test_to_val_mapping:
        avg_similarity = np.mean([v['similarity'] for v in test_to_val_mapping.values()])
        print(f"   Average similarity to closest prototype: {avg_similarity:.3f}")
    else:
        avg_similarity = 0.0

    return {
        'total_test_samples': total_test_samples,
        'num_val_prototypes': len(val_prototypes),
        'test_to_val_mapping': test_to_val_mapping,
        'average_similarity': avg_similarity
    }