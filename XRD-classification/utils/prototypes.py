"""
Prototype computation and management utilities for XRD classification
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def compute_prototypes(model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       device: torch.device) -> Dict[str, np.ndarray]:
    """
    Compute prototype embeddings for compounds in the data loader.

    Args:
        model: Trained model
        data_loader: DataLoader containing patterns
        device: Device to run computation on

    Returns:
        Dictionary mapping compound_id to prototype embedding
    """
    print("Computing prototypes...")
    model.eval()

    compound_embeddings = {}

    with torch.no_grad():
        for xrd_patterns, labels, compound_ids in tqdm(data_loader, desc='Computing embeddings'):
            xrd_patterns = xrd_patterns.to(device)
            embeddings = model.backbone(xrd_patterns)
            embeddings = embeddings.cpu().numpy()

            for i, compound_id in enumerate(compound_ids):
                if compound_id not in compound_embeddings:
                    compound_embeddings[compound_id] = []
                compound_embeddings[compound_id].append(embeddings[i])

    prototypes = {}
    for compound_id, embeddings_list in compound_embeddings.items():
        embeddings_array = np.stack(embeddings_list)
        prototype = embeddings_array.mean(axis=0)
        prototype = prototype / np.linalg.norm(prototype)
        prototypes[compound_id] = prototype

    print(f"✅ Computed {len(prototypes)} prototypes")
    return prototypes


def update_prototype_bank(model: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          device: torch.device) -> Dict[str, np.ndarray]:
    """
    Update prototype bank with current embeddings.

    This is an alias for compute_prototypes but can be extended
    with additional logic for prototype updates during training.

    Args:
        model: Current model
        data_loader: Data loader with patterns
        device: Device for computation

    Returns:
        Dictionary mapping compound_id to prototype embedding
    """
    return compute_prototypes(model, data_loader, device)


def find_nearest_prototypes(query_embedding: np.ndarray,
                             prototypes: Dict[str, np.ndarray],
                             k: int = 5) -> List[Tuple[str, float]]:
    """
    Find k nearest prototypes to a query embedding.

    Args:
        query_embedding: Query embedding vector
        prototypes: Dictionary of prototype embeddings
        k: Number of nearest neighbors to return

    Returns:
        List of tuples (compound_id, similarity_score)
    """
    prototype_ids = list(prototypes.keys())
    prototype_embeddings = np.stack(list(prototypes.values()))

    similarities = np.dot(prototype_embeddings, query_embedding)

    top_indices = np.argsort(similarities)[::-1][:k]

    results = [(prototype_ids[idx], similarities[idx]) for idx in top_indices]
    return results


def prototype_distance_matrix(prototypes: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute pairwise distance matrix between all prototypes.

    Args:
        prototypes: Dictionary of prototype embeddings

    Returns:
        Distance matrix of shape (n_prototypes, n_prototypes)
    """
    prototype_embeddings = np.stack(list(prototypes.values()))
    n_prototypes = len(prototype_embeddings)

    similarity_matrix = np.dot(prototype_embeddings, prototype_embeddings.T)

    distance_matrix = 1 - similarity_matrix

    return distance_matrix


class PrototypeBank:
    """
    Manages a bank of prototypes for classification.
    """

    def __init__(self):
        self.prototypes = {}
        self.compound_ids = []
        self.prototype_matrix = None

    def update(self, prototypes: Dict[str, np.ndarray]):
        """Update the prototype bank with new prototypes."""
        self.prototypes = prototypes
        self.compound_ids = list(prototypes.keys())
        self.prototype_matrix = np.stack(list(prototypes.values())) if prototypes else None

    def classify(self, embedding: np.ndarray, k: int = 1) -> List[Tuple[str, float]]:
        """
        Classify an embedding using the prototype bank.

        Args:
            embedding: Query embedding
            k: Number of top predictions to return

        Returns:
            List of (compound_id, similarity) tuples
        """
        if self.prototype_matrix is None:
            return []

        similarities = np.dot(self.prototype_matrix, embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        results = [(self.compound_ids[idx], similarities[idx]) for idx in top_indices]
        return results

    def get_prototype(self, compound_id: str) -> Optional[np.ndarray]:
        """Get a specific prototype by compound ID."""
        return self.prototypes.get(compound_id)

    def size(self) -> int:
        """Get the number of prototypes in the bank."""
        return len(self.prototypes)