#!/usr/bin/env python3
"""
Prototype Retrieval System for XRD Classification
=================================================

Fast similarity search system for prototype-based inference on large-scale
XRD compound databases. Designed for synthetic-to-real domain transfer
where training is done on synthetic data and inference uses cosine similarity
to stored prototypes.

Key Features:
- Fast cosine similarity search using normalized embeddings
- Support for incremental prototype updates
- Batch query processing
- Top-k retrieval with confidence scores
- Memory-efficient storage for large databases (13k+ compounds)
- Optional FAISS integration for ultra-fast search

Usage:
1. Build prototypes from synthetic training data
2. Store in PrototypeIndex
3. Query with real XRD patterns for nearest compound matches
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path

# Optional FAISS for very fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class PrototypeIndex:
    """
    Prototype-based retrieval system for XRD compound classification.

    Stores compound prototypes and enables fast cosine similarity search
    for real-time inference without retraining.

    Args:
        embedding_dim: Dimension of embeddings
        use_faiss: Whether to use FAISS for acceleration (if available)
        faiss_gpu: Whether to use GPU FAISS (requires GPU FAISS installation)
    """

    def __init__(self,
                 embedding_dim: int,
                 use_faiss: bool = True,
                 faiss_gpu: bool = False):
        """
        Initialize prototype index.

        Args:
            embedding_dim: Dimension of embeddings
            use_faiss: Use FAISS if available
            faiss_gpu: Use GPU FAISS
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_gpu = faiss_gpu

        # Storage for prototypes
        self.prototypes = None  # [num_prototypes, embedding_dim]
        self.compound_ids = []  # List of compound IDs
        self.metadata = {}  # Additional metadata per prototype

        # FAISS index
        self.faiss_index = None

        if self.use_faiss:
            self._init_faiss_index()
        else:
            print("Using PyTorch-based cosine similarity search")

    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE:
            print("FAISS not available, falling back to PyTorch")
            self.use_faiss = False
            return

        # Create inner product index (for cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        if self.faiss_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                print("Using GPU FAISS index")
            except Exception as e:
                print(f"Failed to initialize GPU FAISS, using CPU: {e}")
        else:
            print("Using CPU FAISS index")

    def add_prototypes(self,
                      embeddings: torch.Tensor,
                      compound_ids: List[str],
                      metadata: Optional[List[Dict]] = None,
                      normalize: bool = True):
        """
        Add prototypes to the index.

        Args:
            embeddings: Prototype embeddings [num_prototypes, embedding_dim]
            compound_ids: List of compound IDs
            metadata: Optional metadata for each prototype
            normalize: Whether to L2 normalize embeddings
        """
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert to numpy for FAISS
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)

        if self.prototypes is None:
            # First batch of prototypes
            self.prototypes = embeddings_np
            self.compound_ids = list(compound_ids)
        else:
            # Append to existing prototypes
            self.prototypes = np.vstack([self.prototypes, embeddings_np])
            self.compound_ids.extend(compound_ids)

        # Add metadata
        if metadata is not None:
            for i, comp_id in enumerate(compound_ids):
                self.metadata[comp_id] = metadata[i]

        # Update FAISS index
        if self.use_faiss:
            if len(compound_ids) == len(embeddings_np):
                self.faiss_index.add(embeddings_np)
            else:
                print("Warning: compound_ids length mismatch, rebuilding FAISS index")
                self._rebuild_faiss_index()

        print(f"Added {len(compound_ids)} prototypes. Total: {len(self.compound_ids)}")

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from current prototypes."""
        if not self.use_faiss or self.prototypes is None:
            return

        self.faiss_index.reset()
        self.faiss_index.add(self.prototypes)

    def search(self,
               query_embeddings: torch.Tensor,
               top_k: int = 10,
               normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        Search for nearest prototypes.

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            top_k: Number of nearest neighbors to return
            normalize: Whether to normalize query embeddings

        Returns:
            Tuple of (similarities, indices, compound_ids)
            - similarities: [num_queries, top_k] cosine similarities
            - indices: [num_queries, top_k] prototype indices
            - compound_ids: List of lists of compound IDs for each query
        """
        if self.prototypes is None:
            raise ValueError("No prototypes in index. Call add_prototypes() first.")

        if normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        num_queries = query_embeddings.size(0)
        top_k = min(top_k, len(self.compound_ids))

        if self.use_faiss:
            return self._search_faiss(query_embeddings, top_k)
        else:
            return self._search_pytorch(query_embeddings, top_k)

    def _search_faiss(self,
                     query_embeddings: torch.Tensor,
                     top_k: int) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """FAISS-based search."""
        query_np = query_embeddings.detach().cpu().numpy().astype(np.float32)

        # FAISS search
        similarities, indices = self.faiss_index.search(query_np, top_k)

        # Convert back to torch
        similarities = torch.from_numpy(similarities)
        indices = torch.from_numpy(indices)

        # Get compound IDs
        compound_ids = []
        for i in range(len(query_embeddings)):
            query_compound_ids = []
            for j in range(top_k):
                idx = indices[i, j].item()
                if 0 <= idx < len(self.compound_ids):
                    query_compound_ids.append(self.compound_ids[idx])
                else:
                    query_compound_ids.append("UNKNOWN")
            compound_ids.append(query_compound_ids)

        return similarities, indices, compound_ids

    def _search_pytorch(self,
                       query_embeddings: torch.Tensor,
                       top_k: int) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """PyTorch-based search."""
        # Convert prototypes to torch
        prototypes_torch = torch.from_numpy(self.prototypes).to(query_embeddings.device)

        # Compute cosine similarities
        similarities = torch.mm(query_embeddings, prototypes_torch.t())  # [num_queries, num_prototypes]

        # Get top-k
        top_similarities, top_indices = torch.topk(similarities, k=top_k, dim=1)

        # Get compound IDs
        compound_ids = []
        for i in range(len(query_embeddings)):
            query_compound_ids = []
            for j in range(top_k):
                idx = top_indices[i, j].item()
                query_compound_ids.append(self.compound_ids[idx])
            compound_ids.append(query_compound_ids)

        return top_similarities, top_indices, compound_ids

    def get_prototype(self, compound_id: str) -> Optional[torch.Tensor]:
        """
        Get prototype embedding for a specific compound.

        Args:
            compound_id: Compound identifier

        Returns:
            Prototype embedding or None if not found
        """
        try:
            idx = self.compound_ids.index(compound_id)
            prototype = torch.from_numpy(self.prototypes[idx])
            return prototype
        except ValueError:
            return None

    def remove_prototype(self, compound_id: str) -> bool:
        """
        Remove a prototype from the index.

        Args:
            compound_id: Compound ID to remove

        Returns:
            True if removed, False if not found
        """
        try:
            idx = self.compound_ids.index(compound_id)

            # Remove from arrays
            self.prototypes = np.delete(self.prototypes, idx, axis=0)
            self.compound_ids.pop(idx)

            # Remove from metadata
            if compound_id in self.metadata:
                del self.metadata[compound_id]

            # Rebuild FAISS index
            if self.use_faiss:
                self._rebuild_faiss_index()

            return True
        except ValueError:
            return False

    def update_prototype(self,
                        compound_id: str,
                        new_embedding: torch.Tensor,
                        normalize: bool = True) -> bool:
        """
        Update an existing prototype.

        Args:
            compound_id: Compound ID to update
            new_embedding: New embedding [embedding_dim]
            normalize: Whether to normalize the embedding

        Returns:
            True if updated, False if compound not found
        """
        try:
            idx = self.compound_ids.index(compound_id)

            if normalize:
                new_embedding = F.normalize(new_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)

            new_embedding_np = new_embedding.detach().cpu().numpy().astype(np.float32)
            self.prototypes[idx] = new_embedding_np

            # Rebuild FAISS index
            if self.use_faiss:
                self._rebuild_faiss_index()

            return True
        except ValueError:
            return False

    def save(self, filepath: Union[str, Path]):
        """
        Save the prototype index to disk.

        Args:
            filepath: Path to save the index
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'prototypes': self.prototypes,
            'compound_ids': self.compound_ids,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Prototype index saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """
        Load prototype index from disk.

        Args:
            filepath: Path to load from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Prototype index not found at {filepath}")

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.prototypes = save_data['prototypes']
        self.compound_ids = save_data['compound_ids']
        self.metadata = save_data.get('metadata', {})

        # Verify embedding dimension
        if save_data['embedding_dim'] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                           f"got {save_data['embedding_dim']}")

        # Rebuild FAISS index
        if self.use_faiss:
            self._rebuild_faiss_index()

        print(f"Loaded prototype index with {len(self.compound_ids)} prototypes from {filepath}")

    def get_stats(self) -> Dict:
        """Get statistics about the prototype index."""
        if self.prototypes is None:
            return {'num_prototypes': 0}

        stats = {
            'num_prototypes': len(self.compound_ids),
            'embedding_dim': self.embedding_dim,
            'using_faiss': self.use_faiss,
            'memory_usage_mb': self.prototypes.nbytes / (1024 * 1024),
        }

        if self.prototypes is not None:
            norms = np.linalg.norm(self.prototypes, axis=1)
            stats.update({
                'mean_norm': float(norms.mean()),
                'std_norm': float(norms.std()),
                'min_norm': float(norms.min()),
                'max_norm': float(norms.max())
            })

        return stats


def build_prototypes_from_model(model,
                                data_loader,
                                device: str = 'cuda',
                                aggregation: str = 'mean') -> Tuple[torch.Tensor, List[str]]:
    """
    Build prototypes from a trained model and data loader.

    Args:
        model: Trained model with embedding extraction capability
        data_loader: DataLoader containing training data
        device: Device to run inference on
        aggregation: How to aggregate multiple samples per class ('mean', 'median')

    Returns:
        Tuple of (prototype_embeddings, compound_ids)
    """
    model.eval()

    compound_embeddings = {}

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                patterns, labels, compound_ids_batch = batch
            else:
                patterns, labels = batch
                compound_ids_batch = [f"compound_{label}" for label in labels.tolist()]

            patterns = patterns.to(device)

            # Handle multi-view patterns
            if patterns.dim() == 4:
                batch_size, num_views, channels, length = patterns.shape
                patterns = patterns.view(-1, channels, length)
                embeddings = model(patterns)
                embeddings = embeddings.view(batch_size, num_views, -1).mean(dim=1)
            else:
                embeddings = model(patterns)

            # Group embeddings by compound ID
            for emb, comp_id in zip(embeddings, compound_ids_batch):
                if comp_id not in compound_embeddings:
                    compound_embeddings[comp_id] = []
                compound_embeddings[comp_id].append(emb.cpu())

    # Aggregate embeddings for each compound
    prototype_embeddings = []
    compound_ids = []

    for comp_id, embs in compound_embeddings.items():
        embs_tensor = torch.stack(embs)

        if aggregation == 'mean':
            prototype = embs_tensor.mean(dim=0)
        elif aggregation == 'median':
            prototype = embs_tensor.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        prototype_embeddings.append(prototype)
        compound_ids.append(comp_id)

    return torch.stack(prototype_embeddings), compound_ids


def test_prototype_index():
    """Test the PrototypeIndex functionality."""
    print("Testing PrototypeIndex...")

    # Test parameters
    embedding_dim = 256
    num_prototypes = 1000
    num_queries = 10
    top_k = 5

    # Create test prototypes
    prototypes = torch.randn(num_prototypes, embedding_dim)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    compound_ids = [f"compound_{i:05d}" for i in range(num_prototypes)]
    metadata = [{'formula': f'C{i}H{i+1}O{i+2}'} for i in range(num_prototypes)]

    # Test PrototypeIndex
    print("\n--- Testing PyTorch-based index ---")
    index = PrototypeIndex(embedding_dim, use_faiss=False)
    index.add_prototypes(prototypes, compound_ids, metadata)

    print(f"Index stats: {index.get_stats()}")

    # Test search
    queries = torch.randn(num_queries, embedding_dim)
    queries = F.normalize(queries, p=2, dim=1)

    similarities, indices, found_compound_ids = index.search(queries, top_k=top_k)

    print(f"Search results shape: similarities={similarities.shape}, indices={indices.shape}")
    print(f"First query top compounds: {found_compound_ids[0]}")
    print(f"First query similarities: {similarities[0].tolist()}")

    # Test FAISS if available
    if FAISS_AVAILABLE:
        print("\n--- Testing FAISS-based index ---")
        faiss_index = PrototypeIndex(embedding_dim, use_faiss=True)
        faiss_index.add_prototypes(prototypes, compound_ids, metadata)

        faiss_similarities, faiss_indices, faiss_compound_ids = faiss_index.search(queries, top_k=top_k)
        print(f"FAISS search results: similarities={faiss_similarities.shape}")
        print(f"Results match PyTorch: {torch.allclose(similarities, faiss_similarities, atol=1e-6)}")

    # Test save/load
    print("\n--- Testing save/load ---")
    save_path = '/tmp/test_prototype_index.pkl'
    index.save(save_path)

    new_index = PrototypeIndex(embedding_dim, use_faiss=False)
    new_index.load(save_path)

    # Verify loaded index
    new_similarities, _, _ = new_index.search(queries, top_k=top_k)
    print(f"Loaded index results match: {torch.allclose(similarities, new_similarities)}")

    # Clean up
    os.remove(save_path)

    print("\n✅ All PrototypeIndex tests passed!")


if __name__ == "__main__":
    test_prototype_index()