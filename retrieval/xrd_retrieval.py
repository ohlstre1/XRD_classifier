#!/usr/bin/env python3
"""
XRD Pattern Retrieval Classifier
=================================

Simple and effective retrieval-based classification for XRD patterns.
Given a real (measured) XRD pattern, finds the most similar synthetic
(ideal) patterns and returns the compound names.

Performance: 92.4% Top-1, 99.6% Top-5 accuracy with Gaussian smoothing (σ=10)

Usage:
    python xrd_retrieval.py --query_pattern path/to/pattern.npy
    python xrd_retrieval.py --evaluate  # Run evaluation on test set
"""

import torch
import numpy as np
import re
import argparse
import json
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RetrievalResult:
    """Result from retrieval query."""
    compound: str
    similarity: float
    pattern_idx: int
    cif_file: str


class XRDRetriever:
    """
    Retrieval-based XRD pattern classifier.

    Uses cosine similarity with optional Gaussian smoothing to find
    the most similar synthetic patterns for a given real pattern.
    """

    def __init__(self, data_path: str, sigma: float = 10.0, device: str = 'cpu'):
        """
        Initialize the retriever.

        Args:
            data_path: Path to xrd_dataset_labeled_dtw_window.pt
            sigma: Gaussian smoothing sigma (0 for no smoothing, 10 recommended)
            device: Device for computation
        """
        self.sigma = sigma
        self.device = device

        print(f"Loading dataset from {data_path}...")
        data = torch.load(data_path, map_location=device, weights_only=False)

        self.synth_xrd = data['synth_xrd'].numpy()
        self.real_xrd = data['real_xrd'].numpy()
        self.file_info = data['file_info']
        self.dtw_distances = data['fast_dtw_distance'].numpy()

        # Extract compound names
        self.compounds = []
        self.cif_files = []
        for info in self.file_info:
            if isinstance(info, (list, tuple)):
                cif_name = info[0]
            else:
                cif_name = str(info)
            self.cif_files.append(cif_name)
            compound = re.sub(r'_\d+_cif\.cif$', '', cif_name)
            self.compounds.append(compound)

        self.compounds = np.array(self.compounds)
        self.unique_compounds = sorted(set(self.compounds))

        # Pre-compute smoothed and normalized synthetic patterns
        print(f"Pre-processing {len(self.synth_xrd)} synthetic patterns (σ={sigma})...")
        self._preprocess_database()

        print(f"Ready! {len(self.unique_compounds)} unique compounds in database.")

    def _preprocess_database(self):
        """Pre-compute smoothed and normalized synthetic patterns."""
        if self.sigma > 0:
            self.synth_processed = np.array([
                gaussian_filter1d(p, self.sigma) for p in self.synth_xrd
            ])
        else:
            self.synth_processed = self.synth_xrd.copy()

        # L2 normalize
        norms = np.linalg.norm(self.synth_processed, axis=1, keepdims=True) + 1e-8
        self.synth_normalized = self.synth_processed / norms

    def preprocess_query(self, pattern: np.ndarray) -> np.ndarray:
        """Preprocess a query pattern (smooth + normalize)."""
        if pattern.ndim == 2:
            pattern = pattern.squeeze()

        if self.sigma > 0:
            pattern = gaussian_filter1d(pattern, self.sigma)

        norm = np.linalg.norm(pattern) + 1e-8
        return pattern / norm

    def query(self, pattern: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """
        Find the top-k most similar compounds for a query pattern.

        Args:
            pattern: Query XRD pattern (1D array of length 4500)
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Preprocess query
        query_norm = self.preprocess_query(pattern)

        # Compute similarities
        similarities = np.dot(self.synth_normalized, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                compound=self.compounds[idx],
                similarity=float(similarities[idx]),
                pattern_idx=int(idx),
                cif_file=self.cif_files[idx]
            ))

        return results

    def query_batch(self, patterns: np.ndarray, top_k: int = 5) -> List[List[RetrievalResult]]:
        """Query multiple patterns at once."""
        results = []
        for pattern in tqdm(patterns, desc="Querying"):
            results.append(self.query(pattern, top_k))
        return results

    def evaluate(self, test_indices: Optional[np.ndarray] = None,
                 n_samples: Optional[int] = None) -> dict:
        """
        Evaluate retrieval accuracy on real patterns.

        Args:
            test_indices: Specific indices to test (None for random)
            n_samples: Number of samples to test (None for all)

        Returns:
            Dictionary with accuracy metrics
        """
        if test_indices is None:
            if n_samples is not None:
                test_indices = np.random.RandomState(42).choice(
                    len(self.real_xrd), min(n_samples, len(self.real_xrd)), replace=False
                )
            else:
                test_indices = np.arange(len(self.real_xrd))

        print(f"Evaluating on {len(test_indices)} samples...")

        top1_correct = 0
        top5_correct = 0
        top10_correct = 0

        for idx in tqdm(test_indices, desc="Evaluating"):
            true_compound = self.compounds[idx]
            real_pattern = self.real_xrd[idx]

            results = self.query(real_pattern, top_k=10)

            retrieved_compounds = [r.compound for r in results]

            if retrieved_compounds[0] == true_compound:
                top1_correct += 1
            if true_compound in retrieved_compounds[:5]:
                top5_correct += 1
            if true_compound in retrieved_compounds[:10]:
                top10_correct += 1

        n = len(test_indices)
        metrics = {
            'n_samples': n,
            'sigma': self.sigma,
            'top1_accuracy': 100 * top1_correct / n,
            'top5_accuracy': 100 * top5_correct / n,
            'top10_accuracy': 100 * top10_correct / n,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'top10_correct': top10_correct,
        }

        return metrics


def main():
    parser = argparse.ArgumentParser(description='XRD Pattern Retrieval Classifier')
    parser.add_argument('--data_path', type=str,
                       default='../data/xrd_dataset_labeled_dtw_window.pt',
                       help='Path to dataset')
    parser.add_argument('--sigma', type=float, default=10.0,
                       help='Gaussian smoothing sigma (0 for none, 10 recommended)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on test set')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples for evaluation (None for all)')
    parser.add_argument('--query_pattern', type=str, default=None,
                       help='Path to query pattern (.npy file)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top results to return')
    parser.add_argument('--compare_sigmas', action='store_true',
                       help='Compare different sigma values')
    args = parser.parse_args()

    if args.compare_sigmas:
        print("=== Comparing different sigma values ===\n")
        sigmas = [0, 5, 10, 20, 50]
        results = []

        for sigma in sigmas:
            retriever = XRDRetriever(args.data_path, sigma=sigma)
            metrics = retriever.evaluate(n_samples=args.n_samples or 1000)
            results.append({
                'sigma': sigma,
                **metrics
            })
            print(f"\nσ={sigma}: Top-1={metrics['top1_accuracy']:.1f}%, "
                  f"Top-5={metrics['top5_accuracy']:.1f}%, "
                  f"Top-10={metrics['top10_accuracy']:.1f}%")

        # Save results
        with open('sigma_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to sigma_comparison.json")
        return

    # Initialize retriever
    retriever = XRDRetriever(args.data_path, sigma=args.sigma)

    if args.evaluate:
        print("\n=== Evaluation Mode ===")
        metrics = retriever.evaluate(n_samples=args.n_samples)

        print(f"\n{'='*50}")
        print(f"Results (σ={args.sigma}):")
        print(f"{'='*50}")
        print(f"  Samples tested: {metrics['n_samples']}")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.2f}%")

        # Save metrics
        with open('evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to evaluation_results.json")

    elif args.query_pattern:
        print(f"\n=== Query Mode ===")
        print(f"Loading pattern from {args.query_pattern}...")

        pattern = np.load(args.query_pattern)
        results = retriever.query(pattern, top_k=args.top_k)

        print(f"\nTop {args.top_k} matches:")
        print("-" * 60)
        for i, r in enumerate(results):
            print(f"{i+1:2d}. {r.compound:<30s} (sim={r.similarity:.4f})")
            print(f"    CIF: {r.cif_file}")

    else:
        # Demo mode
        print("\n=== Demo Mode ===")
        print("Testing with a random real pattern...")

        idx = np.random.randint(len(retriever.real_xrd))
        true_compound = retriever.compounds[idx]
        pattern = retriever.real_xrd[idx]

        results = retriever.query(pattern, top_k=5)

        print(f"\nTrue compound: {true_compound}")
        print(f"\nTop 5 matches:")
        print("-" * 60)
        for i, r in enumerate(results):
            match = "✓" if r.compound == true_compound else " "
            print(f"{match} {i+1}. {r.compound:<30s} (sim={r.similarity:.4f})")


if __name__ == '__main__':
    main()
