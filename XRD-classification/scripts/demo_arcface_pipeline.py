#!/usr/bin/env python3
"""
Demo Pipeline: ArcFace + SupCon XRD Classification
==================================================

End-to-end demonstration of the new metric learning approach for synthetic-to-real
XRD domain transfer using ArcFace + Supervised Contrastive Learning.

This script demonstrates:
1. Loading synthetic XRD training data
2. Training ResNet1D backbone with ArcFace + SupCon losses
3. Building prototypes from synthetic embeddings
4. Evaluating on real XRD patterns via cosine similarity retrieval

Usage:
    python demo_arcface_pipeline.py --quick_demo --n_samples 200 --epochs 20

For full demo:
    python demo_arcface_pipeline.py --n_samples 1000 --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
import warnings

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our components
from models.resnet1d import create_resnet1d_18
from models.arcface_head import ArcFaceLoss
from models.contrastive_loss import MultiViewContrastiveLoss
from utils.augmentation import DualXRDAugmenter
from utils.retrieval import PrototypeIndex, build_prototypes_from_model
from utils.data_loading import (
    load_subset_data,
    load_real_test_data
)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


class QuickXRDDataset(Dataset):
    """Simple dataset for demo purposes."""

    def __init__(self, patterns, labels, compound_ids, augmenter=None, num_views=1):
        self.patterns = patterns
        self.labels = labels
        self.compound_ids = compound_ids
        self.augmenter = augmenter
        self.num_views = num_views

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        label = self.labels[idx]
        compound_id = self.compound_ids[idx]

        if self.augmenter and self.num_views > 1:
            # Multi-view training
            views, _ = self.augmenter.augment_pattern_mixed(pattern, num_samples=self.num_views)
            return views, label, compound_id
        else:
            # Single view
            if self.augmenter:
                augmented, _ = self.augmenter.augment_pattern_mixed(pattern, num_samples=1)
                pattern = augmented[0]
            return pattern.unsqueeze(0), label, compound_id


def train_mini_epoch(model, arcface_criterion, contrastive_criterion, train_loader, optimizer, device):
    """Train for one mini epoch."""
    model.train()
    arcface_criterion.train()

    total_loss = 0
    total_arcface_loss = 0
    total_contrastive_loss = 0
    num_batches = 0

    for batch_idx, (patterns, labels, _) in enumerate(train_loader):
        if patterns.dim() == 4:
            # Multi-view: [batch_size, num_views, 1, 4500]
            batch_size, num_views, channels, length = patterns.shape
            patterns = patterns.view(-1, channels, length)
            labels = labels.repeat_interleave(num_views)

        patterns = patterns.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        embeddings = model(patterns)

        # ArcFace loss
        arcface_loss, _ = arcface_criterion(embeddings, labels)

        # Contrastive loss
        contrastive_loss, _ = contrastive_criterion(embeddings, labels)

        # Combined loss
        total_batch_loss = arcface_loss + 0.5 * contrastive_loss

        # Backward pass
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate
        total_loss += total_batch_loss.item()
        total_arcface_loss += arcface_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        num_batches += 1

    return {
        'total_loss': total_loss / num_batches,
        'arcface_loss': total_arcface_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches
    }


def evaluate_retrieval(model, prototype_index, real_patterns, real_labels, real_compound_ids, device, top_k=[1, 5, 10]):
    """Evaluate using prototype retrieval."""
    model.eval()

    with torch.no_grad():
        real_patterns = real_patterns.to(device)
        real_embeddings = model(real_patterns.unsqueeze(1))  # Add channel dim

        # Search prototypes
        similarities, indices, retrieved_compound_ids = prototype_index.search(
            real_embeddings, top_k=max(top_k)
        )

        # Compute accuracies
        results = {}
        for k in top_k:
            correct = 0
            for i, true_compound_id in enumerate(real_compound_ids):
                retrieved_k = retrieved_compound_ids[i][:k]
                if true_compound_id in retrieved_k:
                    correct += 1

            accuracy = correct / len(real_compound_ids)
            results[f'top_{k}_accuracy'] = accuracy

        # Additional metrics
        avg_similarity = similarities.mean().item()
        results['avg_top1_similarity'] = avg_similarity

    return results


def create_demo_plots(train_losses, val_results, save_dir):
    """Create demonstration plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Training losses
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, [x['total_loss'] for x in train_losses], 'b-', label='Total Loss')
    axes[0, 0].plot(epochs, [x['arcface_loss'] for x in train_losses], 'r--', label='ArcFace Loss')
    axes[0, 0].plot(epochs, [x['contrastive_loss'] for x in train_losses], 'g--', label='Contrastive Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Validation accuracies
    val_epochs = list(val_results.keys())
    top1_accs = [val_results[e]['top_1_accuracy'] for e in val_epochs]
    top5_accs = [val_results[e]['top_5_accuracy'] for e in val_epochs]

    axes[0, 1].plot(val_epochs, top1_accs, 'bo-', label='Top-1 Accuracy')
    axes[0, 1].plot(val_epochs, top5_accs, 'ro-', label='Top-5 Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy (Real Patterns)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Final accuracies bar chart
    final_results = val_results[max(val_results.keys())]
    k_values = [1, 5, 10]
    accuracies = [final_results[f'top_{k}_accuracy'] for k in k_values if f'top_{k}_accuracy' in final_results]

    axes[1, 0].bar([f'Top-{k}' for k in k_values[:len(accuracies)]], accuracies, color='skyblue')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Final Retrieval Accuracy')
    axes[1, 0].grid(True, axis='y')

    # Domain transfer illustration
    axes[1, 1].text(0.1, 0.8, 'Domain Transfer Results:', fontsize=12, weight='bold', transform=axes[1, 1].transAxes)

    result_text = f"""Training: Synthetic XRD patterns only
Inference: Real experimental patterns
Method: Cosine similarity to prototypes

Final Performance:
• Top-1 Accuracy: {final_results.get('top_1_accuracy', 0):.1%}
• Top-5 Accuracy: {final_results.get('top_5_accuracy', 0):.1%}
• Top-10 Accuracy: {final_results.get('top_10_accuracy', 0):.1%}

This demonstrates synthetic→real transfer
without any training on real data!"""

    axes[1, 1].text(0.1, 0.1, result_text, fontsize=10, transform=axes[1, 1].transAxes,
                    verticalalignment='bottom', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'demo_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Demo plots saved to {plot_path}")

    return plot_path


def main():
    parser = argparse.ArgumentParser(description='Demo: ArcFace + SupCon XRD Pipeline')
    parser.add_argument('--quick_demo', action='store_true', help='Run quick demo with reduced parameters')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_views', type=int, default=3, help='Number of augmented views per sample')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--save_dir', type=str, default='./demo_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Quick demo settings
    if args.quick_demo:
        args.n_samples = min(args.n_samples, 200)
        args.epochs = min(args.epochs, 20)
        args.num_views = 2
        print("🚀 Running QUICK DEMO with reduced parameters")

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("🔬 ArcFace + Supervised Contrastive XRD Demo")
    print("=" * 70)

    # Load data
    print("\n📊 Loading data...")
    try:
        train_data, train_mapping = load_synthetic_data(
            subset_size=args.n_samples,
            data_dir='data/processed'
        )
        print(f"✅ Training data: {len(train_data['patterns'])} synthetic patterns")

        # Load validation data (real patterns)
        val_data, val_mapping = load_real_val_data(
            subset_size=min(args.n_samples, 300),
            data_dir='data/processed'
        )
        print(f"✅ Validation data: {len(val_data['patterns'])} real patterns")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure data files exist in data/processed/")
        return

    # Initialize augmenter
    print("\n🔄 Initializing augmentation...")
    config = {
        'augmentation': {
            'classical': {'enabled': True},
            'diffusion': {'enabled': False},
            'noise_beta_alpha': 2.0,
            'noise_beta_beta': 5.0,
            'max_noise_level': 0.05
        }
    }
    augmenter = DualXRDAugmenter(config, verbose=False)

    # Create datasets
    train_dataset = QuickXRDDataset(
        train_data['patterns'],
        train_data['labels'],
        train_data['compound_ids'],
        augmenter=augmenter,
        num_views=args.num_views
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    num_classes = len(torch.unique(train_data['labels']))
    print(f"✅ Training setup: {len(train_loader)} batches, {num_classes} classes")

    # Initialize model
    print(f"\n🏗️ Initializing model (embedding_dim={args.embedding_dim})...")
    backbone = create_resnet1d_18(embedding_dim=args.embedding_dim).to(device)

    arcface_criterion = ArcFaceLoss(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        margin=0.5,
        scale=30.0
    ).to(device)

    contrastive_criterion = MultiViewContrastiveLoss(
        temperature=0.07,
        num_views=args.num_views
    )

    # Optimizer
    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(arcface_criterion.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    print(f"✅ Model initialized with {sum(p.numel() for p in backbone.parameters()):,} parameters")

    # Training
    print(f"\n🚂 Training for {args.epochs} epochs...")
    train_losses = []
    val_results = {}

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_metrics = train_mini_epoch(
            backbone, arcface_criterion, contrastive_criterion,
            train_loader, optimizer, device
        )
        train_losses.append(train_metrics)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1:3d}/{args.epochs}: "
              f"Loss={train_metrics['total_loss']:.4f} "
              f"ArcFace={train_metrics['arcface_loss']:.4f} "
              f"SupCon={train_metrics['contrastive_loss']:.4f} "
              f"({epoch_time:.1f}s)")

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"  🔍 Building prototypes and evaluating...")

            # Build prototypes
            prototype_index = PrototypeIndex(embedding_dim=args.embedding_dim, use_faiss=False)

            # Simple prototype building
            backbone.eval()
            compound_embeddings = {}

            with torch.no_grad():
                for patterns, labels, compound_ids in train_loader:
                    if patterns.dim() == 4:
                        batch_size, num_views, channels, length = patterns.shape
                        patterns = patterns.view(-1, channels, length)

                    patterns = patterns.to(device)
                    embeddings = backbone(patterns)

                    if patterns.shape[0] != len(labels):
                        # Multi-view: average embeddings
                        embeddings = embeddings.view(len(labels), -1, args.embedding_dim).mean(dim=1)

                    for emb, comp_id in zip(embeddings, compound_ids):
                        if comp_id not in compound_embeddings:
                            compound_embeddings[comp_id] = []
                        compound_embeddings[comp_id].append(emb.cpu())

            # Build prototype index
            prototypes = []
            prototype_ids = []
            for comp_id, embs in compound_embeddings.items():
                prototype = torch.stack(embs).mean(dim=0)
                prototypes.append(prototype)
                prototype_ids.append(comp_id)

            if prototypes:
                prototype_embeddings = torch.stack(prototypes)
                prototype_index.add_prototypes(prototype_embeddings, prototype_ids)

                # Evaluate on real patterns
                val_metrics = evaluate_retrieval(
                    backbone, prototype_index,
                    val_data['patterns'], val_data['labels'], val_data['compound_ids'],
                    device, top_k=[1, 5, 10]
                )

                val_results[epoch + 1] = val_metrics

                print(f"  📈 Top-1: {val_metrics['top_1_accuracy']:.1%}, "
                      f"Top-5: {val_metrics['top_5_accuracy']:.1%}, "
                      f"Top-10: {val_metrics.get('top_10_accuracy', 0):.1%}")

    # Final results
    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS")
    print("=" * 70)

    final_val = val_results[max(val_results.keys())]

    print(f"💡 Synthetic-to-Real Domain Transfer Results:")
    print(f"   • Training: {len(train_data['patterns'])} synthetic XRD patterns")
    print(f"   • Evaluation: {len(val_data['patterns'])} real XRD patterns")
    print(f"   • Method: ArcFace + Supervised Contrastive Learning")
    print()
    print(f"🏆 Retrieval Accuracy (Real → Synthetic Prototypes):")
    print(f"   • Top-1:  {final_val['top_1_accuracy']:8.1%}")
    print(f"   • Top-5:  {final_val['top_5_accuracy']:8.1%}")
    print(f"   • Top-10: {final_val.get('top_10_accuracy', 0):8.1%}")
    print()
    print(f"📊 Average similarity to top match: {final_val['avg_top1_similarity']:.3f}")

    # Create demo plots
    if train_losses and val_results:
        print("\n📈 Creating demonstration plots...")
        plot_path = create_demo_plots(train_losses, val_results, args.save_dir)

    # Save model
    model_path = os.path.join(args.save_dir, 'demo_model.pt')
    torch.save({
        'backbone': backbone.state_dict(),
        'arcface': arcface_criterion.state_dict(),
        'config': {
            'embedding_dim': args.embedding_dim,
            'num_classes': num_classes,
            'arcface_margin': 0.5,
            'arcface_scale': 30.0
        },
        'final_results': final_val
    }, model_path)

    print(f"\n💾 Model saved to: {model_path}")
    print(f"📁 All results saved to: {args.save_dir}")

    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("\nKey takeaways:")
    print("• Trained ONLY on synthetic data")
    print("• Evaluated on real experimental patterns")
    print("• Used angular margin (ArcFace) + contrastive learning")
    print("• Achieved domain transfer via prototype retrieval")
    print("=" * 70)


if __name__ == "__main__":
    main()