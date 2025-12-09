#!/usr/bin/env python3
"""
ArcFace + Supervised Contrastive Training for XRD Classification
================================================================

Multi-view metric learning approach for synthetic-to-real XRD domain transfer.

Key Components:
- Single shared ResNet1D encoder
- ArcFace head for angular margin classification
- Supervised contrastive loss over multiple augmented views
- Combined loss training on synthetic data only
- Validation on real patterns via prototype retrieval

Training Strategy:
1. For each synthetic pattern, generate N augmented views
2. Pass all views through shared ResNet1D backbone
3. Compute ArcFace loss for classification
4. Compute SupCon loss across augmented views of same compound
5. Validate on real data using cosine similarity to prototypes

Usage:
    python train_arcface_supcon.py --epochs 100 --batch_size 32 --n_samples 1000
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
import warnings
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import existing components
from models.resnet1d import create_resnet1d_18
from models.arcface_head import ArcFaceLoss
from models.contrastive_loss import MultiViewContrastiveLoss
from utils.augmentation import DualXRDAugmenter
from utils.data_loader import create_xrd_datasets
from utils.evaluation import evaluate_on_real_patterns
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")


class MultiViewXRDDataset(torch.utils.data.Dataset):
    """Dataset that generates multiple augmented views per sample."""

    def __init__(self, patterns, labels, compound_ids, augmenter, num_views=5):
        self.patterns = patterns
        self.labels = labels
        self.compound_ids = compound_ids
        self.augmenter = augmenter
        self.num_views = num_views

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        base_pattern = self.patterns[idx]
        label = self.labels[idx]
        compound_id = self.compound_ids[idx]

        # Generate multiple augmented views
        if self.augmenter is not None:
            augmented_views, _ = self.augmenter.augment_pattern_mixed(
                base_pattern, num_samples=self.num_views
            )
            # augmented_views shape: [num_views, 1, 4500]
        else:
            # No augmentation - just repeat the pattern
            augmented_views = base_pattern.unsqueeze(0).repeat(self.num_views, 1, 1)

        return augmented_views, label, compound_id


class MetricLearningTrainer:
    """Trainer for ArcFace + SupCon metric learning."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model components
        self.backbone = create_resnet1d_18(embedding_dim=config['model']['embedding_dim'])
        self.backbone.to(self.device)

        # ArcFace head
        self.arcface_criterion = ArcFaceLoss(
            embedding_dim=config['model']['embedding_dim'],
            num_classes=config['data']['n_samples'],  # Will be set properly after data loading
            margin=config['model']['arcface_margin'],
            scale=config['model']['arcface_scale']
        )
        self.arcface_criterion.to(self.device)

        # Contrastive loss
        self.contrastive_criterion = MultiViewContrastiveLoss(
            temperature=config['model']['contrastive_temperature'],
            num_views=config['training']['num_views']
        )

        # Loss weights
        self.arcface_weight = config['training']['loss_weights']['arcface']
        self.contrastive_weight = config['training']['loss_weights']['contrastive']

        # Optimizer
        all_params = list(self.backbone.parameters()) + list(self.arcface_criterion.parameters())
        self.optimizer = optim.AdamW(
            all_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['min_lr']
        )

        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.prototypes = None

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.backbone.train()
        self.arcface_criterion.train()

        total_loss = 0
        total_arcface_loss = 0
        total_contrastive_loss = 0
        total_arcface_acc = 0
        num_batches = 0

        for batch_idx, (views, labels, compound_ids) in enumerate(train_loader):
            # views: [batch_size, num_views, 1, 4500]
            # labels: [batch_size]

            batch_size, num_views, channels, length = views.shape

            # Reshape to [batch_size * num_views, 1, 4500]
            views_flat = views.view(-1, channels, length).to(self.device)
            labels = labels.to(self.device)

            # Expand labels for all views
            labels_expanded = labels.repeat_interleave(num_views)  # [batch_size * num_views]

            self.optimizer.zero_grad()

            # Forward pass through backbone
            embeddings = self.backbone(views_flat)  # [batch_size * num_views, embedding_dim]

            # ArcFace loss
            arcface_loss, arcface_logits = self.arcface_criterion(embeddings, labels_expanded)

            # Contrastive loss
            contrastive_loss, contrastive_metrics = self.contrastive_criterion(embeddings, labels_expanded)

            # Combined loss
            total_batch_loss = (self.arcface_weight * arcface_loss +
                              self.contrastive_weight * contrastive_loss)

            # Backward pass
            total_batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.arcface_criterion.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                arcface_preds = torch.argmax(arcface_logits, dim=1)
                arcface_acc = (arcface_preds == labels_expanded).float().mean()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_arcface_loss += arcface_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_arcface_acc += arcface_acc.item()
            num_batches += 1

            # Log progress
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Total Loss = {total_batch_loss.item():.4f}, "
                      f"ArcFace = {arcface_loss.item():.4f}, "
                      f"SupCon = {contrastive_loss.item():.4f}, "
                      f"Acc = {arcface_acc.item():.4f}")

        # Average metrics
        avg_metrics = {
            'train/total_loss': total_loss / num_batches,
            'train/arcface_loss': total_arcface_loss / num_batches,
            'train/contrastive_loss': total_contrastive_loss / num_batches,
            'train/arcface_accuracy': total_arcface_acc / num_batches,
            'train/learning_rate': self.optimizer.param_groups[0]['lr']
        }

        return avg_metrics

    def build_prototypes(self, train_loader):
        """Build prototypes from training data for validation."""
        self.backbone.eval()

        compound_embeddings = {}

        with torch.no_grad():
            for views, labels, compound_ids in train_loader:
                batch_size, num_views, channels, length = views.shape
                views_flat = views.view(-1, channels, length).to(self.device)

                # Get embeddings
                embeddings = self.backbone(views_flat)

                # Reshape back to [batch_size, num_views, embedding_dim]
                embeddings = embeddings.view(batch_size, num_views, -1)

                # Average across views to get compound embeddings
                compound_embs = embeddings.mean(dim=1)  # [batch_size, embedding_dim]

                # Store embeddings by compound
                for i, (label, comp_id) in enumerate(zip(labels, compound_ids)):
                    label_item = label.item()
                    if label_item not in compound_embeddings:
                        compound_embeddings[label_item] = []
                    compound_embeddings[label_item].append(compound_embs[i])

        # Compute prototype for each class
        prototypes = {}
        prototype_tensor_list = []
        class_labels = []

        for class_label, embs in compound_embeddings.items():
            # Average embeddings for this class
            class_prototype = torch.stack(embs).mean(dim=0)
            class_prototype = F.normalize(class_prototype, p=2, dim=0)

            prototypes[class_label] = class_prototype
            prototype_tensor_list.append(class_prototype)
            class_labels.append(class_label)

        self.prototypes = torch.stack(prototype_tensor_list)  # [num_classes, embedding_dim]
        self.prototype_labels = torch.tensor(class_labels)

        print(f"Built {len(prototypes)} prototypes")

    def validate_on_real(self, real_patterns, real_labels, top_k_values=[1, 5, 10]):
        """Validate using real patterns and prototype retrieval."""
        if self.prototypes is None:
            print("No prototypes built yet")
            return {}

        self.backbone.eval()

        with torch.no_grad():
            # Get embeddings for real patterns
            real_patterns = real_patterns.to(self.device)

            # Add channel dimension if missing
            if real_patterns.dim() == 2:
                real_patterns = real_patterns.unsqueeze(1)  # [batch, length] -> [batch, 1, length]

            real_embeddings = self.backbone(real_patterns)

            # Normalize embeddings
            real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

            # Compute similarities
            similarities = torch.mm(real_embeddings, prototypes_norm.t())

            # Compute top-k accuracies
            accuracies = {}
            for k in top_k_values:
                if k <= len(self.prototype_labels):
                    _, top_k_indices = torch.topk(similarities, k=k, dim=1)
                    predicted_labels = self.prototype_labels[top_k_indices.cpu()]

                    # Check if true label is in top-k
                    real_labels_expanded = real_labels.unsqueeze(1).expand(-1, k)
                    correct = (predicted_labels == real_labels_expanded).any(dim=1)
                    accuracy = correct.float().mean().item()
                    accuracies[f'val/top_{k}_accuracy'] = accuracy

        return accuracies


def load_config():
    """Load default configuration."""
    return {
        'model': {
            'embedding_dim': 512,
            'arcface_margin': 0.5,
            'arcface_scale': 30.0,
            'contrastive_temperature': 0.07
        },
        'training': {
            'num_views': 5,
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 1e-3,
            'min_lr': 1e-5,
            'weight_decay': 1e-4,
            'loss_weights': {
                'arcface': 1.0,
                'contrastive': 0.5
            }
        },
        'data': {
            'n_samples': 1000,
            'val_samples': 500
        },
        'augmentation': {
            'classical': {
                'enabled': True,
                'probability': 0.8
            },
            'diffusion': {
                'enabled': False,  # Set to True if diffusion model available
                'model_path': '/path/to/diffusion/model.pt'
            },
            'noise_beta_alpha': 2.0,
            'noise_beta_beta': 5.0,
            'max_noise_level': 0.1
        }
    }


def main():
    parser = argparse.ArgumentParser(description='ArcFace + SupCon XRD Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_views', type=int, default=5, help='Number of augmented views per sample')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--arcface_weight', type=float, default=1.0, help='ArcFace loss weight')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()

    # Load and update config
    config = load_config()
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['training']['num_views'] = args.num_views
    config['training']['loss_weights']['arcface'] = args.arcface_weight
    config['training']['loss_weights']['contrastive'] = args.contrastive_weight
    config['model']['embedding_dim'] = args.embedding_dim
    config['data']['n_samples'] = args.n_samples

    print("Configuration:")
    for section, params in config.items():
        print(f"  {section}:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    # Initialize wandb
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.init(
            project='xrd-arcface-supcon',
            config=config,
            name=f'arcface_supcon_{args.n_samples}samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

    # Create datasets using the new data loader
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_xrd_datasets(
        dataset_path='../data/xrd_dataset_labeled_dtw_window.pt',
        test_dataset_path='../data/xrd_test_dataset.pt',
        num_classes=args.n_samples
    )

    # Update number of classes in config
    config['data']['num_classes'] = args.n_samples
    config['model']['num_classes'] = args.n_samples

    # Initialize augmenter
    print("Initializing augmenter...")
    augmenter = DualXRDAugmenter(config, verbose=True)

    # Create data loaders with multi-view augmentation
    train_patterns, train_labels, train_compound_ids = train_dataset.get_train_data()
    val_patterns, val_labels, val_compound_ids = val_dataset.get_val_data()
    test_patterns, test_labels, test_compound_ids = test_dataset.get_test_data()

    train_mv_dataset = MultiViewXRDDataset(
        patterns=train_patterns,
        labels=train_labels,
        compound_ids=train_compound_ids,
        augmenter=augmenter,
        num_views=args.num_views
    )

    train_loader = DataLoader(
        train_mv_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Create validation loader (synthetic patterns for prototype building)
    val_mv_dataset = MultiViewXRDDataset(
        patterns=val_patterns,
        labels=val_labels,
        compound_ids=val_compound_ids,
        augmenter=None,  # No augmentation for validation
        num_views=1
    )

    val_loader = DataLoader(
        val_mv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test patterns: {len(test_patterns)}")

    # Initialize trainer
    trainer = MetricLearningTrainer(config)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        trainer.epoch = epoch

        # Train epoch
        train_metrics = trainer.train_epoch(train_loader)

        # Build prototypes and validate every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print("Building prototypes...")
            trainer.build_prototypes(val_loader)  # Use validation set for prototypes

            print("Evaluating on real test patterns...")
            val_metrics = trainer.validate_on_real(
                test_patterns,
                test_labels,
                top_k_values=[1, 5, 10]
            )

            # Update best validation accuracy
            if 'val/top_1_accuracy' in val_metrics:
                current_val_acc = val_metrics['val/top_1_accuracy']
                if current_val_acc > trainer.best_val_acc:
                    trainer.best_val_acc = current_val_acc

                    # Save best model
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save({
                        'backbone': trainer.backbone.state_dict(),
                        'arcface': trainer.arcface_criterion.state_dict(),
                        'prototypes': trainer.prototypes,
                        'prototype_labels': trainer.prototype_labels,
                        'config': config,
                        'epoch': epoch,
                        'val_accuracy': current_val_acc
                    }, os.path.join(args.save_dir, 'best_model.pt'))

                    print(f"New best validation accuracy: {current_val_acc:.4f}")
        else:
            val_metrics = {}

        # Update scheduler
        trainer.scheduler.step()

        # Log metrics
        all_metrics = {**train_metrics, **val_metrics}

        print(f"Epoch {epoch}/{args.epochs}:")
        for k, v in all_metrics.items():
            print(f"  {k}: {v:.4f}")

        if WANDB_AVAILABLE and not args.disable_wandb:
            wandb.log(all_metrics, step=epoch)

    print(f"Training complete! Best validation accuracy: {trainer.best_val_acc:.4f}")

    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()