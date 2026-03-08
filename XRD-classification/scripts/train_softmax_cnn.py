#!/usr/bin/env python3
"""
ResNet-based Multi-class Softmax Classifier for XRD Patterns
=============================================================

Trains a ResNet-18 backbone with softmax classification head to directly
predict compound classes from XRD patterns.

Training strategy:
- Train on synthetic (ideal) XRD patterns with diffusion augmentation
- Validate on real (measured) patterns from test split (faster)
- Test on real (measured) patterns from train_val split (comprehensive)

Uses the same train/val/test split as the diffusion model to prevent data leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import sys
import re
import json
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet1d import ResNet1D, BasicBlock1D


class ResNet1DClassifier(nn.Module):
    """
    ResNet-18 adapted for multi-class XRD classification with softmax output.

    Unlike the embedding version, this outputs raw logits for CrossEntropyLoss.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3):
        super(ResNet1DClassifier, self).__init__()

        self.in_channels = 64
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.

        Args:
            x: Input XRD patterns [batch, 1, 4500]

        Returns:
            Logits [batch, num_classes]
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)  # Raw logits, no softmax (handled by CrossEntropyLoss)

        return x


class XRDDataset(Dataset):
    """Dataset for XRD patterns with labels."""

    def __init__(self, patterns: torch.Tensor, labels: torch.Tensor):
        self.patterns = patterns
        self.labels = labels

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)  # [1, L]
        return pattern, self.labels[idx]


class AugmentedXRDDataset(Dataset):
    """
    Dataset that applies diffusion augmentation on-the-fly.
    Each sample returns n_augmentations augmented versions.
    """

    def __init__(self, patterns: torch.Tensor, labels: torch.Tensor,
                 augmenter=None, n_augmentations: int = 5, device: str = 'cpu'):
        self.patterns = patterns
        self.labels = labels
        self.augmenter = augmenter
        self.n_augmentations = n_augmentations
        self.device = device

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        label = self.labels[idx]

        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)  # [1, L]

        if self.augmenter is not None and self.augmenter.is_available:
            try:
                # Generate augmented samples
                augmented = self.augmenter.augment_pattern(
                    pattern,
                    num_samples=self.n_augmentations,
                    temp_range=(0.3, 1.0),
                    noise_timestep_range=(0, 50)
                )  # [n_aug, 1, L]

                # Repeat labels
                labels = label.repeat(self.n_augmentations) if isinstance(label, torch.Tensor) else \
                         torch.tensor([label] * self.n_augmentations)

                return augmented, labels
            except Exception as e:
                # Fallback: return original with small noise
                pass

        # Fallback: add small noise variations
        augmented = []
        for _ in range(self.n_augmentations):
            noise = torch.randn_like(pattern) * 0.02
            aug = pattern + noise
            aug = torch.clamp(aug, 0, None)
            augmented.append(aug)

        augmented = torch.stack(augmented, dim=0)  # [n_aug, 1, L]
        labels = torch.tensor([label] * self.n_augmentations)

        return augmented, labels


def extract_compound_labels(file_info: list) -> tuple:
    """
    Extract compound names from file_info and create label mappings.

    Args:
        file_info: List of (cif_filename, diff_filename) tuples

    Returns:
        (compound_names, label_to_idx, idx_to_label)
    """
    compound_names = []
    for info in file_info:
        if isinstance(info, (list, tuple)):
            cif_name = info[0]
        else:
            cif_name = str(info)

        # Extract compound name: remove _XXXXXXX_cif.cif suffix
        compound = re.sub(r'_\d+_cif\.cif$', '', cif_name)
        compound_names.append(compound)

    # Create label mappings
    unique_compounds = sorted(set(compound_names))
    label_to_idx = {name: idx for idx, name in enumerate(unique_compounds)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}

    # Convert to indices
    labels = [label_to_idx[name] for name in compound_names]

    return compound_names, labels, label_to_idx, idx_to_label


def normalize_patterns(patterns: torch.Tensor) -> torch.Tensor:
    """Min-max normalize patterns to [0, 1]."""
    min_vals = patterns.min(dim=-1, keepdim=True)[0]
    max_vals = patterns.max(dim=-1, keepdim=True)[0]
    normalized = (patterns - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized


def load_data(data_dir: str, n_samples: int = None, device: str = 'cpu'):
    """
    Load and prepare datasets following the same split as diffusion model.

    Returns:
        train_patterns, train_labels: Synthetic patterns for training
        val_patterns, val_labels: Real patterns for validation (from test split)
        test_patterns, test_labels: Real patterns for testing (from train_val split)
        label_to_idx, idx_to_label: Label mappings
    """
    print("Loading datasets...")

    # Load original dataset for file_info
    original_data = torch.load(
        os.path.join(data_dir, 'xrd_dataset_labeled_dtw_window.pt'),
        map_location=device,
        weights_only=False
    )
    file_info = original_data['file_info']

    # Load train_val split (for training synthetic and final test on real)
    train_val_data = torch.load(
        os.path.join(data_dir, 'xrd_train_val_dataset.pt'),
        map_location=device,
        weights_only=False
    )

    # Load test split (for validation on real)
    test_split_data = torch.load(
        os.path.join(data_dir, 'xrd_test_dataset.pt'),
        map_location=device,
        weights_only=False
    )

    # Get indices for label extraction
    train_val_indices = train_val_data['indices']
    test_indices = test_split_data['indices']

    # Extract labels using indices
    train_val_file_info = [file_info[i] for i in train_val_indices]
    test_file_info = [file_info[i] for i in test_indices]

    # Get compound labels for train_val
    _, train_val_labels, label_to_idx, idx_to_label = extract_compound_labels(train_val_file_info)

    # Get compound labels for test split (validation)
    val_compound_names = []
    for info in test_file_info:
        if isinstance(info, (list, tuple)):
            cif_name = info[0]
        else:
            cif_name = str(info)
        compound = re.sub(r'_\d+_cif\.cif$', '', cif_name)
        val_compound_names.append(compound)

    # Map to existing labels (compounds not in train_val get -1)
    val_labels = []
    valid_val_mask = []
    for name in val_compound_names:
        if name in label_to_idx:
            val_labels.append(label_to_idx[name])
            valid_val_mask.append(True)
        else:
            val_labels.append(-1)  # Unknown compound
            valid_val_mask.append(False)

    valid_val_mask = torch.tensor(valid_val_mask)

    # Get patterns
    train_synth = train_val_data['synth_xrd']
    test_real = train_val_data['real_xrd']  # Final test
    val_real = test_split_data['real_xrd']   # Validation

    # Normalize
    train_synth = normalize_patterns(train_synth)
    test_real = normalize_patterns(test_real)
    val_real = normalize_patterns(val_real)

    # Apply sample limit if specified
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, len(train_synth))
        train_synth = train_synth[:n_samples]
        train_val_labels = train_val_labels[:n_samples]

        # Also limit test
        n_test = min(n_samples, len(test_real))
        test_real = test_real[:n_test]
        test_labels_subset = train_val_labels[:n_test]
    else:
        test_labels_subset = train_val_labels

    # Filter validation to only known compounds
    val_real = val_real[valid_val_mask]
    val_labels = [l for l, valid in zip(val_labels, valid_val_mask.tolist()) if valid]

    if n_samples is not None and n_samples > 0:
        n_val = min(n_samples // 5 + 1, len(val_real))  # Smaller val set
        val_real = val_real[:n_val]
        val_labels = val_labels[:n_val]

    # Convert labels to tensors
    train_labels = torch.tensor(train_val_labels[:len(train_synth)], dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels_subset, dtype=torch.long)

    print(f"Training samples (synthetic): {len(train_synth)}")
    print(f"Validation samples (real): {len(val_real)}")
    print(f"Test samples (real): {len(test_real)}")
    print(f"Number of classes: {len(label_to_idx)}")

    return (train_synth, train_labels, val_real, val_labels,
            test_real, test_labels, label_to_idx, idx_to_label)


def create_diffusion_augmenter(device: str, verbose: bool = True):
    """Create diffusion augmenter if available."""
    try:
        # Add diffusion modules to path
        diffusion_base = Path(__file__).parent.parent.parent / 'diffusion'
        if str(diffusion_base) not in sys.path:
            sys.path.insert(0, str(diffusion_base))

        # Also add models subdirectory
        models_path = diffusion_base / 'models'
        if str(models_path) not in sys.path:
            sys.path.insert(0, str(models_path))

        from diffusion.process import DiffusionProcess
        from complete_model import ImprovedDiffusionDenoiser

        model_path = diffusion_base / 'models' / 'xrd_diffusion' / 'best_model.pth'

        if not model_path.exists():
            if verbose:
                print(f"Diffusion model not found at {model_path}")
            return None

        # Initialize diffusion process
        diffusion_process = DiffusionProcess(
            num_timesteps=1000,
            schedule_type='cosine',
            device=device
        )

        # Initialize model
        model = ImprovedDiffusionDenoiser(
            in_channels=1,
            hidden_channels=16,
            time_embedding_dim=256,
            num_res_blocks=2,
            attention_levels=[1, 2],
            num_levels=2,
            temperature_condition=True
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create a simple wrapper
        class SimpleDiffusionAugmenter:
            def __init__(self, model, diffusion, device):
                self.model = model
                self.diffusion = diffusion
                self.device = device
                self.is_available = True

            def augment_pattern(self, pattern, num_samples=5, temp_range=(0.3, 1.0),
                              noise_timestep_range=(0, 50)):
                if pattern.dim() == 2:
                    pattern = pattern.unsqueeze(0)  # [1, 1, L]
                elif pattern.dim() == 1:
                    pattern = pattern.unsqueeze(0).unsqueeze(0)

                pattern = pattern.to(self.device)

                augmented = []
                with torch.no_grad():
                    for _ in range(num_samples):
                        # Random temperature
                        temp = torch.rand(1).to(self.device) * (temp_range[1] - temp_range[0]) + temp_range[0]

                        # Random timestep
                        t = torch.randint(noise_timestep_range[0], noise_timestep_range[1] + 1, (1,)).to(self.device)

                        # Get noise prediction
                        noise_pred = self.model(pattern, t, temp)

                        # Create augmented pattern
                        noise_scale = 0.1 + (t.float() / 1000.0) * 0.3
                        aug = pattern + noise_pred * noise_scale
                        aug = torch.clamp(aug, 0, None)
                        augmented.append(aug)

                return torch.cat(augmented, dim=0)

        augmenter = SimpleDiffusionAugmenter(model, diffusion_process, device)

        if verbose:
            epoch = checkpoint.get('epoch', 'Unknown')
            print(f"Diffusion augmenter loaded (epoch {epoch})")

        return augmenter

    except Exception as e:
        if verbose:
            print(f"Failed to load diffusion augmenter: {e}")
        return None


def train_epoch(model, dataloader, criterion, optimizer, device, augmenter=None,
                n_augmentations=5):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for patterns, labels in tqdm(dataloader, desc="Training", leave=False):
        patterns = patterns.to(device)
        labels = labels.to(device)

        # Apply augmentation if available
        if augmenter is not None and augmenter.is_available:
            batch_augmented = []
            batch_labels = []

            for i in range(patterns.size(0)):
                pattern = patterns[i]
                label = labels[i]

                try:
                    aug = augmenter.augment_pattern(
                        pattern,
                        num_samples=n_augmentations,
                        temp_range=(0.3, 1.0),
                        noise_timestep_range=(0, 50)
                    )
                    batch_augmented.append(aug)
                    batch_labels.append(label.repeat(n_augmentations))
                except:
                    # Fallback: just use original with noise
                    aug = pattern.unsqueeze(0) + torch.randn_like(pattern.unsqueeze(0)) * 0.02
                    batch_augmented.append(aug)
                    batch_labels.append(label.unsqueeze(0))

            patterns = torch.cat(batch_augmented, dim=0)
            labels = torch.cat(batch_labels, dim=0)

        # Ensure correct shape
        if patterns.dim() == 2:
            patterns = patterns.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(patterns)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * patterns.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += patterns.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for patterns, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            patterns = patterns.to(device)
            labels = labels.to(device)

            if patterns.dim() == 2:
                patterns = patterns.unsqueeze(1)

            outputs = model(patterns)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * patterns.size(0)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1)
            correct_top5 += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += patterns.size(0)

    return total_loss / total, 100.0 * correct / total, 100.0 * correct_top5 / total


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-based XRD Softmax Classifier')
    parser.add_argument('--data_dir', type=str, default='../../data',
                       help='Data directory')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples to use (None for all)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--n_augmentations', type=int, default=5,
                       help='Number of diffusion augmentations per sample')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='./models/softmax_classifier',
                       help='Directory to save model')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--disable_augmentation', action='store_true',
                       help='Disable diffusion augmentation')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    (train_patterns, train_labels, val_patterns, val_labels,
     test_patterns, test_labels, label_to_idx, idx_to_label) = load_data(
        args.data_dir, args.n_samples, device
    )

    num_classes = len(label_to_idx)
    print(f"Number of classes: {num_classes}")

    # Save label mappings
    with open(os.path.join(args.save_dir, 'label_mappings.json'), 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {str(k): v for k, v in idx_to_label.items()}
        }, f, indent=2)

    # Create augmenter
    augmenter = None
    if not args.disable_augmentation:
        augmenter = create_diffusion_augmenter(device)

    # Create datasets and dataloaders
    train_dataset = XRDDataset(train_patterns, train_labels)
    val_dataset = XRDDataset(val_patterns, val_labels)
    test_dataset = XRDDataset(test_patterns, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    model = ResNet1DClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_top5_acc': []}

    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            augmenter=augmenter, n_augmentations=args.n_augmentations
        )

        # Validate
        val_loss, val_acc, val_top5_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-5: {val_top5_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top5_acc': val_top5_acc,
                'num_classes': num_classes,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  Saved best model (Val Acc: {val_acc:.2f}%)")

        # Always save last model (in case no improvement is ever made)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_top5_acc': val_top5_acc,
            'num_classes': num_classes,
        }, os.path.join(args.save_dir, 'last_model.pth'))

    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)

    # Load best model (or last model if best doesn't exist)
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    last_path = os.path.join(args.save_dir, 'last_model.pth')
    model_path = best_path if os.path.exists(best_path) else last_path

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    test_loss, test_acc, test_top5_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_acc:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5_acc:.2f}%")

    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_top5_acc': test_top5_acc,
        'num_classes': num_classes,
        'train_samples': len(train_patterns),
        'val_samples': len(val_patterns),
        'test_samples': len(test_patterns),
        'epochs': args.epochs,
        'n_augmentations': args.n_augmentations,
        'history': history
    }

    with open(os.path.join(args.save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    main()
