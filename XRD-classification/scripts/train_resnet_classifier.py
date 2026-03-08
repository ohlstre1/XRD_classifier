#!/usr/bin/env python3
"""
ResNet Classifier with Mixed Synthetic + Measured Training Data
================================================================

Training strategy:
- Train on ALL synthetic data + TRAIN portion of measured data
- Validate on measured data only (val split)
- Test on measured data only (test split)

Dataset: data/xrd_dataset_labeled_dtw_window.pt (13,325 paired samples)

Data split:
    synth_xrd [13,325] ─────────────────────────┐
                                                ├──> TRAINING (~22,653 samples)
    real_xrd [13,325] ──┬── 70% train (~9,328) ─┘
                        ├── 15% val   (~1,999) ──> VALIDATION
                        └── 15% test  (~1,998) ──> TESTING
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import sys
import re
import json
from tqdm import tqdm
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet1d import BasicBlock1D


class ResNet1DClassifier(nn.Module):
    """
    ResNet-18 adapted for multi-class XRD classification with softmax output.
    Outputs raw logits for CrossEntropyLoss.
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


class MixedXRDDataset(Dataset):
    """Training dataset combining synthetic and measured data with optional augmentation."""

    def __init__(self, patterns: torch.Tensor, labels: torch.Tensor, augment: bool = True):
        self.patterns = patterns
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        label = self.labels[idx]

        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)  # [1, L]

        # Simple augmentation: add small noise
        if self.augment:
            noise = torch.randn_like(pattern) * 0.02
            pattern = pattern + noise
            pattern = torch.clamp(pattern, 0, None)

        return pattern, label


class EvalXRDDataset(Dataset):
    """Evaluation dataset without augmentation."""

    def __init__(self, patterns: torch.Tensor, labels: torch.Tensor):
        self.patterns = patterns
        self.labels = labels

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        label = self.labels[idx]

        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)  # [1, L]

        return pattern, label


def extract_compound_labels(file_info: list) -> tuple:
    """
    Extract compound names from file_info and create label mappings.

    Args:
        file_info: List of (cif_filename, diff_filename) tuples

    Returns:
        (compound_names, labels, label_to_idx, idx_to_label)
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


def load_and_split_data(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15,
                        n_samples: int = None, seed: int = 42, device: str = 'cpu'):
    """
    Load dataset and create splits.

    Training: ALL synthetic + measured train portion
    Validation: measured val portion only
    Test: measured test portion only

    Args:
        data_path: Path to xrd_dataset_labeled_dtw_window.pt
        train_ratio: Ratio of measured data for training (default 0.7)
        val_ratio: Ratio for validation (default 0.15)
        n_samples: Limit samples for debugging
        seed: Random seed
        device: Device to load data to

    Returns:
        train_patterns, train_labels,
        val_patterns, val_labels,
        test_patterns, test_labels,
        label_to_idx, idx_to_label
    """
    print(f"Loading dataset from {data_path}...")

    # Load dataset
    data = torch.load(data_path, map_location=device, weights_only=False)
    synth_xrd = data['synth_xrd']  # [N, 4500]
    real_xrd = data['real_xrd']    # [N, 4500]
    file_info = data['file_info']

    total_samples = len(synth_xrd)
    print(f"Total paired samples: {total_samples}")

    # Limit samples if specified (for debugging)
    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, total_samples)
        synth_xrd = synth_xrd[:n_samples]
        real_xrd = real_xrd[:n_samples]
        file_info = file_info[:n_samples]
        print(f"Limited to {n_samples} samples for debugging")

    n = len(synth_xrd)

    # Extract labels
    compound_names, labels, label_to_idx, idx_to_label = extract_compound_labels(file_info)
    all_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Number of unique classes: {len(label_to_idx)}")

    # Normalize patterns
    synth_xrd = normalize_patterns(synth_xrd)
    real_xrd = normalize_patterns(real_xrd)

    # Create shuffled indices for measured data split
    np.random.seed(seed)
    indices = np.random.permutation(n)

    # Calculate split points
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"\nMeasured data split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    # Create training set: ALL synthetic + measured train portion
    train_patterns = torch.cat([synth_xrd, real_xrd[train_idx]], dim=0)
    train_labels = torch.cat([all_labels, all_labels[train_idx]], dim=0)

    print(f"\nCombined training set:")
    print(f"  Synthetic: {len(synth_xrd)} samples")
    print(f"  Measured (train): {len(train_idx)} samples")
    print(f"  Total: {len(train_patterns)} samples")

    # Validation: measured val portion only
    val_patterns = real_xrd[val_idx]
    val_labels = all_labels[val_idx]

    # Test: measured test portion only
    test_patterns = real_xrd[test_idx]
    test_labels = all_labels[test_idx]

    print(f"\nFinal dataset sizes:")
    print(f"  Training:   {len(train_patterns)}")
    print(f"  Validation: {len(val_patterns)}")
    print(f"  Test:       {len(test_patterns)}")

    return (train_patterns, train_labels,
            val_patterns, val_labels,
            test_patterns, test_labels,
            label_to_idx, idx_to_label,
            len(synth_xrd), len(train_idx))


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for patterns, labels in tqdm(dataloader, desc="Training", leave=False):
        patterns = patterns.to(device)
        labels = labels.to(device)

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


def evaluate(model, dataloader, criterion, device, top_k=(1, 5, 10)):
    """
    Evaluate model on a dataset.

    Returns loss and accuracy for each k in top_k.
    """
    model.eval()
    total_loss = 0
    total = 0
    correct_k = {k: 0 for k in top_k}

    with torch.no_grad():
        for patterns, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            patterns = patterns.to(device)
            labels = labels.to(device)

            if patterns.dim() == 2:
                patterns = patterns.unsqueeze(1)

            outputs = model(patterns)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * patterns.size(0)

            # Top-k accuracy
            max_k = max(top_k)
            _, topk_pred = outputs.topk(max_k, dim=1)

            for k in top_k:
                correct_k[k] += (topk_pred[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += patterns.size(0)

    acc_k = {k: 100.0 * correct_k[k] / total for k in top_k}
    return total_loss / total, acc_k


def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet Classifier with Mixed Synthetic + Measured Data'
    )
    parser.add_argument('--data_path', type=str,
                       default='../../data/xrd_dataset_labeled_dtw_window.pt',
                       help='Path to dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of measured data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of measured data for validation')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Limit samples for debugging (None = all)')
    parser.add_argument('--save_dir', type=str, default='./models/resnet_classifier',
                       help='Directory to save model')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb if enabled
    if not args.disable_wandb:
        try:
            import wandb
            wandb.init(
                project="xrd-classification",
                name="resnet-mixed-training",
                config=vars(args)
            )
            use_wandb = True
        except ImportError:
            print("wandb not installed, disabling logging")
            use_wandb = False
    else:
        use_wandb = False

    # Load and split data
    (train_patterns, train_labels,
     val_patterns, val_labels,
     test_patterns, test_labels,
     label_to_idx, idx_to_label,
     n_synth, n_real_train) = load_and_split_data(
        args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_samples=args.n_samples,
        seed=args.seed,
        device='cpu'  # Load to CPU first, move to GPU in batches
    )

    num_classes = len(label_to_idx)
    print(f"\nNumber of classes: {num_classes}")

    # Save label mappings
    with open(os.path.join(args.save_dir, 'label_mappings.json'), 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {str(k): v for k, v in idx_to_label.items()}
        }, f, indent=2)

    # Create datasets and dataloaders
    train_dataset = MixedXRDDataset(train_patterns, train_labels, augment=True)
    val_dataset = EvalXRDDataset(val_patterns, val_labels)
    test_dataset = EvalXRDDataset(test_patterns, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    model = ResNet1DClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    start_epoch = 0
    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc_top1': [], 'val_acc_top5': [], 'val_acc_top10': []
    }

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc_top1', 0))
            print(f"  Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

            # Try to load history from training_results.json
            results_path = os.path.join(os.path.dirname(args.resume), 'training_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    prev_results = json.load(f)
                    if 'history' in prev_results:
                        history = prev_results['history']
                        print(f"  Loaded training history ({len(history['train_loss'])} epochs)")
        else:
            print(f"Warning: Checkpoint not found at {args.resume}, starting from scratch")

    print("\n" + "=" * 60)
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}...")
    else:
        print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc_k = evaluate(model, val_loader, criterion, device, top_k=(1, 5, 10))

        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc_top1'].append(val_acc_k[1])
        history['val_acc_top5'].append(val_acc_k[5])
        history['val_acc_top10'].append(val_acc_k[10])

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Top-1: {val_acc_k[1]:.2f}%, "
              f"Top-5: {val_acc_k[5]:.2f}%, Top-10: {val_acc_k[10]:.2f}%")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc_top1': val_acc_k[1],
                'val_acc_top5': val_acc_k[5],
                'val_acc_top10': val_acc_k[10],
                'lr': scheduler.get_last_lr()[0]
            })

        # Save best model
        if val_acc_k[1] > best_val_acc:
            best_val_acc = val_acc_k[1]
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_top1': val_acc_k[1],
                'val_acc_top5': val_acc_k[5],
                'val_acc_top10': val_acc_k[10],
                'num_classes': num_classes,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  -> Saved best model (Val Top-1: {val_acc_k[1]:.2f}%)")

        # Save checkpoint (includes all state needed to resume)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc_top1': val_acc_k[1],
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
        }, os.path.join(args.save_dir, 'last_checkpoint.pth'))

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set (Measured Data Only)")
    print("=" * 60)

    # Load best model
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_loss, test_acc_k = evaluate(model, test_loader, criterion, device, top_k=(1, 5, 10))

    print(f"\nTest Results:")
    print(f"  Loss:    {test_loss:.4f}")
    print(f"  Top-1:   {test_acc_k[1]:.2f}%")
    print(f"  Top-5:   {test_acc_k[5]:.2f}%")
    print(f"  Top-10:  {test_acc_k[10]:.2f}%")

    if use_wandb:
        wandb.log({
            'test_loss': test_loss,
            'test_acc_top1': test_acc_k[1],
            'test_acc_top5': test_acc_k[5],
            'test_acc_top10': test_acc_k[10]
        })
        wandb.finish()

    # Save final results
    results = {
        'config': vars(args),
        'best_val_acc_top1': best_val_acc,
        'test_loss': test_loss,
        'test_acc_top1': test_acc_k[1],
        'test_acc_top5': test_acc_k[5],
        'test_acc_top10': test_acc_k[10],
        'num_classes': num_classes,
        'augmentation': 'none',
        'num_variations': 1,
        'synth_original': n_synth,
        'synth_augmented': n_synth,
        'real_train': n_real_train,
        'train_samples': len(train_patterns),
        'val_samples': len(val_patterns),
        'test_samples': len(test_patterns),
        'history': history
    }

    with open(os.path.join(args.save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    main()
