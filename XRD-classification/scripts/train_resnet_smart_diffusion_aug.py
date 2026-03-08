#!/usr/bin/env python3
"""
ResNet Classifier with Smart Diffusion Augmentation (Class-Aware)
=================================================================

Key differences from train_resnet_diffusion_aug.py:
1. Class-aware augmentation — only augments underrepresented classes (< threshold samples)
2. Keeps ALL original synthetic patterns unchanged
3. Only runs diffusion model on patterns from underrepresented classes

Training strategy:
- Pre-generate diffusion-augmented patterns ONLY for classes with fewer than --aug_threshold samples
- Train on (all original synthetic + diffusion-augmented subset) + TRAIN portion of measured data
- Validate on measured data only (val split)
- Test on measured data only (test split)

Dataset: data/xrd_dataset_labeled_dtw_window.pt (13,325 paired samples)

Data split (aug_threshold=10, num_variations=3):
    synth_xrd [13,325] ── smart diffusion aug (underrepresented only) ── [~20K-40K] ─┐
                                                                                       ├──> TRAINING
    real_xrd [13,325] ──┬── 70% train (noise) ── [9,327] ────────────────────────────┘
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
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add project root for diffusion imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.resnet1d import BasicBlock1D
from diffusion.models.complete_model import DiffusionAugmentor


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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class DiffusionAugMixedXRDDataset(Dataset):
    """
    Training dataset using pre-generated diffusion-augmented synthetic patterns
    and simple Gaussian noise on real patterns.
    """

    def __init__(self, diff_aug_patterns: torch.Tensor, diff_aug_labels: torch.Tensor,
                 real_patterns: torch.Tensor, real_labels: torch.Tensor):
        # Concatenate diffusion-augmented synthetic and real train patterns
        self.patterns = torch.cat([diff_aug_patterns, real_patterns], dim=0)
        self.labels = torch.cat([diff_aug_labels, real_labels], dim=0)

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        label = self.labels[idx]

        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)  # [1, L]

        # Apply light Gaussian noise to all patterns (matching baseline)
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
    """Extract compound names from file_info and create label mappings."""
    compound_names = []
    for info in file_info:
        if isinstance(info, (list, tuple)):
            cif_name = info[0]
        else:
            cif_name = str(info)

        compound = re.sub(r'_\d+_cif\.cif$', '', cif_name)
        compound_names.append(compound)

    unique_compounds = sorted(set(compound_names))
    label_to_idx = {name: idx for idx, name in enumerate(unique_compounds)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}

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
    Load dataset and create splits, also returning DTW distance values.

    Returns:
        synth_patterns, synth_labels, dtw_values,
        real_train_patterns, real_train_labels,
        val_patterns, val_labels,
        test_patterns, test_labels,
        label_to_idx, idx_to_label
    """
    print(f"Loading dataset from {data_path}...")

    data = torch.load(data_path, map_location=device, weights_only=False)
    synth_xrd = data['synth_xrd']  # [N, 4500]
    real_xrd = data['real_xrd']    # [N, 4500]
    file_info = data['file_info']
    dtw_values = data['fast_dtw_distance']  # [N]

    total_samples = len(synth_xrd)
    print(f"Total paired samples: {total_samples}")

    if n_samples is not None and n_samples > 0:
        n_samples = min(n_samples, total_samples)
        synth_xrd = synth_xrd[:n_samples]
        real_xrd = real_xrd[:n_samples]
        file_info = file_info[:n_samples]
        dtw_values = dtw_values[:n_samples]
        print(f"Limited to {n_samples} samples for debugging")

    n = len(synth_xrd)

    # Extract labels
    compound_names, labels, label_to_idx, idx_to_label = extract_compound_labels(file_info)
    all_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Number of unique classes: {len(label_to_idx)}")

    # Normalize patterns
    synth_xrd = normalize_patterns(synth_xrd)
    real_xrd = normalize_patterns(real_xrd)

    # Ensure dtw_values is a tensor
    if not isinstance(dtw_values, torch.Tensor):
        dtw_values = torch.tensor(dtw_values, dtype=torch.float32)

    # Create shuffled indices for measured data split (same seed as baseline)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"\nMeasured data split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    synth_patterns = synth_xrd
    synth_labels = all_labels
    real_train_patterns = real_xrd[train_idx]
    real_train_labels = all_labels[train_idx]

    val_patterns = real_xrd[val_idx]
    val_labels = all_labels[val_idx]

    test_patterns = real_xrd[test_idx]
    test_labels = all_labels[test_idx]

    print(f"\nFinal dataset sizes:")
    print(f"  Synthetic: {len(synth_patterns)} (to be smart diffusion-augmented)")
    print(f"  Measured train: {len(real_train_patterns)}")
    print(f"  Validation: {len(val_patterns)}")
    print(f"  Test:       {len(test_patterns)}")

    return (synth_patterns, synth_labels, dtw_values,
            real_train_patterns, real_train_labels,
            val_patterns, val_labels,
            test_patterns, test_labels,
            label_to_idx, idx_to_label)


def generate_smart_diffusion_augmented(synth_patterns, synth_labels, dtw_values,
                                        diffusion_model_path, num_variations=3,
                                        aug_threshold=10, batch_size=256, device='cpu',
                                        max_timestep=30):
    """
    Pre-generate diffusion-augmented patterns only for underrepresented classes.

    Uses proper forward diffusion + analytical one-step denoising:
    1. Add noise via forward_diffusion(x0, t) at low timesteps
    2. Predict noise with the epsilon-model
    3. Recover x0_pred = (x_t - sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

    Args:
        synth_patterns: Synthetic XRD patterns [N, 4500]
        synth_labels: Class labels for each pattern [N]
        dtw_values: DTW distance values [N]
        diffusion_model_path: Path to trained diffusion model checkpoint
        num_variations: Number of diffusion variations per underrepresented pattern
        aug_threshold: Classes with fewer samples than this get augmented
        batch_size: Batch size for diffusion generation
        device: Device for generation
        max_timestep: Maximum diffusion timestep for augmentation (lower = less noise)

    Returns:
        (all_patterns, all_labels): Combined original + augmented patterns and labels
    """
    print(f"\n{'=' * 60}")
    print("Smart class-aware diffusion augmentation")
    print(f"{'=' * 60}")

    # Count samples per class
    label_counts = Counter(synth_labels.tolist())
    num_classes = len(label_counts)
    classes_to_augment = {cls for cls, count in label_counts.items() if count < aug_threshold}
    classes_skipped = num_classes - len(classes_to_augment)

    print(f"  Total classes: {num_classes}")
    print(f"  Aug threshold: {aug_threshold} samples")
    print(f"  Classes to augment (< {aug_threshold} samples): {len(classes_to_augment)}")
    print(f"  Classes skipped (>= {aug_threshold} samples): {classes_skipped}")

    # Find indices of patterns to augment
    indices_to_aug = [i for i, lbl in enumerate(synth_labels.tolist()) if lbl in classes_to_augment]
    patterns_skipped = len(synth_labels) - len(indices_to_aug)
    expected_new = len(indices_to_aug) * num_variations

    print(f"\n  Patterns to augment: {len(indices_to_aug)}")
    print(f"  Patterns kept as-is: {patterns_skipped}")
    print(f"  Augmented copies to generate: {expected_new}")
    print(f"  Expected total synthetic: {len(synth_patterns) + expected_new}")

    # Load diffusion model
    print(f"\n  Model: {diffusion_model_path}")
    print(f"  Device: {device}")

    diff_model = DiffusionAugmentor(
        in_channels=1,
        hidden_channels=16,
        time_embedding_dim=256,
        num_res_blocks=2,
        attention_levels=[1, 2],
        num_levels=2,
        temperature_condition=True
    ).to(device)

    checkpoint = torch.load(diffusion_model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        diff_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        diff_model.load_state_dict(checkpoint)
    print("  Diffusion model loaded successfully")
    diff_model.eval()  # Deterministic high-quality outputs

    # Start with all original synthetic patterns
    all_patterns = [synth_patterns]
    all_labels = [synth_labels]

    if len(indices_to_aug) > 0:
        # Gather subset of patterns to augment
        indices_tensor = torch.tensor(indices_to_aug, dtype=torch.long)
        patterns_subset = synth_patterns[indices_tensor]  # [M, 4500]
        dtw_subset = dtw_values[indices_tensor]            # [M]
        labels_subset = synth_labels[indices_tensor]        # [M]

        # Prepare DTW values as [M, 1]
        if dtw_subset.dim() == 1:
            dtw_subset = dtw_subset.unsqueeze(1)
        dtw_subset = dtw_subset.float().to(device)

        # Prepare patterns as [M, 1, L]
        if patterns_subset.dim() == 2:
            patterns_3d = patterns_subset.unsqueeze(1).to(device)
        else:
            patterns_3d = patterns_subset.to(device)

        # Run proper diffusion augmentation in batches on subset only
        aug_patterns_list = []
        aug_labels_list = []

        with torch.no_grad():
            for start in tqdm(range(0, len(patterns_subset), batch_size),
                              desc="Smart diffusion augmentation"):
                end = min(start + batch_size, len(patterns_subset))
                batch_patterns = patterns_3d[start:end]   # [B, 1, L]
                batch_dtw = dtw_subset[start:end]          # [B, 1]
                batch_labels = labels_subset[start:end]

                B = batch_patterns.shape[0]
                variation_list = []

                for _ in range(num_variations):
                    t_zero = torch.zeros(B, dtype=torch.long, device=device)

                    # Perturb inputs for variation
                    input_noise = torch.randn_like(batch_patterns) * 0.02
                    noisy_input = (batch_patterns + input_noise).clamp(0, 1)

                    # Direct synthetic->real transformation (model output IS x0_pred)
                    x0_pred = diff_model(noisy_input, t_zero, batch_dtw)
                    x0_pred = x0_pred.clamp(0, 1)

                    variation_list.append(x0_pred.squeeze(1).cpu())

                # Stack variations: [num_variations * B, L]
                aug_batch = torch.cat(variation_list, dim=0)
                aug_patterns_list.append(aug_batch)
                # Repeat labels for each variation
                aug_labels_list.append(batch_labels.repeat(num_variations))

        aug_patterns = torch.cat(aug_patterns_list, dim=0)
        aug_labels_cat = torch.cat(aug_labels_list, dim=0)

        # Normalize augmented patterns to [0, 1]
        aug_patterns = normalize_patterns(aug_patterns)

        all_patterns.append(aug_patterns)
        all_labels.append(aug_labels_cat)

    combined_patterns = torch.cat(all_patterns, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)

    n_augmented = len(combined_patterns) - len(synth_patterns)
    print(f"\n  Generated {n_augmented} diffusion-augmented patterns")
    print(f"  Total synthetic patterns: {len(combined_patterns)}")
    print(f"  Shape: {combined_patterns.shape}")

    return combined_patterns, combined_labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for patterns, labels in tqdm(dataloader, desc="Training", leave=False):
        patterns = patterns.to(device)
        labels = labels.to(device)

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
    """Evaluate model on a dataset."""
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

            max_k = max(top_k)
            _, topk_pred = outputs.topk(max_k, dim=1)

            for k in top_k:
                correct_k[k] += (topk_pred[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += patterns.size(0)

    acc_k = {k: 100.0 * correct_k[k] / total for k in top_k}
    return total_loss / total, acc_k


def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet Classifier with Smart Diffusion Augmentation (Class-Aware)'
    )
    parser.add_argument('--data_path', type=str,
                       default='../../data/xrd_dataset_labeled_dtw_window.pt',
                       help='Path to dataset')
    parser.add_argument('--diffusion_model_path', type=str,
                       default='../../diffusion/models/xrd_diffusion/best_model.pth',
                       help='Path to trained diffusion model checkpoint')
    parser.add_argument('--num_variations', type=int, default=3,
                       help='Number of diffusion variations per pattern (for underrepresented classes)')
    parser.add_argument('--aug_threshold', type=int, default=10,
                       help='Classes with fewer samples than this get augmented')
    parser.add_argument('--diffusion_batch_size', type=int, default=32,
                       help='Batch size for diffusion generation')
    parser.add_argument('--diffusion_max_t', type=int, default=30,
                       help='Max diffusion timestep for augmentation (lower = less noise)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of measured data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of measured data for validation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Limit samples for debugging (None = all)')
    parser.add_argument('--save_dir', type=str, default='./models/smart_diffusion_aug_classifier',
                       help='Directory to save model')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (epochs without val top-1 improvement)')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    if not args.disable_wandb:
        try:
            import wandb
            wandb.init(
                project="xrd-classification",
                name="resnet-smart-diffusion-aug",
                config=vars(args)
            )
            use_wandb = True
        except ImportError:
            print("wandb not installed, disabling logging")
            use_wandb = False
    else:
        use_wandb = False

    # Load and split data (returns synth, real_train, dtw_values separately)
    (synth_patterns, synth_labels, dtw_values,
     real_train_patterns, real_train_labels,
     val_patterns, val_labels,
     test_patterns, test_labels,
     label_to_idx, idx_to_label) = load_and_split_data(
        args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_samples=args.n_samples,
        seed=args.seed,
        device='cpu'
    )

    num_classes = len(label_to_idx)
    print(f"\nNumber of classes: {num_classes}")

    # Save label mappings
    with open(os.path.join(args.save_dir, 'label_mappings.json'), 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {str(k): v for k, v in idx_to_label.items()}
        }, f, indent=2)

    # Pre-generate smart diffusion-augmented patterns
    aug_patterns, aug_labels = generate_smart_diffusion_augmented(
        synth_patterns, synth_labels, dtw_values,
        diffusion_model_path=args.diffusion_model_path,
        num_variations=args.num_variations,
        aug_threshold=args.aug_threshold,
        batch_size=args.diffusion_batch_size,
        device=device,
        max_timestep=args.diffusion_max_t
    )

    n_augmented = len(aug_patterns) - len(synth_patterns)

    print(f"\nCombined training set:")
    print(f"  Smart diffusion-augmented synthetic: {len(aug_patterns)} samples "
          f"({len(synth_patterns)} original + {n_augmented} augmented)")
    print(f"  Measured (train): {len(real_train_patterns)} samples")
    print(f"  Total: {len(aug_patterns) + len(real_train_patterns)} samples")

    # Create datasets and dataloaders
    train_dataset = DiffusionAugMixedXRDDataset(
        aug_patterns, aug_labels,
        real_train_patterns, real_train_labels
    )
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
        print("Starting training with smart diffusion augmentation...")
    print("=" * 60)

    patience_counter = 0
    early_stopped = False

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc_k = evaluate(model, val_loader, criterion, device, top_k=(1, 5, 10))

        scheduler.step()

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

        if val_acc_k[1] > best_val_acc:
            best_val_acc = val_acc_k[1]
            patience_counter = 0
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
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{args.patience} epochs")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc_top1': val_acc_k[1],
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
        }, os.path.join(args.save_dir, 'last_checkpoint.pth'))

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs "
                  f"(no improvement for {args.patience} epochs)")
            early_stopped = True
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set (Measured Data Only)")
    print("=" * 60)

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

    # Count augmentation stats for results
    label_counts = Counter(synth_labels.tolist())
    classes_augmented = sum(1 for c in label_counts.values() if c < args.aug_threshold)
    classes_skipped = len(label_counts) - classes_augmented

    # Save final results
    total_train = len(aug_patterns) + len(real_train_patterns)
    results = {
        'config': vars(args),
        'augmentation': 'smart_diffusion',
        'aug_strategy': {
            'aug_threshold': args.aug_threshold,
            'num_variations': args.num_variations,
            'classes_augmented': classes_augmented,
            'classes_skipped': classes_skipped,
            'augmented_copies': n_augmented,
            'original_synthetic': len(synth_patterns),
            'total_synthetic': len(aug_patterns),
        },
        'real_train': len(real_train_patterns),
        'best_val_acc_top1': best_val_acc,
        'test_loss': test_loss,
        'test_acc_top1': test_acc_k[1],
        'test_acc_top5': test_acc_k[5],
        'test_acc_top10': test_acc_k[10],
        'num_classes': num_classes,
        'early_stopped': early_stopped,
        'stopped_epoch': epoch + 1,
        'train_samples': total_train,
        'val_samples': len(val_patterns),
        'test_samples': len(test_patterns),
        'history': history
    }

    with open(os.path.join(args.save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    main()
