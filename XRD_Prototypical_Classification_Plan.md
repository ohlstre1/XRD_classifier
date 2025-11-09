# XRD Prototypical Classification Pipeline - Implementation Plan

## Project Overview
Build a prototypical learning pipeline for XRD pattern classification using ResNet-18 backbone with 13k compound classes, training on augmented ideal patterns and testing on real measured patterns.

---

## 1. Project Structure

```
xrd_classification/
├── data/
│   ├── raw/
│   │   ├── ideal_patterns/          # 13k ideal XRD patterns
│   │   └── real_patterns/           # 13k real measured XRD patterns (test set)
│   ├── processed/
│   │   ├── train_augmented/         # Augmented training data
│   │   ├── train_val_split.json     # Train/val compound IDs
│   │   └── compound_mapping.json    # ID to file mapping
│   └── prototypes/
│       └── validation_prototypes.pt  # Stored prototypes for 13k classes
├── models/
│   ├── resnet1d.py                  # 1D ResNet-18 implementation
│   ├── prototypical_loss.py         # Prototypical loss function
│   └── xrd_classifier.py            # Main model wrapper
├── utils/
│   ├── data_loader.py               # Dataset and DataLoader
│   ├── augmentation.py              # Augmentation wrapper
│   ├── metrics.py                   # Evaluation metrics
│   └── visualization.py             # Plot XRD patterns and results
├── scripts/
│   ├── 01_prepare_data.py           # Data preprocessing and mapping
│   ├── 02_create_augmentations.py   # Generate augmented training data
│   ├── 03_train.py                  # Training loop
│   ├── 04_compute_prototypes.py     # Generate validation prototypes
│   ├── 05_evaluate.py               # Test on real patterns
│   └── 06_inference.py              # Top-K retrieval for new samples
├── configs/
│   └── config.yaml                  # Hyperparameters and paths
├── checkpoints/                     # Model checkpoints
├── logs/                            # Training logs and tensorboard
└── results/                         # Evaluation results and visualizations
```

---

## 2. Data Preparation Pipeline

### Step 2.1: Create Compound Mapping
**File:** `scripts/01_prepare_data.py`

**Input:**
- 13k ideal XRD patterns (4500,) each
- 13k real XRD patterns (4500,) each

**Output:**
- `compound_mapping.json`:
```json
{
  "compound_0000": {
    "ideal_path": "data/raw/ideal_patterns/compound_0000.npy",
    "real_path": "data/raw/real_patterns/compound_0000.npy"
  },
  ...
}
```
- `train_val_split.json`:
```json
{
  "train": ["compound_0000", "compound_0001", ...],  # 10,400 IDs (80%)
  "val": ["compound_1200", "compound_1201", ...]     # 2,600 IDs (20%)
}
```

**Key Tasks:**
1. Scan ideal and real pattern directories
2. Verify each compound has both ideal and real patterns
3. Create unique compound IDs (0000-12999)
4. Generate 80/20 stratified split (if metadata available) or random split
5. Normalize XRD patterns to [0, 1] range

---

### Step 2.2: Generate Augmented Training Data
**File:** `scripts/02_create_augmentations.py`

**Augmentation Strategy:**
- For each training compound (10,400), create **N_augmentations** versions
- Noise levels sampled from **Beta distribution** to bias toward lower noise:
  ```python
  # Beta(2, 5) distribution gives higher probability to lower values
  noise_levels = np.random.beta(2, 5, size=N_augmentations) * 1000
  # This will generate more samples in 0-400 range, fewer in 600-1000
  ```

**Recommended N_augmentations:**
- Start with **10 augmentations** per compound → 104k training samples
- Can increase to 20-50 if training is too fast or underfitting

**Process:**
```python
for compound_id in train_compounds:
    ideal_pattern = load_ideal(compound_id)  # (4500,)
    
    for aug_idx in range(N_augmentations):
        noise_level = np.random.beta(2, 5) * 1000  # 0-1000, biased low
        augmented = your_augmentation_module.augment(ideal_pattern, noise_level)
        
        save_path = f"data/processed/train_augmented/{compound_id}_aug{aug_idx}.npy"
        np.save(save_path, augmented)
```

**Output:**
- Augmented patterns saved as `.npy` files
- Metadata: `augmentation_metadata.json` with noise levels used

---

## 3. Model Architecture

### Step 3.1: 1D ResNet-18 Implementation
**File:** `models/resnet1d.py`

**Architecture Adaptation:**

```python
class BasicBlock1D(nn.Module):
    """Basic ResNet block adapted for 1D signals"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.RU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """ResNet-18 adapted for 1D XRD signals"""
    
    def __init__(self, block=BasicBlock1D, layers=[2, 2, 2, 2], 
                 in_channels=1, embedding_dim=256):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution: (batch, 1, 4500) -> (batch, 64, 2250)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Embedding head
        self.fc = nn.Linear(512, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, 1, 4500)
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 64, 2250)
        x = self.maxpool(x)                      # (batch, 64, 1125)
        
        x = self.layer1(x)  # (batch, 64, 1125)
        x = self.layer2(x)  # (batch, 128, 563)
        x = self.layer3(x)  # (batch, 256, 282)
        x = self.layer4(x)  # (batch, 512, 141)
        
        x = self.avgpool(x)  # (batch, 512, 1)
        x = torch.flatten(x, 1)  # (batch, 512)
        
        x = self.fc(x)  # (batch, embedding_dim)
        x = self.bn_fc(x)
        
        # L2 normalize embeddings to unit sphere
        x = F.normalize(x, p=2, dim=1)
        
        return x
```

**Key Design Decisions:**
- Input shape: `(batch, 1, 4500)` - single channel 1D signal
- Embedding dimension: `256` (configurable, can try 128, 512)
- L2 normalization ensures embeddings lie on unit hypersphere
- Output: `(batch, 256)` normalized embeddings

---

### Step 3.2: Prototypical Loss
**File:** `models/prototypical_loss.py`

**Concept:**
- In each batch, compute prototype (centroid) for each compound class
- Pull embeddings toward their class prototype
- Push embeddings away from other class prototypes

**Implementation:**

```python
class PrototypicalLoss(nn.Module):
    """
    Prototypical Networks loss for metric learning
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2 normalized
            labels: (batch_size,) - compound IDs
        
        Returns:
            loss: scalar tensor
        """
        unique_labels = torch.unique(labels)
        
        # Compute prototypes for each class in the batch
        prototypes = []
        prototype_labels = []
        
        for label in unique_labels:
            mask = (labels == label)
            class_embeddings = embeddings[mask]
            prototype = class_embeddings.mean(dim=0)  # (embedding_dim,)
            prototype = F.normalize(prototype, p=2, dim=0)  # Re-normalize
            prototypes.append(prototype)
            prototype_labels.append(label)
        
        prototypes = torch.stack(prototypes)  # (n_classes_in_batch, embedding_dim)
        
        # Compute distances from each embedding to all prototypes
        # Using negative cosine similarity (since embeddings are normalized)
        distances = -torch.mm(embeddings, prototypes.t())  # (batch_size, n_classes_in_batch)
        distances = distances / self.temperature
        
        # Create target indices for cross-entropy
        target_indices = torch.zeros(len(labels), dtype=torch.long, device=embeddings.device)
        for i, label in enumerate(labels):
            target_indices[i] = (torch.tensor(prototype_labels, device=embeddings.device) == label).nonzero(as_tuple=True)[0]
        
        # Cross-entropy loss
        loss = F.cross_entropy(distances, target_indices)
        
        return loss


class PrototypicalWithTripletLoss(nn.Module):
    """
    Combined loss: Prototypical + Hard Triplet Mining
    Provides stronger gradients and faster convergence
    """
    def __init__(self, proto_weight=1.0, triplet_weight=0.5, 
                 triplet_margin=0.2, temperature=0.1):
        super().__init__()
        self.proto_loss = PrototypicalLoss(temperature)
        self.triplet_margin = triplet_margin
        self.proto_weight = proto_weight
        self.triplet_weight = triplet_weight
    
    def forward(self, embeddings, labels):
        # Prototypical loss
        proto_loss = self.proto_loss(embeddings, labels)
        
        # Hard triplet mining
        triplet_loss = self._batch_hard_triplet_loss(embeddings, labels)
        
        total_loss = self.proto_weight * proto_loss + self.triplet_weight * triplet_loss
        
        return total_loss, proto_loss, triplet_loss
    
    def _batch_hard_triplet_loss(self, embeddings, labels):
        """
        Hard triplet mining: for each anchor, select hardest positive and negative
        """
        # Pairwise distances (using cosine distance since embeddings are normalized)
        distances = 1 - torch.mm(embeddings, embeddings.t())  # (batch, batch)
        
        triplet_losses = []
        
        for i in range(len(embeddings)):
            anchor_label = labels[i]
            
            # Positive mask: same class, excluding self
            positive_mask = (labels == anchor_label)
            positive_mask[i] = False
            
            if positive_mask.sum() == 0:
                continue  # No positives for this anchor
            
            # Negative mask: different class
            negative_mask = (labels != anchor_label)
            
            if negative_mask.sum() == 0:
                continue  # No negatives for this anchor
            
            # Hardest positive: farthest sample from same class
            hardest_positive_dist = distances[i][positive_mask].max()
            
            # Hardest negative: closest sample from different class
            hardest_negative_dist = distances[i][negative_mask].min()
            
            # Triplet loss with margin
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.triplet_margin)
            triplet_losses.append(triplet_loss)
        
        if len(triplet_losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return torch.stack(triplet_losses).mean()
```

**Loss Selection Recommendation:**
- **Start with:** `PrototypicalWithTripletLoss` (more stable, faster convergence)
- **Alternative:** Pure `PrototypicalLoss` if training is stable

---

### Step 3.3: Main Model Wrapper
**File:** `models/xrd_classifier.py`

```python
class XRDPrototypicalClassifier(nn.Module):
    """
    Complete XRD classification model with prototypical learning
    """
    def __init__(self, embedding_dim=256, temperature=0.1):
        super().__init__()
        self.backbone = ResNet1D(embedding_dim=embedding_dim)
        self.criterion = PrototypicalWithTripletLoss(
            proto_weight=1.0,
            triplet_weight=0.5,
            triplet_margin=0.2,
            temperature=temperature
        )
    
    def forward(self, x, labels=None):
        """
        Args:
            x: (batch, 1, 4500) - XRD patterns
            labels: (batch,) - compound IDs (only during training)
        
        Returns:
            embeddings: (batch, embedding_dim)
            loss: scalar (if labels provided)
        """
        embeddings = self.backbone(x)
        
        if labels is not None:
            loss, proto_loss, triplet_loss = self.criterion(embeddings, labels)
            return embeddings, loss, proto_loss, triplet_loss
        
        return embeddings
```

---

## 4. Data Loading

### Step 4.1: Dataset Class
**File:** `utils/data_loader.py`

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

class XRDDataset(Dataset):
    """
    Dataset for augmented XRD patterns
    """
    def __init__(self, compound_ids, augmentation_dir, 
                 n_augmentations=10, mode='train'):
        """
        Args:
            compound_ids: List of compound IDs
            augmentation_dir: Path to augmented data
            n_augmentations: Number of augmentations per compound
            mode: 'train' or 'val'
        """
        self.compound_ids = compound_ids
        self.augmentation_dir = augmentation_dir
        self.n_augmentations = n_augmentations
        self.mode = mode
        
        # Create mapping: index -> (compound_id, aug_idx)
        self.samples = []
        for compound_id in compound_ids:
            for aug_idx in range(n_augmentations):
                self.samples.append((compound_id, aug_idx))
        
        # Create label encoding: compound_id -> integer label
        unique_ids = sorted(set(compound_ids))
        self.id_to_label = {cid: idx for idx, cid in enumerate(unique_ids)}
        self.label_to_id = {idx: cid for cid, idx in self.id_to_label.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        compound_id, aug_idx = self.samples[idx]
        
        # Load augmented pattern
        file_path = f"{self.augmentation_dir}/{compound_id}_aug{aug_idx}.npy"
        xrd_pattern = np.load(file_path).astype(np.float32)  # (4500,)
        
        # Convert to tensor and add channel dimension: (1, 4500)
        xrd_tensor = torch.from_numpy(xrd_pattern).unsqueeze(0)
        
        # Get integer label
        label = self.id_to_label[compound_id]
        
        return xrd_tensor, label, compound_id


class XRDRealDataset(Dataset):
    """
    Dataset for real measured XRD patterns (test set)
    """
    def __init__(self, compound_mapping, compound_ids):
        """
        Args:
            compound_mapping: Dict mapping compound_id -> file paths
            compound_ids: List of compound IDs to include
        """
        self.compound_mapping = compound_mapping
        self.compound_ids = compound_ids
    
    def __len__(self):
        return len(self.compound_ids)
    
    def __getitem__(self, idx):
        compound_id = self.compound_ids[idx]
        
        # Load real pattern
        real_path = self.compound_mapping[compound_id]['real_path']
        xrd_pattern = np.load(real_path).astype(np.float32)  # (4500,)
        
        # Convert to tensor: (1, 4500)
        xrd_tensor = torch.from_numpy(xrd_pattern).unsqueeze(0)
        
        return xrd_tensor, compound_id


def create_data_loaders(train_ids, val_ids, config):
    """
    Create train and validation data loaders
    """
    train_dataset = XRDDataset(
        compound_ids=train_ids,
        augmentation_dir=config['augmentation_dir'],
        n_augmentations=config['n_augmentations'],
        mode='train'
    )
    
    val_dataset = XRDDataset(
        compound_ids=val_ids,
        augmentation_dir=config['augmentation_dir'],
        n_augmentations=config['n_augmentations'],
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.id_to_label
```

---

## 5. Training Pipeline

### Step 5.1: Training Script
**File:** `scripts/03_train.py`

```python
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
from tqdm import tqdm
import os

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Single training epoch"""
    model.train()
    
    total_loss = 0
    total_proto_loss = 0
    total_triplet_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (xrd_patterns, labels, compound_ids) in enumerate(pbar):
        xrd_patterns = xrd_patterns.to(device)  # (batch, 1, 4500)
        labels = labels.to(device)  # (batch,)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings, loss, proto_loss, triplet_loss = model(xrd_patterns, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_proto_loss += proto_loss.item()
        total_triplet_loss += triplet_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'proto': f'{proto_loss.item():.4f}',
            'triplet': f'{triplet_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_proto_loss = total_proto_loss / len(train_loader)
    avg_triplet_loss = total_triplet_loss / len(train_loader)
    
    return avg_loss, avg_proto_loss, avg_triplet_loss


def validate_epoch(model, val_loader, device):
    """Validation epoch"""
    model.eval()
    
    total_loss = 0
    total_proto_loss = 0
    total_triplet_loss = 0
    
    with torch.no_grad():
        for xrd_patterns, labels, compound_ids in tqdm(val_loader, desc='Validation'):
            xrd_patterns = xrd_patterns.to(device)
            labels = labels.to(device)
            
            embeddings, loss, proto_loss, triplet_loss = model(xrd_patterns, labels)
            
            total_loss += loss.item()
            total_proto_loss += proto_loss.item()
            total_triplet_loss += triplet_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_proto_loss = total_proto_loss / len(val_loader)
    avg_triplet_loss = total_triplet_loss / len(val_loader)
    
    return avg_loss, avg_proto_loss, avg_triplet_loss


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load train/val split
    with open('data/processed/train_val_split.json', 'r') as f:
        split = json.load(f)
    
    train_ids = split['train']
    val_ids = split['val']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, id_to_label = create_data_loaders(
        train_ids, val_ids, config
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(id_to_label)}")
    
    # Create model
    model = XRDPrototypicalClassifier(
        embedding_dim=config['embedding_dim'],
        temperature=config['temperature']
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir='logs')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_proto, train_triplet = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_proto, val_triplet = validate_epoch(
            model, val_loader, device
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        print(f"\nTrain Loss: {train_loss:.4f} (Proto: {train_proto:.4f}, Triplet: {train_triplet:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Proto: {val_proto:.4f}, Triplet: {val_triplet:.4f})")
        print(f"Learning Rate: {current_lr:.6f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_proto', train_proto, epoch)
        writer.add_scalar('Loss/val_proto', val_proto, epoch)
        writer.add_scalar('Loss/train_triplet', train_triplet, epoch)
        writer.add_scalar('Loss/val_triplet', val_triplet, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        # Save latest
        torch.save(checkpoint, 'checkpoints/latest.pth')
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'checkpoints/best.pth')
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic
        if epoch % config['save_every'] == 0:
            torch.save(checkpoint, f'checkpoints/epoch_{epoch}.pth')
    
    writer.close()
    print("\n✓ Training completed!")


if __name__ == '__main__':
    main()
```

---

## 6. Prototype Computation

### Step 6.1: Compute Validation Prototypes
**File:** `scripts/04_compute_prototypes.py`

```python
import torch
import numpy as np
import json
from tqdm import tqdm

def compute_prototypes(model, val_loader, device, n_augmentations=10):
    """
    Compute prototype embeddings for each compound class
    by averaging embeddings from all augmented versions
    """
    model.eval()
    
    # Dictionary to accumulate embeddings per compound
    compound_embeddings = {}
    
    with torch.no_grad():
        for xrd_patterns, labels, compound_ids in tqdm(val_loader, desc='Computing embeddings'):
            xrd_patterns = xrd_patterns.to(device)
            
            # Get embeddings
            embeddings = model.backbone(xrd_patterns)  # (batch, embedding_dim)
            embeddings = embeddings.cpu().numpy()
            
            # Group by compound_id
            for i, compound_id in enumerate(compound_ids):
                if compound_id not in compound_embeddings:
                    compound_embeddings[compound_id] = []
                compound_embeddings[compound_id].append(embeddings[i])
    
    # Compute prototypes (mean of all augmented embeddings)
    prototypes = {}
    
    for compound_id, embeddings_list in tqdm(compound_embeddings.items(), 
                                             desc='Computing prototypes'):
        embeddings_array = np.stack(embeddings_list)  # (n_samples, embedding_dim)
        prototype = embeddings_array.mean(axis=0)  # (embedding_dim,)
        
        # Re-normalize to unit sphere
        prototype = prototype / np.linalg.norm(prototype)
        
        prototypes[compound_id] = prototype
    
    return prototypes


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load train/val split
    with open('data/processed/train_val_split.json', 'r') as f:
        split = json.load(f)
    
    val_ids = split['val']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create validation data loader
    _, val_loader, _ = create_data_loaders(
        train_ids=split['train'],  # Not used, just for consistency
        val_ids=val_ids,
        config=config
    )
    
    # Load trained model
    model = XRDPrototypicalClassifier(
        embedding_dim=config['embedding_dim'],
        temperature=config['temperature']
    ).to(device)
    
    checkpoint = torch.load('checkpoints/best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Compute prototypes
    print(f"\nComputing prototypes for {len(val_ids)} validation compounds...")
    prototypes = compute_prototypes(model, val_loader, device, config['n_augmentations'])
    
    # Save prototypes
    output_path = 'data/prototypes/validation_prototypes.pt'
    os.makedirs('data/prototypes', exist_ok=True)
    
    torch.save({
        'prototypes': prototypes,
        'compound_ids': list(prototypes.keys()),
        'embedding_dim': config['embedding_dim']
    }, output_path)
    
    print(f"✓ Saved {len(prototypes)} prototypes to {output_path}")
    
    # Verify prototype properties
    prototype_array = np.stack(list(prototypes.values()))
    print(f"\nPrototype statistics:")
    print(f"  Shape: {prototype_array.shape}")
    print(f"  Mean norm: {np.linalg.norm(prototype_array, axis=1).mean():.6f}")
    print(f"  Std norm: {np.linalg.norm(prototype_array, axis=1).std():.6f}")


if __name__ == '__main__':
    main()
```

---

## 7. Evaluation on Real Test Data

### Step 7.1: Evaluation Metrics
**File:** `utils/metrics.py`

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_topk_accuracy(similarities, true_label_idx, k):
    """
    Compute top-k accuracy
    
    Args:
        similarities: (n_test,) array of similarity scores
        true_label_idx: Index of true label in similarity array
        k: Top-k to consider
    
    Returns:
        1 if true label in top-k, 0 otherwise
    """
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return 1 if true_label_idx in top_k_indices else 0


def compute_reciprocal_rank(similarities, true_label_idx):
    """
    Compute reciprocal rank (1 / rank of true label)
    
    Returns:
        Reciprocal rank (1 for rank 1, 0.5 for rank 2, etc.)
    """
    sorted_indices = np.argsort(similarities)[::-1]
    rank = np.where(sorted_indices == true_label_idx)[0][0] + 1  # 1-indexed
    return 1.0 / rank


def evaluate_retrieval(test_embeddings, test_labels, prototypes, prototype_ids, k_values=[1, 5, 10, 20]):
    """
    Evaluate top-k retrieval performance
    
    Args:
        test_embeddings: (n_test, embedding_dim) array
        test_labels: (n_test,) list of compound IDs
        prototypes: (n_prototypes, embedding_dim) array
        prototype_ids: (n_prototypes,) list of compound IDs
        k_values: List of k values for top-k accuracy
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    n_test = len(test_embeddings)
    
    # Initialize metrics
    topk_correct = {k: 0 for k in k_values}
    reciprocal_ranks = []
    all_top1_predictions = []
    
    print(f"Evaluating {n_test} test samples...")
    
    for i in range(n_test):
        test_emb = test_embeddings[i]
        true_label = test_labels[i]
        
        # Compute cosine similarities with all prototypes
        similarities = np.dot(prototypes, test_emb)  # (n_prototypes,)
        
        # Find index of true label in prototype list
        try:
            true_label_idx = prototype_ids.index(true_label)
        except ValueError:
            print(f"Warning: True label {true_label} not in prototype list")
            continue
        
        # Top-k accuracy
        for k in k_values:
            topk_correct[k] += compute_topk_accuracy(similarities, true_label_idx, k)
        
        # Reciprocal rank
        rr = compute_reciprocal_rank(similarities, true_label_idx)
        reciprocal_ranks.append(rr)
        
        # Top-1 prediction for confusion matrix
        top1_idx = np.argmax(similarities)
        all_top1_predictions.append(prototype_ids[top1_idx])
    
    # Compute final metrics
    metrics = {
        'top_k_accuracy': {k: topk_correct[k] / n_test for k in k_values},
        'mean_reciprocal_rank': np.mean(reciprocal_ranks),
        'top1_predictions': all_top1_predictions,
        'true_labels': test_labels
    }
    
    return metrics


def print_metrics(metrics):
    """Pretty print evaluation metrics"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nTop-K Accuracy:")
    for k, acc in sorted(metrics['top_k_accuracy'].items()):
        print(f"  Top-{k:2d}: {acc*100:6.2f}%")
    
    print(f"\nMean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f}")
    print("="*50 + "\n")


def plot_topk_curve(metrics, save_path='results/topk_curve.png'):
    """Plot top-k accuracy curve"""
    k_values = sorted(metrics['top_k_accuracy'].keys())
    accuracies = [metrics['top_k_accuracy'][k] * 100 for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('K (Top-K)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Top-K Retrieval Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved top-k curve to {save_path}")
```

---

### Step 7.2: Evaluation Script
**File:** `scripts/05_evaluate.py`

```python
import torch
import numpy as np
import json
import yaml
from tqdm import tqdm
import os

def extract_test_embeddings(model, test_loader, device):
    """Extract embeddings for all test samples"""
    model.eval()
    
    all_embeddings = []
    all_compound_ids = []
    
    with torch.no_grad():
        for xrd_patterns, compound_ids in tqdm(test_loader, desc='Extracting test embeddings'):
            xrd_patterns = xrd_patterns.to(device)
            
            embeddings = model.backbone(xrd_patterns)  # (batch, embedding_dim)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_compound_ids.extend(compound_ids)
    
    all_embeddings = np.vstack(all_embeddings)  # (n_test, embedding_dim)
    
    return all_embeddings, all_compound_ids


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load compound mapping
    with open('data/processed/compound_mapping.json', 'r') as f:
        compound_mapping = json.load(f)
    
    # Load train/val split to get validation compound IDs
    with open('data/processed/train_val_split.json', 'r') as f:
        split = json.load(f)
    
    val_ids = split['val']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data loader (real measured patterns)
    test_dataset = XRDRealDataset(compound_mapping, val_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load trained model
    model = XRDPrototypicalClassifier(
        embedding_dim=config['embedding_dim'],
        temperature=config['temperature']
    ).to(device)
    
    checkpoint = torch.load('checkpoints/best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Extract test embeddings
    print("\nExtracting embeddings for real test patterns...")
    test_embeddings, test_labels = extract_test_embeddings(model, test_loader, device)
    
    # Load validation prototypes
    prototype_data = torch.load('data/prototypes/validation_prototypes.pt')
    prototypes_dict = prototype_data['prototypes']
    prototype_ids = prototype_data['compound_ids']
    
    # Convert prototypes to array
    prototypes = np.stack([prototypes_dict[cid] for cid in prototype_ids])
    
    print(f"\nTest embeddings: {test_embeddings.shape}")
    print(f"Prototypes: {prototypes.shape}")
    
    # Evaluate retrieval performance
    metrics = evaluate_retrieval(
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        prototypes=prototypes,
        prototype_ids=prototype_ids,
        k_values=[1, 5, 10, 20, 50]
    )
    
    # Print results
    print_metrics(metrics)
    
    # Plot top-k curve
    os.makedirs('results', exist_ok=True)
    plot_topk_curve(metrics, 'results/topk_accuracy_curve.png')
    
    # Save detailed results
    results_output = {
        'top_k_accuracy': metrics['top_k_accuracy'],
        'mean_reciprocal_rank': metrics['mean_reciprocal_rank'],
        'n_test_samples': len(test_labels),
        'n_prototypes': len(prototype_ids)
    }
    
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results_output, f, indent=2)
    
    print("✓ Saved results to results/evaluation_results.json")
    
    # Analyze failure cases (samples where top-1 is incorrect)
    incorrect_indices = []
    for i, (pred, true) in enumerate(zip(metrics['top1_predictions'], test_labels)):
        if pred != true:
            incorrect_indices.append(i)
    
    if len(incorrect_indices) > 0:
        print(f"\nFound {len(incorrect_indices)} incorrect top-1 predictions")
        print(f"Top-1 accuracy: {(1 - len(incorrect_indices)/len(test_labels))*100:.2f}%")
        
        # Save failure cases for analysis
        failure_cases = {
            'indices': incorrect_indices,
            'true_labels': [test_labels[i] for i in incorrect_indices],
            'predicted_labels': [metrics['top1_predictions'][i] for i in incorrect_indices]
        }
        
        with open('results/failure_cases.json', 'w') as f:
            json.dump(failure_cases, f, indent=2)
        
        print("✓ Saved failure cases to results/failure_cases.json")


if __name__ == '__main__':
    main()
```

---

## 8. Inference Pipeline (Top-K Retrieval)

### Step 8.1: Inference Script
**File:** `scripts/06_inference.py`

```python
import torch
import numpy as np
import json
import yaml
import argparse
import matplotlib.pyplot as plt

def load_inference_model(checkpoint_path, config):
    """Load trained model for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = XRDPrototypicalClassifier(
        embedding_dim=config['embedding_dim'],
        temperature=config['temperature']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def retrieve_top_k(query_embedding, prototypes, prototype_ids, k=10):
    """
    Retrieve top-K similar compounds
    
    Args:
        query_embedding: (embedding_dim,) numpy array
        prototypes: (n_prototypes, embedding_dim) numpy array
        prototype_ids: List of compound IDs
        k: Number of results to return
    
    Returns:
        results: List of (compound_id, similarity_score) tuples
    """
    # Compute cosine similarities
    similarities = np.dot(prototypes, query_embedding)  # (n_prototypes,)
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Create results
    results = [
        {
            'compound_id': prototype_ids[idx],
            'similarity_score': float(similarities[idx]),
            'rank': rank + 1
        }
        for rank, idx in enumerate(top_k_indices)
    ]
    
    return results


def visualize_retrieval(query_pattern, retrieved_patterns, results, save_path):
    """
    Visualize query pattern and top-K retrieved patterns
    
    Args:
        query_pattern: (4500,) numpy array
        retrieved_patterns: List of (4500,) numpy arrays
        results: List of retrieval results
        save_path: Path to save figure
    """
    k = len(results)
    fig, axes = plt.subplots(k + 1, 1, figsize=(12, 2 * (k + 1)))
    
    # Plot query
    axes[0].plot(query_pattern, linewidth=0.5)
    axes[0].set_title('Query Pattern', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Intensity')
    axes[0].grid(True, alpha=0.3)
    
    # Plot top-k results
    for i, (pattern, result) in enumerate(zip(retrieved_patterns, results)):
        axes[i + 1].plot(pattern, linewidth=0.5)
        axes[i + 1].set_title(
            f"Rank {result['rank']}: {result['compound_id']} "
            f"(Similarity: {result['similarity_score']:.4f})",
            fontsize=10
        )
        axes[i + 1].set_ylabel('Intensity')
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('2θ (degrees)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='XRD Pattern Inference')
    parser.add_argument('--query', type=str, required=True, 
                       help='Path to query XRD pattern (.npy file)')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of top results to retrieve')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    args = parser.parse_args()
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    model, device = load_inference_model('checkpoints/best.pth', config)
    
    # Load prototypes
    print("Loading prototypes...")
    prototype_data = torch.load('data/prototypes/validation_prototypes.pt')
    prototypes_dict = prototype_data['prototypes']
    prototype_ids = prototype_data['compound_ids']
    prototypes = np.stack([prototypes_dict[cid] for cid in prototype_ids])
    
    print(f"Loaded {len(prototype_ids)} prototypes")
    
    # Load query pattern
    print(f"\nLoading query pattern: {args.query}")
    query_pattern = np.load(args.query).astype(np.float32)  # (4500,)
    
    # Prepare query tensor
    query_tensor = torch.from_numpy(query_pattern).unsqueeze(0).unsqueeze(0)  # (1, 1, 4500)
    query_tensor = query_tensor.to(device)
    
    # Extract query embedding
    print("Extracting query embedding...")
    with torch.no_grad():
        query_embedding = model.backbone(query_tensor)  # (1, embedding_dim)
        query_embedding = query_embedding.cpu().numpy()[0]  # (embedding_dim,)
    
    # Retrieve top-k
    print(f"\nRetrieving top-{args.k} matches...")
    results = retrieve_top_k(query_embedding, prototypes, prototype_ids, k=args.k)
    
    # Print results
    print("\n" + "="*70)
    print(f"TOP-{args.k} RETRIEVAL RESULTS")
    print("="*70)
    for result in results:
        print(f"Rank {result['rank']:2d}: {result['compound_id']:15s} "
              f"(Similarity: {result['similarity_score']:.4f})")
    print("="*70 + "\n")
    
    # Save results
    output_path = 'results/inference_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'query_file': args.query,
            'k': args.k,
            'results': results
        }, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Visualization
    if args.visualize:
        print("\nCreating visualization...")
        
        # Load compound mapping to get file paths
        with open('data/processed/compound_mapping.json', 'r') as f:
            compound_mapping = json.load(f)
        
        # Load retrieved patterns
        retrieved_patterns = []
        for result in results:
            compound_id = result['compound_id']
            pattern_path = compound_mapping[compound_id]['ideal_path']  # Or 'real_path'
            pattern = np.load(pattern_path)
            retrieved_patterns.append(pattern)
        
        visualize_retrieval(
            query_pattern,
            retrieved_patterns,
            results,
            'results/retrieval_visualization.png'
        )


if __name__ == '__main__':
    main()
```

---

## 9. Configuration File

### Config Template
**File:** `configs/config.yaml`

```yaml
# Data paths
augmentation_dir: 'data/processed/train_augmented'
compound_mapping_path: 'data/processed/compound_mapping.json'
split_path: 'data/processed/train_val_split.json'

# Augmentation settings
n_augmentations: 10  # Number of augmented versions per compound
noise_beta_alpha: 2  # Beta distribution alpha parameter (higher = more low noise)
noise_beta_beta: 5   # Beta distribution beta parameter
max_noise_level: 1000

# Model architecture
embedding_dim: 256  # Embedding vector dimension (128, 256, or 512)
temperature: 0.1    # Temperature for prototypical loss

# Training hyperparameters
batch_size: 128
epochs: 100
learning_rate: 0.001
min_lr: 0.00001
weight_decay: 0.0001

# Loss weights (for combined prototypical + triplet loss)
proto_weight: 1.0
triplet_weight: 0.5
triplet_margin: 0.2

# Data loading
num_workers: 4
pin_memory: true

# Checkpointing
save_every: 10  # Save checkpoint every N epochs

# Evaluation
top_k_values: [1, 5, 10, 20, 50]
```

---

## 10. Execution Workflow

### Complete Pipeline Execution Order

```bash
# Step 1: Setup project structure
mkdir -p xrd_classification/{data/{raw/{ideal_patterns,real_patterns},processed,prototypes},models,utils,scripts,configs,checkpoints,logs,results}

# Step 2: Prepare data and create mapping
python scripts/01_prepare_data.py

# Step 3: Generate augmented training data
python scripts/02_create_augmentations.py

# Step 4: Train the model
python scripts/03_train.py

# Step 5: Compute validation prototypes
python scripts/04_compute_prototypes.py

# Step 6: Evaluate on real test patterns
python scripts/05_evaluate.py

# Step 7: Inference on new samples
python scripts/06_inference.py --query data/raw/real_patterns/compound_0001.npy --k 10 --visualize
```

---

## 11. Expected Results & Troubleshooting

### Expected Performance
Based on similar XRD classification tasks:
- **Top-1 accuracy:** 70-85% (depends on pattern similarity and noise)
- **Top-5 accuracy:** 85-95%
- **Top-10 accuracy:** 90-98%
- **Mean Reciprocal Rank:** 0.75-0.85

### Training Monitoring
Monitor these metrics during training:
1. **Loss decreasing steadily:** Both prototypical and triplet losses should decrease
2. **Validation loss not diverging:** If val_loss >> train_loss, increase augmentation or regularization
3. **Learning rate:** Should decrease smoothly with cosine schedule
4. **GPU memory:** ~10-15 GB for batch_size=128 with ResNet-18

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss not decreasing | Learning rate too high/low | Try lr=0.0001 or 0.01 |
| Overfitting (val_loss increasing) | Not enough augmentation | Increase n_augmentations to 20-50 |
| Low top-1 accuracy but high top-5 | Similar compounds hard to distinguish | Expected behavior, focus on top-K |
| Out of memory | Batch size too large | Reduce batch_size to 64 or 32 |
| Training too slow | CPU bottleneck | Increase num_workers, use pin_memory |
| Model doesn't converge | Triplet loss dominating | Reduce triplet_weight to 0.1 |

### Hyperparameter Tuning Priorities
1. **Embedding dimension:** Try 128, 256, 512 (256 is good default)
2. **Temperature:** Try 0.05, 0.1, 0.2 (lower = harder separation)
3. **N_augmentations:** Try 10, 20, 50 (more = better generalization)
4. **Triplet margin:** Try 0.1, 0.2, 0.3 (larger = more separation)

---

## 12. Future Improvements

### Potential Enhancements
1. **Self-supervised pre-training:** Pre-train on full 500k dataset with SimCLR
2. **Attention mechanisms:** Add self-attention layers after ResNet blocks
3. **Multi-scale features:** Combine features from different ResNet layers
4. **Hard negative mining:** Improve triplet loss with online hard negative mining
5. **Data augmentation:** Add peak shifting, intensity scaling, background noise
6. **Ensemble:** Train multiple models with different random seeds and average embeddings
7. **Active learning:** Identify most uncertain predictions and request labels

### Scaling to Full 500k Dataset
When ready to scale:
1. Use same pipeline, just change train/val split to include all 500k
2. Increase batch size to 256-512 (multi-GPU)
3. Use distributed training (torch.nn.DataParallel or DistributedDataParallel)
4. Increase embedding_dim to 512
5. Consider hierarchical prototypes (group similar compounds)

---

## 13. Key Implementation Notes

### Critical Points to Remember
1. **L2 normalization:** Always normalize embeddings before computing similarities
2. **Beta distribution for noise:** Ensures more training samples have lower noise
3. **Prototype re-normalization:** After averaging, renormalize prototypes
4. **Cosine similarity:** Use dot product (since embeddings are normalized)
5. **GPU memory management:** Use `pin_memory=True` and appropriate batch size
6. **Reproducibility:** Set random seeds in all scripts

### Code Quality Checklist
- [ ] All scripts have proper error handling
- [ ] Data loaders verified with small subset
- [ ] Model architecture tested with dummy input
- [ ] Checkpoint saving/loading works correctly
- [ ] Metrics calculation verified manually
- [ ] Visualization functions tested
- [ ] Config file complete and valid
- [ ] Documentation comments in code

---

## 14. Questions to Ask Before Starting Implementation

1. **Data format:** Confirm XRD patterns are .npy files with shape (4500,)
2. **Augmentation module:** Get exact import path and function signature
3. **Computing resources:** Verify Tesla V100 32GB access
4. **Storage:** Ensure sufficient disk space for augmented data (~10-50 GB)
5. **Timeline:** Estimated ~2-3 days for full pipeline implementation and testing

---

## Summary Checklist

### Implementation Steps
- [ ] Create project structure
- [ ] Implement data preparation script
- [ ] Implement augmentation generation
- [ ] Implement 1D ResNet-18
- [ ] Implement prototypical loss
- [ ] Implement training loop
- [ ] Implement prototype computation
- [ ] Implement evaluation pipeline
- [ ] Implement inference script
- [ ] Test on small subset (100 compounds)
- [ ] Train on full dataset (13k compounds)
- [ ] Evaluate and analyze results
- [ ] Document findings

**Estimated Time:** 3-5 days for complete implementation and initial training

---

**This plan is ready for Claude Code implementation. Each script can be implemented incrementally and tested before moving to the next step.**
