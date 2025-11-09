# XRD Prototypical Classification Pipeline

A complete implementation of prototypical learning for X-ray diffraction (XRD) pattern classification using deep neural networks. This pipeline combines ResNet-18 1D architecture with prototypical learning to classify 13,325 XRD compound patterns using dual augmentation strategies.

## 🚀 Overview

This project implements a state-of-the-art prototypical learning pipeline specifically designed for XRD pattern classification. It addresses the challenge of learning robust representations from synthetic XRD patterns while generalizing to real measured patterns through metric learning and sophisticated augmentation strategies.

### Key Features

- **🔬 Prototypical Learning**: Learn embeddings that cluster similar compounds and separate different ones
- **🏗️ ResNet-18 1D Architecture**: Adapted for 1D XRD signals (4500 data points)
- **🔄 Dual Augmentation System**: Combines classical signal processing with diffusion model augmentation
- **📊 Comprehensive Evaluation**: Top-K accuracy, Mean Reciprocal Rank, and similarity analysis
- **⚡ Ready-to-Use Pipeline**: From raw data to trained classifier in one workflow

### Problem Statement

XRD pattern classification faces several challenges:
- **Domain Gap**: Synthetic patterns vs. real measured patterns
- **Limited Training Data**: Need for data augmentation
- **High Dimensionality**: 4500-point signals require specialized architectures
- **Metric Learning**: Need to learn meaningful similarity measures

This pipeline addresses these challenges through prototypical learning combined with sophisticated augmentation.

## 📁 Project Structure

```
XRD-classification/
├── configs/
│   └── config.yaml                 # Configuration file with all hyperparameters
├── data/
│   ├── processed/
│   │   ├── compound_mapping.json   # Compound ID to pattern mapping
│   │   ├── train_val_split.json    # Train/validation split
│   │   └── dataset_statistics.json # Dataset analysis
│   ├── prototypes/
│   │   └── validation_prototypes.pt # Computed prototypes for inference
│   └── raw/                        # Raw data directory
├── models/
│   ├── __init__.py                 # Package initialization
│   ├── resnet1d.py                 # 1D ResNet-18 implementation
│   ├── prototypical_loss.py        # Loss functions for metric learning
│   └── xrd_classifier.py           # Main classifier wrapper
├── utils/
│   └── augmentation.py             # Dual augmentation system
├── scripts/
│   ├── 01_prepare_data.py          # Data preprocessing and splitting
│   ├── 02_create_augmentations.py  # Generate augmented training data
│   ├── 03_train.py                 # Training pipeline
│   ├── 04_compute_prototypes.py    # Generate validation prototypes
│   ├── 05_evaluate.py              # Evaluation on real test patterns
│   └── 06_inference.py             # Top-K retrieval for new samples
├── checkpoints/                    # Model checkpoints
├── logs/                          # Training logs and tensorboard
└── results/                       # Evaluation results and visualizations
```

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for full dataset processing

### Environment Setup

```bash
# Create virtual environment
python -m venv xrd_env
source xrd_env/bin/activate  # On Windows: xrd_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib seaborn
pip install pyyaml tqdm tensorboard
pip install scikit-learn pandas
```

### Data Requirements

This pipeline expects the preprocessed XRD dataset at:
```
../data/xrd_dataset_labeled_dtw_window.pt
```

The dataset should contain:
- `synth_xrd`: Synthetic XRD patterns [13325, 4500]
- `real_xrd`: Real measured XRD patterns [13325, 4500]
- `file_info`: Compound information
- `fast_dtw_distance`: DTW distances between synthetic and real patterns

## 🚀 Quick Start

### 1. Configure the Pipeline

Edit `configs/config.yaml` to customize hyperparameters:

```yaml
# Key configuration options
model:
  embedding_dim: 256          # Embedding dimension
  temperature: 0.1            # Prototypical loss temperature

training:
  batch_size: 128             # Training batch size
  epochs: 100                 # Number of training epochs
  learning_rate: 0.001        # Initial learning rate

augmentation:
  n_augmentations: 10         # Augmented samples per compound
  classical:
    enabled: true             # Enable classical augmentation
    samples_ratio: 0.5        # 50% classical samples
  diffusion:
    enabled: true             # Enable diffusion augmentation
    samples_ratio: 0.5        # 50% diffusion samples
```

### 2. Prepare the Data

```bash
cd scripts
python 01_prepare_data.py \
    --config ../configs/config.yaml \
    --dataset_path ../../data/xrd_dataset_labeled_dtw_window.pt \
    --output_dir ../data/processed
```

This creates:
- **10,660 training compounds** (80%)
- **2,665 validation compounds** (20%)
- Normalized XRD patterns [0, 1]
- Compound mapping with metadata

### 3. Generate Augmented Training Data

```bash
python 02_create_augmentations.py \
    --config ../configs/config.yaml \
    --output_dir ../data/processed/train_augmented
```

This generates ~106,600 augmented training samples using dual augmentation.

### 4. Train the Model

```bash
python 03_train.py --config ../configs/config.yaml
```

Training progress is logged to:
- Console output with progress bars
- TensorBoard logs in `logs/`
- Model checkpoints in `checkpoints/`

### 5. Compute Validation Prototypes

```bash
python 04_compute_prototypes.py --config ../configs/config.yaml
```

This computes prototype embeddings for all validation compounds.

### 6. Evaluate on Real Test Patterns

```bash
python 05_evaluate.py --config ../configs/config.yaml
```

Evaluation metrics:
- **Top-1 Accuracy**: 70-85% (expected)
- **Top-5 Accuracy**: 85-95% (expected)
- **Top-10 Accuracy**: 90-98% (expected)
- **Mean Reciprocal Rank**: 0.75-0.85 (expected)

### 7. Run Inference on New Patterns

```bash
python 06_inference.py \
    --query path/to/new_pattern.npy \
    --k 10 \
    --visualize
```

## 🏗️ Architecture Deep Dive

### ResNet-18 1D Architecture

The backbone network adapts ResNet-18 for 1D XRD signals:

```
Input: [batch, 1, 4500]
├── Conv1D(1→64, k=7, s=2) → [batch, 64, 2250]
├── MaxPool1D(k=3, s=2) → [batch, 64, 1125]
├── Layer1: 2×BasicBlock1D → [batch, 64, 1125]
├── Layer2: 2×BasicBlock1D → [batch, 128, 563]
├── Layer3: 2×BasicBlock1D → [batch, 256, 282]
├── Layer4: 2×BasicBlock1D → [batch, 512, 141]
├── AdaptiveAvgPool1D → [batch, 512, 1]
├── Linear(512→256) → [batch, 256]
└── L2 Normalize → [batch, 256] (unit sphere)
```

**Key Design Choices**:
- **1D Convolutions**: Adapted from 2D to handle sequential XRD data
- **Progressive Downsampling**: 4500 → 141 points through strided convolutions
- **L2 Normalization**: Embeddings lie on unit hypersphere for cosine similarity
- **~4M Parameters**: Efficient architecture for XRD classification

### Prototypical Learning

The loss function encourages embeddings to cluster by compound class:

```python
# For each batch:
# 1. Compute class prototypes (centroids)
prototypes = embeddings.group_by(labels).mean()

# 2. Compute distances to all prototypes
distances = -cosine_similarity(embeddings, prototypes) / temperature

# 3. Cross-entropy loss for correct prototype matching
loss = cross_entropy(distances, target_prototype_indices)
```

**Enhanced with Hard Triplet Mining**:
- **Hardest Positive**: Farthest sample from same class
- **Hardest Negative**: Closest sample from different class
- **Triplet Loss**: `max(0, d_pos - d_neg + margin)`

### Dual Augmentation System

Combines two complementary augmentation strategies:

#### 1. Classical Augmentation (50% of samples)
- **Peak Broadening**: Gaussian/Lorentzian/Voigt convolution
- **Intensity Scaling**: Random multiplicative factors
- **Background Noise**: Additive Gaussian noise
- **Peak Shifting**: Position jitter simulation
- **Baseline Drift**: Low-frequency variations

#### 2. Diffusion Model Augmentation (50% of samples)
- **Learned Artifacts**: Realistic experimental variations
- **Temperature Conditioning**: Instrument-specific effects
- **Noise Timestep Control**: Varying degradation levels
- **Fallback to Classical**: Robust operation

**Benefits of Dual Approach**:
- **Immediate Deployment**: Classical works without model training
- **Maximum Realism**: Diffusion adds learned experimental artifacts
- **Robust Training**: 106,600 diverse samples from 10,660 compounds
- **Configurable Mixing**: Adjust ratios based on performance

## 📊 Performance Analysis

### Expected Results

Based on similar XRD classification tasks:

| Metric | Expected Range | Description |
|--------|---------------|-------------|
| **Top-1 Accuracy** | 70-85% | Exact compound match |
| **Top-5 Accuracy** | 85-95% | Correct compound in top 5 |
| **Top-10 Accuracy** | 90-98% | Correct compound in top 10 |
| **Mean Reciprocal Rank** | 0.75-0.85 | Average inverse rank |

### Training Monitoring

Key metrics to monitor during training:

```python
# Loss components
- total_loss: Combined prototypical + triplet loss
- proto_loss: Prototypical learning component
- triplet_loss: Hard triplet mining component

# Accuracy metrics
- proto_accuracy: Batch-level prototype matching
- intra_class_similarity: Within-class cohesion
- inter_class_similarity: Between-class separation
```

### Evaluation Outputs

The evaluation script generates:

1. **Quantitative Results** (`results/evaluation_results.json`)
   ```json
   {
     "top_k_accuracy": {
       "1": 0.78, "5": 0.92, "10": 0.96, "20": 0.98
     },
     "mean_reciprocal_rank": 0.82,
     "n_test_samples": 2665
   }
   ```

2. **Visualizations** (`results/`)
   - Top-K accuracy curves
   - Confusion matrices for top-1 predictions
   - Embedding space visualizations (t-SNE/UMAP)
   - Failure case analysis

3. **Detailed Analysis**
   - Per-compound performance statistics
   - DTW distance correlation analysis
   - Augmentation method effectiveness

## 🔧 Configuration Guide

### Model Architecture Options

```yaml
model:
  embedding_dim: [128, 256, 512]    # Embedding dimensions to try
  temperature: [0.05, 0.1, 0.2]    # Lower = sharper distributions

  # Loss function configuration
  loss:
    proto_weight: 1.0               # Prototypical loss weight
    triplet_weight: 0.5             # Triplet loss weight
    triplet_margin: 0.2             # Triplet loss margin
```

### Training Hyperparameters

```yaml
training:
  batch_size: [64, 128, 256]        # GPU memory dependent
  learning_rate: [0.0001, 0.001, 0.01]  # Learning rate sweep
  weight_decay: [0.0001, 0.001]    # Regularization strength
  epochs: [50, 100, 200]           # Training duration
```

### Augmentation Tuning

```yaml
augmentation:
  n_augmentations: [5, 10, 20, 50] # More = better generalization

  classical:
    samples_ratio: [0.3, 0.5, 0.7] # Classical vs diffusion ratio
    noise_level_range: [0.01, 0.1] # Background noise strength

  diffusion:
    temp_range: [0.1, 2.0]          # Temperature conditioning range
    noise_timestep_range: [0, 50]   # Diffusion noise levels
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors
```bash
# Reduce batch size
training:
  batch_size: 64  # or 32

# Process fewer augmentations
augmentation:
  n_augmentations: 5
```

#### 2. Training Not Converging
```bash
# Lower learning rate
training:
  learning_rate: 0.0001

# Increase augmentation
augmentation:
  n_augmentations: 20
```

#### 3. Low Top-1 Accuracy but High Top-5
This is expected behavior - similar compounds are hard to distinguish. Focus on Top-K metrics rather than Top-1.

#### 4. Diffusion Model Unavailable
The pipeline automatically falls back to classical augmentation if diffusion models are unavailable.

#### 5. CUDA Out of Memory
```bash
# Enable mixed precision
hardware:
  mixed_precision: true

# Reduce workers
training:
  dataloader:
    num_workers: 2
```

### Performance Optimization

#### 1. GPU Utilization
```python
# Monitor GPU usage
nvidia-smi -l 1

# Enable compiled models (PyTorch 2.0+)
performance:
  compile_model: true
```

#### 2. Data Loading
```python
# Increase workers for faster data loading
training:
  dataloader:
    num_workers: 8
    pin_memory: true
```

#### 3. Mixed Precision Training
```python
# Reduce memory usage and increase speed
hardware:
  mixed_precision: true
```

## 🧪 Advanced Usage

### Custom Augmentation Strategies

```python
from utils.augmentation import DualXRDAugmenter

# Create custom augmenter
config = {
    'augmentation': {
        'classical': {'enabled': True, 'samples_ratio': 0.7},
        'diffusion': {'enabled': True, 'samples_ratio': 0.3}
    }
}

augmenter = DualXRDAugmenter(config)
augmented, methods = augmenter.augment_pattern_mixed(pattern, num_samples=15)
```

### Model Architecture Experiments

```python
from models import create_resnet1d_34, XRDPrototypicalClassifier

# Try ResNet-34 for larger capacity
model = XRDPrototypicalClassifier(
    embedding_dim=512,
    loss_type='prototypical_triplet',
    temperature=0.05
)
```

### Ensemble Methods

```python
# Train multiple models with different seeds
models = []
for seed in [42, 123, 456]:
    model = train_model(seed=seed)
    models.append(model)

# Average embeddings for inference
ensemble_embedding = torch.stack([m(x) for m in models]).mean(dim=0)
```

## 📈 Scaling to Full Dataset

When ready to scale beyond 13k compounds:

### 1. Distributed Training
```python
# Use DataParallel for multi-GPU
model = torch.nn.DataParallel(model)

# Or DistributedDataParallel for multi-node
model = torch.nn.parallel.DistributedDataParallel(model)
```

### 2. Hierarchical Prototypes
```python
# Group similar compounds to reduce memory
hierarchical_prototypes = {
    'organic': {...},
    'inorganic': {...},
    'metallic': {...}
}
```

### 3. Incremental Learning
```python
# Add new compounds without full retraining
model.add_new_prototypes(new_compound_embeddings)
```

## 🤝 Contributing

### Code Quality Standards

- **Type Hints**: All functions use type annotations
- **Docstrings**: Google-style documentation
- **Testing**: Unit tests for core components
- **Formatting**: Black code formatting
- **Linting**: Pylint compliance

### Development Workflow

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Submit** pull request

### Adding New Features

#### New Loss Functions
```python
# Add to models/prototypical_loss.py
class CustomPrototypicalLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Implementation
```

#### New Augmentation Methods
```python
# Add to utils/augmentation.py
def custom_augmentation(pattern, params):
    # Implementation
    return augmented_pattern
```

## 📚 Citation and References

If you use this pipeline in your research, please cite:

```bibtex
@software{xrd_prototypical_classification,
  title={XRD Prototypical Classification Pipeline},
  author={XRD Classification Team},
  year={2025},
  url={https://github.com/your-repo/xrd-classification}
}
```

### Key References

1. **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
2. **Hard Triplet Mining**: Hermans et al., "In Defense of the Triplet Loss for Person Re-Identification", arXiv 2017
3. **ResNet Architecture**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
4. **XRD Analysis**: Powder Diffraction File (PDF) Database, ICDD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **XRD Community** for domain knowledge and datasets
- **Research Collaborators** for insights and feedback

---

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@domain.com

---

*Last Updated: November 2024*