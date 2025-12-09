# XRD Classification Status - November 16, 2025

## High-Level Architecture

### Problem Statement
Train a deep learning model on **synthetic XRD patterns** to classify **real measured XRD patterns** of the same compounds. This is a domain adaptation challenge where we bridge the gap between computational simulations and experimental measurements.

### Data Pipeline

**Training Data**: Synthetic XRD patterns from computational simulations (~13k compounds)
- Source: `xrd_dataset_labeled_dtw_window.pt`
- Contains both `synth_xrd` and `real_xrd` arrays for each compound
- Training uses only synthetic patterns with augmentation

**Validation/Test Data**: Real measured XRD patterns
- Validation: `xrd_train_val_dataset.pt` (~9k compounds)
- Test: `xrd_test_dataset.pt` (~3k compounds)
- Contains `real_xrd` arrays of compund
- Contains actual experimental measurements

**Data Augmentation**: Diffusion-based augmentation to make synthetic patterns more "real-like"
- Classical augmentation: disabled
- Diffusion augmentation: enabled
- 5 augmented samples per base synthetic pattern

### Model Architecture

**XRDPrototypicalClassifier**:
- **Backbone**: ResNet1D-18 (1D CNN adapted for XRD patterns)
- **Embedding Dimension**: 1024
- **Loss Function**: Combined Prototypical + Triplet Loss
  - Prototypical loss: Groups same compounds, separates different compounds
  - Triplet loss: Hard negative mining for better embeddings
  - Weights: proto_weight=1.0, triplet_weight=0.5

**Training Process**:
1. Load synthetic patterns with augmentation (750 samples from 50 compounds)
2. Create compound prototypes during training
3. Validate against real patterns of same compounds
4. Use cosine similarity for prototype matching

### Evaluation Strategy

**Same-Compound Transfer Learning**:
- Train: Synthetic pattern of compound_00005
- Test: Real measured pattern of compound_00005
- Metric: Top-k accuracy (k=1,5,10)
- Goal: Synthetic training should help identify real measurements

### Current Performance Issues

**Problem**: Very low accuracy (2-4% validation, random=2% for 50 classes)
- Training accuracy: 0-2% (model barely learning synthetic patterns)
- Loss barely decreasing (6.31 → 6.30 over 5 epochs)
- Suggests fundamental training issues, not just domain gap

**Potential Causes**:
1. **Learning rate issues**: May be too low or decaying too fast
2. **Diffusion augmentation**: May not be bridging synthetic→real gap effectively
3. **Architecture mismatch**: ResNet1D may not be optimal for XRD patterns
4. **Data preprocessing**: Normalization or scaling issues
5. **Loss function**: Prototypical+triplet may be too complex for this domain

### Key Components

**Scripts**:
- `train_modular.py`: Main training pipeline with proper synthetic→real evaluation
- `train_500_samples.py`: Legacy script (accidentally did synthetic→synthetic validation)

**Utils**:
- `data_loading.py`: Handles separate dataset loading and compound ID mapping
- `datasets.py`: Custom PyTorch datasets for synthetic vs real patterns
- `evaluation.py`: Cross-domain evaluation functions
- `augmentation.py`: Diffusion-based pattern augmentation


to use python you need to actiavate the env by 
$ source myenv/bin/activate