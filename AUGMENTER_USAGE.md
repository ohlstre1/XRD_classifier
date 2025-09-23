# XRD Pattern Augmenter Usage Guide

This guide explains how to use the XRD Pattern Augmenter to generate realistic experimental-like patterns from synthetic XRD data.

## 🚀 Quick Start

### Basic Command Line Usage

```bash
# NO TRAINING REQUIRED - Classical augmentation (RECOMMENDED for immediate use)
python xrd_pattern_augmenter.py \
    --input_file data/synthetic_patterns.pt \
    --output_dir ./augmented_patterns \
    --samples_per_pattern 5 \
    --no_training

# Alternative: Dedicated classical augmenter
python classical_xrd_augmenter.py \
    --input_file data/synthetic_patterns.pt \
    --output_dir ./classical_augmented \
    --samples_per_pattern 5

# Model-based augmentation (requires trained model)
python xrd_pattern_augmenter.py \
    --input_file data/xrd_dataset_labeled_dtw_window.pt \
    --output_dir ./augmented_patterns \
    --samples_per_pattern 5

# Advanced usage with custom parameters
python xrd_pattern_augmenter.py \
    --input_file data/synthetic_patterns.pt \
    --output_dir ./realistic_patterns \
    --samples_per_pattern 10 \
    --temp_range 0.1 2.5 \
    --temp_mode random \
    --noise_timestep_range 0 100 \
    --base_seed 42 \
    --visualize \
    --batch_size 16
```

### Example Scripts

```bash
# Run comprehensive example (requires trained model)
python run_augmentation_example.py

# Demo no-training augmentation (WORKS IMMEDIATELY)
python demo_no_training_augmentation.py
```

## 🎯 Two Augmentation Modes

### 1. Classical Mode (No Training Required) ⚡ RECOMMENDED
- **Instant usage** - no model training needed
- **Physical realism** - based on XRD physics and measurement principles
- **Fast processing** - CPU friendly
- **Highly configurable** - many parameters to tune

### 2. Model-Based Mode (Requires Trained Model)
- Uses your trained diffusion model
- Learned augmentation patterns from training data
- Requires pre-trained model checkpoint
- GPU recommended for speed

## 📋 Command Line Parameters

### Required Parameters
- `--input_file`: Path to synthetic XRD patterns file (.pt or .npy)

### Mode Selection
- `--no_training`: Use classical augmentation without trained model (RECOMMENDED)

### Output Parameters
- `--output_dir`: Output directory (default: `./augmented_patterns`)
- `--model_path`: Path to trained diffusion model (default: `./models/xrd_diffusion/improved_diffusion_model_best.pth`, ignored if `--no_training` used)

### Augmentation Parameters
- `--samples_per_pattern`: Number of augmented samples per input (default: 5)
- `--temp_range`: Temperature conditioning range, e.g., `0.1 2.0` (default: [0.1, 2.0])
- `--temp_mode`: Temperature sampling mode: `random`, `linear`, `exponential` (default: random)
- `--noise_timestep_range`: Diffusion timestep range, e.g., `0 50` (default: [0, 50])
- `--base_seed`: Base random seed for reproducibility (default: 42)

### Processing Parameters
- `--batch_size`: Processing batch size (default: 8)
- `--max_patterns`: Maximum patterns to process (default: all)
- `--device`: Device to use: `auto`, `cpu`, `cuda` (default: auto)

### Output Options
- `--visualize`: Create visualization plots
- `--vis_samples`: Number of patterns to visualize (default: 3)
- `--quiet`: Suppress verbose output

## 🐍 Python API Usage

### Basic Usage

```python
from xrd_pattern_augmenter import XRDPatternAugmenter
import torch

# Option 1: Classical augmentation (NO TRAINING REQUIRED)
augmenter = XRDPatternAugmenter(
    model_path=None,           # No model needed!
    use_classical=True,        # Use classical augmentation
    device='auto'
)

# Option 2: Model-based augmentation (requires trained model)
augmenter = XRDPatternAugmenter(
    model_path="./models/xrd_diffusion/improved_diffusion_model_best.pth",
    device='auto'
)

# Load synthetic patterns
synth_patterns = torch.randn(10, 1000)  # 10 patterns, 1000 points each

# Augment single pattern
augmented = augmenter.augment_pattern(
    synth_pattern=synth_patterns[0:1],
    num_samples=5,
    temp_range=(0.1, 1.0),
    base_seed=42
)

print(f"Input: {synth_patterns[0:1].shape}")
print(f"Output: {augmented.shape}")  # [5, 1, 1000]
```

### Batch Processing

```python
# Augment multiple patterns
augmented_batch, metadata = augmenter.augment_batch(
    synth_patterns=synth_patterns,
    samples_per_pattern=3,
    temp_range=(0.2, 1.5),
    temp_mode='random',
    base_seed=123,
    progress_bar=True
)

print(f"Input batch: {synth_patterns.shape}")      # [10, 1000]
print(f"Output batch: {augmented_batch.shape}")    # [30, 1, 1000]
print(f"Metadata entries: {len(metadata)}")        # 10
```

### Advanced Configuration

```python
# Custom temperature conditions
augmented_custom = augmenter.augment_pattern(
    synth_pattern=synth_patterns[0:1],
    num_samples=8,
    temp_range=(0.1, 3.0),
    temp_mode='exponential',  # More low temperatures
    noise_timestep_range=(10, 80),  # Higher noise range
    base_seed=999
)

# Get metadata for analysis
augmented_with_meta, metadata = augmenter.augment_pattern(
    synth_pattern=synth_patterns[0:1],
    num_samples=5,
    return_metadata=True
)

print("Temperature values:", metadata['temperatures'])
print("Timesteps used:", metadata['timesteps'])
print("Noise levels:", metadata['noise_levels'])
```

## 🎯 Temperature Modes

### Random Mode (default)
```python
temp_mode='random'  # Uniform random distribution in temp_range
```

### Linear Mode
```python
temp_mode='linear'  # Linear spacing across temp_range
```

### Exponential Mode
```python
temp_mode='exponential'  # More low temperature values
```

### Fixed Temperature
```python
temp_mode=1.5  # Fixed temperature value
```

## 📊 Understanding Output

### File Structure
```
output_dir/
├── augmented_patterns_YYYYMMDD_HHMMSS.pt    # PyTorch tensor file
├── augmented_metadata_YYYYMMDD_HHMMSS.json  # Metadata JSON
└── visualizations/                          # (if --visualize used)
    ├── augmentation_example_1.png
    ├── augmentation_example_2.png
    └── ...
```

### PyTorch File Contents
```python
data = torch.load("augmented_patterns_20231215_143022.pt")

# Available data
data['augmented_patterns']  # [N*samples_per_pattern, 1, L] - Generated patterns
data['original_patterns']   # [N, 1, L] - Original synthetic patterns
data['metadata']            # List of metadata dicts
data['generation_info']     # Generation timestamp and parameters
```

### Metadata Structure
```python
metadata = {
    'temperatures': np.array,     # Temperature values used [samples_per_pattern, 1]
    'seeds': list,               # Random seeds for each sample
    'timesteps': list,           # Diffusion timesteps used
    'noise_levels': list         # Noise scaling factors applied
}
```

## 🔬 Advanced Features

### Custom Augmentation Pipeline

```python
class CustomAugmenter(XRDPatternAugmenter):
    def custom_augment_pattern(self, pattern, custom_params):
        # Your custom augmentation logic
        # Can access self.model, self.diffusion, etc.
        pass

# Use custom augmenter
custom_aug = CustomAugmenter(model_path="model.pth")
```

### Batch Saving and Loading

```python
# Save augmented patterns for later use
augmenter.save_results(
    augmented_patterns=augmented_batch,
    metadata_list=metadata,
    original_patterns=synth_patterns,
    output_dir="./my_augmented_data",
    prefix="training_data"
)

# Load saved patterns
import torch
saved_data = torch.load("training_data_patterns_20231215_143022.pt")
patterns = saved_data['augmented_patterns']
```

### Creating Visualizations

```python
# Visualize augmentation results
augmenter.visualize_augmentation(
    original_pattern=synth_patterns[0],
    augmented_patterns=augmented_batch[0:5],  # First 5 augmented samples
    metadata=metadata[0],
    save_path="./my_visualization.png",
    show_plot=True
)
```

## ⚙️ Configuration Examples

### High-Quality Augmentation
```python
# For training data augmentation
augmented = augmenter.augment_batch(
    synth_patterns=patterns,
    samples_per_pattern=10,      # Many samples
    temp_range=(0.1, 1.0),      # Conservative temperature range
    temp_mode='random',         # Diverse conditions
    noise_timestep_range=(0, 30), # Light noise addition
    base_seed=42               # Reproducible
)
```

### Stress Testing
```python
# For robustness testing
stress_patterns = augmenter.augment_batch(
    synth_patterns=patterns,
    samples_per_pattern=5,
    temp_range=(0.5, 3.0),      # Extreme temperatures
    temp_mode='exponential',    # Favor challenging conditions
    noise_timestep_range=(50, 100), # Heavy noise
    base_seed=999
)
```

### Instrument Simulation
```python
# Simulate specific instrument conditions
instrument_patterns = augmenter.augment_pattern(
    synth_pattern=clean_pattern,
    num_samples=50,             # Statistical sampling
    temp_mode=1.2,             # Fixed instrument temperature
    noise_timestep_range=(20, 40), # Typical instrument noise
    base_seed=instrument_id    # Reproducible per instrument
)
```

## 🔧 Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce batch size
--batch_size 2

# Process fewer patterns
--max_patterns 100
```

**Model Loading Issues**
```bash
# Check model path
ls -la ./models/xrd_diffusion/improved_diffusion_model_best.pth

# Use CPU if GPU issues
--device cpu
```

**Input Format Issues**
```python
# Ensure correct input format
patterns = patterns.float()  # Convert to float32
if patterns.dim() == 2:
    patterns = patterns.unsqueeze(1)  # Add channel dimension
```

### Performance Tips

1. **Use GPU**: Significantly faster than CPU
2. **Optimize batch size**: Balance memory usage and speed
3. **Limit patterns**: Use `--max_patterns` for testing
4. **Disable visualization**: Skip `--visualize` for production runs

## 📚 Scientific Applications

### Training Data Augmentation
Generate diverse training samples from limited synthetic data:
```python
# Create 1000 training samples from 100 synthetic patterns
training_data = augmenter.augment_batch(
    synth_patterns=synthetic_patterns,
    samples_per_pattern=10,
    temp_range=(0.1, 2.0),
    temp_mode='random'
)
```

### Instrument Noise Modeling
Simulate different measurement conditions:
```python
# Model different noise levels
low_noise = augmenter.augment_pattern(pattern, noise_timestep_range=(0, 20))
high_noise = augmenter.augment_pattern(pattern, noise_timestep_range=(50, 100))
```

### Robustness Testing
Test algorithm performance under realistic conditions:
```python
# Create challenging test cases
test_patterns = augmenter.augment_batch(
    synth_patterns=test_set,
    samples_per_pattern=5,
    temp_range=(1.0, 3.0),      # Challenging conditions
    temp_mode='exponential'
)
```

## 🎯 Best Practices

1. **Start small**: Test with a few patterns first
2. **Use visualization**: Verify augmentation quality
3. **Save metadata**: Important for reproducibility
4. **Set seeds**: Enable reproducible research
5. **Monitor quality**: Check correlation and peak preservation
6. **Validate results**: Compare with real experimental data

This augmenter transforms your synthetic XRD patterns into realistic experimental-like data, perfect for training robust machine learning models and simulating real-world measurement conditions!