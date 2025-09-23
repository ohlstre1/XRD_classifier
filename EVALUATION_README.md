# XRD Diffusion Model Evaluation with Standard Deviation Metrics

This package provides comprehensive evaluation tools for the XRD diffusion model with statistical analysis including standard deviation calculations.

## 📁 Files Overview

- **`evaluate_diffusion_std.py`** - Core evaluation script with statistical analysis
- **`run_evaluation.py`** - User-friendly runner script
- **`diffusion_model_0.1.5.py`** - Original diffusion model (required dependency)

## 🚀 Quick Start

### Option 1: Simple Run (Recommended)
```bash
python run_evaluation.py
```

### Option 2: Direct Evaluation
```bash
python evaluate_diffusion_std.py
```

## 📊 What Gets Evaluated

### Core Performance Metrics
- **Total Loss** - Combined diffusion and reconstruction loss
- **Diffusion Loss** - Noise prediction accuracy
- **Reconstruction Loss** - Real→synthetic pattern transformation accuracy

### Accuracy Metrics
- **MSE Score** - Mean squared error between predictions and targets
- **MAE Score** - Mean absolute error
- **Correlation Score** - Pearson correlation coefficient

### XRD-Specific Metrics
- **Peak Position Error** - Accuracy of peak location detection
- **Peak Intensity Error** - Accuracy of peak height prediction
- **Peak Detection Rate** - Fraction of true peaks successfully identified

### Signal Quality Metrics
- **SNR Improvement** - Signal-to-noise ratio enhancement (dB)

## 📈 Statistical Analysis

For each metric, the system calculates:
- **Mean (μ)** - Average performance across multiple runs
- **Standard Deviation (σ)** - Consistency measure (lower = more reliable)
- **Min/Max** - Performance range bounds

## 🔧 Configuration

Edit the following parameters in `evaluate_diffusion_std.py`:

```python
# Number of evaluation runs for std calculation
num_runs = 10  # Increase for more statistical confidence

# Dataset size for faster evaluation
sample_limit = 200  # Reduce for faster evaluation

# Model parameters (must match your trained model)
hidden_channels = 16
time_embedding_dim = 256
num_res_blocks = 2
attention_levels = [1, 2]
num_levels = 2
```

## 📁 Required Files

1. **Model checkpoint** (optional): `./models/xrd_diffusion/improved_diffusion_model_best.pth`
   - If missing, uses randomly initialized model for demonstration

2. **Dataset** (required): `data/xrd_dataset_labeled_dtw_window.pt`
   - Must contain: `synth_xrd`, `real_xrd`, `fast_dtw_distance`

3. **Original model** (required): `diffusion_model_0.1.5.py`
   - Contains the model architecture definitions

## 📂 Output Files

Results are saved to `./evaluation_results/`:

- **`evaluation_results_with_std.json`** - Complete numerical results
- **`metrics_with_std.png`** - Overview bar chart with error bars
- **`distribution_*.png`** - Distribution analysis for each metric type

## 🎯 Interpreting Results

### Good Performance Indicators
- **Low reconstruction loss** (< 0.01) - Good real→synthetic transformation
- **High correlation scores** (> 0.95) - Strong pattern similarity
- **Low standard deviation** - Consistent, reliable performance
- **High peak detection rate** (> 0.8) - Accurate peak identification
- **Positive SNR improvement** - Effective noise reduction

### Performance Analysis Examples

```
Total Loss........................ μ=0.004523 ± σ=0.000341 [0.004102, 0.004891]
Peak Detection Rate............... μ=0.847000 ± σ=0.023100 [0.812000, 0.891000]
SNR Improvement (dB).............. μ=12.340000 ± σ=1.230000 [10.890000, 14.120000]
```

**Interpretation:**
- Total loss is low and consistent (small σ)
- Peak detection is good (~85%) with reasonable consistency
- SNR improvement is substantial (~12dB) with moderate variation

## 🔬 Advanced Usage

### Custom Evaluation Function

```python
from evaluate_diffusion_std import PerformanceEvaluator, load_model_and_data

# Load your model and data
model, diffusion, dataloader = load_model_and_data(
    model_path="path/to/model.pth",
    data_path="path/to/data.pt",
    device='cuda'
)

# Initialize evaluator
evaluator = PerformanceEvaluator(model, diffusion, device='cuda')

# Run single evaluation
single_results = evaluator.evaluate_single_run(dataloader, seed=42)

# Run multiple evaluations for std
results, all_results = evaluator.evaluate_with_std(
    dataloader,
    num_runs=20,  # More runs = better statistics
    base_seed=42
)
```

### Custom Metrics

You can extend the `PerformanceEvaluator` class to add custom metrics:

```python
def calculate_custom_metric(self, predicted, target):
    # Your custom metric calculation
    return metric_value

# Add to evaluate_single_run method:
custom_score = self.calculate_custom_metric(pred_sample, target_sample)
metrics['custom_scores'].append(custom_score)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure `diffusion_model_0.1.5.py` is in the same directory
2. **Memory errors**: Reduce `sample_limit` or `batch_size`
3. **CUDA errors**: Set `device='cpu'` for CPU-only evaluation
4. **Missing data**: Check that dataset file exists and has correct keys

### Performance Tips

- **GPU acceleration**: Use CUDA for faster evaluation
- **Batch size**: Adjust based on available memory
- **Sample limit**: Reduce for faster debugging/testing
- **Number of runs**: Start with 3-5 runs for quick testing

## 📚 References

This evaluation system is designed for the XRD diffusion model described in `diffusion_model_0.1.5.py`, which implements:
- U-Net architecture for XRD pattern denoising
- Temperature-conditioned diffusion process
- Scherrer equation-based peak broadening simulation
- Progressive training phases

For detailed model information, refer to the original diffusion model documentation.