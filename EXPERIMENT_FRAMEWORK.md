# XRD Classification Experiment Framework

## Overview
This framework helps track different experimental approaches for XRD pattern classification using prototypical learning.

## Current Best Results (Baseline to Beat)
- **File**: `results/500_sample_results_20251110_154538.json`
- **Performance**: 90% Top-1, 97% Top-5 accuracy
- **Configuration**:
  - 500 samples, 15 epochs, batch_size=16
  - learning_rate=1e-6, embedding_dim=256
  - Diffusion augmentation only (5x per sample)
  - Train: 2000 samples (400 compounds), Val: 100 samples

## Experimental Branches Strategy

### 1. Current Working Approaches
```
main branch: Latest stable version
├── approach-ideal-patterns: Each pattern = unique compound (current unstaged)
├── approach-duplicated-patterns: Pattern duplication with noise (staged)
└── approach-compound-grouping: Smart compound clustering (future)
```

### 2. Key Parameters to Track
- **Data Processing**: ideal vs duplicated vs compound-grouped patterns
- **Augmentation**: classical only, diffusion only, or mixed
- **Architecture**: embedding dimensions, loss weights
- **Training**: learning rates, batch sizes, epochs

### 3. Experimental Tracking Template

Each experiment should document:
```json
{
  "experiment_id": "YYYYMMDD_HHMMSS",
  "approach": "ideal|duplicated|compound-grouped",
  "hypothesis": "Why we expect this to work",
  "changes_from_baseline": ["specific changes made"],
  "configuration": {
    "data_approach": "description",
    "augmentation": "classical/diffusion/mixed",
    "key_parameters": {}
  },
  "results": {
    "top1_accuracy": 0.0,
    "top5_accuracy": 0.0,
    "training_time": 0.0
  },
  "analysis": "What worked/didn't work",
  "next_steps": ["what to try next"]
}
```

## Git Workflow for Experiments

### Before Each Experiment:
1. Commit current changes: `git add -A && git commit -m "experiment: [description]"`
2. Create branch: `git checkout -b exp-[approach]-[date]`
3. Run experiment and save results
4. Document findings in this file

### After Each Experiment:
1. Add results to git: `git add results/`
2. Update this framework: document what worked/didn't work
3. Commit: `git commit -m "results: [approach] - [performance]"`
4. Return to main: `git checkout main`

## Current Status & Next Actions

### What We Know Works (Baseline):
- **Diffusion-only augmentation**: 5x per sample
- **Ideal patterns**: Each treated as unique compound
- **Learning rate**: 1e-6 with cosine annealing
- **Architecture**: ResNet1D-18, 256 embedding dim

### Active Questions:
1. **Pattern Processing**: Does duplication help or hurt?
2. **Augmentation Mix**: Classical + Diffusion vs Diffusion-only?
3. **Validation Samples**: Why only 100 val samples vs 2000 train?

### Immediate Experiments to Run:
1. **Reproduce Baseline**: Ensure current setup can achieve 90% again
2. **Augmentation Ablation**: Test classical-only vs mixed approaches
3. **Data Balance**: Try equal val samples (500 compounds → 400 train, 500 val)
4. **Pattern Approach**: Compare ideal vs duplicated systematically

## Configuration Files to Track
- `configs/config.yaml`: Main configuration
- `scripts/train_500_samples.py`: Training script versions
- `utils/augmentation.py`: Augmentation strategies