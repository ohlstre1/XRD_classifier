# XRD Classifier Project

## Environment Setup (using uv)

```bash
cd /home/bert-linux-2nd/Documents/XRD_classifier

# Create virtual environment with uv
uv venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Data Setup

The scripts expect data at `data/` directory. Create symlink to existing data:

```bash
mkdir -p data
ln -s ../xrd_patterns_final/xrd_ams_patterns/xrd_dataset_labeled_dtw_window.pt data/xrd_dataset_labeled_dtw_window.pt
```

## Running Scripts

### Diffusion Model Training (GPU test)
```bash
cd diffusion
WANDB_MODE=disabled python main.py
```

### Classifier Training
```bash
cd XRD-classification/scripts
python train_modular.py --n_samples 100 --epochs 2 --disable_wandb
```

## Project Structure

- `diffusion/` - Diffusion model for XRD pattern augmentation
- `XRD-classification/` - Main classification scripts and configs
- `data/` - Dataset directory (symlinked)
- `xrd_patterns_final/` - Raw XRD pattern data files
- `data_generation/` - Scripts for generating synthetic XRD data
