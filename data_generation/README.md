# XRD Dataset Generation

This directory contains all scripts and documentation needed to generate the XRD pattern datasets used in this project.

## Overview

The datasets in `xrd_patterns_final/` were generated from three primary sources:
- **COD** (Crystallography Open Database) - Large database of crystal structures
- **RRUFF** - Mineral database with measured XRD patterns
- **AMS** (American Mineralogist Structure Database) - Mineral crystal structures with measured diffraction data

## Dataset Inventory

### COD Datasets (`xrd_patterns_final/`)

| File | Size | Description |
|------|------|-------------|
| `COD_xrd_patterns_and_ID.pt` | 7.8 GB | Full dataset with compound IDs |
| `COD_xrd_patterns_100000.pt` | 1.8 GB | 100,000 sample subset |
| `COD_xrd_patterns_50000.pt` | 901 MB | 50,000 sample subset |
| `COD_xrd_patterns_10000.pt` | 180 MB | 10,000 sample subset |
| `COD_xrd_patterns_1000.pt` | 18 MB | 1,000 sample subset |

### RRUFF Dataset (`xrd_patterns_final/`)

| File | Size | Description |
|------|------|-------------|
| `RRUFF_xrd_dataset_normalized.pt` | 24 MB | Normalized RRUFF mineral patterns |

### AMS Datasets (`xrd_patterns_final/xrd_ams_patterns/`)

| File | Size | Description |
|------|------|-------------|
| `xrd_dataset_labeled_dtw_window.pt` | 480 MB | Patterns with DTW distance labels |
| `xrd_dataset_labeled_fastdw.pt` | 480 MB | Patterns with FastDTW labels |
| `xrd_dataset_large.pt` | 480 MB | Large pattern dataset |
| `xrd_dataset.pt` | 13 MB | Standard dataset |
| `xrd_dataset_dev.pt` | 1 MB | Development subset |

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ COD Database    │ AMS Database    │ RRUFF Database              │
│ (CIF files)     │ (CIF + measured)│ (measured patterns)         │
└────────┬────────┴────────┬────────┴─────────────┬───────────────┘
         │                 │                      │
         ▼                 ▼                      │
┌─────────────────────────────────────────┐      │
│        cod_scraper.py                   │      │
│  Downloads CIF files from COD           │      │
└────────────────┬────────────────────────┘      │
                 │                               │
                 ▼                               │
┌─────────────────────────────────────────┐      │
│        xrd_generator.py                 │      │
│  - Parses CIF files (pymatgen)          │      │
│  - Generates synthetic XRD patterns     │      │
│  - Parses measured diffraction files    │      │
│  - Zero-pads to fixed grid (4500 pts)   │      │
│  - Outputs .pt tensor files             │◄─────┘
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│        prepare_data.py                  │
│  - Normalizes patterns to [0,1]         │
│  - Creates compound mappings            │
│  - Generates train/val splits           │
│  - Computes DTW statistics              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     Training-Ready Datasets (.pt)       │
│  - synth_xrd: [N, 4500] tensor          │
│  - real_xrd: [N, 4500] tensor           │
│  - file_info: list of filenames         │
│  - fast_dtw_distance: similarity scores │
└─────────────────────────────────────────┘
```

## Scripts

### 1. `scripts/cod_scraper.py`
Downloads CIF files from the Crystallography Open Database.

**Input:**
- `data/matching_compositions.csv` - List of COD IDs to download

**Output:**
- `downloads/` folder with `.cif` files

**Usage:**
```bash
cd data_generation
python scripts/cod_scraper.py
```

### 2. `scripts/xrd_generator.py`
Main data generation script. Generates synthetic XRD patterns from CIF files and pairs them with measured diffraction data.

**Input:**
- Folder containing paired `*_cif.cif` and `*_diffraction.txt` files

**Output:**
- PyTorch `.pt` file with dictionary:
  - `real_xrd`: Measured XRD patterns [N, 4500]
  - `synth_xrd`: Synthetic XRD patterns [N, 4500]
  - `file_info`: List of (cif_filename, diffraction_filename) tuples

**Key Parameters:**
- `num_points`: Grid resolution (default: 4500)
- `max_angle`: Maximum 2θ angle (default: 90°)
- `wavelength`: X-ray wavelength (default: 1.54184 Å for Cu Kα)

**Usage:**
```bash
python scripts/xrd_generator.py
# Edit the script to set src_folder and output_path
```

**Example (in Python):**
```python
from scripts.xrd_generator import process_and_save_torch

process_and_save_torch(
    src_folder="path/to/cif_and_diffraction_files",
    output_path="xrd_dataset.pt",
    num_points=4500,
    max_angle=90.0,
    wavelength=1.54184
)
```

### 3. `scripts/xrd_peak_generator.py`
Simpler script that generates XRD patterns from CIF files only (no measured data).

**Input:**
- Folder with `.cif` files

**Output:**
- `.pt` file with list of (peak_params, index, filename) tuples

**Usage:**
```bash
python scripts/xrd_peak_generator.py
```

### 4. `scripts/generate_xy_data.py`
Generates fully synthetic XRD-like data for testing/development.

**Usage:**
```bash
python scripts/generate_xy_data.py <num_files>
# Example: python scripts/generate_xy_data.py 1000
```

### 5. `scripts/prepare_data.py`
Preprocesses raw XRD datasets for model training.

**Input:**
- Raw `.pt` dataset file

**Output:**
- `compound_mapping.json` - Compound ID to pattern mapping
- `train_val_split.json` - Train/validation split indices
- `dataset_statistics.json` - Dataset statistics

**Usage:**
```bash
python scripts/prepare_data.py \
    --dataset_path ../xrd_patterns_final/xrd_ams_patterns/xrd_dataset_labeled_dtw_window.pt \
    --output_dir processed/ \
    --stratify_by_dtw
```

## How to Regenerate Datasets

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download CIF Files (for COD datasets)
```bash
# Edit data/matching_compositions.csv with desired COD IDs
python scripts/cod_scraper.py
```

### Step 3: Generate XRD Patterns
```bash
# For paired CIF + measured diffraction data:
python scripts/xrd_generator.py

# For CIF-only synthetic patterns:
python scripts/xrd_peak_generator.py
```

### Step 4: Prepare for Training
```bash
python scripts/prepare_data.py \
    --dataset_path <path_to_raw_dataset.pt> \
    --output_dir processed/
```

## Data Format Details

### Raw Dataset Format (from xrd_generator.py)
```python
{
    'real_xrd': torch.Tensor,      # Shape: [N, 4500] - Measured patterns
    'synth_xrd': torch.Tensor,     # Shape: [N, 4500] - Synthetic patterns
    'file_info': list              # List of (cif_file, diffraction_file) tuples
}
```

### Prepared Dataset Format (from prepare_data.py)
```python
# compound_mapping.json
{
    "compound_00000": {
        "index": 0,
        "synth_pattern": [...],    # 4500 float values
        "real_pattern": [...],     # 4500 float values
        "file_info": "...",
        "dtw_distance": 0.123,
        "synth_stats": {"mean": ..., "std": ..., "min": ..., "max": ...},
        "real_stats": {"mean": ..., "std": ..., "min": ..., "max": ...}
    },
    ...
}

# train_val_split.json
{
    "train": ["compound_00001", "compound_00003", ...],
    "val": ["compound_00002", "compound_00005", ...],
    "split_config": {
        "train_ratio": 0.8,
        "random_seed": 42,
        "stratify_by_dtw": true
    }
}
```

## Missing Scripts / Manual Steps

### AMS Data Acquisition (NOT AUTOMATED)

The AMS (American Mineralogist Structure Database) data was obtained manually. There is **no scraper script** for AMS in this repository.

**To obtain AMS data:**
1. Visit the American Mineralogist Crystal Structure Database: http://rruff.geo.arizona.edu/AMS/amcsd.php
2. Manually download CIF files and their corresponding measured diffraction files
3. Name files in pairs:
   - `<name>_cif.cif` - Crystal structure file
   - `<name>_diffraction.txt` - Measured XRD data with "2-THETA" and "INTENSITY" columns
4. Place all files in a single folder (e.g., `AMS_Downloads/`)
5. Run `xrd_generator.py` pointing to that folder

**TODO:** Create an `ams_scraper.py` script to automate this process.

### RRUFF Data Acquisition (NOT AUTOMATED)

Similarly, the RRUFF data (`RRUFF_xrd_dataset_normalized.pt`) was obtained through a process not captured in these scripts.

**RRUFF Database:** https://rruff.info/

## Notes

- XRD patterns are normalized to have max intensity = 1.0
- 2θ range: 0° to 90° with 4500 data points (0.02° resolution)
- Wavelength: Cu Kα (1.54184 Å) by default
- Measured diffraction files must have "2-THETA" and "INTENSITY" columns
