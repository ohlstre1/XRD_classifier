# XRD Dataset Generation

This directory contains all scripts and documentation needed to generate the XRD pattern datasets used in this project.

## Data Organization

Datasets are organized by source in `data/`:

```
data_generation/
├── data/
│   ├── COD/                       ← Open license, redistributable
│   │   ├── COD_xrd_patterns_*.pt
│   │   └── README.md
│   │
│   ├── AMCSD/                     ← Restricted (requires MSA permission)
│   │   ├── xrd_dataset*.pt
│   │   └── README.md
│   │
│   ├── RRUFF/                     ← Research/educational use
│   │   ├── RRUFF_xrd_dataset_normalized.pt
│   │   └── README.md
│   │
│   ├── matching_compositions.csv  ← RRUFF↔COD ID mappings
│   └── matching_compositions.json
│
├── scripts/                       ← Data generation scripts
└── README.md                      ← This file
```

## Data Sources Summary

| Source | Size | License | Redistributable? |
|--------|------|---------|------------------|
| **COD** | ~10.7 GB | Open | Yes |
| **AMCSD** | ~1.4 GB | MSA/MAC | Needs permission |
| **RRUFF** | ~24 MB | Research use | Verify terms |

### COD (Crystallography Open Database)
- **Contents:** Synthetic XRD patterns calculated from CIF structures
- **License:** Open, freely redistributable
- **Source:** https://www.crystallography.net/cod/

### AMCSD (American Mineralogist Crystal Structure Database)
- **Contents:** Paired synthetic + measured XRD patterns
- **License:** Requires MSA/MAC permission to redistribute
- **Source:** http://rruff.geo.arizona.edu/AMS/amcsd.php
- **Note:** See `data/AMCSD/README.md` for regeneration instructions

### RRUFF
- **Contents:** Measured experimental XRD patterns
- **License:** Research/educational use
- **Source:** https://rruff.info/

## Dataset Inventory

### COD Datasets (`data/COD/`)

| File | Size | Description |
|------|------|-------------|
| `COD_xrd_patterns_and_ID.pt` | 7.4 GB | Full dataset with compound IDs |
| `COD_xrd_patterns_100000.pt` | 1.7 GB | 100,000 sample subset |
| `COD_xrd_patterns_50000.pt` | 860 MB | 50,000 sample subset |
| `COD_xrd_patterns_10000.pt` | 172 MB | 10,000 sample subset |
| `COD_xrd_patterns_1000.pt` | 18 MB | 1,000 sample subset |

### AMCSD Datasets (`data/AMCSD/`)

| File | Size | Description |
|------|------|-------------|
| `xrd_dataset_labeled_dtw_window.pt` | 459 MB | Patterns with DTW distance labels |
| `xrd_dataset_labeled_fastdw.pt` | 459 MB | Patterns with FastDTW labels |
| `xrd_dataset_large.pt` | 459 MB | Large pattern dataset |
| `xrd_dataset.pt` | 13 MB | Standard dataset |
| `xrd_dataset_dev.pt` | 1.1 MB | Development subset |

### RRUFF Dataset (`data/RRUFF/`)

| File | Size | Description |
|------|------|-------------|
| `RRUFF_xrd_dataset_normalized.pt` | 24 MB | Normalized RRUFF mineral patterns |

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

### 3. `scripts/xrd_peak_generator.py`
Simpler script that generates XRD patterns from CIF files only (no measured data).

**Input:**
- Folder with `.cif` files

**Output:**
- `.pt` file with list of (peak_params, index, filename) tuples

### 4. `scripts/generate_xy_data.py`
Generates fully synthetic XRD-like data for testing/development.

**Usage:**
```bash
python scripts/generate_xy_data.py <num_files>
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
    --dataset_path data/AMCSD/xrd_dataset_labeled_dtw_window.pt \
    --output_dir processed/ \
    --stratify_by_dtw
```

## How to Regenerate Datasets

### COD Data (Automated)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download CIF files from COD
python scripts/cod_scraper.py

# 3. Generate XRD patterns
python scripts/xrd_peak_generator.py
```

### AMCSD Data (Manual Download Required)

See `data/AMCSD/README.md` for detailed instructions. Summary:

1. Download CIF + diffraction files from http://rruff.geo.arizona.edu/AMS/amcsd.php
2. Run `python scripts/xrd_generator.py`
3. Run `python scripts/prepare_data.py`

### RRUFF Data (Manual)

See `data/RRUFF/README.md` for instructions.

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

## Publishing Guide

### What You CAN Publish (Open)
- `data/COD/` - All COD .pt files (open license)
- `scripts/` - All preprocessing scripts (your code)
- `data/matching_compositions.csv` - RRUFF↔COD mapping

### What Requires Permission
- `data/AMCSD/` - Needs MSA/MAC approval before redistribution

### Recommended Approach
1. Publish COD data and scripts freely
2. For AMCSD, provide regeneration instructions (in `data/AMCSD/README.md`)
3. Users can download from source and run your scripts

## Notes

- XRD patterns are normalized to have max intensity = 1.0
- 2θ range: 0° to 90° with 4500 data points (0.02° resolution)
- Wavelength: Cu Kα (1.54184 Å) by default
- Measured diffraction files must have "2-THETA" and "INTENSITY" columns
