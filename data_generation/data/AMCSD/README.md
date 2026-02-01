# AMCSD (American Mineralogist Crystal Structure Database) XRD Patterns

## License
**Restricted** - Redistribution requires permission from MSA/MAC.

The AMCSD data is maintained by the Mineralogical Society of America (MSA) and the Mineralogical Association of Canada (MAC). Redistribution of derived data requires explicit permission.

- Source: http://rruff.geo.arizona.edu/AMS/amcsd.php
- Contact for permissions: MSA/MAC

## Contents

| File | Description | Size |
|------|-------------|------|
| `xrd_dataset.pt` | Base paired dataset | ~13 MB |
| `xrd_dataset_dev.pt` | Development subset | ~1.1 MB |
| `xrd_dataset_large.pt` | Extended dataset | ~459 MB |
| `xrd_dataset_labeled_dtw_window.pt` | DTW-windowed labeled pairs | ~459 MB |
| `xrd_dataset_labeled_fastdw.pt` | FastDTW labeled pairs | ~459 MB |

## Why This Data is Special

AMCSD uniquely provides **paired synthetic and measured XRD patterns** for the same mineral specimens. This enables:
- Training models to map synthetic → measured patterns
- Validating synthetic XRD generation methods
- Domain adaptation between simulation and experiment

## How to Obtain This Data

Since we cannot redistribute AMCSD data directly, follow these steps to generate it yourself:

### Step 1: Download from AMCSD

1. Visit http://rruff.geo.arizona.edu/AMS/amcsd.php
2. Download the CIF structure files
3. Download the corresponding measured diffraction data (DIF format)
4. Place files in `data_generation/raw/amcsd/`

### Step 2: Generate Dataset

```bash
cd data_generation/scripts

# Generate XRD patterns from CIF + measured pairs
python xrd_generator.py --source amcsd --input_dir ../raw/amcsd/

# Create train/val splits and labeled datasets
python prepare_data.py --source amcsd
```

### Step 3: Apply DTW Alignment (Optional)

For the labeled datasets with DTW alignment:
```bash
python prepare_data.py --source amcsd --dtw_window 50
```

## Data Format

Each `.pt` file contains:
- `synthetic_patterns`: XRD patterns calculated from CIF structures
- `measured_patterns`: Corresponding experimental XRD measurements
- `mineral_names`: Mineral identification labels
- `amcsd_ids`: Original AMCSD database IDs

