# RRUFF XRD Patterns

## License
RRUFF data is provided for research and educational purposes. Check current terms at the RRUFF website before redistribution.

- Source: https://rruff.info/
- Terms: https://rruff.info/about/

## Contents

| File | Description | Size |
|------|-------------|------|
| `RRUFF_xrd_dataset_normalized.pt` | Normalized measured XRD patterns | ~24 MB |

## About RRUFF

RRUFF is a comprehensive database of mineral data including:
- Raman spectra
- X-ray diffraction patterns
- Chemistry data
- Cell parameters

The XRD patterns in this dataset are **measured experimental patterns**, not synthetic calculations.

## Data Generation

These patterns were extracted and normalized from RRUFF's publicly available XRD measurements.

### Regeneration Steps

1. Download XRD data from RRUFF:
   - Visit https://rruff.info/
   - Navigate to the XRD data section
   - Download powder diffraction files

2. Process and normalize:
```bash
cd data_generation/scripts
python xrd_generator.py --source rruff --input_dir ../raw/rruff/
```

## RRUFF ↔ COD Mapping

The file `matching_compositions.csv` in the parent directory contains mappings between RRUFF mineral IDs and corresponding COD structure IDs (14,255 entries). This enables:
- Cross-referencing measured (RRUFF) and synthetic (COD) patterns
- Validating synthetic pattern generation
- Building paired datasets

## Data Format

The `.pt` file contains:
- `patterns`: Normalized XRD intensity patterns
- `two_theta`: 2-theta angle values
- `mineral_names`: RRUFF mineral identifications
- `rruff_ids`: Original RRUFF database IDs

## Citation

If you use RRUFF data, please cite:
> Lafuente, B., Downs, R.T., Yang, H., and Stone, N. (2015). The power of databases: the RRUFF project. In: Highlights in Mineralogical Crystallography, T. Armbruster and R.M. Danisi, eds. Berlin, Germany, W. De Gruyter, pp 1-30.
