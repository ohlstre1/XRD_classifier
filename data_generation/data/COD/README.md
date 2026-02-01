---
license: cc0-1.0
task_categories:
  - feature-extraction
  - other
tags:
  - chemistry
  - crystallography
  - materials-science
  - xrd
  - diffraction
pretty_name: COD XRD Patterns
size_categories:
  - 100K<n<1M
---

# COD XRD Patterns

Simulated X-ray diffraction (XRD) patterns generated from crystal structures in the [Crystallography Open Database (COD)](https://www.crystallography.net/cod/).

## What is XRD?

X-ray diffraction is a technique used to determine the atomic structure of crystalline materials. When X-rays scatter off a crystal lattice, they produce a characteristic pattern of peaks that acts as a "fingerprint" for that material. This dataset contains simulated XRD patterns computed from known crystal structures.

## Dataset Files

| File | Samples | Size | Description |
|------|---------|------|-------------|
| `COD_xrd_patterns_1000.pt` | 1,000 | 18 MB | Small subset for testing |
| `COD_xrd_patterns_10000.pt` | 10,000 | 172 MB | Medium subset |
| `COD_xrd_patterns_50000.pt` | 50,000 | 860 MB | Large subset |
| `COD_xrd_patterns_100000.pt` | 100,000 | 1.7 GB | Very large subset |
| `COD_xrd_patterns_and_ID.pt` | 436,196 | 7.4 GB | Full dataset with COD IDs |

## Data Format

Each `.pt` file is a PyTorch dictionary with the following structure:

```python
{
    'patterns': torch.Tensor,   # Shape: [N, 4500]
    'filenames': list           # COD IDs (e.g., "4326570.cif")
}
```

### Pattern Specifications

| Property | Value |
|----------|-------|
| 2θ range | 0° to 90° |
| Resolution | 0.02° per point (4500 total points) |
| Normalization | Intensities scaled to [0, 1] |
| X-ray source | Cu Kα (λ = 1.54184 Å) |

## Usage Example

```python
import torch
import matplotlib.pyplot as plt

# Load dataset
data = torch.load("COD_xrd_patterns_1000.pt", weights_only=False)
patterns = data['patterns']    # Shape: [1000, 4500]
filenames = data['filenames']  # COD IDs

print(f"Loaded {len(patterns)} patterns")
print(f"First pattern from: {filenames[0]}")

# Plot the first pattern
two_theta = torch.linspace(0, 90, 4500)
plt.figure(figsize=(10, 4))
plt.plot(two_theta, patterns[0])
plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity (normalized)')
plt.title(f'XRD Pattern - COD ID: {filenames[0].replace(".cif", "")}')
plt.show()
```

## Data Generation

Data generation scripts can be found at git (TODO: )

The XRD patterns were generated using [pymatgen](https://pymatgen.org/)'s XRDCalculator from CIF files downloaded from COD.

### Scripts

Two scripts are provided to regenerate or extend this dataset:

**1. `cod_scraper.py`** - Downloads CIF files from COD

```python
# Downloads CIF files for a list of COD IDs
# Requires: requests, pandas
# Input: CSV file with 'cod_id' column
# Output: CIF files in downloads/ directory
```

**2. `xrd_generator.py`** - Converts CIF files to XRD patterns

```python
# Key function: generate_synthetic_xrd()
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def generate_synthetic_xrd(cif_path, wavelength=1.54184):
    structure = Structure.from_file(cif_path)
    xrd_calc = XRDCalculator(wavelength=wavelength)
    pattern = xrd_calc.get_pattern(structure)
    return list(pattern.x), list(pattern.y)
```

### Requirements

```
pymatgen
torch
numpy
requests
pandas
```

## Related Databases (Not Included)

Due to licensing restrictions, patterns from these databases are not included but can be generated using the provided scripts:

### RRUFF (Measured XRD Patterns)

- **Source:** https://rruff.info/
- **Contains:** Experimental XRD from mineral specimens
- **License:** Research/educational use - verify terms before redistribution

### AMCSD (Paired Synthetic + Measured)

- **Source:** http://rruff.geo.arizona.edu/AMS/amcsd.php
- **Contains:** Paired CIF structures with measured diffraction data
- **License:** MSA/MAC - requires permission to redistribute
- **Note:** Use the provided scripts to regenerate from source


Please also cite the Crystallography Open Database (more info at https://www.crystallography.net/cod/)

## License

This dataset is released under **CC0 1.0 Universal (Public Domain Dedication)**.

The underlying crystal structure data comes from the Crystallography Open Database, which makes its data freely available for any purpose.
