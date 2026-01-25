from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

"""
This script processes CIF files to compute and store their X-ray diffraction (XRD) patterns.

Function Overview:
    - Scans the "../CIF_files/" directory for CIF files.
    - For each CIF file:
        * Loads the crystal structure using pymatgen's Structure.from_file.
        * Calculates the XRD pattern using XRDCalculator with the CuKa wavelength (1.5406 Å).
        * Extracts the diffraction pattern:
            - pattern.x: 2θ values (angles).
            - pattern.y: Corresponding intensities.
        * Pairs these into a list of (2θ, intensity) tuples.
    - Each file's data is stored as a tuple containing:
        (list of (2θ, intensity) pairs, file index, file name)
    - All tuples are collected into a dataset, which is then saved as a PyTorch .pt file
      (named "dataset_of_<number_of_files>_files.pt").

Example of an entry in the output dataset:
    ([(10.5, 120.0), (20.3, 450.0), (30.1, 300.0), ...], 0, "example_structure.cif")
    
Output:
-------
The script outputs a .pt file containing a list of tuples. Each tuple holds:
    1. A list of tuples representing the diffraction peaks (2θ, intensity).
    2. The index of the CIF file.
    3. The filename of the CIF file.
"""

def main():
    # Load structure from CIF file
    cif_folder_path = "../CIF_files/"
    dataset = []
    for index, file_name  in enumerate(os.listdir(cif_folder_path)):
        if file_name.endswith(".cif"):
            file_path = os.path.join(cif_folder_path, file_name)
            structure = Structure.from_file(file_path)

            xrd_calculator = XRDCalculator(wavelength="CuKa")  # Use "CuKa" (1.5406 Å) or another source

            pattern = xrd_calculator.get_pattern(structure)


            X = np.array(pattern.x)  # 2θ values
            Y = np.array(pattern.y)  # Intensities

            peak_parms = list(zip(X, Y))
            dataset.append((peak_parms, index, file_name))
    
    torch.save(dataset, f"dataset_of_{len(dataset)}_files.pt")


main()