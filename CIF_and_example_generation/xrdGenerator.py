import os
import re
import numpy as np
import torch

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

#############################
# Parsing & Utility Functions
#############################

def parse_and_zip_files(folder_path):
    """
    Parse all filenames from a folder and zip them into pairs of
    (CIF file, diffraction file) based on their prefixes.
    
    Returns:
        A list of (cif_file, diffraction_file) tuples.
    """
    all_files = os.listdir(folder_path)
    
    # Identify CIF and diffraction files
    cif_files = [f for f in all_files if f.endswith("_cif.cif")]
    diffraction_files = [f for f in all_files if f.endswith("_diffraction.txt")]
    
    cif_files.sort()
    diffraction_files.sort()
    
    # Pair them in sorted order
    return list(zip(cif_files, diffraction_files))

def parse_diffraction_file(file_path):
    """
    Parse a measured diffraction text file to extract 2 columns:
      - 2-THETA
      - INTENSITY
    Returns:
      (two_theta_values, intensities) as lists of floats.
    """
    two_theta_values = []
    intensities = []
    start_parsing = False
    
    # Regex to detect the header (e.g., "2-THETA  INTENSITY")
    header_pattern = re.compile(r'\b2-THETA\s+INTENSITY\b', re.IGNORECASE)
    
    with open(file_path, 'r') as f:
        for line in f:
            if not start_parsing and header_pattern.search(line):
                start_parsing = True
                continue
            
            if start_parsing:
                # Stop parsing if lines start with "XPOW" or "="
                if line.strip().startswith("XPOW") or line.strip().startswith("="):
                    break
                columns = line.split()
                if len(columns) >= 2:
                    try:
                        two_theta = float(columns[0])
                        intensity = float(columns[1])
                        two_theta_values.append(two_theta)
                        intensities.append(intensity)
                    except ValueError:
                        pass

    return two_theta_values, intensities

def zero_pad_pattern(two_theta, intensities, num_points=4500, max_angle=90.0):
    """
    Normalizes and zero-pads an XRD pattern onto a fixed grid:
      - Range: 0 to `max_angle` in 2θ
      - Length: `num_points`
    Returns:
        (two_theta_grid, intens_grid) as 1D np.arrays
    """
    two_theta = np.array(two_theta)
    intensities = np.array(intensities)
    
    # If no data, return empty
    if len(two_theta) == 0:
        return np.array([]), np.array([])
    
    # Normalize intensities so max = 1.0
    max_intensity = intensities.max()
    if max_intensity > 0:
        intensities = intensities / max_intensity
    
    # Create uniform 2θ grid
    two_theta_grid = np.linspace(0, max_angle, num_points)
    intens_grid = np.zeros(num_points, dtype=float)
    
    indices = np.searchsorted(two_theta_grid, two_theta)
    indices = np.clip(indices, 0, num_points - 1)
    intens_grid[indices] = intensities
    
    return two_theta_grid, intens_grid

def generate_synthetic_xrd(cif_path, wavelength=1.54184):
    """
    Generate a synthetic XRD pattern from a CIF using pymatgen's XRDCalculator.
    Returns (two_theta_list, intensity_list).
    """
    structure = Structure.from_file(cif_path)
    xrd_calc = XRDCalculator(wavelength=wavelength)
    pattern = xrd_calc.get_pattern(structure)
    
    return list(pattern.x), list(pattern.y)

############################
# Main Processing + Saving
############################

def process_and_save_torch(
    src_folder, 
    output_path="xrd_dataset.pt",
    num_points=4500,
    max_angle=90.0,
    wavelength=1.54184
):
    """
    1. Finds all (CIF, measured diffraction) pairs in src_folder.
    2. For each pair:
        - Parse the measured XRD file -> (2θ, intensity).
        - Generate the synthetic XRD from the CIF -> (2θ, intensity).
        - Zero-pad both to the same grid of length `num_points`.
    3. Stores all pairs in a single dictionary of torch Tensors, then torch.save().

    At any failure in the process, increments a fail counter and skips.
    Finally, prints # successes, # failures.

    Args:
        src_folder (str): Folder containing `_cif.cif` and `_diffraction.txt` files.
        output_path (str): Where to save the final .pt file.
        num_points (int): Number of points for zero-padding each XRD curve.
        max_angle (float): The maximum 2θ angle in degrees for the zero-padding.
        wavelength (float): X-ray wavelength for synthetic pattern generation.
    """
    file_pairs = parse_and_zip_files(src_folder)
    print(f"Found {len(file_pairs)} pairs in '{src_folder}'.")

    real_patterns = []      # measured intensities
    synth_patterns = []     # synthetic intensities
    file_info = []          # store pair info if needed (filenames, etc.)

    success_count = 0
    fail_count = 0

    for i, (cif_file, diff_file) in enumerate(file_pairs, start=1):
        cif_path = os.path.join(src_folder, cif_file)
        diff_path = os.path.join(src_folder, diff_file)

        ###########################
        # 1) Basic file checks
        ###########################
        if not (os.path.isfile(cif_path) and os.path.isfile(diff_path)):
            print(f"Skipping pair due to missing file (CIF: {cif_file}, Diff: {diff_file})")
            fail_count += 1
            continue
        
        ###########################
        # 2) Parse measured XRD
        ###########################
        try:
            exp_2theta, exp_intens = parse_diffraction_file(diff_path)
        except Exception as e:
            print(f"Failed to parse measured XRD from {diff_file}: {e}")
            fail_count += 1
            continue
        
        if not exp_2theta:
            print(f"Warning: No valid data from {diff_file}")
            fail_count += 1
            continue
        
        ###########################
        # 3) Generate synthetic XRD
        ###########################
        try:
            synth_2theta, synth_intens = generate_synthetic_xrd(cif_path, wavelength=wavelength)
        except Exception as e:
            print(f"Failed to generate synthetic XRD from {cif_file}: {e}")
            fail_count += 1
            continue

        if not synth_2theta:
            print(f"Warning: Empty synthetic XRD from {cif_file}")
            fail_count += 1
            continue

        ###########################
        # 4) Zero-pad both
        ###########################
        try:
            _, exp_intens_grid = zero_pad_pattern(
                exp_2theta, exp_intens, 
                num_points=num_points, 
                max_angle=max_angle
            )
            _, synth_intens_grid = zero_pad_pattern(
                synth_2theta, synth_intens, 
                num_points=num_points, 
                max_angle=max_angle
            )
        except Exception as e:
            print(f"Failed to zero-pad data for {cif_file} & {diff_file}: {e}")
            fail_count += 1
            continue
        
        if len(exp_intens_grid) == 0 or len(synth_intens_grid) == 0:
            print(f"Warning: Zero-length data encountered for {diff_file}, {cif_file}")
            fail_count += 1
            continue

        ###########################
        # 5) Store data
        ###########################
        real_patterns.append(exp_intens_grid)
        synth_patterns.append(synth_intens_grid)
        file_info.append((cif_file, diff_file))
        success_count += 1

        if i % 50 == 0:
            print(f"... processed {i} pairs so far")

    ###########################
    # Convert lists to Tensors
    ###########################
    if len(real_patterns) == 0:
        print("No successful patterns were processed. Exiting without saving.")
        print(f"Successes: {success_count}, Failures: {fail_count}")
        return

    real_tensor  = torch.tensor(real_patterns,  dtype=torch.float32)
    synth_tensor = torch.tensor(synth_patterns, dtype=torch.float32)

    dataset_dict = {
        "real_xrd": real_tensor,     # shape: (N, num_points)
        "synth_xrd": synth_tensor,   # shape: (N, num_points)
        "file_info": file_info       # list of (cif_filename, diffraction_filename)
    }

    #################################
    # Save & Print Summary
    #################################
    try:
        torch.save(dataset_dict, output_path)
        print(f"\nSuccessfully saved dataset to {output_path}")
        print(f" - real_xrd shape:  {real_tensor.shape}")
        print(f" - synth_xrd shape: {synth_tensor.shape}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")
    
    print(f"\nProcess complete. Successes: {success_count}, Failures: {fail_count}")

############################
# Example Usage in a Script
############################
if __name__ == "__main__":
    src_folder = "AMS_Downloads_test"             # Folder with your .cif and .txt files
    output_path = "xrd_dataset.pt"     # Where to store the final PyTorch dataset

    process_and_save_torch(
        src_folder=src_folder,
        output_path=output_path,
        num_points=4500,
        max_angle=90.0,
        wavelength=1.54184  # Common Cu K-alpha
    )
