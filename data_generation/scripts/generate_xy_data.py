#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import random

def generate_xy_data(x_range=(0, 90), num_points=900, num_peaks=50,
                      high_frac=0.1, mid_frac=0.4,
                      high_range=(50, 100), mid_range=(20, 50), low_range=(2, 20), save=False):
    """
    Generate discrete (x, y) datapoints corresponding to XRD peak positions with adjustable intensity groups.
s
    Parameters:
        x_range (tuple): The range (min, max) of the 2Theta angle.
        num_points (int): Number of grid points between x_range[0] and x_range[1].
        num_peaks (int): Total number of peaks (datapoints) to simulate.
        
        high_frac (float): Fraction of peaks that have high intensity.
        mid_frac (float): Fraction of peaks that have medium intensity.
                         (The remaining fraction is assigned to low intensity.)
                         
        high_range (tuple): Intensity range for high intensity peaks.
        mid_range (tuple): Intensity range for medium intensity peaks.
        low_range (tuple): Intensity range for low intensity peaks.
        
    Returns:
        x_peaks (np.ndarray): Array of 2Theta values for the peaks.
        y_peaks (np.ndarray): Array of intensity values for the peaks.
        peak_params (list of tuples): Each tuple is (2Theta, intensity) for a peak.
    """
    # Validate that the specified fractions do not exceed 1.
    if high_frac + mid_frac > 1:
        raise ValueError("The sum of high_frac and mid_frac should not exceed 1.")
    
    # Calculate the fraction for low intensity peaks.
    low_frac = 1 - (high_frac + mid_frac)
    
    # Determine the number of peaks in each category.
    num_high = int(round(num_peaks * high_frac))
    num_mid = int(round(num_peaks * mid_frac))
    num_low = num_peaks - num_high - num_mid  # Ensure the total equals num_peaks.
    
    # Create a grid of x values with the desired resolution.
    x_grid = np.linspace(x_range[0], x_range[1], num_points)
    
    # Randomly choose num_peaks positions from the grid (without replacement).
    chosen_indices = np.random.choice(len(x_grid), size=num_peaks, replace=False)
    chosen_indices.sort()  # Sort so that the x values are in order.
    x_peaks = x_grid[chosen_indices]
    
    # Generate intensities for each group:
    intensities_high = np.random.uniform(high_range[0], high_range[1], num_high) if num_high > 0 else np.array([])
    intensities_mid = np.random.uniform(mid_range[0], mid_range[1], num_mid) if num_mid > 0 else np.array([])
    intensities_low = np.random.uniform(low_range[0], low_range[1], num_low) if num_low > 0 else np.array([])
    
    # Combine all intensity values and shuffle so that the intensity groups are randomly assigned to the x positions.
    y_peaks = np.concatenate((intensities_high, intensities_mid, intensities_low))
    np.random.shuffle(y_peaks)
    
    # Bundle the peak parameters for verification.
    peak_params = list(zip(x_peaks, y_peaks))
    
    # Save the generated data to a file if requested.


    return x_peaks, y_peaks, peak_params

def generate_labeled_datapoints(num_of_files):
    dataset = []
    x_range = (0, 90)
    num_points = 900
    high_frac = 0.1
    mid_frac = 0.4
    high_range = (50, 100)
    mid_range = (20, 50)
    low_range = (2, 20)


    for i in range(num_of_files):
        num_peaks = random.randint(20, 60)
        _, _, peak_params = generate_xy_data(
            x_range, num_points, num_peaks,
            high_frac, mid_frac,
            high_range, mid_range, low_range
        )
        dataset.append((peak_params, i))
    
    torch.save(dataset, f"dataset_of_{num_of_files}_files.pt")

def main():
    parser = argparse.ArgumentParser(description='Generate labeled XRD data files.')
    parser.add_argument('num_files', type=int, help='Number of files to generate')
    args = parser.parse_args()
    
    generate_labeled_datapoints(args.num_files)

if __name__ == '__main__':
    main()