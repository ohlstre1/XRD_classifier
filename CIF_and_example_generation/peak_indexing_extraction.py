import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def index_xrd_peaks(csv_path, angle_col='angle', intensity_col='intensity', prominence=10, height=50):
    """
    Reads an XRD dataset from a CSV file, detects peaks, and returns a DataFrame with peak angles and intensities.
    
    Parameters:
    - csv_path (str): Path to the CSV file.
    - angle_col (str): Name of the column representing angles (2θ).
    - intensity_col (str): Name of the column representing intensity values.
    - prominence (float): Minimum prominence of peaks to filter noise.
    - height (float): Minimum height of peaks to be considered.

    Returns:
    - pd.DataFrame: DataFrame containing indexed peaks (angle and intensity).
    """
    xrd = pd.read_csv(csv_path)

    peaks, _ = find_peaks(xrd[intensity_col], prominence=prominence, height=height)

    peak_angles = xrd[angle_col].iloc[peaks].values
    peak_intensities = xrd[intensity_col].iloc[peaks].values

    peaks_df = pd.DataFrame({'Angle (2θ)': peak_angles, 'Intensity': peak_intensities})

    return peaks_df  


