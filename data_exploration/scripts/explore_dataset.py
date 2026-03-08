#!/usr/bin/env python3
"""
XRD Dataset Exploration Script
Generates comprehensive statistics and visualizations for thesis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import json

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / 'xrd_patterns_final/xrd_ams_patterns/xrd_dataset_labeled_dtw_window.pt'
FIGURES_DIR = BASE_DIR / 'data_exploration/figures'
OUTPUT_DIR = BASE_DIR / 'data_exploration/output'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_compound(file_info_tuple):
    """Extract compound name from file_info tuple."""
    filename = file_info_tuple[0]
    parts = filename.split('_')
    return parts[0]


def load_data():
    """Load the XRD dataset."""
    print(f"Loading data from {DATA_PATH}")
    data = torch.load(DATA_PATH, weights_only=False)
    return data


def compute_statistics(data):
    """Compute comprehensive statistics about the dataset."""
    real_xrd = data['real_xrd']
    synth_xrd = data['synth_xrd']
    dtw_distances = data['fast_dtw_distance']
    file_info = data['file_info']

    # Extract compound names
    compounds = [extract_compound(fi) for fi in file_info]
    compound_counts = Counter(compounds)
    counts = list(compound_counts.values())

    stats = {
        'dataset': {
            'total_patterns': len(compounds),
            'unique_compounds': len(compound_counts),
            'pattern_length': real_xrd.shape[1],
        },
        'class_distribution': {
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': float(np.mean(counts)),
            'median_samples': float(np.median(counts)),
            'std_samples': float(np.std(counts)),
            'classes_with_1_sample': sum(1 for c in counts if c == 1),
            'classes_with_le_5_samples': sum(1 for c in counts if c <= 5),
            'classes_with_ge_50_samples': sum(1 for c in counts if c >= 50),
        },
        'real_xrd': {
            'min': float(real_xrd.min()),
            'max': float(real_xrd.max()),
            'mean': float(real_xrd.mean()),
            'std': float(real_xrd.std()),
        },
        'synth_xrd': {
            'min': float(synth_xrd.min()),
            'max': float(synth_xrd.max()),
            'mean': float(synth_xrd.mean()),
            'std': float(synth_xrd.std()),
        },
        'dtw_distance': {
            'min': float(dtw_distances.min()),
            'max': float(dtw_distances.max()),
            'mean': float(dtw_distances.mean()),
            'std': float(dtw_distances.std()),
            'median': float(torch.median(dtw_distances)),
        },
        'top_compounds': dict(compound_counts.most_common(20)),
    }

    return stats, compounds, compound_counts


def plot_class_distribution(compound_counts, save_path):
    """Plot histogram of class sizes."""
    counts = list(compound_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full distribution
    ax1 = axes[0]
    ax1.hist(counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Number of Samples per Class')
    ax1.set_ylabel('Number of Classes')
    ax1.set_title('(a) Full Distribution')
    ax1.axvline(np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    ax1.axvline(np.median(counts), color='orange', linestyle='--', label=f'Median: {np.median(counts):.0f}')
    ax1.legend()

    # Zoomed distribution (1-20 samples)
    ax2 = axes[1]
    small_counts = [c for c in counts if c <= 20]
    ax2.hist(small_counts, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Number of Samples per Class')
    ax2.set_ylabel('Number of Classes')
    ax2.set_title('(b) Distribution for Classes with $\\leq$20 Samples')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cumulative_distribution(compound_counts, save_path):
    """Plot cumulative distribution of class sizes."""
    counts = sorted(compound_counts.values())
    cumulative = np.arange(1, len(counts) + 1) / len(counts) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, cumulative, 'b-', linewidth=2)
    ax.fill_between(counts, cumulative, alpha=0.3)
    ax.set_xlabel('Samples per Class')
    ax.set_ylabel('Cumulative Percentage of Classes (%)')
    ax.set_title('Cumulative Distribution of Class Sizes')

    # Add reference lines
    for threshold in [1, 5, 10]:
        pct = sum(1 for c in counts if c <= threshold) / len(counts) * 100
        ax.axvline(threshold, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(pct, color='gray', linestyle=':', alpha=0.7)
        ax.annotate(f'{pct:.1f}%', xy=(threshold, pct),
                   xytext=(threshold+2, pct+3), fontsize=10)

    ax.set_xlim(0, max(counts))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_compounds(compound_counts, save_path, n=20):
    """Plot bar chart of top N compounds by sample count."""
    top_n = compound_counts.most_common(n)
    compounds, counts = zip(*top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(compounds)), counts, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(compounds)))
    ax.set_yticklabels(compounds)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Samples')
    ax.set_title(f'Top {n} Mineral Classes by Sample Count')

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 1, i, str(count), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_patterns(data, save_path, n_patterns=4):
    """Plot sample XRD patterns comparing real vs synthetic."""
    real_xrd = data['real_xrd']
    synth_xrd = data['synth_xrd']
    file_info = data['file_info']
    dtw_distances = data['fast_dtw_distance']

    # Select diverse patterns (low, medium, high DTW distance)
    sorted_indices = torch.argsort(dtw_distances)
    indices = [
        sorted_indices[0].item(),  # Best match
        sorted_indices[len(sorted_indices)//3].item(),  # Low-mid
        sorted_indices[2*len(sorted_indices)//3].item(),  # Mid-high
        sorted_indices[-100].item(),  # Poor match (not worst to avoid outliers)
    ]

    # 2-theta range (assuming 5-90 degrees with 4500 points)
    two_theta = np.linspace(5, 90, 4500)

    fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 3*n_patterns))

    for i, idx in enumerate(indices):
        ax = axes[i]
        compound = extract_compound(file_info[idx])
        dtw = dtw_distances[idx].item()

        ax.plot(two_theta, real_xrd[idx].numpy(), 'b-', alpha=0.8,
                label='Experimental', linewidth=1)
        ax.plot(two_theta, synth_xrd[idx].numpy(), 'r-', alpha=0.6,
                label='Simulated', linewidth=1)
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'{compound} (DTW Distance: {dtw:.2f})')
        ax.legend(loc='upper right')
        ax.set_xlim(5, 90)

        if i == n_patterns - 1:
            ax.set_xlabel(r'2$\theta$ (degrees)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_dtw_distribution(data, save_path):
    """Plot distribution of DTW distances."""
    dtw_distances = data['fast_dtw_distance'].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(dtw_distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('DTW Distance')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Distribution of DTW Distances')
    ax1.axvline(np.mean(dtw_distances), color='red', linestyle='--',
                label=f'Mean: {np.mean(dtw_distances):.2f}')
    ax1.axvline(np.median(dtw_distances), color='orange', linestyle='--',
                label=f'Median: {np.median(dtw_distances):.2f}')
    ax1.legend()

    # Box plot with log scale
    ax2 = axes[1]
    ax2.boxplot(dtw_distances, vert=True)
    ax2.set_ylabel('DTW Distance')
    ax2.set_title('(b) Box Plot of DTW Distances')
    ax2.set_xticklabels(['All Patterns'])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pattern_intensity_stats(data, save_path):
    """Plot statistics about pattern intensities."""
    real_xrd = data['real_xrd']

    # Compute per-pattern statistics
    pattern_maxes = real_xrd.max(dim=1).values.numpy()
    pattern_means = real_xrd.mean(dim=1).numpy()
    pattern_sums = real_xrd.sum(dim=1).numpy()

    # Find number of peaks per pattern (simplified: values > 0.1 * max)
    peaks_per_pattern = []
    for i in range(real_xrd.shape[0]):
        pattern = real_xrd[i].numpy()
        threshold = 0.1 * pattern.max()
        # Count local maxima above threshold
        peaks = 0
        for j in range(1, len(pattern)-1):
            if pattern[j] > threshold and pattern[j] > pattern[j-1] and pattern[j] > pattern[j+1]:
                peaks += 1
        peaks_per_pattern.append(peaks)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.hist(pattern_maxes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Maximum Intensity')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Distribution of Maximum Intensities')

    ax2 = axes[0, 1]
    ax2.hist(pattern_sums, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Integrated Intensity')
    ax2.set_ylabel('Count')
    ax2.set_title('(b) Distribution of Integrated Intensities')

    ax3 = axes[1, 0]
    ax3.hist(peaks_per_pattern, bins=range(0, max(peaks_per_pattern)+2),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax3.set_xlabel('Number of Peaks')
    ax3.set_ylabel('Count')
    ax3.set_title('(c) Distribution of Peak Counts per Pattern')

    # Average pattern
    ax4 = axes[1, 1]
    two_theta = np.linspace(5, 90, 4500)
    mean_pattern = real_xrd.mean(dim=0).numpy()
    std_pattern = real_xrd.std(dim=0).numpy()
    ax4.plot(two_theta, mean_pattern, 'b-', linewidth=1.5, label='Mean')
    ax4.fill_between(two_theta, mean_pattern - std_pattern, mean_pattern + std_pattern,
                     alpha=0.3, label='$\\pm$ 1 Std')
    ax4.set_xlabel(r'2$\theta$ (degrees)')
    ax4.set_ylabel('Intensity (a.u.)')
    ax4.set_title('(d) Average XRD Pattern Across Dataset')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_latex_table(stats):
    """Generate LaTeX table of dataset statistics."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Summary statistics of the XRD dataset.}
\label{tab:dataset_stats}
\begin{tabular}{lr}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Total XRD patterns & """ + f"{stats['dataset']['total_patterns']:,}" + r""" \\
Unique mineral classes & """ + f"{stats['dataset']['unique_compounds']:,}" + r""" \\
Pattern length (data points) & """ + f"{stats['dataset']['pattern_length']:,}" + r""" \\
2$\theta$ range & 5--90$^\circ$ \\
\midrule
\multicolumn{2}{l}{\textit{Class Distribution}} \\
\midrule
Minimum samples per class & """ + f"{stats['class_distribution']['min_samples']}" + r""" \\
Maximum samples per class & """ + f"{stats['class_distribution']['max_samples']}" + r""" \\
Mean samples per class & """ + f"{stats['class_distribution']['mean_samples']:.2f}" + r""" \\
Median samples per class & """ + f"{stats['class_distribution']['median_samples']:.0f}" + r""" \\
Classes with only 1 sample & """ + f"{stats['class_distribution']['classes_with_1_sample']:,}" + r""" (""" + f"{stats['class_distribution']['classes_with_1_sample']/stats['dataset']['unique_compounds']*100:.1f}" + r"""\%) \\
Classes with $\leq$5 samples & """ + f"{stats['class_distribution']['classes_with_le_5_samples']:,}" + r""" (""" + f"{stats['class_distribution']['classes_with_le_5_samples']/stats['dataset']['unique_compounds']*100:.1f}" + r"""\%) \\
Classes with $\geq$50 samples & """ + f"{stats['class_distribution']['classes_with_ge_50_samples']}" + r""" (""" + f"{stats['class_distribution']['classes_with_ge_50_samples']/stats['dataset']['unique_compounds']*100:.1f}" + r"""\%) \\
\midrule
\multicolumn{2}{l}{\textit{DTW Distance (Experimental vs. Simulated)}} \\
\midrule
Mean DTW distance & """ + f"{stats['dtw_distance']['mean']:.2f}" + r""" \\
Median DTW distance & """ + f"{stats['dtw_distance']['median']:.2f}" + r""" \\
Std. deviation & """ + f"{stats['dtw_distance']['std']:.2f}" + r""" \\
Range & """ + f"{stats['dtw_distance']['min']:.2f}--{stats['dtw_distance']['max']:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_latex_section(stats):
    """Generate complete LaTeX section for thesis."""
    latex = r"""\section{Dataset Description}
\label{sec:dataset}

This study utilizes a curated X-ray diffraction (XRD) pattern dataset derived from the American Mineralogist Crystal Structure Database (AMCSD). The dataset comprises paired experimental and simulated XRD patterns for mineral identification and classification tasks.

\subsection{Dataset Composition}

The dataset contains """ + f"{stats['dataset']['total_patterns']:,}" + r""" XRD patterns representing """ + f"{stats['dataset']['unique_compounds']:,}" + r""" unique mineral classes. Each pattern consists of """ + f"{stats['dataset']['pattern_length']:,}" + r""" intensity values spanning a $2\theta$ range of 5--90$^\circ$, providing comprehensive coverage of the diffraction angles typically used in powder XRD analysis. Table~\ref{tab:dataset_stats} summarizes the key statistics of the dataset.

""" + generate_latex_table(stats) + r"""

\subsection{Class Distribution Analysis}

The dataset exhibits significant class imbalance, a common challenge in mineral classification. As shown in Figure~\ref{fig:class_distribution}, the distribution of samples per class is heavily right-skewed. The median number of samples per class is """ + f"{stats['class_distribution']['median_samples']:.0f}" + r""", while the mean is """ + f"{stats['class_distribution']['mean_samples']:.2f}" + r""", indicating that most classes have very few samples while a small number of classes dominate the dataset.

Specifically, """ + f"{stats['class_distribution']['classes_with_1_sample']:,}" + r""" classes (""" + f"{stats['class_distribution']['classes_with_1_sample']/stats['dataset']['unique_compounds']*100:.1f}" + r"""\%) contain only a single sample, and """ + f"{stats['class_distribution']['classes_with_le_5_samples']:,}" + r""" classes (""" + f"{stats['class_distribution']['classes_with_le_5_samples']/stats['dataset']['unique_compounds']*100:.1f}" + r"""\%) have five or fewer samples. In contrast, only """ + f"{stats['class_distribution']['classes_with_ge_50_samples']}" + r""" classes (""" + f"{stats['class_distribution']['classes_with_ge_50_samples']/stats['dataset']['unique_compounds']*100:.1f}" + r"""\%) contain 50 or more samples. The most represented minerals include Iron, Periclase, Pyroxene-ideal, Pyroxene, and Spinel, each with approximately 100 samples.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/class_distribution.pdf}
    \caption{Distribution of samples per mineral class. (a) Full distribution showing the extreme right-skew characteristic of the dataset. (b) Zoomed view of classes with 20 or fewer samples, highlighting that the majority of classes fall in this range.}
    \label{fig:class_distribution}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/cumulative_distribution.pdf}
    \caption{Cumulative distribution of class sizes, showing the percentage of classes with a given number of samples or fewer.}
    \label{fig:cumulative}
\end{figure}

\subsection{XRD Pattern Characteristics}

Each entry in the dataset contains both an experimental XRD pattern and a corresponding simulated pattern calculated from the crystal structure. All patterns are normalized to have a maximum intensity of 1.0 and minimum of 0.0. Figure~\ref{fig:sample_patterns} presents representative examples of experimental-simulated pattern pairs, demonstrating varying degrees of agreement between the two.

The Dynamic Time Warping (DTW) distance metric quantifies the similarity between experimental and simulated patterns. The mean DTW distance is """ + f"{stats['dtw_distance']['mean']:.2f}" + r""" with a standard deviation of """ + f"{stats['dtw_distance']['std']:.2f}" + r""" (Figure~\ref{fig:dtw_distribution}). This variation reflects factors such as preferred orientation effects, peak broadening, background contributions, and structural differences between the ideal crystal structure and real samples.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/sample_patterns.pdf}
    \caption{Representative XRD patterns showing experimental (blue) and simulated (red) diffraction patterns for selected minerals. The DTW distance indicates the degree of agreement between experimental and theoretical patterns.}
    \label{fig:sample_patterns}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/dtw_distribution.pdf}
    \caption{Distribution of DTW distances between experimental and simulated XRD patterns. (a) Histogram with mean and median indicated. (b) Box plot showing the spread and outliers.}
    \label{fig:dtw_distribution}
\end{figure}

\subsection{Data Quality and Preprocessing}

The patterns in the dataset are clean, with no missing values (NaN) or infinite values detected. The intensity normalization ensures consistent scaling across all patterns, which is essential for neural network training. Figure~\ref{fig:pattern_stats} presents statistics on pattern characteristics across the dataset, including the distribution of integrated intensities and peak counts.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/pattern_intensity_stats.pdf}
    \caption{Statistical analysis of XRD pattern characteristics. (a) Distribution of maximum intensities. (b) Distribution of integrated intensities. (c) Distribution of peak counts per pattern. (d) Average XRD pattern across the entire dataset with standard deviation envelope.}
    \label{fig:pattern_stats}
\end{figure}

\subsection{Implications for Classification}

The extreme class imbalance presents significant challenges for training machine learning classifiers. Several strategies can be employed to address this:

\begin{itemize}
    \item \textbf{Data augmentation}: Generating synthetic XRD patterns using diffusion models or traditional augmentation techniques to balance the class distribution.
    \item \textbf{Few-shot learning}: Employing metric learning approaches such as Siamese networks or prototypical networks that can learn from limited examples.
    \item \textbf{Class weighting}: Applying inverse frequency weighting during training to prevent the model from being biased toward majority classes.
    \item \textbf{Hierarchical classification}: Leveraging mineralogical taxonomies to group related minerals and perform multi-level classification.
\end{itemize}

The paired experimental-simulated data also enables semi-supervised and self-supervised learning approaches, where the relationship between the two pattern types can provide additional training signal.

"""
    return latex


def main():
    """Main function to run all analyses."""
    print("=" * 60)
    print("XRD Dataset Exploration")
    print("=" * 60)

    # Load data
    data = load_data()

    # Compute statistics
    print("\nComputing statistics...")
    stats, compounds, compound_counts = compute_statistics(data)

    # Save statistics as JSON
    stats_path = OUTPUT_DIR / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_class_distribution(compound_counts, FIGURES_DIR / 'class_distribution.pdf')
    plot_cumulative_distribution(compound_counts, FIGURES_DIR / 'cumulative_distribution.pdf')
    plot_top_compounds(compound_counts, FIGURES_DIR / 'top_compounds.pdf')
    plot_sample_patterns(data, FIGURES_DIR / 'sample_patterns.pdf')
    plot_dtw_distribution(data, FIGURES_DIR / 'dtw_distribution.pdf')
    plot_pattern_intensity_stats(data, FIGURES_DIR / 'pattern_intensity_stats.pdf')

    # Generate LaTeX
    print("\nGenerating LaTeX output...")
    latex_table = generate_latex_table(stats)
    table_path = OUTPUT_DIR / 'dataset_table.tex'
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved: {table_path}")

    latex_section = generate_latex_section(stats)
    section_path = OUTPUT_DIR / 'dataset_section.tex'
    with open(section_path, 'w') as f:
        f.write(latex_section)
    print(f"Saved: {section_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"LaTeX output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
