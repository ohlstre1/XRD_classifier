#!/usr/bin/env python
"""
Generate validation figures and metrics for the diffusion model thesis chapter.

Produces:
  figures/overlay_comparison.pdf      – RQ1: visual replication quality
  figures/quantitative_metrics.pdf    – RQ2: RMSE & Pearson over test set
  figures/stochasticity_spread.pdf    – RQ3: N=1000 spread analysis
  figures/mae_histogram.pdf           – RQ3: MAE distribution
  figures/cv_peak_positions.pdf       – RQ4: peak position stability
  figures/denoising_waterfall.pdf     – Supplementary: noise level effect
  metrics_summary.json                – Numeric metrics for the tex files
"""

import sys
import os
import json
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(DIFFUSION_DIR)
sys.path.insert(0, DIFFUSION_DIR)

from models.complete_model import ImprovedDiffusionDenoiser
from diffusion.process import DiffusionProcess

FIGURE_DIR = os.path.join(SCRIPT_DIR, 'figures')
MODEL_PATH = os.path.join(DIFFUSION_DIR, 'models', 'xrd_diffusion', 'best_model.pth')
TEST_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'xrd_test_dataset.pt')

# Matplotlib defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_and_diffusion(device):
    """Load trained model and diffusion process."""
    model = ImprovedDiffusionDenoiser(
        in_channels=1,
        hidden_channels=16,
        time_embedding_dim=256,
        num_res_blocks=2,
        attention_levels=[1, 2],
        num_levels=2,
        temperature_condition=True,
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded from epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?')}")

    diffusion = DiffusionProcess(
        num_timesteps=1000,
        schedule_type='cosine',
        device=device,
    )
    return model, diffusion


def load_test_data(device):
    """Load test dataset and return CPU tensors."""
    data = torch.load(TEST_DATA_PATH, map_location='cpu', weights_only=False)
    synth = data['synth_xrd']            # [N, L]
    real = data['real_xrd']              # [N, L]
    dtw = data['fast_dtw_distance']      # [N]
    return synth, real, dtw


def select_representative_samples(dtw, percentiles=(10, 35, 65, 90)):
    """Return sample indices closest to the given DTW percentiles."""
    values = dtw.numpy()
    indices = []
    for p in percentiles:
        target = np.percentile(values, p)
        idx = int(np.argmin(np.abs(values - target)))
        indices.append(idx)
    return indices


def denoise(model, diffusion, noisy_x, t, temp):
    """Analytical one-step denoising:
       x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    """
    noise_pred = model(noisy_x, t, temp)
    alpha_bar_t = diffusion.alpha_bars[t].view(-1, 1, 1)
    x0_pred = (noisy_x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
    return x0_pred.clamp(0, 1)


# ---------------------------------------------------------------------------
# Graph 1 – Overlay Comparison  (RQ1)
# ---------------------------------------------------------------------------
def plot_overlay_comparison(model, diffusion, synth, real, dtw, indices, device):
    """2x2 grid: ideal input (black) -> model output (red) vs measured (blue dashed)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    percentile_labels = ['10th', '35th', '65th', '90th']

    for ax, idx, plabel in zip(axes, indices, percentile_labels):
        s = synth[idx].unsqueeze(0).unsqueeze(0).to(device)    # [1,1,L]
        temp = dtw[idx].unsqueeze(0).unsqueeze(0).to(device)   # [1,1]
        t_zero = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            pred = model(s, t_zero, temp)

        pred_np = pred.cpu().numpy().reshape(-1)
        synth_np = synth[idx].numpy()
        real_np = real[idx].numpy()
        x_axis = np.arange(len(synth_np))

        ax.plot(x_axis, synth_np, color='black', linewidth=1.0, label='Ideal (Rietveld)')
        ax.plot(x_axis, pred_np, color='red', linewidth=1.0, alpha=0.9, label='Model output')
        ax.plot(x_axis, real_np, color='blue', linewidth=0.8,
                linestyle='--', alpha=0.7, label='Measured')

        dtw_val = dtw[idx].item()
        ax.set_title(f'{plabel} percentile (DTW = {dtw_val:.1f})')
        ax.set_ylabel('Intensity (a.u.)')
        if ax in axes[2:]:
            ax.set_xlabel('Channel index')

    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Overlay comparison: ideal \u2192 model output vs measured pattern',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'overlay_comparison.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved overlay_comparison.pdf")


# ---------------------------------------------------------------------------
# Graph 6 – Quantitative Test-Set Metrics  (RQ2)
# ---------------------------------------------------------------------------
def compute_test_set_metrics(model, diffusion, synth, real, dtw, device,
                             batch_size=64):
    """RMSE, Pearson r, normalised cross-correlation per test sample at t=0."""
    n = len(synth)
    rmses, pearsons, cross_corrs = [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s = synth[start:end].unsqueeze(1).to(device)          # [B,1,L]
        temp = dtw[start:end].unsqueeze(1).to(device)          # [B,1]
        t_zero = torch.zeros(end - start, dtype=torch.long, device=device)

        with torch.no_grad():
            pred = model(s, t_zero, temp)

        pred_np = pred.cpu().numpy().squeeze(1)                # [B,L]
        real_np = real[start:end].numpy()                      # [B,L]

        for i in range(end - start):
            p, m = pred_np[i], real_np[i]
            rmses.append(float(np.sqrt(np.mean((p - m) ** 2))))
            r, _ = pearsonr(p, m)
            pearsons.append(float(r))
            # Normalised cross-correlation (max lag-0 value)
            ncc = np.sum((p - p.mean()) * (m - m.mean())) / \
                  (len(p) * p.std() * m.std() + 1e-12)
            cross_corrs.append(float(ncc))

    return np.array(rmses), np.array(pearsons), np.array(cross_corrs)


def plot_quantitative_summary(rmses, pearsons):
    """Side-by-side histograms of RMSE and Pearson r."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(rmses, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.axvline(np.mean(rmses), color='red', linestyle='--',
                label=f'Mean = {np.mean(rmses):.4f}')
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Count')
    ax1.set_title('RMSE distribution (model output vs measured)')
    ax1.legend()

    ax2.hist(pearsons, bins=30, color='coral', edgecolor='black', alpha=0.8)
    ax2.axvline(np.mean(pearsons), color='red', linestyle='--',
                label=f'Mean = {np.mean(pearsons):.4f}')
    ax2.set_xlabel('Pearson $r$')
    ax2.set_ylabel('Count')
    ax2.set_title('Pearson correlation distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'quantitative_metrics.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved quantitative_metrics.pdf")


# ---------------------------------------------------------------------------
# Graphs 2 & 3 – Stochastic Spread + MAE Histogram  (RQ3)
# ---------------------------------------------------------------------------
def run_stochasticity_analysis(model, diffusion, synth, dtw,
                               sample_idx, device,
                               n_runs=1000, t_step=50, batch_size=100):
    """N forward-diffusion → denoise passes at t=t_step on one fixed sample."""
    s = synth[sample_idx]            # [L]
    temp_val = dtw[sample_idx]
    all_outputs = []

    for start in range(0, n_runs, batch_size):
        bs = min(batch_size, n_runs - start)
        s_batch = s.unsqueeze(0).unsqueeze(0).expand(bs, 1, -1).clone().to(device)
        temp_batch = temp_val.view(1, 1).expand(bs, 1).clone().to(device)
        t_batch = torch.full((bs,), t_step, dtype=torch.long, device=device)

        with torch.no_grad():
            noisy_x, _ = diffusion.forward_diffusion(s_batch, t_batch)
            denoised = denoise(model, diffusion, noisy_x, t_batch, temp_batch)

        all_outputs.append(denoised.cpu().numpy().squeeze(1))   # [bs, L]

    return np.concatenate(all_outputs, axis=0)                   # [n_runs, L]


def plot_spread(outputs, synth_np, real_np, sample_idx, dtw_val):
    """Mean ± 1 sigma band with ideal and measured overlaid."""
    mean_out = outputs.mean(axis=0)
    std_out = outputs.std(axis=0)
    x_axis = np.arange(len(mean_out))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_axis, mean_out - std_out, mean_out + std_out,
                    alpha=0.3, color='red', label=r'$\pm 1\sigma$ band ($N$=1,000)')
    ax.plot(x_axis, mean_out, color='red', linewidth=1.0, label='Mean output')
    ax.plot(x_axis, synth_np, color='black', linewidth=0.8, label='Ideal (Rietveld)')
    ax.plot(x_axis, real_np, color='blue', linewidth=0.8,
            linestyle='--', alpha=0.7, label='Measured')

    ax.set_xlabel('Channel index')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'Stochastic spread: $N$=1,000 runs at $t$=50 '
                 f'(sample {sample_idx}, DTW={dtw_val:.1f})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'stochasticity_spread.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved stochasticity_spread.pdf")


def plot_mae_histogram(outputs, synth_np):
    """Histogram of MAE across N runs (each run vs ideal input)."""
    maes = np.mean(np.abs(outputs - synth_np), axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(maes, bins=40, color='mediumpurple', edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(maes), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean MAE = {np.mean(maes):.4f}')
    ax.set_xlabel('Mean Absolute Error')
    ax.set_ylabel('Count')
    ax.set_title('MAE distribution across $N$=1,000 stochastic runs ($t$=50)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'mae_histogram.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved mae_histogram.pdf")
    return maes


# ---------------------------------------------------------------------------
# Graph 4 – Peak Position Coefficient of Variation  (RQ4)
# ---------------------------------------------------------------------------
def plot_cv_analysis(outputs, synth_np):
    """CV of detected peak positions across N runs.

    Reference peaks are detected on the *ideal* (synth) pattern, which has
    full-range intensities, rather than on the mean stochastic output whose
    amplitude is compressed by the denoising formula.
    """
    # Detect reference peaks on the ideal pattern (full [0,1] range)
    peaks, _ = signal.find_peaks(synth_np, prominence=0.02, distance=20)
    if len(peaks) == 0:
        print("  Warning: no peaks at prominence=0.02, trying 0.005")
        peaks, _ = signal.find_peaks(synth_np, prominence=0.005, distance=10)
    if len(peaks) == 0:
        print("  Error: no peaks detected — skipping CV analysis")
        return np.array([])

    window = 15   # ± channels to search for local max
    peak_positions = np.zeros((len(outputs), len(peaks)))

    for run_idx in range(len(outputs)):
        spectrum = outputs[run_idx]
        for pk_idx, ref_pos in enumerate(peaks):
            lo = max(0, ref_pos - window)
            hi = min(len(spectrum), ref_pos + window + 1)
            local_max_offset = int(np.argmax(spectrum[lo:hi]))
            peak_positions[run_idx, pk_idx] = lo + local_max_offset

    # CV (%) per peak
    cvs = peak_positions.std(axis=0) / (peak_positions.mean(axis=0) + 1e-10) * 100

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(peaks)), 5))
    x = np.arange(len(peaks))
    ax.bar(x, cvs, color='teal', edgecolor='black', alpha=0.8)
    ax.axhline(5.0, color='red', linestyle='--', linewidth=1.5, label='5% threshold')
    ax.set_xlabel('Peak index')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title(f'Peak position stability across $N$=1,000 runs '
                 f'({len(peaks)} peaks detected)')
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'cv_peak_positions.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved cv_peak_positions.pdf "
          f"({len(peaks)} peaks, max CV = {cvs.max():.2f}%)")
    return cvs


# ---------------------------------------------------------------------------
# Graph 5 – Denoising Waterfall / Heatmap  (Supplementary)
# ---------------------------------------------------------------------------
def plot_waterfall(model, diffusion, synth, dtw, sample_idx, device):
    """Heatmap of model output at different noise timesteps."""
    timesteps = [0, 50, 100, 200, 500, 900]
    s = synth[sample_idx].unsqueeze(0).unsqueeze(0).to(device)
    temp = dtw[sample_idx].unsqueeze(0).unsqueeze(0).to(device)

    rows = []
    for t_val in timesteps:
        t = torch.full((1,), t_val, dtype=torch.long, device=device)
        with torch.no_grad():
            if t_val == 0:
                pred = model(s, t, temp)
            else:
                noisy_x, _ = diffusion.forward_diffusion(s, t)
                pred = denoise(model, diffusion, noisy_x, t, temp)
        rows.append(pred.cpu().numpy().reshape(-1))

    matrix = np.array(rows)                      # [len(timesteps), L]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_yticks(range(len(timesteps)))
    ax.set_yticklabels([f'$t$ = {t}' for t in timesteps])
    ax.set_xlabel('Channel index')
    ax.set_title('Model output at increasing noise levels')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'denoising_waterfall.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved denoising_waterfall.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ---- Load ----
    print("Loading model …")
    model, diffusion = load_model_and_diffusion(device)

    print("Loading test data …")
    synth, real, dtw = load_test_data(device)
    print(f"  {len(synth)} samples, pattern length {synth.shape[1]}\n")

    # Representative samples (DTW percentiles 10/35/65/90)
    indices = select_representative_samples(dtw)
    print("Representative DTW values:",
          [f"{dtw[i].item():.1f}" for i in indices])
    # Use 35th-percentile sample for single-sample analyses
    stoch_idx = indices[1]

    # ---- RQ1: overlay comparison ----
    print("\n[RQ1] Overlay comparison …")
    plot_overlay_comparison(model, diffusion, synth, real, dtw, indices, device)

    # ---- RQ2: quantitative metrics ----
    print("\n[RQ2] Test-set metrics …")
    rmses, pearsons, cross_corrs = compute_test_set_metrics(
        model, diffusion, synth, real, dtw, device)
    plot_quantitative_summary(rmses, pearsons)
    print(f"  RMSE     : {rmses.mean():.4f} ± {rmses.std():.4f}")
    print(f"  Pearson r: {pearsons.mean():.4f} ± {pearsons.std():.4f}")
    print(f"  NCC      : {cross_corrs.mean():.4f} ± {cross_corrs.std():.4f}")

    # ---- RQ3: stochastic variation ----
    print(f"\n[RQ3] Stochastic analysis (N=1000, t=50, sample {stoch_idx}) …")
    outputs = run_stochasticity_analysis(
        model, diffusion, synth, dtw, stoch_idx, device)
    plot_spread(outputs, synth[stoch_idx].numpy(), real[stoch_idx].numpy(),
                stoch_idx, dtw[stoch_idx].item())
    maes = plot_mae_histogram(outputs, synth[stoch_idx].numpy())

    # ---- RQ4: peak-position CV ----
    print("\n[RQ4] Peak position CV …")
    cvs = plot_cv_analysis(outputs, synth[stoch_idx].numpy())

    # ---- Supplementary: waterfall ----
    print("\n[Supp] Denoising waterfall …")
    plot_waterfall(model, diffusion, synth, dtw, stoch_idx, device)

    # ---- Metrics JSON ----
    metrics = {
        'test_set_size': int(len(synth)),
        'pattern_length': int(synth.shape[1]),
        'rmse_mean': float(rmses.mean()),
        'rmse_std': float(rmses.std()),
        'pearson_mean': float(pearsons.mean()),
        'pearson_std': float(pearsons.std()),
        'cross_corr_mean': float(cross_corrs.mean()),
        'cross_corr_std': float(cross_corrs.std()),
        'stochastic_n_runs': 1000,
        'stochastic_timestep': 50,
        'stochastic_sample_idx': int(stoch_idx),
        'stochastic_dtw_value': float(dtw[stoch_idx].item()),
        'mae_mean': float(maes.mean()),
        'mae_std': float(maes.std()),
        'num_peaks_detected': int(len(cvs)),
        'cv_max_pct': float(cvs.max()) if len(cvs) > 0 else None,
        'cv_mean_pct': float(cvs.mean()) if len(cvs) > 0 else None,
        'all_cv_below_5pct': bool(np.all(cvs < 5)) if len(cvs) > 0 else None,
    }

    json_path = os.path.join(SCRIPT_DIR, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics_summary.json")
    print("All figures saved to:", FIGURE_DIR)


if __name__ == '__main__':
    main()
