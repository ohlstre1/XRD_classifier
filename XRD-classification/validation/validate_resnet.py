"""
Generate publication-quality figures from ResNet-18 training results.

Produces three figures in XRD-classification/validation/figures/:
  - training_curves.pdf:   2-panel (train/val loss, train accuracy)
  - val_topk_accuracy.pdf: validation top-1/5/10 accuracy vs epoch
  - test_results_bar.pdf:  bar chart of final test top-k accuracy
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR.parent / "scripts" / "models" / "resnet_classifier" / "training_results.json"
FIG_DIR = SCRIPT_DIR / "figures"


# ── Style ──────────────────────────────────────────────────────────────
def setup_style():
    """IEEE-friendly matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.4,
    })


# ── Figures ────────────────────────────────────────────────────────────
def plot_training_curves(history, fig_dir):
    """2-panel: (left) train + val loss, (right) train accuracy."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()
    ax1.set_title("(a) Training and validation loss")

    # Accuracy
    ax2.plot(epochs, history["train_acc"], color="C2")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 accuracy (%)")
    ax2.set_title("(b) Training accuracy")

    fig.tight_layout()
    out = fig_dir / "training_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_val_topk(history, fig_dir):
    """Validation top-1 / top-5 / top-10 accuracy vs epoch."""
    epochs = np.arange(1, len(history["val_acc_top1"]) + 1)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(epochs, history["val_acc_top1"], label="Top-1")
    ax.plot(epochs, history["val_acc_top5"], label="Top-5")
    ax.plot(epochs, history["val_acc_top10"], label="Top-10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation top-$k$ accuracy")
    ax.legend()

    fig.tight_layout()
    out = fig_dir / "val_topk_accuracy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_test_bar(results, fig_dir):
    """Bar chart of final test top-1 / top-5 / top-10 accuracy."""
    labels = ["Top-1", "Top-5", "Top-10"]
    values = [results["test_acc_top1"], results["test_acc_top5"], results["test_acc_top10"]]
    colors = ["C0", "C1", "C2"]

    fig, ax = plt.subplots(figsize=(3.5, 3))
    bars = ax.bar(labels, values, color=colors, width=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test set accuracy")
    ax.set_ylim(0, 105)

    fig.tight_layout()
    out = fig_dir / "test_results_bar.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    history = data["history"]
    plot_training_curves(history, FIG_DIR)
    plot_val_topk(history, FIG_DIR)
    plot_test_bar(data, FIG_DIR)

    print(f"\nTest results: top-1 = {data['test_acc_top1']:.1f}%,  "
          f"top-5 = {data['test_acc_top5']:.1f}%,  "
          f"top-10 = {data['test_acc_top10']:.1f}%")


if __name__ == "__main__":
    main()
