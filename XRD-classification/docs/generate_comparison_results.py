#!/usr/bin/env python
"""
Generate comparison figures and metrics for the classification chapter.

Produces:
  figures/test_accuracy_comparison.pdf   – grouped bar chart of test accuracy
  figures/training_curves_comparison.pdf – 2-panel loss overlay
  figures/val_top1_comparison.pdf        – validation top-1 accuracy
  figures/val_top5_comparison.pdf        – validation top-5 accuracy
  figures/val_top10_comparison.pdf       – validation top-10 accuracy
  comparison_metrics.json                – numeric summary
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FIG_DIR = SCRIPT_DIR / "figures"

RESULTS = {
    "Baseline": PROJECT_ROOT / "XRD-classification" / "scripts" / "models" / "baseline_comparison" / "training_results.json",
    "Smart Classical Aug": PROJECT_ROOT / "XRD-classification" / "scripts" / "models" / "smart_aug_comparison" / "training_results.json",
    "Smart Diffusion Aug": PROJECT_ROOT / "XRD-classification" / "scripts" / "models" / "smart_diffusion_aug_classifier" / "training_results.json",
}

RETRIEVAL_PATH = PROJECT_ROOT / "retrieval" / "evaluation_results.json"

LOG_FILES = {
    "Baseline": PROJECT_ROOT / "XRD-classification" / "scripts" / "comparison_output.log",
    "Smart Classical Aug": PROJECT_ROOT / "XRD-classification" / "scripts" / "nohup_run_smart_aug.out",
    "Smart Diffusion Aug": PROJECT_ROOT / "XRD-classification" / "scripts" / "nohup_run_smart_diffusion_aug.out",
}

# For comparison_output.log, only use the first 200 epochs (baseline run)
LOG_MAX_EPOCHS = {
    "Baseline": 200,
    "Smart Classical Aug": None,
    "Smart Diffusion Aug": None,
}


# ── Style ─────────────────────────────────────────────────────────────
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


# ── Log parsing ───────────────────────────────────────────────────────
RE_TRAIN = re.compile(r"Train Loss:\s*([\d.]+),\s*Train Acc:\s*([\d.]+)%")
RE_VAL = re.compile(
    r"Val Loss:\s*([\d.]+),\s*Val Top-1:\s*([\d.]+)%,\s*Top-5:\s*([\d.]+)%,\s*Top-10:\s*([\d.]+)%"
)


def parse_train_log(log_path, max_epochs=None):
    """Extract per-epoch metrics from a training log file."""
    train_loss, train_acc = [], []
    val_loss, val_top1, val_top5, val_top10 = [], [], [], []

    with open(log_path) as f:
        for line in f:
            m = RE_TRAIN.search(line)
            if m:
                train_loss.append(float(m.group(1)))
                train_acc.append(float(m.group(2)))
            m = RE_VAL.search(line)
            if m:
                val_loss.append(float(m.group(1)))
                val_top1.append(float(m.group(2)))
                val_top5.append(float(m.group(3)))
                val_top10.append(float(m.group(4)))

    if max_epochs:
        train_loss = train_loss[:max_epochs]
        train_acc = train_acc[:max_epochs]
        val_loss = val_loss[:max_epochs]
        val_top1 = val_top1[:max_epochs]
        val_top5 = val_top5[:max_epochs]
        val_top10 = val_top10[:max_epochs]

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc_top1": val_top1,
        "val_acc_top5": val_top5,
        "val_acc_top10": val_top10,
    }


def load_history(name):
    """Load training history from JSON if complete, otherwise parse log."""
    json_path = RESULTS[name]
    with open(json_path) as f:
        data = json.load(f)

    history = data.get("history", {})
    # Use JSON history if it has more than 1 epoch
    if len(history.get("train_loss", [])) > 1:
        return history

    # Fall back to log parsing
    log_path = LOG_FILES[name]
    max_ep = LOG_MAX_EPOCHS.get(name)
    print(f"  Parsing log for {name}: {log_path}")
    return parse_train_log(log_path, max_epochs=max_ep)


# ── Figure 1: Test accuracy grouped bar chart ─────────────────────────
def plot_test_accuracy(test_metrics, fig_dir):
    """Grouped bar chart: 4 methods x 3 top-k metrics."""
    methods = list(test_metrics.keys())
    k_labels = ["Top-1", "Top-5", "Top-10"]
    n_methods = len(methods)
    n_k = len(k_labels)

    x = np.arange(n_methods)
    width = 0.25
    colors = ["C0", "C1", "C2"]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, (k_label, color) in enumerate(zip(k_labels, colors)):
        key = f"top{[1, 5, 10][i]}"
        values = [test_metrics[m][key] for m in methods]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=k_label, color=color, alpha=0.85)

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test set accuracy comparison")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = fig_dir / "test_accuracy_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: Training curves comparison ──────────────────────────────
def plot_training_curves(histories, fig_dir):
    """2-panel overlay: train loss and val loss for 3 trained models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    colors = {"Baseline": "C0", "Smart Classical Aug": "C1", "Smart Diffusion Aug": "C3"}

    for name, hist in histories.items():
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        color = colors[name]

        ax1.plot(epochs, hist["train_loss"], label=name, color=color)
        ax2.plot(epochs, hist["val_loss"], label=name, color=color)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("(a) Training loss")
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("(b) Validation loss")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out = fig_dir / "training_curves_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Validation top-k comparison (one per k) ─────────────────
def plot_val_topk(histories, fig_dir):
    """One figure per top-k metric, each with 3 model lines."""
    colors = {"Baseline": "C0", "Smart Classical Aug": "C1", "Smart Diffusion Aug": "C3"}
    k_configs = [
        ("top1", 1, "(a) Top-1"),
        ("top5", 5, "(b) Top-5"),
        ("top10", 10, "(c) Top-10"),
    ]

    for suffix, k, title in k_configs:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        key = f"val_acc_{suffix}"

        for name, hist in histories.items():
            epochs = np.arange(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], label=name, color=colors[name])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=7)

        fig.tight_layout()
        out = fig_dir / f"val_{suffix}_comparison.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test metrics ──
    print("Loading test metrics...")
    test_metrics = {}
    for name, path in RESULTS.items():
        with open(path) as f:
            data = json.load(f)
        test_metrics[name] = {
            "top1": data["test_acc_top1"],
            "top5": data["test_acc_top5"],
            "top10": data["test_acc_top10"],
            "best_val": data["best_val_acc_top1"],
            "train_samples": data["train_samples"],
        }

    # Add retrieval
    with open(RETRIEVAL_PATH) as f:
        ret = json.load(f)
    test_metrics = {
        "Retrieval": {
            "top1": ret["top1_accuracy"],
            "top5": ret["top5_accuracy"],
            "top10": ret["top10_accuracy"],
            "best_val": None,
            "train_samples": None,
        },
        **test_metrics,
    }

    # Print summary table
    print(f"\n{'Method':<22} {'Top-1':>7} {'Top-5':>7} {'Top-10':>7} {'Best Val':>9} {'Train':>7}")
    print("-" * 65)
    for name, m in test_metrics.items():
        bv = f"{m['best_val']:.2f}" if m["best_val"] else "—"
        ts = str(m["train_samples"]) if m["train_samples"] else "—"
        print(f"{name:<22} {m['top1']:>7.2f} {m['top5']:>7.2f} {m['top10']:>7.2f} {bv:>9} {ts:>7}")

    # ── Load training histories ──
    print("\nLoading training histories...")
    histories = {}
    for name in ["Baseline", "Smart Classical Aug", "Smart Diffusion Aug"]:
        histories[name] = load_history(name)
        n = len(histories[name]["train_loss"])
        print(f"  {name}: {n} epochs")

    # ── Generate figures ──
    print("\nGenerating figures...")
    plot_test_accuracy(test_metrics, FIG_DIR)
    plot_training_curves(histories, FIG_DIR)
    plot_val_topk(histories, FIG_DIR)

    # ── Save metrics JSON ──
    metrics = {
        "test_metrics": {
            name: {k: round(v, 2) if isinstance(v, float) else v
                   for k, v in m.items()}
            for name, m in test_metrics.items()
        },
        "training_epochs": {
            name: len(h["train_loss"]) for name, h in histories.items()
        },
    }
    json_path = SCRIPT_DIR / "comparison_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {json_path}")
    print("All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
