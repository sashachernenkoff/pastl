"""
Metric and plotting functions for PASTL evaluation.

Pure functions (no model I/O) live here so they can be imported by
train.py for in-loop validation as well as run standalone.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Colour palette ("sunset bio")
# ---------------------------------------------------------------------------

SB_COLORS = {
    'DeepTeal': '#006D77',
    'Coral':    '#E29578',
    'Sea':      '#83C5BE',
    'Plum':     '#6D597A',
    'Apricot':  '#FFDDD2',
    'BG':       '#EDF6F9',
}


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

def compute_r2(labels: list | np.ndarray, preds: list | np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    return float(r2_score(labels, preds))


def compute_pearson(labels: list | np.ndarray, preds: list | np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return float(pearsonr(np.asarray(labels), np.asarray(preds))[0])


def compute_spearman(labels: list | np.ndarray, preds: list | np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    return float(spearmanr(np.asarray(labels), np.asarray(preds))[0])


def compute_all_metrics(labels: list | np.ndarray, preds: list | np.ndarray) -> dict[str, float]:
    """Returns a dict with r2, pearson, and spearman."""
    return {
        "r2":      compute_r2(labels, preds),
        "pearson": compute_pearson(labels, preds),
        "spearman": compute_spearman(labels, preds),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    val_r2_scores: list[float],
    save_dir: str,
) -> None:
    """Saves a two-panel figure: loss curves and validation R²."""
    actual_epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Loss
    ax1.plot(actual_epochs, train_losses, label='Train loss', color=SB_COLORS['DeepTeal'], linewidth=2)
    ax1.plot(actual_epochs, val_losses,   label='Val loss',   color=SB_COLORS['Coral'],    linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Huber)')
    ax1.set_title('Training and Validation Loss', fontweight='bold', pad=15)
    ax1.legend(frameon=False)
    ax1.set_xlim(left=0)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.spines[['left', 'bottom']].set_edgecolor('black')
    ax1.tick_params(axis='both', which='minor', length=0)

    # Panel B: Validation R²
    ax2.plot(actual_epochs, val_r2_scores, label='Val $R^2$', color=SB_COLORS['DeepTeal'], linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('$R^2$ Score')
    ax2.set_title(r'Validation Accuracy ($\mathbf{R^2}$)', fontweight='bold', pad=15)
    ax2.legend(frameon=False)
    ax2.set_xlim(left=0)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_edgecolor('black')
    ax2.tick_params(axis='both', which='minor', length=0)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved training curves → {out_path}")


def plot_scatter(
    labels:   list | np.ndarray,
    preds:    list | np.ndarray,
    metrics:  dict[str, float],
    save_dir: str,
    filename: str = "test_preds.png",
) -> None:
    """Saves a predicted-vs-observed scatter plot."""
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    r2     = metrics["r2"]

    plt.figure(figsize=(6, 4))
    plt.scatter(labels, preds, alpha=1.0, s=20, c=SB_COLORS['DeepTeal'],
                edgecolors='black', linewidths=0.5, zorder=2)

    lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
    plt.plot(lims, lims, color=SB_COLORS['Coral'], linewidth=1.5, linestyle='--', zorder=1)

    plt.xlabel('Observed Protein Abundance (z-score)')
    plt.ylabel('Predicted Protein Abundance (z-score)')
    plt.title(rf'Test Set: Predicted vs Observed ($\mathbf{{R^2}}$ = {r2:.3f})', fontweight='bold', pad=15)

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    ax.spines[['left', 'bottom']].set_edgecolor('black')
    ax.grid(False)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot → {out_path}")


# ---------------------------------------------------------------------------
# Full test-set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_set(
    model:      torch.nn.Module,
    test_loader,
    device:     torch.device,
    save_dir:   str,
    bottleneck_dim: int,
) -> dict[str, float]:
    """
    Run inference on the test set. Saves predictions, latent features, and
    a scatter plot to save_dir. Returns a dict of r2, pearson, spearman.
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_prots  = []
    all_feats  = []

    print("\nRunning test inference...")
    for batch in test_loader:
        dna       = batch['dna'].to(device, non_blocking=True)
        gene_mask = batch['gene_mask'].to(device, non_blocking=True).float() if 'gene_mask' in batch else None
        label     = batch['label'].to(device, non_blocking=True).float()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds, features = model(dna, gene_mask=gene_mask, return_features=True)

        all_preds.extend(preds.squeeze(-1).float().cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_prots.extend(batch['prots'])
        all_feats.append(features.float().cpu().numpy())

    # --- Save latent features ---
    feats_arr = np.concatenate(all_feats, axis=0)
    feat_cols = [f"Dim_{i + 1}" for i in range(feats_arr.shape[1])]
    feats_df  = pd.DataFrame(feats_arr, columns=feat_cols)
    feats_df.insert(0, 'Protein', all_prots)
    feats_path = os.path.join(save_dir, "test_feats.csv")
    feats_df.to_csv(feats_path, index=False)
    print(f"Saved {bottleneck_dim}D latent features → {feats_path}")

    # --- Metrics ---
    metrics = compute_all_metrics(all_labels, all_preds)

    print("\n" + "=" * 40)
    print("Final test metrics:")
    print(f"  R²:       {metrics['r2']:.4f}")
    print(f"  Pearson:  {metrics['pearson']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print("=" * 40)

    # --- Save predictions ---
    results_df = pd.DataFrame({
        'Protein':     all_prots,
        'Observed_z':  all_labels,
        'Predicted_z': all_preds,
    })
    results_df['Abs_error'] = np.abs(results_df['Observed_z'] - results_df['Predicted_z'])
    results_df = results_df.sort_values('Abs_error')
    preds_path = os.path.join(save_dir, "test_preds.csv")
    results_df.to_csv(preds_path, index=False)
    print(f"Saved predictions → {preds_path}")

    # --- Scatter plot ---
    plot_scatter(all_labels, all_preds, metrics, save_dir)

    # --- Best / worst proteins ---
    print("\nTop 10 best predictions (lowest error):")
    print(results_df.head(10)[['Protein', 'Observed_z', 'Predicted_z']].to_string(index=False))
    print("\nTop 10 worst predictions (highest error):")
    print(results_df.tail(10)[['Protein', 'Observed_z', 'Predicted_z']].to_string(index=False))

    model.train()
    return metrics
