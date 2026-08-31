r"""
RNA / protein / prediction correlation analysis.

Quantifies how much of the cross-gene variation in observed protein abundance
is explained by (a) transcript level alone and (b) the model's sequence-based
predictions, and tests whether predictions add signal beyond RNA via a partial
correlation controlling for transcript level.

Three cross-gene Pearson correlations are reported and plotted:
  1. Observed RNA (per-gene median log2(TPM+1), z-scored) vs Observed Protein
     (z-score from predictions.pkl true_label).
  2. Prediction (z-score) vs Observed Protein (z-score).
  3. Prediction (z-score) vs Observed RNA (z-score).
  + Partial correlation of Prediction vs Protein controlling for RNA.

Protein values in predictions.pkl are already z-scored at training time.

Inputs:
  predictions.pkl       from analysis/predict.py (protein_id, true_label, prediction)
  rna_aligned_log2.csv  genes x patients RNA matrix from harmonize_rna_prot.py

Usage:
  python plot_decomp.py \
      --wandb_run_id m7h1162n --checkpoint best.pt \
      --rna_path out/analysis/decomp/rna_aligned_log2.csv \
      --output_dir out/analysis/m7h1162n_best/decomp/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import palette

_BOLD = palette.PALETTES["bold"]

_PREDICTIONS_DIR = "out/predictions"


def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.rename(columns={"protein_id": "gene"})
    return df[["gene", "true_label", "prediction"]].dropna()


def _draw_scatter(ax, x, y, color, title, xlabel, ylabel, stats_lines):
    """Scatter with OLS fit line and a stats annotation box."""
    ax.scatter(x, y, facecolors=color, alpha=0.45, s=12, linewidths=0.3)

    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.array([x.min(), x.max()])
    ax.plot(x_range, slope * x_range + intercept, color="#888888", linewidth=0.9)

    ax.text(0.05, 0.95, "\n".join(stats_lines),
            transform=ax.transAxes,
            fontsize=plt.rcParams["xtick.labelsize"],
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wandb_run_id",
                      help="WandB run ID for a PASTL model.")
    mode.add_argument("--enformer", action="store_true",
                      help="Use the Enformer elastic-net baseline.")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename; its stem forms the slug "
                             "(default: best.pt). Ignored for --enformer.")
    parser.add_argument("--rna_path", required=True,
                        help="Path to rna_aligned_log2.csv from "
                             "analysis/decomp/harmonize_rna_prot.py.")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Root predictions directory (default: %(default)s).")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory; figures/, tables/, and logs/ "
                             "will be created inside it.")
    args = parser.parse_args()

    palette.use_style()

    out_dir = Path(args.output_dir)
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────────
    if args.enformer:
        slug = "enformer"
        pred_path = Path(args.predictions_dir) / "enformer" / "predictions.pkl"
        model_label = "Enformer"
    else:
        slug = ckpt_slug(args.wandb_run_id, args.checkpoint)
        pred_path = Path(args.predictions_dir) / slug / "predictions.pkl"
        model_label = "PASTL"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions at {pred_path}")

    preds = load_predictions(pred_path).set_index("gene")
    print(f"Predictions: {len(preds):,} test-set genes from {pred_path}")

    # ── Load RNA matrix and compute per-gene medians ──────────────────────────
    print(f"Loading RNA: {args.rna_path}")
    rna_df = pd.read_csv(args.rna_path, index_col=0)
    rna_median = rna_df.median(axis=1)
    print(f"  RNA matrix: {rna_df.shape[0]:,} genes × {rna_df.shape[1]:,} patients")

    # ── Intersect genes ───────────────────────────────────────────────────────
    common = sorted(set(preds.index) & set(rna_median.index))
    if not common:
        raise ValueError(
            "No gene overlap between predictions and RNA matrix. "
            "Check that gene symbols match (e.g. no version suffixes)."
        )
    print(f"  Gene intersection: {len(common):,}")

    merged = pd.DataFrame({
        "pred": preds.loc[common, "prediction"],
        "prot": preds.loc[common, "true_label"],
        "rna_raw": rna_median.loc[common],
    }).dropna()
    print(f"  Genes after dropping NaN: {len(merged):,}")

    # Z-score RNA medians cross-gene to match the z-space of prot / pred
    rna_mean = merged["rna_raw"].mean()
    rna_std = merged["rna_raw"].std()
    merged["rna"] = (merged["rna_raw"] - rna_mean) / rna_std

    n = len(merged)

    # ── Correlations ──────────────────────────────────────────────────────────
    r_rna_prot,  p_rna_prot  = pearsonr(merged["rna"],  merged["prot"])
    r2_rna_prot  = r_rna_prot ** 2

    r_pred_prot,  p_pred_prot  = pearsonr(merged["pred"], merged["prot"])
    r2_pred_prot = r_pred_prot ** 2

    r_pred_rna,  p_pred_rna  = pearsonr(merged["pred"], merged["rna"])
    r2_pred_rna  = r_pred_rna ** 2

    # Partial: pred vs prot | rna
    X_rna = sm.add_constant(merged["rna"].values)
    resid_pred = sm.OLS(merged["pred"].values, X_rna).fit().resid
    resid_prot = sm.OLS(merged["prot"].values, X_rna).fit().resid
    r_partial, p_partial = pearsonr(resid_pred, resid_prot)

    print(f"\nCorrelation results (n = {n:,} genes):")
    print(f"  RNA vs Protein             R² = {r2_rna_prot:.4f}  r = {r_rna_prot:.4f}  p = {p_rna_prot:.2e}")
    print(f"  Pred vs Protein            R² = {r2_pred_prot:.4f}  r = {r_pred_prot:.4f}  p = {p_pred_prot:.2e}")
    print(f"  Pred vs RNA                R² = {r2_pred_rna:.4f}  r = {r_pred_rna:.4f}  p = {p_pred_rna:.2e}")
    print(f"  Pred vs Protein | RNA      r = {r_partial:.4f}  p = {p_partial:.2e}")

    # ── Tables ────────────────────────────────────────────────────────────────
    corr_df = pd.DataFrame([
        {
            "comparison": "RNA vs Protein (baseline)",
            "r2": r2_rna_prot,  "pearson_r": r_rna_prot,  "pearson_p": p_rna_prot,
        },
        {
            "comparison": f"{model_label} Pred vs Protein",
            "r2": r2_pred_prot, "pearson_r": r_pred_prot, "pearson_p": p_pred_prot,
        },
        {
            "comparison": f"{model_label} Pred vs RNA",
            "r2": r2_pred_rna,  "pearson_r": r_pred_rna,  "pearson_p": p_pred_rna,
        },
        {
            "comparison": f"{model_label} Pred vs Protein | RNA (partial)",
            "r2": np.nan,       "pearson_r": r_partial,   "pearson_p": p_partial,
        },
    ])
    corr_df.to_csv(tables_dir / "correlation_results.csv", index=False)
    merged.drop(columns=["rna_raw"]).to_csv(tables_dir / "per_gene_values.csv")

    # ── Logs ─────────────────────────────────────────────────────────────────
    with open(logs_dir / "decomp_summary.txt", "w") as f:
        f.write("=== RNA / Protein / Prediction Decomposition ===\n")
        f.write(f"Model slug:   {slug}\n")
        f.write(f"Model label:  {model_label}\n")
        f.write(f"RNA path:     {args.rna_path}\n")
        f.write(f"n genes:      {n}\n\n")
        f.write(
            "Protein values:  z-score (from predictions.pkl true_label, "
            "standardized at training time)\n"
            "RNA values:      per-gene median log2(TPM+1), z-scored cross-gene\n\n"
            "Note: CPTAC TMT normalization deflates the cross-gene RNA–protein\n"
            "baseline correlation. Protein z-scores from the model sidestep\n"
            "this because z-scoring per gene removes the TMT-induced mean\n"
            "compression. The baseline r reported here reflects this; it is\n"
            "expected to be lower than in label-free proteomics.\n\n"
        )
        f.write(f"RNA vs Protein (baseline):           R² = {r2_rna_prot:.4f}  r = {r_rna_prot:.4f}  p = {p_rna_prot:.2e}\n")
        f.write(f"{model_label} Pred vs Protein:       R² = {r2_pred_prot:.4f}  r = {r_pred_prot:.4f}  p = {p_pred_prot:.2e}\n")
        f.write(f"{model_label} Pred vs RNA:            R² = {r2_pred_rna:.4f}  r = {r_pred_rna:.4f}  p = {p_pred_rna:.2e}\n")
        f.write(f"{model_label} Pred vs Protein | RNA:  r = {r_partial:.4f}  p = {p_partial:.2e}\n")

    # =========================================================================
    # Figure 1: Three-panel scatter
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    panels = [
        (
            axes[0],
            merged["rna"].values, merged["prot"].values,
            _BOLD["cat3"],
            "RNA vs Observed Protein",
            "Observed RNA (z-score)", "Observed Protein (z-score)",
            [f"R² = {r2_rna_prot:.3f}", f"r = {r_rna_prot:.3f}", f"n = {n:,}"],
        ),
        (
            axes[1],
            merged["pred"].values, merged["prot"].values,
            _BOLD["cat4"],
            f"{model_label} Prediction vs Observed Protein",
            f"{model_label} Prediction (z-score)", "Observed Protein (z-score)",
            [f"R² = {r2_pred_prot:.3f}", f"r = {r_pred_prot:.3f}", f"n = {n:,}",
             f"partial r = {r_partial:.3f}"],
        ),
        (
            axes[2],
            merged["pred"].values, merged["rna"].values,
            _BOLD["cat5"],
            f"{model_label} Prediction vs Observed RNA",
            f"{model_label} Prediction (z-score)", "Observed RNA (z-score)",
            [f"R² = {r2_pred_rna:.3f}", f"r = {r_pred_rna:.3f}", f"n = {n:,}"],
        ),
    ]

    for ax, x, y, color, title, xlabel, ylabel, stats_lines in panels:
        _draw_scatter(ax, x, y, color, title, xlabel, ylabel, stats_lines)

    fig.tight_layout()
    fig.savefig(figs_dir / "scatter_correlations.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "scatter_correlations.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter_correlations")

    # =========================================================================
    # Figure 2: Correlation summary bar chart
    # =========================================================================
    comparisons = [
        ("RNA vs\nProtein",                   r_rna_prot,  _BOLD["cat3"]),
        (f"Pred vs\nProtein",                 r_pred_prot, palette.MODEL_FILL["pastl"]),
        (f"Pred vs\nRNA",                     r_pred_rna,  _BOLD["cat2"]),
        (f"Pred vs Protein\n| RNA (partial)", r_partial,   _BOLD["cat4"]),
    ]
    labels, r_vals, fills = zip(*comparisons)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(comparisons))

    bars = ax.bar(x, r_vals, width=0.55, color=fills, zorder=2)
    ax.axhline(0, color="black", linewidth=0.4, zorder=1)

    for bar, r_val in zip(bars, r_vals):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.008 if h >= 0 else -0.008
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                f"r = {r_val:.3f}", ha="center", va=va,
                fontsize=plt.rcParams["xtick.labelsize"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r")
    ax.set_title("Cross-gene RNA / Protein / Prediction Correlations")
    ymin = min(0, min(r_vals)) - 0.08
    ymax = max(r_vals) + 0.10
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(figs_dir / "correlation_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "correlation_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved correlation_summary")

    # =========================================================================
    # Figure 3: Unique signal beyond the other model (PASTL mode only)
    # =========================================================================
    if args.wandb_run_id:
        enf_path = Path(args.predictions_dir) / "enformer" / "predictions.pkl"
        if not enf_path.exists():
            print(f"\nSkipping Figure 3: Enformer predictions not found at {enf_path}")
        else:
            enf_preds = load_predictions(enf_path).set_index("gene")

            common3 = sorted(set(merged.index) & set(enf_preds.index))
            tlp3 = merged.loc[common3, "pred"].values
            enf3 = enf_preds.loc[common3, "prediction"].values
            prot3 = merged.loc[common3, "prot"].values
            n3 = len(common3)

            # r(PASTL predictions, Enformer predictions)
            r_models, p_models = pearsonr(tlp3, enf3)

            # Partial r: PASTL vs protein, controlling for Enformer
            X_enf = sm.add_constant(enf3)
            resid_tlp_on_enf  = sm.OLS(tlp3,  X_enf).fit().resid
            resid_prot_on_enf = sm.OLS(prot3, X_enf).fit().resid
            r_tlp_partial, p_tlp_partial = pearsonr(resid_tlp_on_enf, resid_prot_on_enf)

            # Partial r: Enformer vs protein, controlling for PASTL
            X_tlp = sm.add_constant(tlp3)
            resid_enf_on_tlp  = sm.OLS(enf3,  X_tlp).fit().resid
            resid_prot_on_tlp = sm.OLS(prot3, X_tlp).fit().resid
            r_enf_partial, p_enf_partial = pearsonr(resid_enf_on_tlp, resid_prot_on_tlp)

            print(f"\nFigure 3 (n = {n3:,} genes):")
            print(f"  PASTL vs Enformer (model–model):          r = {r_models:.4f}  p = {p_models:.2e}")
            print(f"  PASTL vs Protein | Enformer (partial):    r = {r_tlp_partial:.4f}  p = {p_tlp_partial:.2e}")
            print(f"  Enformer vs Protein | PASTL (partial):    r = {r_enf_partial:.4f}  p = {p_enf_partial:.2e}")

            fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

            _draw_scatter(
                ax_left,
                resid_tlp_on_enf, resid_prot_on_enf,
                palette.MODEL_FILL["pastl"],
                "PASTL unique signal vs Protein\n(controlling for Enformer)",
                "PASTL prediction residual", "Protein residual",
                [f"partial r = {r_tlp_partial:.3f}", f"n = {n3:,}"],
            )
            _draw_scatter(
                ax_right,
                resid_enf_on_tlp, resid_prot_on_tlp,
                palette.MODEL_FILL["enformer"],
                "Enformer unique signal vs Protein\n(controlling for PASTL)",
                "Enformer prediction residual", "Protein residual",
                [f"partial r = {r_enf_partial:.3f}", f"n = {n3:,}"],
            )

            fig3.tight_layout()
            fig3.savefig(figs_dir / "unique_signal_scatter.png", dpi=300, bbox_inches="tight")
            fig3.savefig(figs_dir / "unique_signal_scatter.pdf", bbox_inches="tight")
            plt.close(fig3)
            print("  Saved unique_signal_scatter")

            extra_rows = pd.DataFrame([
                {
                    "comparison": "PASTL vs Enformer (model–model)",
                    "r2": r_models ** 2, "pearson_r": r_models, "pearson_p": p_models,
                },
                {
                    "comparison": "PASTL vs Protein | Enformer (partial)",
                    "r2": np.nan, "pearson_r": r_tlp_partial, "pearson_p": p_tlp_partial,
                },
                {
                    "comparison": "Enformer vs Protein | PASTL (partial)",
                    "r2": np.nan, "pearson_r": r_enf_partial, "pearson_p": p_enf_partial,
                },
            ])
            corr_csv = tables_dir / "correlation_results.csv"
            existing = pd.read_csv(corr_csv)
            pd.concat([existing, extra_rows], ignore_index=True).to_csv(corr_csv, index=False)

            with open(logs_dir / "decomp_summary.txt", "a") as f:
                f.write("\n=== Unique Signal Beyond Each Model ===\n")
                f.write(f"n genes (3-way intersection):  {n3}\n\n")
                f.write(f"PASTL vs Enformer (model–model):       r = {r_models:.4f}  p = {p_models:.2e}\n")
                f.write(f"PASTL vs Protein | Enformer (partial): r = {r_tlp_partial:.4f}  p = {p_tlp_partial:.2e}\n")
                f.write(f"Enformer vs Protein | PASTL (partial): r = {r_enf_partial:.4f}  p = {p_enf_partial:.2e}\n")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
