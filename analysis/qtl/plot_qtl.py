r"""
QTL comparison bar charts: PASTL vs base Enformer on fine-mapped cis-eQTLs.

Loads the per-dataset variant predictions written by analysis/predict_variants.py
and evaluates each model as a zero-shot causal-variant scorer:

  AUROC   — discriminating SuSiE causal (PIP > 0.9) from spurious (PIP < 0.01)
            variants using the model's absolute effect magnitude
            (|DELTA_Z| for PASTL, |LOG2FC| for Enformer). Reported for both
            GTEx lung and PancanQTL LUAD eQTLs.
  Directional concordance — fraction of causal variants whose predicted effect
            sign matches the empirical allelic-fold-change sign. Only GTEx
            carries a signed empirical effect, so this panel is GTEx-only.

Outputs figures/ tables/ logs/ under --output_dir (figures: grouped bar chart;
tables: per-dataset metrics CSV + JSON).

Usage:
  python plot_qtl.py --wandb_run_id s5fo3aak --checkpoint best.pt \
      --output_dir out/analysis/s5fo3aak_best/qtl_eval/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import palette

_PREDICTIONS_DIR = "out/predictions"

_DATASET_LABELS = {"eqtl_gtex": "GTEx eQTL", "eqtl_sqtl": "GTEx sQTL",
                   "eqtl_pancan": "PancanQTL\nLUAD eQTL"}
_MODEL_STYLE = {
    "pastl":   {"label": "PASTL",  "fill": palette.MODEL_FILL["pastl"],   "score": "DELTA_Z"},
    "enformer": {"label": "Enformer", "fill": palette.MODEL_FILL["enformer"], "score": "LOG2FC"},
}


def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def prediction_path(pred_dir: Path, model: str, slug: str, dataset: str) -> Path:
    sub = "enformer" if model == "enformer" else slug
    return pred_dir / sub / "variants" / f"{dataset}.tsv"


def evaluate(df: pd.DataFrame, score_col: str) -> dict:
    """AUROC/AUPRC (causal vs spurious) + directional concordance (if signed effect)."""
    d = df.dropna(subset=["PIP"]).copy()
    d["abs_score"] = d[score_col].abs()
    bench = d[(d["PIP"] > 0.9) | (d["PIP"] < 0.01)].copy()
    bench["is_causal"] = (bench["PIP"] > 0.9).astype(int)

    out = {"n_causal": int(bench["is_causal"].sum()),
           "n_spurious": int((1 - bench["is_causal"]).sum()),
           "auroc": np.nan, "auprc": np.nan, "concordance_pct": np.nan,
           "n_concordance": 0, "n_zero_score": 0}
    if bench["is_causal"].nunique() > 1:
        out["auroc"] = float(roc_auc_score(bench["is_causal"], bench["abs_score"]))
        out["auprc"] = float(average_precision_score(bench["is_causal"], bench["abs_score"]))

    if "EMPIRICAL_EFFECT" in d.columns:
        # Keep zero-score predictions (a score of 0 counts as non-concordant via
        # np.sign); only require a signed empirical effect to compare direction to.
        causal = bench[bench["is_causal"] == 1].copy()
        causal = causal[causal["EMPIRICAL_EFFECT"].fillna(0) != 0]
        if len(causal):
            conc = (np.sign(causal[score_col]) == np.sign(causal["EMPIRICAL_EFFECT"])).mean()
            out["concordance_pct"] = float(conc * 100)
            out["n_concordance"] = int(len(causal))
            out["n_zero_score"] = int((causal[score_col] == 0).sum())
    return out


def grouped_bars(ax, categories, pastl_vals, enf_vals, ylabel, title, baseline, ylim, fmt):
    offsets, bar_width = palette.grouped_bar_positions(len(categories), 2, bar_width=0.07)
    x = np.arange(len(categories))
    models = [("pastl", pastl_vals), ("enformer", enf_vals)]
    for offset, (model, vals) in zip(offsets, models):
        ax.bar(x + offset, vals, bar_width,
               color=_MODEL_STYLE[model]["fill"],
               label=_MODEL_STYLE[model]["label"],
               zorder=3)
    ax.axhline(baseline, ls="--", color="gray", lw=0.5, zorder=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_ylim(*ylim)
    palette.square_legend(ax, [_MODEL_STYLE[m]["label"] for m in ("pastl", "enformer")],
                          [_MODEL_STYLE[m]["fill"] for m in ("pastl", "enformer")],
                          frameon=False)
    for c in ax.containers:
        lbls = ax.bar_label(c, fmt=fmt, fontsize=7, padding=4)
        for lbl in lbls:
            lbl.set_rotation(90)
            lbl.set_va("bottom")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--auroc_datasets", nargs="+",
                        default=["eqtl_gtex", "eqtl_sqtl", "eqtl_pancan"])
    parser.add_argument("--concordance_datasets", nargs="+",
                        default=["eqtl_gtex"])   # sQTL SuSiE has no signed effect (no afc)
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    palette.use_style()

    out_dir = Path(args.output_dir)
    figs_dir, tables_dir, logs_dir = out_dir / "figures", out_dir / "tables", out_dir / "logs"
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.predictions_dir)
    slug = ckpt_slug(args.wandb_run_id, args.checkpoint)

    # --- Compute metrics for every dataset × model (skip datasets not yet scored) ---
    metrics: dict[str, dict[str, dict]] = {}
    datasets = list(dict.fromkeys(args.auroc_datasets + args.concordance_datasets))
    for dataset in datasets:
        model_metrics, ok = {}, True
        for model, style in _MODEL_STYLE.items():
            path = prediction_path(pred_dir, model, slug, dataset)
            if not path.exists():
                print(f"  skipping '{dataset}': missing {model} predictions at {path}")
                ok = False
                break
            model_metrics[model] = evaluate(pd.read_csv(path, sep="\t"), style["score"])
        if ok:
            metrics[dataset] = model_metrics

    auroc_datasets = [d for d in args.auroc_datasets if d in metrics]
    conc_datasets = [d for d in args.concordance_datasets if d in metrics]
    if not auroc_datasets:
        raise SystemExit("No datasets with predictions available — run predict_variants first.")

    # --- Tables ---
    rows = [{"dataset": d, "model": m, **metrics[d][m]}
            for d in metrics for m in _MODEL_STYLE]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "qtl_metrics.csv", index=False)
    with open(tables_dir / "qtl_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    grouped_bars(
        axes[0], [_DATASET_LABELS.get(d, d) for d in auroc_datasets],
        [metrics[d]["pastl"]["auroc"] for d in auroc_datasets],
        [metrics[d]["enformer"]["auroc"] for d in auroc_datasets],
        "AUROC", "Causal-variant discrimination (AUROC)", 0.5, (0, 1), "%.3f",
    )

    if conc_datasets:
        grouped_bars(
            axes[1], [_DATASET_LABELS.get(d, d) for d in conc_datasets],
            [metrics[d]["pastl"]["concordance_pct"] for d in conc_datasets],
            [metrics[d]["enformer"]["concordance_pct"] for d in conc_datasets],
            "Concordance (%)", "Directional concordance", 50, (0, 100), "%.1f%%",
        )

    fig.tight_layout()
    fig.savefig(figs_dir / "qtl_model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "qtl_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(logs_dir / "qtl_summary.txt", "w") as f:
        f.write("=== QTL Model Comparison ===\n")
        f.write(f"Run/checkpoint slug: {slug}\n\n")
        f.write(metrics_df.to_string(index=False) + "\n")

    print(f"Saved to {out_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
