r"""
Generate observed-vs-predicted scatter plots comparing PASTL and Enformer models.

Predictions for PASTL models live in a checkpoint-specific slug directory:
  {_PREDICTIONS_DIR}/{wandb_run_id}_{ckpt_stem}/predictions.pkl
Enformer predictions are shared (checkpoint-independent):
  {_PREDICTIONS_DIR}/enformer/predictions.pkl

Usage (all three panels):
  python plot_correlations.py \
      --attention_run_id <wandb_run_id> --attention_checkpoint best.pt \
      --linear_run_id   <wandb_run_id> --linear_checkpoint   best.pt \
      --enformer \
      --output_dir out/analysis/<slug1>__vs__<slug2>/correlation_eval/

Single-model example:
  python plot_correlations.py \
      --attention_run_id m7h1162n --attention_checkpoint best.pt \
      --output_dir out/analysis/m7h1162n_best/correlation_eval/
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

import palette
from evaluate import compute_all_metrics

_PREDICTIONS_DIR = "out/predictions"

_MODEL_LABELS = {
    "attention": "PASTL Attention",
    "linear":    "PASTL Linear",
    "enformer":  "Enformer Elastic Net",
}

_MODEL_COLORS = {
    "attention": palette.MODEL_FILL["pastl"],
    "linear":    palette.FILL["cat3"],
    "enformer":  palette.MODEL_FILL["enformer"],
}


def ckpt_slug(run_id: str, checkpoint: str) -> str:
    """Derive the predictions subdirectory name from run ID and checkpoint filename."""
    return f"{run_id}_{Path(checkpoint).stem}"


def load_predictions(path: Path):
    df = pd.read_pickle(path)
    return df["true_label"].values, df["prediction"].values


def draw_panel(ax, true_vals, pred_vals, title, color, show_xlabel, show_ylabel, y_lim=None):
    ax.scatter(true_vals, pred_vals, facecolors=color, alpha=0.6, s=10, linewidths=0.3)

    slope, intercept = np.polyfit(true_vals, pred_vals, 1)
    x_range = np.array([true_vals.min(), true_vals.max()])
    ax.plot(x_range, slope * x_range + intercept, color="#888888", linewidth=0.9)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("Predicted")
    if show_xlabel:
        ax.set_xlabel("Observed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_run_id",    help="WandB run ID for PASTL Attention")
    parser.add_argument("--attention_checkpoint", default="best.pt",
                        help="Checkpoint filename for PASTL Attention (default: best.pt)")
    parser.add_argument("--linear_run_id",    help="WandB run ID for PASTL Linear")
    parser.add_argument("--linear_checkpoint", default="best.pt",
                        help="Checkpoint filename for PASTL Linear (default: best.pt)")
    parser.add_argument("--enformer", action="store_true",
                        help="Include Enformer Elastic Net panel")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for figures, tables, and logs. "
                             "Caller constructs the slug-based path "
                             "(e.g. .../analysis/m7h1162n_best__vs__x9k3lp2_best/correlation_eval/).")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Base directory containing prediction outputs")
    args = parser.parse_args()

    panels = []
    if args.attention_run_id:
        panels.append("attention")
    if args.linear_run_id:
        panels.append("linear")
    if args.enformer:
        panels.append("enformer")

    if not panels:
        parser.error("Specify at least one model: --attention_run_id, --linear_run_id, or --enformer")

    palette.use_style()

    out_dir    = Path(args.output_dir)
    figs_dir   = out_dir / "figures"
    tables_dir = out_dir / "tables"
    logs_dir   = out_dir / "logs"
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.predictions_dir)

    def predictions_path(model):
        if model == "enformer":
            # Shared, checkpoint-independent baseline
            return pred_dir / "enformer" / "predictions.pkl"
        run_id = getattr(args, f"{model}_run_id")
        ckpt   = getattr(args, f"{model}_checkpoint")
        slug   = ckpt_slug(run_id, ckpt)
        return pred_dir / slug / "predictions.pkl"

    data = {}
    metrics = {}
    for model in panels:
        path = predictions_path(model)
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for {_MODEL_LABELS[model]} at {path}")
        true_vals, pred_vals = load_predictions(path)
        data[model]    = (true_vals, pred_vals)
        metrics[model] = compute_all_metrics(true_vals, pred_vals)

    results_df = pd.DataFrame([
        {"Model": _MODEL_LABELS[m], "R2": metrics[m]["r2"], "Pearson r": metrics[m]["pearson"]}
        for m in panels
    ])
    results_df.to_csv(tables_dir / "correlation_results.csv", index=False)

    with open(logs_dir / "correlation_summary.txt", "w") as f:
        f.write("=== Correlation Evaluation ===\n")
        f.write(f"Test Set Size: {len(data[panels[0]][0])} proteins\n\n")
        for m in panels:
            f.write(f"{_MODEL_LABELS[m]}:\n")
            f.write(f"  R2:       {metrics[m]['r2']:.4f}\n")
            f.write(f"  Pearson:  {metrics[m]['pearson']:.4f}\n\n")

    all_preds = np.concatenate([data[m][1] for m in panels])
    pad = (all_preds.max() - all_preds.min()) * 0.05
    y_lim = (all_preds.min() - pad, all_preds.max() + pad)

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.subplots_adjust(hspace=0.2)

    center = n // 2
    for i, model in enumerate(panels):
        true_vals, pred_vals = data[model]
        m = metrics[model]
        title = f"{_MODEL_LABELS[model]} (R² = {m['r2']:.3f}, r = {m['pearson']:.3f})"
        draw_panel(axes[i], true_vals, pred_vals, title, _MODEL_COLORS[model],
                   show_xlabel=(i == n - 1), show_ylabel=(i == center), y_lim=y_lim)
        if i < n - 1:
            axes[i].tick_params(axis="x", which="both", bottom=False)

    fig.savefig(figs_dir / "scatter_comparisons.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "scatter_comparisons.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved to {out_dir}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
