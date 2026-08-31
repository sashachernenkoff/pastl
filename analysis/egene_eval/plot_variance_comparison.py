r"""
Compare PASTL vs Enformer elastic-net Pearson r across across-patient
variability strata (Low / Mid / High IQR), using the metric tables written by
plot_egene_stratification.py.

Requires plot_egene_stratification.py to have been run for both models. Output
is saved to the PASTL model's egene_eval/figures/ directory.

Usage:
  python plot_variance_comparison.py \
      --wandb_run_id m7h1162n --checkpoint best.pt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import palette

_ANALYSIS_DIR = "out/analysis"

_STRATA_ORDER = ["Low", "Mid", "High"]

_COLORS = {
    "pastl":   palette.MODEL_FILL["pastl"],
    "enformer": palette.MODEL_FILL["enformer"],
}


def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def load_variance_metrics(egene_eval_dir: Path) -> pd.DataFrame:
    path = egene_eval_dir / "tables" / "variance_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing variance metrics at {path}\n"
            "Run plot_egene_stratification.py first."
        )
    df = pd.read_csv(path)
    df["Stratum"] = pd.Categorical(df["Stratum"], categories=_STRATA_ORDER, ordered=True)
    df = df[df["Stratum"].isin(_STRATA_ORDER)].sort_values("Stratum")
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb_run_id", required=True,
                        help="WandB run ID for the PASTL model.")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename (default: best.pt).")
    parser.add_argument("--analysis_dir", default=_ANALYSIS_DIR,
                        help="Root analysis output directory (default: %(default)s).")
    args = parser.parse_args()

    slug = ckpt_slug(args.wandb_run_id, args.checkpoint)
    analysis_dir = Path(args.analysis_dir)

    pastl_dir   = analysis_dir / slug / "egene_eval"
    enformer_dir = analysis_dir / "enformer" / "egene_eval"
    figs_dir     = pastl_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    pastl_df   = load_variance_metrics(pastl_dir)
    enformer_df = load_variance_metrics(enformer_dir)

    palette.use_style()

    # Display Low at top -> High at bottom (reads naturally top-to-bottom).
    strata = list(reversed(_STRATA_ORDER))
    n = len(strata)
    y = np.arange(n)
    offsets, bar_w = palette.grouped_bar_positions(n, n_bars_per_group=2)
    offsets = list(reversed(offsets))  # PASTL (i=0) → top bar, Enformer (i=1) → bottom

    fig, ax = plt.subplots(figsize=(7, 3))

    models = [
        ("pastl",   pastl_df,   f"PASTL ({slug})"),
        ("enformer", enformer_df, "Enformer Elastic Net"),
    ]
    for i, (key, df, label) in enumerate(models):
        r_vals = [
            float(df.loc[df["Stratum"] == s, "pearson"].iloc[0])
            if s in df["Stratum"].values else float("nan")
            for s in strata
        ]
        ax.barh(y + offsets[i], r_vals, bar_w,
                color=_COLORS[key],
                label=label)
        for yi, r in zip(y + offsets[i], r_vals):
            if not np.isnan(r):
                ax.text(r + 0.005, yi, f"{r:.2f}", ha="left", va="center",
                        color="black",
                        fontsize=6)

    # Fold n into the y-tick labels (universe is the same for both models).
    tick_labels = []
    for s in strata:
        row = pastl_df[pastl_df["Stratum"] == s]
        n_str = f"  (n={int(row['n'].iloc[0])})" if not row.empty else ""
        tick_labels.append(f"{s}{n_str}")

    ax.set_ylim(*palette.bar_limits(n))
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Across-Patient Variability\n(IQR of log₂ abundance)")
    ax.set_xlabel("Pearson r")
    ax.set_title("Prediction Accuracy by Protein Variability", pad=15)
    xlim = ax.get_xlim()[1]
    ax.set_xlim(0, xlim * 1.12)
    palette.square_legend(ax,
                          labels=[m[2] for m in models],
                          colors=[_COLORS[m[0]] for m in models],
                          loc="center right")

    out_stem = figs_dir / "variance_model_comparison"
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
