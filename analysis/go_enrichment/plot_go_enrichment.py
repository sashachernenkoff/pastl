r"""
GO over-representation analysis for PASTL predictions.

Loads predictions_all.pkl (all splits) for a given model checkpoint, splits
proteins into best-predicted / over-predicted / under-predicted tail sets,
runs unordered ORA via the g:Profiler API against the full protein set as
background, and saves one bar-plot figure per set.

Usage:
  python plot_go_enrichment.py \
      --run_id s5fo3aak --checkpoint best.pt \
      --output_dir out/analysis/s5fo3aak_best/go_enrichment
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gprofiler import GProfiler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import palette

_PREDICTIONS_DIR = "out/predictions"

_SOURCES = ["GO:BP", "GO:MF", "GO:CC"]

_SOURCE_LABELS = {
    "GO:BP": "Biological Process",
    "GO:MF": "Molecular Function",
    "GO:CC": "Cellular Component",
}

_SOURCE_COLORS = {
    "GO:BP": palette.FILL["cat3"],   # blue
    "GO:MF": palette.FILL["cat4"],   # purple
    "GO:CC": palette.FILL["cat5"],   # gold
}

_SET_LABELS = {
    "best":  "Best-Predicted Proteins",
    "over":  "Most Over-Predicted Proteins",
    "under": "Most Under-Predicted Proteins",
}


def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def build_ranked_lists(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["residual"] = df["prediction"] - df["true_label"]
    df["abs_error"] = df["residual"].abs()
    return {
        "best":  df.sort_values("abs_error", ascending=True)["protein_id"].tolist(),
        "over":  df.sort_values("residual",  ascending=False)["protein_id"].tolist(),
        "under": df.sort_values("residual",  ascending=True)["protein_id"].tolist(),
    }


def run_enrichment(query: list, background: list) -> pd.DataFrame:
    gp = GProfiler(return_dataframe=True)
    results = gp.profile(
        organism="hsapiens",
        query=query,
        background=background,
        sources=_SOURCES,
        ordered=False,
        significance_threshold_method="fdr",
        user_threshold=0.05,
        no_evidences=True,
    )
    if results.empty or "significant" not in results.columns:
        return pd.DataFrame()
    return results[results["significant"]]


def top_terms(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if results.empty or "source" not in results.columns:
        return pd.DataFrame()
    rows = []
    for source in _SOURCES:
        sub = results[results["source"] == source].nsmallest(n, "p_value")
        if not sub.empty:
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_go_bars(terms: pd.DataFrame, title: str, out_path: Path):
    if terms.empty or "source" not in terms.columns:
        print(f"No significant terms — skipping: {title}")
        return

    palette.use_style()

    bar_h = 0.17
    step  = 0.30  # distance between bar centres
    y_pos = 0
    y_ticks, y_labels, bar_data = [], [], []

    for source in _SOURCES:
        sub = terms[terms["source"] == source]
        for _, row in sub[::-1].iterrows():
            score = -np.log10(row["p_value"])
            label = row["name"].title()
            bar_data.append((y_pos, score, _SOURCE_COLORS[source]))
            y_ticks.append(y_pos)
            y_labels.append(label)
            y_pos += step

    if not bar_data:
        print(f"No significant terms — skipping: {title}")
        return

    n_bars = len(bar_data)
    fig_h = max(3, 0.14 * n_bars + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    for y, score, color in bar_data:
        ax.barh(y, score, bar_h, color=color)

    ax.axvline(x=-np.log10(0.05), color="#888888", linestyle="--", linewidth=0.5)
    ax.set_ylim(-step / 2, (n_bars - 1) * step + step / 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(r"$-\log_{10}(\mathrm{FDR})$")
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    present = [s for s in reversed(_SOURCES) if any(d[2] == _SOURCE_COLORS[s] for d in bar_data)]
    palette.square_legend(
        ax,
        labels=[_SOURCE_LABELS[s] for s in present],
        colors=[_SOURCE_COLORS[s] for s in present],
        loc="lower right",
    )

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}.{{png,pdf}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--percentile", type=float, default=10,
                        help="Tail percentile cutoff for over/under/best sets (default: 10)")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Base directory containing prediction outputs")
    args = parser.parse_args()

    slug = ckpt_slug(args.run_id, args.checkpoint)
    pred_path = Path(args.predictions_dir) / slug / "predictions_all.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions at {pred_path}")

    df = pd.read_pickle(pred_path)
    background = df["protein_id"].tolist()
    n_tail = max(1, int(np.ceil(len(df) * args.percentile / 100)))
    print(f"Loaded {len(df):,} proteins as background; tail n={n_tail} ({args.percentile}%)")

    ranked = build_ranked_lists(df)
    subsets = {
        "best":  ranked["best"][:n_tail],
        "over":  ranked["over"][:n_tail],
        "under": ranked["under"][:n_tail],
    }

    out_dir = Path(args.output_dir)
    figures_dir = out_dir / "figures"
    tables_dir  = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for set_name, label in _SET_LABELS.items():
        query = subsets[set_name]
        print(f"\nRunning ORA: {label} (n={len(query)}, background={len(background)})")
        results = run_enrichment(query, background)
        print(f"  {len(results)} significant terms")
        terms = top_terms(results)
        plot_go_bars(terms, f"GO Over-representation: {label}", figures_dir / f"go_{set_name}")

        if not terms.empty:
            terms.to_csv(tables_dir / f"go_{set_name}_terms.csv", index=False)


if __name__ == "__main__":
    main()
