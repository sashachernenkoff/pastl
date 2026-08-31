r"""
Identify top-100 PASTL-improved and 50 control genes for ISM analysis.

Loads test-set predictions from both models, computes per-gene squared error,
and ranks by SE improvement (SE_enformer - SE_pastl). The top 100 genes with
the largest improvement form the "improved" group; 50 genes with near-zero
improvement (smallest |SE_enformer - SE_pastl|) form the "control" group.

Inputs:
  PASTL:  {predictions_dir}/{run}_{ckpt}/predictions.pkl
              columns: protein_id, true_label, prediction
  Enformer: {predictions_dir}/enformer/predictions.pkl
              columns: protein_id, true_label, prediction  (ElasticNet on embeddings)

Output:
  {output_dir}/priority_genes.tsv
    Columns: symbol, true_label, pastl_pred, enformer_pred,
             pastl_se, enformer_se, se_improvement, group

Usage:
  python prioritize_genes.py \
      --pastl_pkl out/predictions/s5fo3aak_best/predictions.pkl \
      --enformer_pkl out/predictions/enformer/predictions.pkl \
      --output_dir out/analysis/ism
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_PREDICTIONS_DIR = "out/predictions"
_ISM_DIR = "out/analysis/ism"

_ENF_DEFAULT = f"{_PREDICTIONS_DIR}/enformer/predictions.pkl"

N_IMPROVED = 100
N_CONTROL = 50


def prioritize_genes(pastl_pkl, enformer_pkl, n_improved, n_control, output_dir):
    tlp = pd.read_pickle(pastl_pkl).rename(
        columns={"protein_id": "symbol", "prediction": "pastl_pred"}
    )
    enf = pd.read_pickle(enformer_pkl).rename(
        columns={"protein_id": "symbol", "prediction": "enformer_pred"}
    )

    df = tlp[["symbol", "true_label", "pastl_pred"]].merge(
        enf[["symbol", "enformer_pred"]], on="symbol", how="inner"
    )
    print(f"Merged {len(df):,} genes present in both prediction files")

    df["pastl_se"] = (df["true_label"] - df["pastl_pred"]) ** 2
    df["enformer_se"] = (df["true_label"] - df["enformer_pred"]) ** 2
    df["se_improvement"] = df["enformer_se"] - df["pastl_se"]

    df = df.sort_values("se_improvement", ascending=False).reset_index(drop=True)

    improved = df.head(n_improved).copy()
    improved["group"] = "improved"

    # Control: genes where both models perform equivalently (smallest |se_improvement|).
    # Draw only from genes not already in the improved set.
    remaining = df.iloc[n_improved:].copy()
    abs_impr = remaining["se_improvement"].abs()
    control = remaining.loc[abs_impr.nsmallest(n_control).index].copy()
    control["group"] = "control"

    out = pd.concat([improved, control], ignore_index=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "priority_genes.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    print(f"Improved group  ({n_improved} genes): "
          f"SE improvement {improved['se_improvement'].min():.4f} – "
          f"{improved['se_improvement'].max():.4f}")
    print(f"Control group   ({n_control} genes): "
          f"|SE improvement| ≤ {control['se_improvement'].abs().max():.4f}")
    print(f"Saved -> {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pastl_pkl", required=True,
                        help="Path to PASTL predictions.pkl")
    parser.add_argument("--enformer_pkl", default=_ENF_DEFAULT,
                        help="Path to Enformer (ElasticNet) predictions.pkl")
    parser.add_argument("--n_improved", type=int, default=N_IMPROVED,
                        help="Number of top-improved genes (default: 100)")
    parser.add_argument("--n_control", type=int, default=N_CONTROL,
                        help="Number of control genes (default: 50)")
    parser.add_argument("--output_dir", default=_ISM_DIR)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: top 10 improved genes only, no control set. "
                             "Submit the array job with --array=0-9.")
    args = parser.parse_args()

    if args.test:
        print("*** TEST MODE: top 10 improved genes, no control set ***")
        args.n_improved = 10
        args.n_control = 0

    prioritize_genes(
        args.pastl_pkl, args.enformer_pkl,
        args.n_improved, args.n_control,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
