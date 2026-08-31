r"""
Build the per-protein across-patient variability table consumed by
plot_egene_stratification.py (--variance_table).

Uses preprocessing.build_sequences_bed.load_gct_matrix so cohort and sample
selection match the training target: CPTAC LUAD discovery ∩ confirmatory,
Solid-Tissue-Normal samples only, replicate-averaged, protein × patient log2
abundance. Variability is computed on the observed values (before the
left-censored imputation used for the mean target).

Emits gene, iqr, std where:
  * iqr = interquartile range of log2 abundance across patients.
  * std = standard deviation of log2 abundance across patients.
CV (std/mean) is omitted because log2 abundance crosses zero.

Usage:
  python analysis/egene_eval/build_variance_table.py \
      --gct_paths /path/to/CPTAC_LUAD_discovery_proteome.gct \
                  /path/to/CPTAC_LUAD_confirmatory_proteome.gct \
      --output_dir out/analysis/egene_eval
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "preprocessing"))

from build_sequences_bed import load_gct_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gct_paths", nargs="+", required=True,
                    help="CPTAC proteomics GCT files (discovery + confirmatory)")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write protein_variance.tsv")
    args = ap.parse_args()

    out_path = Path(args.output_dir) / "protein_variance.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = load_gct_matrix(args.gct_paths)              # protein × patient, log2, NaNs kept
    iqr = mat.quantile(0.75, axis=1) - mat.quantile(0.25, axis=1)
    std = mat.std(axis=1)

    out = pd.DataFrame({"gene": mat.index, "iqr": iqr.values, "std": std.values})
    out = out.dropna(subset=["iqr", "std"])
    out.to_csv(out_path, sep="\t", index=False)

    print(f"Wrote {len(out):,} proteins to {out_path}")
    print(out[["iqr", "std"]].quantile([0.5, 0.9, 0.95, 1.0]).to_string())


if __name__ == "__main__":
    main()
