r"""
Filter per-gene genotype matrices to priority SNV sets for PWAS.

Loads each gene's dosage matrix from the genotype directory, then for each
priority set (pastl/enformer × 250K/500K/1M plus an unfiltered baseline)
retains only columns whose variant ID appears in the priority set.  Genes with
fewer than 2 variants after filtering are skipped.

Priority sets: pastl_250K, pastl_500K, pastl_1M,
               enformer_250K, enformer_500K, enformer_1M, all (baseline)

Inputs:
  {genotype_dir}/{gene_symbol}.tsv.gz   dosage matrices from build_genotype_matrix.py
  {variant_index}                        global variant index
  {priority_dir}/pastl_top{N}.tsv      PASTL priority SNV sets
  {priority_dir}/enformer_top{N}.tsv    Enformer priority SNV sets

Outputs (under --output_dir):
  {model}_{threshold}/{gene_symbol}.tsv.gz   filtered dosage matrices
  {model}_{threshold}_summary.tsv            per-gene variant counts

Usage:
  python filter_by_priority.py \
      --genotype_dir out/analysis/pwas/genotypes/ \
      --variant_index out/analysis/pwas/genotypes/variant_index.tsv \
      --priority_dir out/analysis/pwas/priority_snvs/ \
      --output_dir out/analysis/pwas/filtered_genotypes/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_GENO_DIR  = "out/analysis/pwas/genotypes/"
_DEFAULT_INDEX     = "out/analysis/pwas/genotypes/variant_index.tsv"
_DEFAULT_PRIOR_DIR = "out/analysis/pwas/priority_snvs/"
_DEFAULT_OUT       = "out/analysis/pwas/filtered_genotypes/"

THRESHOLDS = ["250K", "500K", "1M"]
MODELS     = ["pastl", "enformer"]
MIN_VARIANTS = 2


# ---------------------------------------------------------------------------
# Load priority sets
# ---------------------------------------------------------------------------

def _vid(chrom, pos, ref, alt) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


def load_priority_sets(priority_dir: Path) -> dict[str, set[str]]:
    """
    Returns a dict mapping set name → frozenset of 'chrom:pos:ref:alt' strings.
    Includes 'all' as a sentinel (None means no filtering).
    """
    sets: dict[str, set[str]] = {"all": None}
    for model in MODELS:
        for thresh in THRESHOLDS:
            name = f"{model}_{thresh}"
            path = priority_dir / f"{model}_top{thresh}.tsv"
            df = pd.read_csv(path, sep="\t", usecols=["CHROM", "POS", "REF", "ALT"])
            vids = set(
                _vid(c, p, r, a)
                for c, p, r, a in zip(df["CHROM"], df["POS"], df["REF"], df["ALT"])
            )
            sets[name] = vids
            print(f"  Loaded {name}: {len(vids):,} variants")
    return sets


# ---------------------------------------------------------------------------
# Per-gene filtering
# ---------------------------------------------------------------------------

def filter_gene(
    gene: str,
    geno_path: Path,
    priority_sets: dict[str, set[str]],
    out_dirs: dict[str, Path],
    summary_rows: dict[str, list[dict]],
):
    try:
        mat = pd.read_csv(geno_path, sep="\t", index_col=0, low_memory=False)
    except Exception as e:
        print(f"  [WARN] {gene}: could not read {geno_path}: {e}")
        return

    all_cols = set(mat.columns)
    n_total = len(mat.columns)

    for set_name, vid_set in priority_sets.items():
        if vid_set is None:
            # baseline: keep all columns
            keep = mat.columns.tolist()
        else:
            keep = [c for c in mat.columns if c in vid_set]

        n_filtered = len(keep)
        if n_filtered < MIN_VARIANTS:
            continue

        out_path = out_dirs[set_name] / f"{gene}.tsv.gz"
        mat[keep].to_csv(out_path, sep="\t", index=True, compression="gzip")

        summary_rows[set_name].append({
            "gene_symbol":        gene,
            "n_variants_total":   n_total,
            "n_variants_filtered": n_filtered,
            "fraction_retained":  round(n_filtered / n_total, 4) if n_total else 0.0,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--genotype_dir", default=_DEFAULT_GENO_DIR)
    parser.add_argument("--variant_index", default=_DEFAULT_INDEX)
    parser.add_argument("--priority_dir",  default=_DEFAULT_PRIOR_DIR)
    parser.add_argument("--output_dir",    default=_DEFAULT_OUT)
    args = parser.parse_args()

    geno_dir   = Path(args.genotype_dir)
    prior_dir  = Path(args.priority_dir)
    out_dir    = Path(args.output_dir)

    # Load priority sets.
    print("Loading priority sets ...")
    priority_sets = load_priority_sets(prior_dir)

    # Create output subdirectories.
    set_names = list(priority_sets.keys())
    out_dirs: dict[str, Path] = {}
    for name in set_names:
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[name] = d

    # Load variant index to enumerate genes that have any variants.
    print(f"Loading variant index from {args.variant_index} ...")
    idx = pd.read_csv(args.variant_index, sep="\t", usecols=["gene_symbol"])
    genes_with_variants = sorted(idx["gene_symbol"].unique())
    print(f"  {len(genes_with_variants):,} genes with at least one variant")

    # Per-set summary accumulators.
    summary_rows: dict[str, list[dict]] = {name: [] for name in set_names}

    print(f"\nFiltering {len(genes_with_variants):,} genes × {len(set_names)} sets ...")
    for i, gene in enumerate(genes_with_variants):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,} / {len(genes_with_variants):,} genes processed ...")
        geno_path = geno_dir / f"{gene}.tsv.gz"
        if not geno_path.exists():
            print(f"  [WARN] missing: {geno_path}")
            continue
        filter_gene(gene, geno_path, priority_sets, out_dirs, summary_rows)

    # Write per-set summary TSVs and log stats.
    print("\nSummary:")
    print(f"  {'Set':<20s}  {'Genes retained':>15s}  {'Median variants':>15s}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
    for name in set_names:
        rows = summary_rows[name]
        if rows:
            summ = pd.DataFrame(rows)
            med_var = np.median(summ["n_variants_filtered"])
        else:
            summ = pd.DataFrame(columns=[
                "gene_symbol", "n_variants_total",
                "n_variants_filtered", "fraction_retained",
            ])
            med_var = 0
        summ_path = out_dir / f"{name}_summary.tsv"
        summ.to_csv(summ_path, sep="\t", index=False)
        print(f"  {name:<20s}  {len(rows):>15,}  {med_var:>15.0f}")

    print(f"\nTotal genes processed : {len(genes_with_variants):,}")
    print(f"Output dir            : {out_dir}")


if __name__ == "__main__":
    main()
