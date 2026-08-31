r"""
Build per-gene cis-window genotype dosage matrices from the CPTAC germline VCF.

For each gene in sequences_protein.bed, extracts all biallelic PASS SNVs within
its 196,608 bp cis-window, computes alt-allele dosage per sample (0/1/2; NaN
for missing genotypes), applies MAF and missingness filters, and saves a gzipped
dosage matrix TSV plus a variant index shard.

Designed for SLURM array jobs: use --gene_index and --n_genes to process a
contiguous slice of genes per task. Each task writes its own variant index shard
(variant_index_{gene_index}.tsv); concatenate shards after all tasks finish to
get the global index.

Inputs:
  VCF: cohort.clean.norm.vcf.gz  (tabix-indexed, biallelic PASS-normalized)
  BED: sequences_protein.bed     (chrom, start, end, split, strand, symbol, ...)

Outputs (under --output_dir):
  {gene_symbol}.tsv.gz            dosage matrix: rows=samples, cols=chrom:pos:ref:alt
  variant_index.tsv               full variant index (single-job / all-genes mode)
  variant_index_{gene_index}.tsv  per-task shard (SLURM array mode)

Usage (single job):
  python build_genotype_matrix.py \
      --vcf /path/to/data/CPTAC_germline/results/genotyped/cohort.clean.norm.vcf.gz \
      --bed data/sequences_protein.bed \
      --output_dir out/analysis/pwas/genotypes/

Usage (SLURM array — each task processes 50 genes):
  python build_genotype_matrix.py \
      --vcf ... --bed ... --output_dir ... \
      --gene_index $((SLURM_ARRAY_TASK_ID * 50)) \
      --n_genes 50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

_DEFAULT_BED = "data/sequences_protein.bed"
_DEFAULT_OUT = "out/analysis/pwas/genotypes/"

HALF_WINDOW = 98_304   # 196_608 // 2


# ---------------------------------------------------------------------------
# Gene windows
# ---------------------------------------------------------------------------

def load_genes(bed_path: str) -> pd.DataFrame:
    bed = pd.read_csv(bed_path, sep="\t")
    return pd.DataFrame({
        "symbol": bed["symbol"].astype(str),
        "chrom":  bed["chrom"].astype(str),
        "start":  bed["start"].astype(int),
        "end":    bed["end"].astype(int),
        "strand": bed["strand"].astype(str),
        "tss":    bed["start"].astype(int) + HALF_WINDOW + 1,  # 1-based to match VCF POS
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# VCF / dosage helpers
# ---------------------------------------------------------------------------

def _resolve_chrom(vcf: pysam.VariantFile, chrom: str) -> str | None:
    """Return the chromosome name as present in the VCF index, or None."""
    contigs = set(vcf.header.contigs.keys())
    if chrom in contigs:
        return chrom
    bare = chrom.replace("chr", "")
    if bare in contigs:
        return bare
    prefixed = chrom if chrom.startswith("chr") else "chr" + chrom
    if prefixed in contigs:
        return prefixed
    return None


def _extract_variants(
    vcf: pysam.VariantFile,
    samples: list[str],
    vcf_chrom: str,
    start: int,
    end: int,
    maf_min: float,
    miss_max: float,
    bed_chrom: str,
    tss: int,
    gene_symbol: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fetch biallelic PASS SNVs in [vcf_chrom:start-end] (0-based BED coords).

    Returns
    -------
    dosage : DataFrame  shape (n_samples, n_passing_variants); empty if none pass
    index_rows : list of dicts with variant metadata
    """
    dosage_cols: dict[str, np.ndarray] = {}
    index_rows: list[dict] = []

    for rec in vcf.fetch(vcf_chrom, max(0, start), end):
        # Require exactly one ALT allele (biallelic).
        if len(rec.alts) != 1:
            continue
        ref, alt = rec.ref, rec.alts[0]
        # SNV only: single-base ref and alt.
        if len(ref) != 1 or len(alt) != 1:
            continue
        # PASS filter.
        if "PASS" not in rec.filter:
            continue

        gt_array = [rec.samples[s].allele_indices for s in samples]
        dosages = np.array(
            [float(sum(a > 0 for a in gt)) if (gt is not None and None not in gt)
             else np.nan
             for gt in gt_array],
            dtype=float,
        )
        n_valid = int(np.sum(~np.isnan(dosages)))
        if n_valid == 0:
            continue

        miss = 1.0 - n_valid / len(samples)
        alt_freq = float(np.nansum(dosages)) / (2 * n_valid)
        maf = min(alt_freq, 1.0 - alt_freq)

        if miss > miss_max or maf < maf_min:
            continue

        # rec.pos is 1-based VCF POS.
        vid = f"{bed_chrom}:{rec.pos}:{ref}:{alt}"
        dosage_cols[vid] = dosages
        index_rows.append({
            "CHROM":          bed_chrom,
            "POS":            rec.pos,
            "REF":            ref,
            "ALT":            alt,
            "gene_symbol":    gene_symbol,
            "maf":            round(maf, 6),
            "missingness":    round(miss, 6),
            "distance_to_tss": abs(rec.pos - tss),
        })

    if not dosage_cols:
        return pd.DataFrame(index=samples), index_rows

    dosage_df = pd.DataFrame(dosage_cols, index=samples)
    return dosage_df, index_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vcf", required=True,
                        help="Tabix-indexed germline VCF.gz")
    parser.add_argument("--bed", default=_DEFAULT_BED,
                        help="sequences_protein.bed")
    parser.add_argument("--output_dir", default=_DEFAULT_OUT,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--maf_min", type=float, default=0.01,
                        help="Minimum minor allele frequency (default: 0.01)")
    parser.add_argument("--miss_max", type=float, default=0.10,
                        help="Maximum missingness rate (default: 0.10)")
    parser.add_argument("--gene_index", type=int, default=None,
                        help="0-based start index for SLURM array slicing")
    parser.add_argument("--n_genes", type=int, default=None,
                        help="Genes to process per job; required when --gene_index is set")
    args = parser.parse_args()

    if args.gene_index is not None and args.n_genes is None:
        parser.error("--n_genes is required when --gene_index is set")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = load_genes(args.bed)
    total_genes = len(genes)
    print(f"Loaded {total_genes:,} genes from {args.bed}")

    # SLURM array slicing.
    if args.gene_index is not None:
        lo = args.gene_index
        hi = min(lo + args.n_genes, total_genes)
        genes = genes.iloc[lo:hi].reset_index(drop=True)
        index_suffix = f"_{lo}"
        print(f"Array slice: genes[{lo}:{hi}]  ({len(genes):,} genes this task)")
    else:
        index_suffix = ""

    vcf = pysam.VariantFile(args.vcf)
    samples = list(vcf.header.samples)
    n_samples = len(samples)
    print(f"VCF samples : {n_samples:,}")
    print(f"Output dir  : {out_dir}")
    print(f"Filters     : MAF >= {args.maf_min}  missingness <= {args.miss_max}")
    print()

    all_index_rows: list[dict] = []
    per_gene_counts: list[int] = []

    for _, gene in genes.iterrows():
        symbol = str(gene["symbol"])
        chrom  = str(gene["chrom"])
        start  = int(gene["start"])
        end    = int(gene["end"])
        tss    = int(gene["tss"])

        vcf_chrom = _resolve_chrom(vcf, chrom)
        if vcf_chrom is None:
            print(f"  {symbol:<20s}  {chrom} not in VCF contigs — skipping")
            per_gene_counts.append(0)
            pd.DataFrame(index=samples).to_csv(
                out_dir / f"{symbol}.tsv.gz", sep="\t",
                index=True, index_label="sample_id", compression="gzip",
            )
            continue

        dosage_df, index_rows = _extract_variants(
            vcf, samples, vcf_chrom, start, end,
            args.maf_min, args.miss_max, chrom, tss, symbol,
        )

        n_var = len(index_rows)
        per_gene_counts.append(n_var)
        print(f"  {symbol:<20s}  {chrom}:{start}-{end}  variants={n_var:,}")

        dosage_df.to_csv(
            out_dir / f"{symbol}.tsv.gz", sep="\t",
            index=True, index_label="sample_id", compression="gzip",
        )
        all_index_rows.extend(index_rows)

    vcf.close()

    # Write variant index shard.
    index_cols = ["CHROM", "POS", "REF", "ALT", "gene_symbol",
                  "maf", "missingness", "distance_to_tss"]
    if all_index_rows:
        idx = pd.DataFrame(all_index_rows)[index_cols]
    else:
        idx = pd.DataFrame(columns=index_cols)

    index_path = out_dir / f"variant_index{index_suffix}.tsv"
    idx.to_csv(index_path, sep="\t", index=False)

    # Summary log.
    counts = np.array(per_gene_counts, dtype=int)
    total_var = int(counts.sum())
    n_processed = len(genes)
    print()
    print("=" * 50)
    print(f"Genes processed  : {n_processed:,}")
    print(f"Samples          : {n_samples:,}")
    print(f"Total variants   : {total_var:,}")
    if n_processed > 0:
        print(
            f"Variants / gene  : "
            f"median={np.median(counts):.0f}  "
            f"min={counts.min():,}  "
            f"max={counts.max():,}"
        )
    print(f"Variant index    : {index_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
