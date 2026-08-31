r"""
Build the ISM regions manifest from priority genes, sequences_protein.bed,
and the GENCODE v36 annotation GTF.

For each priority gene, extracts intervals for:
  tss       — TSS ± --tss_flank bp (default 2000)
  utr5      — 5' UTR exons from GENCODE transcript (inferred from UTR/CDS boundaries)
  utr3      — 3' UTR exons from GENCODE transcript
  gene_body — first exon start to last exon end (written for optional --include_gene_body
              mode in run_ism.py; filtered out by default)

All intervals are clipped to the Enformer window [tss - 98304, tss + 98304).
Coordinates are 0-based, half-open (BED convention).

Output: {output_dir}/ism_regions.tsv
  Columns: symbol, chrom, start, end, region_type, strand, tss

Usage:
  python build_ism_regions.py \
      --priority_genes out/analysis/ism/priority_genes.tsv \
      --output_dir out/analysis/ism
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

WINDOW_SIZE = 196_608
HALF_WINDOW = WINDOW_SIZE // 2
TSS_FLANK_DEFAULT = 2_000

_SEQUENCES_BED = "data/sequences_protein.bed"
_ISM_DIR = "out/analysis/ism"


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def _parse_attr(attr_str, key):
    m = re.search(rf'{key} "([^"]+)"', attr_str)
    return m.group(1) if m else None


def load_utr_cds_from_gtf(gtf_path, tx_ids):
    """
    Stream the GENCODE GTF and collect UTR + CDS feature intervals for each
    transcript in tx_ids (unversioned ENST strings).

    Returns dict: tx_id -> {'utr': [(start, end),...], 'cds': [(start, end),...]}
    Coordinates are 0-based, half-open.
    """
    tx_set = set(tx_ids)
    records = {tx: {"utr": [], "cds": []} for tx in tx_set}

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    n_matched = 0
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in ("UTR", "CDS"):
                continue
            attr = fields[8]
            tx_versioned = _parse_attr(attr, "transcript_id")
            if tx_versioned is None:
                continue
            tx = tx_versioned.split(".")[0]   # strip GENCODE version suffix
            if tx not in tx_set:
                continue
            # GTF uses 1-based inclusive; convert to 0-based half-open
            start = int(fields[3]) - 1
            end = int(fields[4])
            if feature == "UTR":
                records[tx]["utr"].append((start, end))
            else:
                records[tx]["cds"].append((start, end))
            n_matched += 1

    found = sum(1 for v in records.values() if v["utr"] or v["cds"])
    print(f"  GTF: {n_matched:,} UTR/CDS records matched across {found}/{len(tx_ids)} transcripts")
    return records


def classify_utrs(utr_intervals, cds_intervals, strand):
    """
    Split UTR intervals into 5' and 3' lists based on CDS boundaries.

    On + strand: UTRs ending before the first CDS start are 5'; the rest are 3'.
    On - strand: UTRs starting after the last CDS end are 5'; the rest are 3'.
    """
    if not utr_intervals:
        return [], []
    if not cds_intervals:
        return utr_intervals, []

    cds_min = min(s for s, e in cds_intervals)
    cds_max = max(e for s, e in cds_intervals)

    utr5, utr3 = [], []
    for s, e in utr_intervals:
        if strand == "+":
            (utr5 if e <= cds_min else utr3).append((s, e))
        else:
            (utr5 if s >= cds_max else utr3).append((s, e))
    return utr5, utr3


def clip_to_window(intervals, win_start, win_end):
    """Clip intervals to [win_start, win_end); drop empty results."""
    out = []
    for s, e in intervals:
        cs, ce = max(s, win_start), min(e, win_end)
        if cs < ce:
            out.append((cs, ce))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--priority_genes", required=True,
                        help="Path to priority_genes.tsv (from prioritize_genes.py)")
    parser.add_argument("--sequences_bed", default=_SEQUENCES_BED)
    parser.add_argument("--gtf", required=True,
                        help="GENCODE v36 annotation GTF (gzipped)")
    parser.add_argument("--tss_flank", type=int, default=TSS_FLANK_DEFAULT,
                        help="bp on each side of TSS to include (default: 2000)")
    parser.add_argument("--output_dir", default=_ISM_DIR)
    args = parser.parse_args()

    genes = pd.read_csv(args.priority_genes, sep="\t")
    symbols = set(genes["symbol"].tolist())
    print(f"Loaded {len(symbols)} priority genes")

    bed = pd.read_csv(args.sequences_bed, sep="\t")
    bed = bed[bed["symbol"].isin(symbols)].reset_index(drop=True)
    missing = symbols - set(bed["symbol"])
    if missing:
        print(f"  WARNING: {len(missing)} priority genes not in BED file: {missing}")

    # Strip version suffix from transcript IDs (sequences_protein.bed stores unversioned,
    # but defensive split handles both cases)
    tx_ids = bed["txid"].astype(str).str.split(".").str[0].tolist()

    print(f"Parsing GTF for {len(tx_ids)} transcripts from:\n  {args.gtf}")
    gtf_records = load_utr_cds_from_gtf(args.gtf, tx_ids)
    print("GTF parsing complete.")

    rows = []
    for _, g in bed.iterrows():
        sym = g["symbol"]
        chrom = str(g["chrom"])
        strand = str(g["strand"])
        win_start = int(g["start"])
        win_end = int(g["end"])
        tss = win_start + HALF_WINDOW

        # TSS ± flank (clipped to window)
        tss_s = max(tss - args.tss_flank, win_start)
        tss_e = min(tss + args.tss_flank, win_end)
        if tss_s < tss_e:
            rows.append((sym, chrom, tss_s, tss_e, "tss", strand, tss))

        # 5' and 3' UTRs from GENCODE
        tx = str(g["txid"]).split(".")[0]
        rec = gtf_records.get(tx, {"utr": [], "cds": []})
        utr5, utr3 = classify_utrs(rec["utr"], rec["cds"], strand)

        for s, e in clip_to_window(utr5, win_start, win_end):
            rows.append((sym, chrom, s, e, "utr5", strand, tss))
        for s, e in clip_to_window(utr3, win_start, win_end):
            rows.append((sym, chrom, s, e, "utr3", strand, tss))

        # Gene body (first exon start to last exon end) — included so
        # run_ism.py can optionally activate it via --include_gene_body.
        try:
            exon_starts = [int(x) for x in str(g["exonstarts"]).split(",") if x]
            exon_ends = [int(x) for x in str(g["exonends"]).split(",") if x]
            if exon_starts and exon_ends:
                gb_s, gb_e = min(exon_starts), max(exon_ends)
                for s, e in clip_to_window([(gb_s, gb_e)], win_start, win_end):
                    rows.append((sym, chrom, s, e, "gene_body", strand, tss))
        except (ValueError, AttributeError):
            pass

    out_df = pd.DataFrame(
        rows, columns=["symbol", "chrom", "start", "end", "region_type", "strand", "tss"]
    )

    print("\nRegion counts per type:")
    print(out_df.groupby("region_type")["symbol"].count().to_string())

    n_no_utr = len(set(bed["symbol"]) - set(out_df[out_df["region_type"].isin(["utr5", "utr3"])]["symbol"]))
    if n_no_utr:
        print(f"\n  NOTE: {n_no_utr} genes have no UTR annotations in the GTF "
              f"(TSS region will still be mutated)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / "ism_regions.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved {len(out_df)} region intervals -> {out_path}")


if __name__ == "__main__":
    main()
