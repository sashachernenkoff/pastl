r"""
Convert TargetScan predicted target sites from UTR-relative coordinates to
hg38 genomic BED format.

TargetScan stores site positions as 1-based offsets within each transcript's
3'UTR sequence (columns UTR_start / UTR_end). This script maps those offsets
to hg38 genomic coordinates using the 3'UTR exon structure from the GENCODE
v36 GTF, resolving multi-exon UTRs correctly for both strands.

Only human targets (Gene Tax ID = 9606) are kept. Only transcripts in
sequences_protein.bed are converted (our analysis universe).

Output: {output_path} — BED6, sorted by chrom/start.
  chrom  start  end  miRNA_name  context++_score  strand

Usage:
  python convert_targetscan.py \
      --targetscan  /path/to/data/motifs/Predicted_Targets_Context_Scores.default_predictions.txt \
      --gtf         /path/to/data/gencode.v36.annotation.gtf.gz \
      --sequences_bed data/sequences_protein.bed \
      --output      out/analysis/ism/targetscan_hg38.bed
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

_SEQ_BED     = "data/sequences_protein.bed"
_OUTPUT      = "out/analysis/ism/targetscan_hg38.bed"

HUMAN_TAX_ID = "9606"


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def _parse_attr(attr_str, key):
    m = re.search(rf'{key} "([^"]+)"', attr_str)
    return m.group(1) if m else None


def load_utr_cds_from_gtf(gtf_path, tx_ids):
    """
    Stream GENCODE GTF and collect UTR + CDS intervals for transcripts in tx_ids
    (unversioned ENST strings).

    Returns dict: tx_id -> {'utr': [(start, end),...], 'cds': [(start, end),...],
                             'chrom': str, 'strand': str}
    Coordinates are 0-based half-open.
    """
    tx_set = set(tx_ids)
    records = {tx: {"utr": [], "cds": [], "chrom": None, "strand": None}
               for tx in tx_set}

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
            tx_ver = _parse_attr(attr, "transcript_id")
            if tx_ver is None:
                continue
            tx = tx_ver.split(".")[0]
            if tx not in tx_set:
                continue
            chrom  = fields[0]
            strand = fields[6]
            start  = int(fields[3]) - 1   # 0-based half-open
            end    = int(fields[4])
            rec = records[tx]
            rec["chrom"]  = chrom
            rec["strand"] = strand
            if feature == "UTR":
                rec["utr"].append((start, end))
            else:
                rec["cds"].append((start, end))
            n_matched += 1

    found = sum(1 for v in records.values() if v["chrom"] is not None)
    print(f"  GTF: {n_matched:,} UTR/CDS records for {found}/{len(tx_ids)} transcripts")
    return records


def classify_utr3(utr_intervals, cds_intervals, strand):
    """Return only the 3' UTR intervals (downstream of CDS)."""
    if not utr_intervals or not cds_intervals:
        return []
    cds_min = min(s for s, e in cds_intervals)
    cds_max = max(e for s, e in cds_intervals)
    utr3 = []
    for s, e in utr_intervals:
        if strand == "+":
            if s >= cds_max:
                utr3.append((s, e))
        else:
            if e <= cds_min:
                utr3.append((s, e))
    return utr3


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def _sorted_utr3_exons(utr3_intervals, strand):
    """Sort 3'UTR exons in transcript order (5'→3')."""
    if strand == "+":
        return sorted(utr3_intervals, key=lambda x: x[0])        # ascending
    else:
        return sorted(utr3_intervals, key=lambda x: x[0], reverse=True)  # descending


def _map_utr_pos(utr_pos_1based, exons_ordered, strand):
    """
    Map a 1-based position within the spliced 3'UTR to a 0-based genomic position.
    Returns None if the position exceeds the total UTR length.
    """
    remaining = utr_pos_1based
    for exon_start, exon_end in exons_ordered:
        exon_len = exon_end - exon_start
        if remaining <= exon_len:
            offset = remaining - 1   # 0-based within exon
            if strand == "+":
                return exon_start + offset
            else:
                return exon_end - 1 - offset
        remaining -= exon_len
    return None


def utr_site_to_genomic(utr_start_1, utr_end_1, exons_ordered, strand):
    """
    Convert a TargetScan 1-based UTR site (inclusive both ends) to a
    0-based half-open genomic interval [gstart, gend).
    Returns (gstart, gend) or None if mapping fails.
    """
    g_s = _map_utr_pos(utr_start_1, exons_ordered, strand)
    g_e = _map_utr_pos(utr_end_1,   exons_ordered, strand)
    if g_s is None or g_e is None:
        return None
    return min(g_s, g_e), max(g_s, g_e) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targetscan",     required=True,
                        help="TargetScan Predicted_Targets_Context_Scores.default_predictions.txt")
    parser.add_argument("--gtf",            required=True,
                        help="GENCODE v36 annotation GTF (gzipped)")
    parser.add_argument("--sequences_bed",  default=_SEQ_BED)
    parser.add_argument("--output",         default=_OUTPUT)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Transcript universe from sequences_protein.bed ---
    bed = pd.read_csv(args.sequences_bed, sep="\t")
    tx_ids = set(bed["txid"].astype(str).str.split(".").str[0].tolist())
    chrom_map = dict(zip(
        bed["txid"].astype(str).str.split(".").str[0],
        bed["chrom"].astype(str)
    ))
    strand_map = dict(zip(
        bed["txid"].astype(str).str.split(".").str[0],
        bed["strand"].astype(str)
    ))
    print(f"Transcript universe: {len(tx_ids):,} transcripts from sequences_protein.bed")

    # --- Load TargetScan predictions (human, our transcripts) ---
    print(f"Loading TargetScan predictions from {args.targetscan} ...")
    cols = ["gene_id", "gene_symbol", "transcript_id", "tax_id", "mirna",
            "site_type", "utr_start", "utr_end", "context_score",
            "context_pct", "weighted_score", "weighted_pct", "rel_kd"]
    ts = pd.read_csv(args.targetscan, sep="\t", header=0, names=cols,
                     usecols=["transcript_id", "tax_id", "mirna",
                               "utr_start", "utr_end", "context_score"])
    ts = ts[ts["tax_id"].astype(str) == HUMAN_TAX_ID].copy()
    ts["tx"] = ts["transcript_id"].astype(str).str.split(".").str[0]
    ts = ts[ts["tx"].isin(tx_ids)].copy()
    print(f"  Human entries in our transcript set: {len(ts):,}")

    # --- Parse GTF for 3'UTR exon structure ---
    needed_tx = set(ts["tx"].unique())
    print(f"Parsing GTF for {len(needed_tx):,} transcripts ...")
    gtf_records = load_utr_cds_from_gtf(args.gtf, needed_tx)

    # --- Map UTR positions to genomic coordinates ---
    print("Mapping UTR-relative positions to hg38 genomic coordinates ...")
    bed_rows = []
    n_ok = n_skip = 0

    for tx, grp in ts.groupby("tx"):
        rec = gtf_records.get(tx)
        if rec is None or rec["chrom"] is None:
            n_skip += len(grp)
            continue
        chrom  = rec["chrom"]
        strand = rec["strand"]
        utr3   = classify_utr3(rec["utr"], rec["cds"], strand)
        if not utr3:
            n_skip += len(grp)
            continue
        exons_ordered = _sorted_utr3_exons(utr3, strand)

        for _, row in grp.iterrows():
            coords = utr_site_to_genomic(
                int(row["utr_start"]), int(row["utr_end"]),
                exons_ordered, strand
            )
            if coords is None:
                n_skip += 1
                continue
            gstart, gend = coords
            score = row["context_score"] if pd.notna(row["context_score"]) else 0.0
            bed_rows.append((chrom, gstart, gend, row["mirna"], score, strand))
            n_ok += 1

    print(f"  Mapped: {n_ok:,}   Skipped (no UTR annotation or out-of-range): {n_skip:,}")

    out_df = pd.DataFrame(bed_rows, columns=["chrom", "start", "end",
                                              "mirna", "score", "strand"])
    out_df = out_df.sort_values(["chrom", "start"]).reset_index(drop=True)
    out_df.to_csv(out_path, sep="\t", index=False, header=False)
    print(f"Written {len(out_df):,} sites -> {out_path}")


if __name__ == "__main__":
    main()
