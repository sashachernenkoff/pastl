r"""
Identify PASTL-dominant ISM positions, export as BED, and scan for
post-transcriptional regulatory motifs using external databases.

A position is called PASTL-dominant when:
    (pastl_ism_z - enformer_ism_z) >= ratio_threshold

Motif scanning (each layer is optional and layers stack):
  TargetScan  — bedtools intersect of dominant positions against a TargetScan
                predicted-target-sites BED (miRNA family name in column 4).
  ATtRACT     — FIMO PWM scan of 21-bp reference windows against the ATtRACT
                human RBP PWM database (MEME format).
  polyA       — Canonical polyadenylation signals (AATAAA/ATTAAA); always runs.

Enrichment is computed against a matched background (z_diff < 0, same region
type distribution, same N) using Fisher's exact test.

Outputs:
  {output_dir}/pastl_dominant_positions.bed  — sorted BED6
  {output_dir}/motif_hits.tsv                 — long-form hits table (fg only)
  {output_dir}/motif_enrichment.tsv           — fold enrichment + Fisher p-value

Usage:
  python intersect_pastl_dominant.py \
      --scores_dir    out/analysis/s5fo3aak_best/ism/scores \
      --output_dir    out/analysis/s5fo3aak_best/ism/motifs \
      --fasta         /path/to/GRCh38.d1.vd1.fa \
      --targetscan_bed /path/to/targetscan_hg38.bed \
      --attract_pwm   /path/to/ATtRACT_human.meme
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.stats import fisher_exact

# load_scores lives in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from plot_ism import load_scores

WINDOW_HALF = 10   # ± bp around each position → 21-bp window

# Canonical polyadenylation signals — always run, no database needed
POLYA_PATTERNS = {
    "PAS_AATAAA": "AATAAA",
    "PAS_ATTAAA": "ATTAAA",
}

def _rc(seq):
    """Reverse-complement a DNA string."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


_TARGETSCAN_URL = (
    "https://www.targetscan.org/vert_80/vert_80_data_download/"
    "Predicted_Targets_Context_Scores.default_predictions.txt.zip"
)
_ATTRACT_URL = "https://attract.cnic.es/download  (Human PWMs, MEME format)"


# ---------------------------------------------------------------------------
# Sequence / window helpers  (unchanged)
# ---------------------------------------------------------------------------

def _fetch_window(fa, chrom, pos, window_half=WINDOW_HALF):
    """Return a DNA string of length 2*window_half+1 centred on pos (0-based)."""
    chrom_key = chrom if chrom in fa else f"chr{chrom}"
    chrom_len = len(fa[chrom_key])
    start = max(0, pos - window_half)
    end = min(chrom_len, pos + window_half + 1)
    seq = str(fa[chrom_key][start:end].seq.upper())
    left_pad  = max(0, window_half - pos)
    right_pad = (2 * window_half + 1) - len(seq) - left_pad
    return ("N" * left_pad) + seq + ("N" * max(0, right_pad))


def _parse_region_filter(region_filter_str):
    if region_filter_str.strip().lower() == "all":
        return None
    return {r.strip() for r in region_filter_str.split(",")}


# ---------------------------------------------------------------------------
# PolyA signal scan (always-run)
# ---------------------------------------------------------------------------

def _scan_polya_patterns(window):
    """Return list of (motif_name, motif_type) for canonical polyA signal hits."""
    return [
        (name, "polyA_signal")
        for name, pat in POLYA_PATTERNS.items()
        if pat in window
    ]


# ---------------------------------------------------------------------------
# TargetScan intersect (bedtools)
# ---------------------------------------------------------------------------

def _write_temp_bed(df, path):
    """Write a minimal BED6 for a set of ISM positions (internal use)."""
    out = df.copy()
    out["_start"] = out["position"].astype(int)
    out["_end"]   = out["position"].astype(int) + 1
    out["_name"]  = (
        out["symbol"].astype(str) + ":" + out["ref_base"].astype(str) + ">" + out["alt_base"].astype(str)
        + ":" + out["region_type"].astype(str) + ":" + out["position"].astype(str)
    )
    out[["chrom", "_start", "_end", "_name", "z_diff", "strand"]].to_csv(
        path, sep="\t", index=False, header=False
    )


def _targetscan_intersect(bed_path, targetscan_bed):
    """
    bedtools intersect dominant positions against the TargetScan BED.

    Returns dict: internal_name (symbol:ref>alt:region_type:pos)
                  -> set of miRNA family strings.
    Column 4 (0-based index 3) of the TargetScan BED is the miRNA family name.
    After -wa -wb with a 6-column dominant BED, TargetScan col 4 lands at index 9.
    """
    result = subprocess.run(
        ["bedtools", "intersect",
         "-a", str(bed_path), "-b", str(targetscan_bed),
         "-wa", "-wb"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: bedtools intersect failed:\n    {result.stderr.strip()[:400]}")
        return {}

    hits = {}
    for line in result.stdout.strip().splitlines():
        cols = line.split("\t")
        if len(cols) < 10:
            continue
        pos_name    = cols[3]   # _name col from dominant BED
        mirna_fam   = cols[9]   # col 4 of TargetScan BED (0-indexed: 6 dom cols + 3)
        hits.setdefault(pos_name, set()).add(mirna_fam)
    return hits


# ---------------------------------------------------------------------------
# ATtRACT / FIMO scan
# ---------------------------------------------------------------------------

def _fimo_seq_name(row):
    """Sequence name safe for FIMO: no '>' (FASTA delimiter) or ':' (FIMO coord separator)."""
    return (
        f"{row['symbol']}__{row['ref_base']}to{row['alt_base']}"
        f"__{row['region_type']}__{int(row['position'])}"
    )


def _write_fasta_windows(df, fa, fasta_path):
    """Write 21-bp windows for all positions in df to a FASTA file for FIMO."""
    with open(fasta_path, "w") as fh:
        for _, row in df.iterrows():
            window = _fetch_window(fa, str(row["chrom"]), int(row["position"]))
            fh.write(f">{_fimo_seq_name(row)}\n{window}\n")


def _run_fimo(attract_pwm, fasta_path, fimo_threshold, work_dir):
    """
    Run FIMO against the ATtRACT MEME-format PWM database.

    Creates output in {work_dir}/fimo_out/ and parses fimo.tsv.
    Returns dict: seq_name -> set of motif_id strings with p-value <= threshold.
    """
    fimo_out = Path(work_dir) / "fimo_out"
    fimo_out.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "fimo",
            "--thresh",    str(fimo_threshold),
            "--oc",        str(fimo_out),
            "--verbosity", "1",
            str(attract_pwm),
            str(fasta_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: FIMO failed:\n    {result.stderr.strip()[:400]}")
        return {}

    tsv_path = fimo_out / "fimo.tsv"
    if not tsv_path.exists():
        print(f"  WARNING: FIMO output not found at {tsv_path}")
        return {}

    try:
        fimo_df = pd.read_csv(tsv_path, sep="\t", comment="#")
    except Exception as e:
        print(f"  WARNING: could not parse FIMO output: {e}")
        return {}

    hits = {}
    for _, row in fimo_df.iterrows():
        seq_name  = str(row.get("sequence_name", ""))
        motif_id  = str(row.get("motif_id", ""))
        hits.setdefault(seq_name, set()).add(motif_id)
    return hits


# ---------------------------------------------------------------------------
# Background set
# ---------------------------------------------------------------------------

def _build_background(df, dominant, region_keep, seed=42):
    """
    Sample background positions: z_diff < 0, same region type distribution
    as dominant, same N per region type.
    """
    bg_pool = df[df["z_diff"] < 0].copy()
    if region_keep is not None:
        bg_pool = bg_pool[bg_pool["region_type"].isin(region_keep)]

    parts = []
    for rtype, cnt in dominant["region_type"].value_counts().items():
        pool = bg_pool[bg_pool["region_type"] == rtype]
        n = min(cnt, len(pool))
        if n > 0:
            parts.append(pool.sample(n=n, random_state=seed))

    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

def _fisher_enrichment(n_fg_hit, n_fg, n_bg_hit, n_bg):
    """
    One-sided Fisher's exact test (greater) for motif enrichment in foreground.
    Returns (fold_enrichment, p_value).
    """
    fg_rate = n_fg_hit / n_fg if n_fg > 0 else 0.0
    bg_rate = n_bg_hit / n_bg if n_bg > 0 else 0.0
    fold    = (fg_rate / bg_rate) if bg_rate > 0 else (float("inf") if fg_rate > 0 else 1.0)
    _, pvalue = fisher_exact(
        [[n_fg_hit, n_fg - n_fg_hit],
         [n_bg_hit, n_bg - n_bg_hit]],
        alternative="greater",
    )
    return round(fold, 3), float(pvalue)


# ---------------------------------------------------------------------------
# GENCODE feature annotation
# ---------------------------------------------------------------------------

def _gencode_gtf_to_bed(gtf_path, bed_path,
                         features=("gene", "exon", "CDS", "UTR")):
    """Stream GENCODE GTF and write a sorted BED6 of selected feature types."""
    import gzip as _gzip
    gene_name_re = re.compile(r'gene_name "([^"]+)"')
    feature_set = set(features)
    rows = []
    opener = _gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in feature_set:
                continue
            m = gene_name_re.search(parts[8])
            gname = m.group(1) if m else "."
            rows.append((parts[0], int(parts[3]) - 1, int(parts[4]),
                         f"{parts[2]}:{gname}", ".", parts[6]))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(bed_path, "w") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"    GTF → BED: {len(rows):,} features written")


def _run_gencode_annotation(bed_path, gtf_path, output_dir, work_dir):
    """
    Intersect dominant positions against GENCODE feature BED.
    Writes {output_dir}/gencode_annotation_hits.bed and prints a feature summary.
    Returns a DataFrame of hits, or an empty DataFrame on failure.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Building GENCODE feature BED from {gtf_path} ...")
    gtf_bed = work_dir / "gencode_features.bed"
    _gencode_gtf_to_bed(gtf_path, gtf_bed)

    result = subprocess.run(
        ["bedtools", "intersect",
         "-a", str(bed_path), "-b", str(gtf_bed),
         "-wa", "-wb"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: GENCODE bedtools intersect failed:\n"
              f"    {result.stderr.strip()[:300]}")
        return pd.DataFrame()

    hits = []
    for line in result.stdout.strip().splitlines():
        cols = line.split("\t")
        if len(cols) < 10:
            continue
        feat_field = cols[9]                              # "exon:BRCA1"
        feature   = feat_field.split(":")[0]
        gene_name = feat_field.split(":", 1)[1] if ":" in feat_field else feat_field
        hits.append({
            "chrom":     cols[0],
            "start":     int(cols[1]),
            "end":       int(cols[2]),
            "name":      cols[3],
            "z_diff":    float(cols[4]),
            "strand":    cols[5],
            "feature":   feature,
            "gene_name": gene_name,
        })

    hits_df = pd.DataFrame(hits)
    out_path = Path(output_dir) / "gencode_annotation_hits.bed"
    hits_df.to_csv(out_path, sep="\t", index=False)

    if hits_df.empty:
        print("  No GENCODE feature overlaps found.")
    else:
        print(f"  GENCODE annotation hits -> {out_path}  ({len(hits_df):,} rows)")
        print("  Feature type breakdown:")
        for feat, cnt in hits_df["feature"].value_counts().items():
            print(f"    {feat:<12} {cnt:>6,}")

    return hits_df


# ---------------------------------------------------------------------------
# Per-cohort motif scanning
# ---------------------------------------------------------------------------

def _make_hit_row(row, motif_name, motif_type, window_seq):
    return {
        "chrom":       row["chrom"],
        "start":       int(row["position"]),
        "end":         int(row["position"]) + 1,
        "symbol":      row["symbol"],
        "region_type": row["region_type"],
        "z_diff":      round(float(row["z_diff"]), 4),
        "strand":      row["strand"],
        "ref_base":    row["ref_base"],
        "alt_base":    row["alt_base"],
        "motif_name":  motif_name,
        "motif_type":  motif_type,
        "window_seq":  window_seq if window_seq is not None else "",
    }


def _scan_cohort(df, fa, label, targetscan_bed, attract_pwm, fimo_threshold, work_dir):
    """
    Run all enabled motif scans for a cohort (foreground or background).
    Returns list of hit-row dicts.
    """
    hit_rows = []
    work_dir = Path(work_dir) / label
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. PolyA signals (always)
    if fa is not None:
        print(f"  [{label}] Scanning {len(df):,} positions for polyA signals ...")
        for _, row in df.iterrows():
            window = _fetch_window(fa, str(row["chrom"]), int(row["position"]))
            if str(row["strand"]) == "-":
                window = _rc(window)
            for motif_name, motif_type in _scan_polya_patterns(window):
                hit_rows.append(_make_hit_row(row, motif_name, motif_type, window))

    # 2. TargetScan bedtools intersect
    if targetscan_bed is not None:
        print(f"  [{label}] bedtools intersect vs TargetScan ({len(df):,} positions) ...")
        tmp_bed = work_dir / "positions.bed"
        _write_temp_bed(df, tmp_bed)
        ts_hits = _targetscan_intersect(tmp_bed, targetscan_bed)
        df = df.copy()
        df["_name"] = (
            df["symbol"].astype(str) + ":" + df["ref_base"].astype(str) + ">" + df["alt_base"].astype(str)
            + ":" + df["region_type"].astype(str) + ":" + df["position"].astype(str)
        )
        for _, row in df.iterrows():
            for mirna_fam in ts_hits.get(row["_name"], []):
                hit_rows.append(_make_hit_row(row, mirna_fam, "miRNA_targetscan", None))

    # 3. ATtRACT / FIMO
    if attract_pwm is not None and fa is not None:
        print(f"  [{label}] Extracting windows and running FIMO ...")
        tmp_fa = work_dir / "windows.fa"
        _write_fasta_windows(df, fa, tmp_fa)
        fimo_hits = _run_fimo(attract_pwm, tmp_fa, fimo_threshold, work_dir)
        df = df.copy()
        for _, row in df.iterrows():
            for motif_id in fimo_hits.get(_fimo_seq_name(row), []):
                hit_rows.append(_make_hit_row(row, motif_id, "RBP_fimo", None))

    return hit_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores_dir", required=True,
                        help="Directory containing per-gene .h5 files from run_ism.py")
    parser.add_argument("--priority_genes", default=None,
                        help="Path to priority_genes.tsv (optional; adds group labels)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fasta", required=True,
                        help="Reference genome FASTA for window extraction")
    parser.add_argument("--ratio_threshold", type=float, default=1.0,
                        help="Minimum z_diff (pastl_ism_z - enformer_ism_z) to call "
                             "a position PASTL-dominant (default: 1.0 SD)")
    parser.add_argument("--pseudocount", type=float, default=1e-6,
                        help="Pseudocount added before ratio computation (default: 1e-6)")
    parser.add_argument("--region_filter", default="utr3",
                        help="Comma-separated region types to focus on, or 'all' "
                             "(default: utr3)")
    parser.add_argument("--targetscan_bed", default=None,
                        help="TargetScan predicted target sites BED file "
                             "(miRNA family name in column 4).")
    parser.add_argument("--attract_pwm", default=None,
                        help="ATtRACT human RBP PWMs in MEME format (for FIMO).")
    parser.add_argument("--fimo_threshold", type=float, default=1e-4,
                        help="FIMO p-value threshold (default: 1e-4)")
    parser.add_argument("--gtf", default=None,
                        help="GENCODE GTF (gzipped) for feature annotation of dominant "
                             "positions (gene/exon/UTR/CDS). Optional.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    region_keep = _parse_region_filter(args.region_filter)

    # --- Print download instructions for missing databases ---
    if args.targetscan_bed is None:
        print(
            "\nNOTE: --targetscan_bed not provided. TargetScan miRNA intersection skipped.\n"
            f"  Download: {_TARGETSCAN_URL}\n"
        )
    if args.attract_pwm is None:
        print(
            "NOTE: --attract_pwm not provided. FIMO/ATtRACT RBP scan skipped.\n"
            f"  Download: {_ATTRACT_URL}\n"
        )

    # --- Load ISM scores ---
    print("Loading ISM scores ...")
    df = load_scores(args.scores_dir, args.priority_genes)

    # --- Z-score each model globally, then compute difference ---
    df["pastl_ism_z"]   = (df["pastl_ism"]   - df["pastl_ism"].mean())   / df["pastl_ism"].std()
    df["enformer_ism_z"] = (df["enformer_ism"] - df["enformer_ism"].mean()) / df["enformer_ism"].std()
    df["z_diff"] = df["pastl_ism_z"] - df["enformer_ism_z"]

    # --- Filter ---
    mask = df["z_diff"] >= args.ratio_threshold
    if region_keep is not None:
        mask &= df["region_type"].isin(region_keep)
    dominant = df[mask].copy().reset_index(drop=True)

    print(f"\nPASTL-dominant positions (z_diff >= {args.ratio_threshold}"
          f", regions: {args.region_filter}):")
    print(f"  Total passing filter : {len(dominant):,}")

    if dominant.empty:
        print("  No positions pass the filter — try lowering --ratio_threshold.")
        return

    print("\n  By region type:")
    for rtype, cnt in dominant["region_type"].value_counts().items():
        print(f"    {rtype:<12} {cnt:>6,}")

    print("\n  By gene (top 10):")
    for sym, cnt in dominant["symbol"].value_counts().head(10).items():
        print(f"    {sym:<16} {cnt:>6,}")

    print(f"\n  Median z_diff : {dominant['z_diff'].median():.3f}")
    print(f"  Max    z_diff : {dominant['z_diff'].max():.3f}")

    # --- Export BED ---
    bed_path = Path(args.output_dir) / "pastl_dominant_positions.bed"
    bed = dominant.sort_values(["chrom", "position"])[
        ["chrom", "position", "symbol", "ref_base", "alt_base", "region_type",
         "z_diff", "strand"]
    ].copy()
    bed["start"] = bed["position"]
    bed["end"]   = bed["position"] + 1
    bed["name"]  = (
        bed["symbol"].astype(str) + ":" + bed["ref_base"].astype(str) + ">" + bed["alt_base"].astype(str)
        + ":" + bed["region_type"].astype(str)
    )
    bed[["chrom", "start", "end", "name", "z_diff", "strand"]].to_csv(
        bed_path, sep="\t", index=False, header=False
    )
    print(f"\nBED written -> {bed_path}")

    # --- GENCODE feature annotation ---
    if args.gtf and Path(args.gtf).exists():
        print("\nAnnotating dominant positions with GENCODE features ...")
        _run_gencode_annotation(
            bed_path, args.gtf, args.output_dir,
            work_dir=Path(args.output_dir) / "tmp" / "gencode",
        )
    elif args.gtf:
        print(f"\nWARNING: GTF not found at {args.gtf} — GENCODE annotation skipped.")

    # --- FASTA handle ---
    fasta_path = Path(args.fasta)
    fa = None
    if fasta_path.exists():
        fa = Fasta(str(fasta_path))
    else:
        print(f"\nWARNING: FASTA not found at {fasta_path} — window-based scans skipped.")
        if args.attract_pwm is not None:
            print("  FIMO requires --fasta; ATtRACT scan will be skipped.")

    targetscan_bed = Path(args.targetscan_bed) if args.targetscan_bed else None
    attract_pwm    = Path(args.attract_pwm)    if args.attract_pwm    else None

    # --- Build background ---
    background = _build_background(df, dominant, region_keep)
    print(f"\nBackground set: {len(background):,} positions (z_diff < 0, matched by region type)")

    # --- Scan foreground ---
    work_dir = Path(args.output_dir) / "tmp"
    print("\nScanning foreground ...")
    fg_hits = _scan_cohort(
        dominant, fa, "foreground",
        targetscan_bed, attract_pwm, args.fimo_threshold, work_dir,
    )

    # --- Scan background ---
    bg_hits = []
    if not background.empty:
        print("\nScanning background ...")
        bg_hits = _scan_cohort(
            background, fa, "background",
            targetscan_bed, attract_pwm, args.fimo_threshold, work_dir,
        )

    # --- Write foreground motif hits ---
    hits_df   = pd.DataFrame(fg_hits)
    hits_path = Path(args.output_dir) / "motif_hits.tsv"
    hits_df.to_csv(hits_path, sep="\t", index=False)
    if hits_df.empty:
        print("\n  No motif hits found in foreground.")
    else:
        print(f"\n  Foreground motif hits -> {hits_path}  ({len(hits_df):,} rows)")

    # --- Enrichment summary (foreground vs background, Fisher's exact) ---
    bg_hits_df = pd.DataFrame(bg_hits)
    n_fg = len(dominant)
    n_bg = len(background)

    all_motifs = set()
    if not hits_df.empty:
        all_motifs.update(hits_df["motif_name"].unique())
    if not bg_hits_df.empty:
        all_motifs.update(bg_hits_df["motif_name"].unique())
    # Always include polyA signals in summary
    all_motifs.update(POLYA_PATTERNS.keys())

    def _motif_type(name):
        if name in POLYA_PATTERNS:
            return "polyA_signal"
        return "miRNA_targetscan" if "miR" in name or "let-" in name else "RBP_fimo"

    def _n_positions_hit(hdf, motif):
        if hdf.empty or motif not in hdf["motif_name"].values:
            return 0
        return hdf[hdf["motif_name"] == motif]["start"].nunique()

    enrich_rows = []
    for motif in sorted(all_motifs):
        n_fg_hit = _n_positions_hit(hits_df,    motif)
        n_bg_hit = _n_positions_hit(bg_hits_df, motif)
        fold, pval = _fisher_enrichment(n_fg_hit, n_fg, n_bg_hit, n_bg)
        enrich_rows.append({
            "motif_name":          motif,
            "motif_type":          _motif_type(motif),
            "n_dominant_hits":     n_fg_hit,
            "n_dominant_total":    n_fg,
            "n_background_hits":   n_bg_hit,
            "n_background_total":  n_bg,
            "fold_enrichment":     fold,
            "pvalue_fisher":       round(pval, 6),
        })

    enrich_df = (
        pd.DataFrame(enrich_rows)
        .sort_values(["pvalue_fisher", "fold_enrichment"], ascending=[True, False])
        .reset_index(drop=True)
    )
    enrich_path = Path(args.output_dir) / "motif_enrichment.tsv"
    enrich_df.to_csv(enrich_path, sep="\t", index=False)

    print(f"\nMotif enrichment summary -> {enrich_path}")
    sig = enrich_df[enrich_df["n_dominant_hits"] > 0]
    print(sig.to_string(index=False) if not sig.empty else "  (no hits)")


if __name__ == "__main__":
    main()