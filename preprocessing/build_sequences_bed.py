#!/usr/bin/env python3
r"""
Build sequences_protein.bed for PASTL training.

Steps:
  1. Load CPTAC GCT (discovery ∩ confirmatory) → protein abundance matrix.
  2. Apply 50% missingness threshold.
  3. Identify dominant lung isoforms via GTEx transcript median TPM.
  4. Parse GENCODE v36 GTF → TSS and exon coordinates for each dominant isoform.
  5. Refine TSS with FANTOM5 CAGE peaks (≤ 500 bp).
  6. Filter: gene body (first exonStart → last exonEnd) must overlap the
     Enformer prediction window (TSS ± 57,344 bp = centre 114,688 bp).
  7. Load Enformer intervals → assign split (train/valid/test); drop any
     protein whose prediction window overlaps windows from two different splits.
  8. Precompute targets: left-censored imputation using train-set statistics,
     per-protein mean, z-score with train mean/std.
  9. Write sequences_protein.bed.

Output columns:
    chrom  start  end  split  strand  symbol  ensembl_gene_id  txid
    exonstarts  exonends  target_value

Usage:
    python preprocessing/build_sequences_bed.py \
        --config  model/configs/config.yaml \
        --gtf     /path/to/gencode.v36.annotation.gtf.gz \
        --gtex    /path/to/GTEx_Analysis_v8_Transcript_Median_TPM.txt.gz \
        --cage    /path/to/hg38_CAGE_peaks.bed.gz

    python build_sequences_bed.py \
        --gct_discovery    /path/to/CPTAC_LUAD_discovery_proteome.gct \
        --gct_confirmatory /path/to/CPTAC_LUAD_confirmatory_proteome.gct \
        --gtf     /path/to/gencode.v36.annotation.gtf.gz \
        --gtex    /path/to/GTEx_Analysis_v8_Transcript_Median_TPM.txt.gz \
        --cage    /path/to/hg38_CAGE_peaks.bed.gz \
        --intervals /path/to/enformer/human_intervals.tsv \
        --out     data/sequences_protein.bed
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Enformer / PASTL geometry constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 196_608
HALF_WINDOW = WINDOW_SIZE // 2   # 98,304

PRED_SIZE = 896 * 128            # 114,688 bp
PRED_HALF = PRED_SIZE // 2       # 57,344

CAGE_RADIUS = 500                # bp around annotated TSS to accept a CAGE peak


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Step 3: dominant lung isoform selection via GTEx
# ---------------------------------------------------------------------------

def get_dominant_isoforms(gtex_path: str, gencode_path: str, target_genes: set[str]) -> dict[str, str]:
    """
    For each gene in target_genes, return the transcript ID of the protein-coding
    isoform with the highest median Lung TPM in GTEx v8.

    Returns {gene_symbol: transcript_id_base (no version suffix)}.
    """
    print(f"[{_ts()}]   Scanning GENCODE for protein-coding transcripts ...")
    protein_coding_txs: set[str] = set()
    with gzip.open(gencode_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            if "protein_coding" not in fields[8]:
                continue
            attr: dict[str, str] = {}
            for item in fields[8].strip().split(";"):
                parts = item.strip().split(" ", 1)
                if len(parts) == 2:
                    attr[parts[0]] = parts[1].strip('"')
            tx_id = attr.get("transcript_id", "").split(".")[0]
            if tx_id:
                protein_coding_txs.add(tx_id)

    print(f"[{_ts()}]   {len(protein_coding_txs):,} protein-coding transcripts found")
    print(f"[{_ts()}]   Loading GTEx transcript median TPM ...")
    df = pd.read_csv(gtex_path, sep="\t")
    df["Clean_Name"] = df["Name"].apply(lambda x: x.split(".")[0])
    df = df[df["Description"].isin(target_genes) & df["Clean_Name"].isin(protein_coding_txs)]

    if df.empty:
        print(f"[{_ts()}]   Warning: no transcripts survived the protein-coding filter")
        return {}

    idx = df.groupby("Description")["Lung"].idxmax()
    dominant = df.loc[idx, ["Description", "Clean_Name"]].set_index("Description")["Clean_Name"].to_dict()
    print(f"[{_ts()}]   {len(dominant):,} dominant lung isoforms identified")
    return dominant


# ---------------------------------------------------------------------------
# Step 5 (helper): FANTOM5 CAGE peaks
# ---------------------------------------------------------------------------

def get_cage_peaks(cage_path: str) -> dict[str, list[int]]:
    """Parse FANTOM5 CAGE peaks BED → {chrom: [peak_centre, ...]}."""
    print(f"[{_ts()}] Step 5: loading FANTOM5 CAGE peaks ...")
    peaks: dict[str, list[int]] = defaultdict(list)
    with gzip.open(cage_path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            try:
                chrom  = fields[0]
                centre = (int(fields[1]) + int(fields[2])) // 2
                peaks[chrom].append(centre)
            except ValueError:
                continue
    print(f"[{_ts()}] Step 5 done: CAGE peaks on {len(peaks):,} chromosomes")
    return dict(peaks)


# ---------------------------------------------------------------------------
# Step 1–2: GCT loading + missingness filter
# ---------------------------------------------------------------------------

def load_gct_matrix(paths: list[str],
                    sample_type: str = "Solid Tissue Normal") -> pd.DataFrame:
    """
    Merge CPTAC GCT files (inner join on proteins), keeping only samples whose
    sample_type matches the given value.

    sample_type options from CPTAC GCT metadata:
      "Solid Tissue Normal"  – adjacent normal (default; used for PASTL training)
      "Primary Tumor"        – tumour samples (use for PWAS imputation models)

    Returns protein × patient log₂-abundance DataFrame.
    """
    print(f"[{_ts()}] Step 1: loading GCT files (sample_type='{sample_type}') ...")
    combined = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", skiprows=2, low_memory=False)

        type_mask  = df[df["id"] == "sample_type"].iloc[0, 4:]
        keep       = type_mask == sample_type

        id_row    = df[df["id"] == "diagnosis_submitter_id"].iloc[0, 4:]
        clean_ids = id_row[keep].str.replace("-DX", "", regex=False).values

        data_start = df[df["id"] == "A1BG"].index[0]
        data_df    = df.iloc[data_start:].set_index("id").iloc[:, 3:]
        data_df    = data_df.loc[:, keep.values]
        data_df.columns = clean_ids
        data_df    = data_df.apply(pd.to_numeric, errors="coerce")
        combined.append(data_df)

    # Inner join on proteins (axis=0 index) → only proteins in both files
    merged = pd.concat(combined, axis=1, join="inner")
    merged = merged.T.groupby(level=0).mean().T   # average patient replicates

    print(f"[{_ts()}] Step 1 done: {merged.shape[0]:,} proteins × {merged.shape[1]:,} patients")

    threshold = 0.50
    min_valid = int(merged.shape[1] * threshold)
    n_before  = merged.shape[0]
    merged    = merged.dropna(thresh=min_valid)
    print(f"[{_ts()}] Step 2: missingness filter ({threshold*100:.0f}%): "
          f"{n_before:,} → {merged.shape[0]:,} proteins retained")
    return merged


# ---------------------------------------------------------------------------
# Step 4: GTF parsing (TSS + exon coordinates for dominant isoforms)
# ---------------------------------------------------------------------------

def parse_gtf(
    gtf_path: str,
    dominant_isoforms: dict[str, str],
    cage_peaks: dict[str, list[int]],
) -> dict[str, dict]:
    """
    Parse GENCODE GTF for the dominant isoform of each target gene.

    Returns {symbol: {Chrom, Strand, TSS, EnsemblGeneID, TxID,
                       ExonStarts, ExonEnds}}

    TSS is CAGE-refined where a FANTOM5 peak falls within CAGE_RADIUS bp.
    """
    print(f"[{_ts()}] Step 4: parsing GENCODE GTF for TSS + exon coordinates ...")

    gene_anchors: dict[str, dict] = {}
    gene_exon_starts: dict[str, list[int]] = defaultdict(list)
    gene_exon_ends:   dict[str, list[int]] = defaultdict(list)

    target_symbols = set(dominant_isoforms.keys())

    opener = gzip.open if str(gtf_path).endswith(".gz") else open

    # Single pass: collect transcript anchors and exon coordinates together
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            feat_type = fields[2]
            if feat_type not in ("transcript", "exon"):
                continue

            attr_str = fields[8]
            attr: dict[str, str] = {}
            for item in attr_str.strip().split(";"):
                item = item.strip()
                if not item:
                    continue
                parts = item.split(" ", 1)
                if len(parts) == 2:
                    attr[parts[0]] = parts[1].strip('"')

            symbol = attr.get("gene_name")
            if not symbol or symbol not in target_symbols:
                continue

            tx_id_full = attr.get("transcript_id", "")
            tx_id      = tx_id_full.split(".")[0]

            if tx_id != dominant_isoforms.get(symbol):
                continue

            chrom_raw = fields[0]
            chrom = f"chr{chrom_raw}" if not chrom_raw.startswith("chr") else chrom_raw
            if chrom == "chrMT":
                chrom = "chrM"
            if "_" in chrom or "v" in chrom.lower():
                continue  # skip scaffolds / patches

            start  = int(fields[3])
            end    = int(fields[4])
            strand = fields[6]

            if feat_type == "transcript":
                raw_tss = start if strand == "+" else end

                # CAGE refinement
                refined_tss = raw_tss
                if chrom in cage_peaks:
                    closest = min(cage_peaks[chrom], key=lambda p: abs(p - raw_tss))
                    if abs(closest - raw_tss) <= CAGE_RADIUS:
                        refined_tss = closest

                gene_anchors[symbol] = {
                    "Chrom":         chrom,
                    "Strand":        strand,
                    "TSS":           refined_tss,
                    "EnsemblGeneID": attr.get("gene_id", "").split(".")[0],
                    "TxID":          tx_id,
                }

            else:  # exon
                gene_exon_starts[symbol].append(start)
                gene_exon_ends[symbol].append(end)

    # Consolidate exon coordinates
    result: dict[str, dict] = {}
    for symbol, info in gene_anchors.items():
        starts = sorted(gene_exon_starts.get(symbol, []))
        ends   = sorted(gene_exon_ends.get(symbol, []))
        if not starts:
            continue
        result[symbol] = {
            **info,
            "ExonStarts": ",".join(map(str, starts)),
            "ExonEnds":   ",".join(map(str, ends)),
        }

    print(f"[{_ts()}] Step 4 done: {len(result):,} proteins with GTF anchors "
          f"({len(dominant_isoforms) - len(result):,} dominant isoforms not found in GTF)")
    return result


# ---------------------------------------------------------------------------
# Step 6: gene-body / prediction-window overlap filter
# ---------------------------------------------------------------------------

def overlaps_prediction_window(tss: int, exon_starts_str: str, exon_ends_str: str) -> bool:
    """True if the gene body overlaps the Enformer prediction window (TSS ± PRED_HALF)."""
    try:
        starts = [int(x) for x in exon_starts_str.split(",") if x]
        ends   = [int(x) for x in exon_ends_str.split(",")   if x]
    except ValueError:
        return False
    if not starts or not ends:
        return False
    gene_start = min(starts)
    gene_end   = max(ends)
    pred_start = tss - PRED_HALF
    pred_end   = tss + PRED_HALF
    return gene_end > pred_start and gene_start < pred_end


# ---------------------------------------------------------------------------
# Step 7: Enformer interval index + split assignment
# ---------------------------------------------------------------------------

EnformerIndex = dict[str, tuple[list[tuple[int, int, str]], list[int]]]


def load_enformer_index(intervals_path: str) -> EnformerIndex:
    print(f"[{_ts()}] Step 7: building Enformer interval index ...")
    raw: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    df = pd.read_csv(intervals_path, sep="\t")
    for _, row in df.iterrows():
        raw[str(row["chrom"])].append((int(row["start"]), int(row["end"]), str(row["split"])))
    index: EnformerIndex = {}
    for chrom, ivals in raw.items():
        ivals.sort(key=lambda x: x[0])
        index[chrom] = (ivals, [iv[0] for iv in ivals])
    n = sum(len(v[0]) for v in index.values())
    print(f"[{_ts()}] Step 7 done: {n:,} Enformer intervals across {len(index):,} chromosomes")
    return index


def find_tss_split(chrom: str, tss: int, index: EnformerIndex) -> str | None:
    """Split of the nearest-centre Enformer window containing the TSS, or None."""
    entry = index.get(chrom)
    if entry is None:
        return None
    intervals, starts = entry
    i = bisect.bisect_right(starts, tss) - 1
    candidates: list[tuple[int, str]] = []
    while i >= 0:
        start, end, split = intervals[i]
        if start < tss - WINDOW_SIZE:
            break
        if start <= tss < end:
            candidates.append((abs((start + end) // 2 - tss), split))
        i -= 1
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def find_overlapping_splits(
    chrom: str, pred_start: int, pred_end: int, index: EnformerIndex
) -> set[str]:
    """Set of distinct splits for all Enformer windows overlapping [pred_start, pred_end)."""
    entry = index.get(chrom)
    if entry is None:
        return set()
    intervals, starts = entry
    splits: set[str] = set()
    i = bisect.bisect_left(starts, pred_end) - 1
    while i >= 0:
        start, end, split = intervals[i]
        if start < pred_start - WINDOW_SIZE:
            break
        if start < pred_end and end > pred_start:
            splits.add(split)
        i -= 1
    return splits


# ---------------------------------------------------------------------------
# Step 8: target precomputation
# ---------------------------------------------------------------------------

def compute_targets(prot_df: pd.DataFrame, train_proteins: list[str]) -> dict[str, float]:
    """
    Left-censored imputation (train stats only) → per-protein mean → z-score.
    Returns {symbol: z_scored_mean}.
    """
    print(f"[{_ts()}] Step 8: computing targets ...")
    train_df  = prot_df.loc[train_proteins]
    floor_val = np.floor(train_df.min().min()) - 1.0
    imputed   = prot_df.copy()

    np.random.seed(42)
    for col in imputed.columns:
        mask = imputed[col].isnull()
        if not mask.any():
            continue
        s_mean = train_df[col].mean()
        s_std  = train_df[col].std()
        if pd.isna(s_std) or s_std == 0:
            imputed.loc[mask, col] = floor_val
        else:
            imputed.loc[mask, col] = np.random.normal(
                s_mean - 1.8 * s_std, 0.3 * s_std, size=mask.sum()
            )

    prot_means  = imputed.mean(axis=1)
    train_mean  = float(prot_means.loc[train_proteins].mean())
    train_std   = float(prot_means.loc[train_proteins].std())
    z_scores    = (prot_means - train_mean) / train_std

    print(f"[{_ts()}] Step 8 done: train mean={train_mean:.4f}  std={train_std:.4f}")
    return z_scores.to_dict()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build sequences_protein.bed for PASTL")

    parser.add_argument("--config",           type=str, default=None)
    parser.add_argument("--gct_discovery",    type=str, default=None)
    parser.add_argument("--gct_confirmatory", type=str, default=None)
    parser.add_argument("--gtf",    required=True,
                        help="GENCODE v36 annotation GTF (may be .gz)")
    parser.add_argument("--gtex",   required=True,
                        help="GTEx_Analysis_v8_Transcript_Median_TPM.txt.gz")
    parser.add_argument("--cage",   required=True,
                        help="FANTOM5 CAGE peaks BED (.gz)")
    parser.add_argument("--intervals", type=str, default=None,
                        help="Enformer human_intervals.tsv")
    parser.add_argument("--out",       type=str, default=None,
                        help="Output path (default: data/sequences_protein.bed)")
    args = parser.parse_args()

    gct_discovery    = args.gct_discovery
    gct_confirmatory = args.gct_confirmatory
    intervals_path   = args.intervals
    out_path         = args.out

    if args.config:
        sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
        from config import load_config  # type: ignore
        cfg = load_config(args.config)
        gct_discovery    = gct_discovery    or cfg.gct_discovery
        gct_confirmatory = gct_confirmatory or cfg.gct_confirmatory
        intervals_path   = intervals_path   or cfg.sequences_file
        out_path         = out_path or str(Path(cfg.project_dir) / "data" / "sequences_protein.bed")

    for name, val in [("--gct_discovery", gct_discovery),
                      ("--gct_confirmatory", gct_confirmatory),
                      ("--intervals", intervals_path)]:
        if not val:
            parser.error(f"{name} is required (or provide --config)")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Steps 1–2: GCT loading + missingness filter
    # ------------------------------------------------------------------
    prot_df = load_gct_matrix([gct_discovery, gct_confirmatory])
    proteins_in_gct = set(prot_df.index)

    # ------------------------------------------------------------------
    # Step 3: dominant lung isoforms via GTEx
    # ------------------------------------------------------------------
    print(f"[{_ts()}] Step 3: identifying dominant lung isoforms via GTEx ...")
    dominant_isoforms = get_dominant_isoforms(args.gtex, args.gtf, proteins_in_gct)
    print(f"[{_ts()}] Step 3 done: {len(dominant_isoforms):,} dominant isoforms identified")

    # ------------------------------------------------------------------
    # Step 4/5: GTF parsing + CAGE refinement
    # ------------------------------------------------------------------
    print(f"[{_ts()}] Step 5: loading FANTOM5 CAGE peaks ...")
    cage_peaks = get_cage_peaks(args.cage)
    gene_info  = parse_gtf(args.gtf, dominant_isoforms, cage_peaks)

    # ------------------------------------------------------------------
    # Step 6: gene-body / prediction-window overlap filter
    # ------------------------------------------------------------------
    print(f"[{_ts()}] Step 6: filtering by prediction-window overlap ...")
    n_before = len(gene_info)
    gene_info = {
        sym: info for sym, info in gene_info.items()
        if overlaps_prediction_window(info["TSS"], info["ExonStarts"], info["ExonEnds"])
    }
    print(f"[{_ts()}] Step 6 done: {n_before - len(gene_info):,} dropped "
          f"(gene body outside prediction window), {len(gene_info):,} retained")

    # Restrict abundance matrix to proteins with genomic anchors
    retained_prots = sorted(set(prot_df.index) & set(gene_info.keys()))
    prot_df = prot_df.loc[retained_prots]
    print(f"[{_ts()}] GCT ∩ GTF intersection: {len(retained_prots):,} proteins")

    # ------------------------------------------------------------------
    # Step 7: Enformer split assignment + contamination filter
    # ------------------------------------------------------------------
    enformer = load_enformer_index(intervals_path)

    rows: list[dict] = []
    n_no_interval = 0
    n_split_leak  = 0

    for sym in retained_prots:
        info   = gene_info[sym]
        chrom  = info["Chrom"]
        tss    = info["TSS"]
        strand = info["Strand"]

        split = find_tss_split(chrom, tss, enformer)
        if split is None:
            n_no_interval += 1
            continue

        pred_start = tss - PRED_HALF
        pred_end   = tss + PRED_HALF
        overlapping = find_overlapping_splits(chrom, pred_start, pred_end, enformer)
        if len(overlapping) > 1:
            n_split_leak += 1
            continue

        rows.append({
            "chrom":           chrom,
            "start":           tss - HALF_WINDOW,
            "end":             tss + HALF_WINDOW,
            "split":           split,
            "strand":          strand,
            "symbol":          sym,
            "ensembl_gene_id": info["EnsemblGeneID"],
            "txid":            info["TxID"],
            "exonstarts":      info["ExonStarts"],
            "exonends":        info["ExonEnds"],
        })

    print(f"[{_ts()}] Step 7: {len(rows):,} proteins retained  "
          f"({n_no_interval} no covering interval, {n_split_leak} split-contaminated)")

    if not rows:
        raise SystemExit(f"[{_ts()}] ERROR: no proteins survived filtering")

    # ------------------------------------------------------------------
    # Step 8: target precomputation
    # ------------------------------------------------------------------
    final_syms   = [r["symbol"] for r in rows]
    train_prots  = [r["symbol"] for r in rows if r["split"] == "train"]
    targets      = compute_targets(prot_df.loc[final_syms], train_prots)

    # ------------------------------------------------------------------
    # Step 9: write BED
    # ------------------------------------------------------------------
    print(f"[{_ts()}] Step 9: writing {out_path} ...")
    split_counts: dict[str, int] = defaultdict(int)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("chrom\tstart\tend\tsplit\tstrand\tsymbol\t"
                 "ensembl_gene_id\ttxid\texonstarts\texonends\ttarget_value\n")
        for r in sorted(rows, key=lambda x: (x["chrom"], x["start"])):
            sym = r["symbol"]
            split_counts[r["split"]] += 1
            fh.write(
                f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['split']}\t"
                f"{r['strand']}\t{sym}\t{r['ensembl_gene_id']}\t{r['txid']}\t"
                f"{r['exonstarts']}\t{r['exonends']}\t{targets[sym]:.6f}\n"
            )

    elapsed = time.time() - t0
    print(f"[{_ts()}] Done in {elapsed/60:.1f} min — "
          f"{len(rows):,} proteins written to {out_path}")
    for sp in ("train", "valid", "test"):
        print(f"  {sp}: {split_counts[sp]:,}")


if __name__ == "__main__":
    main()
