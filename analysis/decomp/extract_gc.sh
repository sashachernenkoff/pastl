#!/usr/bin/env bash
#
# extract_gc.sh – Extract gene and 3' UTR coordinates from GENCODE GTF and
#                 compute per-interval GC content using bedtools nuc.
#
# 3' UTR intervals are identified by comparing each UTR position to the
# per-transcript CDS boundaries: downstream of CDS on the forward strand,
# or upstream on the reverse strand.
#
# Outputs:
#   genes_gc.bed  – one row per gene with bedtools nuc columns appended
#   utrs_gc.bed   – one row per 3' UTR interval with bedtools nuc columns appended
#
# Usage:
#   bash extract_gc.sh \
#       --gtf        /path/to/gencode.v36.annotation.gtf.gz \
#       --fasta      /path/to/GRCh38.d1.vd1.fa \
#       --output_dir out/analysis/decomp

set -euo pipefail

module load bedtools/2.31.0

GTF=""
FASTA=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gtf)        GTF="$2";        shift 2 ;;
        --fasta)      FASTA="$2";      shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GTF" || -z "$FASTA" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --gtf <gtf.gz> --fasta <genome.fa> --output_dir <dir>" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ─── Step 1: Decompress GTF ──────────────────────────────────────────────────
echo "[1/5] Decompressing GTF..."
gzip -cd "$GTF" > "$TMP/annotation.gtf"

# ─── Step 2: Extract gene intervals (BED 0-based half-open) ──────────────────
echo "[2/5] Extracting gene intervals..."
awk '
    $0 !~ /^#/ && $3 == "gene" {
        gene = ""
        for (i = 9; i <= NF; i++) {
            if ($i == "gene_name") {
                gene = $(i+1); gsub(/"/, "", gene); gsub(/;/, "", gene)
            }
        }
        if (gene != "")
            print $1 "\t" ($4 - 1) "\t" $5 "\t" gene "\t.\t" $7
    }
' "$TMP/annotation.gtf" | sort -k1,1 -k2,2n > "$TMP/genes.bed"
echo "  $(wc -l < "$TMP/genes.bed") gene records extracted"

# ─── Step 3: Build per-transcript CDS boundary table ─────────────────────────
echo "[3/5] Building CDS boundary table..."
awk '
    $0 !~ /^#/ && $3 == "CDS" {
        tx = ""
        for (i = 9; i <= NF; i++) {
            if ($i == "transcript_id") {
                tx = $(i+1); gsub(/"/, "", tx); gsub(/;/, "", tx)
            }
        }
        if (tx != "") {
            if (!(tx in cds_min) || $4 < cds_min[tx]) cds_min[tx] = $4
            if (!(tx in cds_max) || $5 > cds_max[tx]) cds_max[tx] = $5
            tx_strand[tx] = $7
        }
    }
    END {
        for (tx in cds_min)
            print tx "\t" tx_strand[tx] "\t" cds_min[tx] "\t" cds_max[tx]
    }
' "$TMP/annotation.gtf" > "$TMP/cds_bounds.tsv"
echo "  $(wc -l < "$TMP/cds_bounds.tsv") transcripts with CDS annotation"

# ─── Step 4: Extract 3' UTR intervals ────────────────────────────────────────
# A UTR interval is 3' if it lies downstream of CDS on '+' strand (start > CDS max)
# or upstream of CDS on '-' strand (end < CDS min).
echo "[4/5] Extracting 3' UTR intervals..."
awk -v cds_file="$TMP/cds_bounds.tsv" '
    BEGIN {
        while ((getline line < cds_file) > 0) {
            n = split(line, a, "\t")
            tx = a[1]
            cds_min[tx] = a[3] + 0
            cds_max[tx] = a[4] + 0
            tx_strand[tx] = a[2]
        }
    }
    $0 !~ /^#/ && $3 == "UTR" {
        tx = ""; gene = ""; strand = $7
        for (i = 9; i <= NF; i++) {
            if ($i == "transcript_id") { tx = $(i+1); gsub(/"/, "", tx); gsub(/;/, "", tx) }
            if ($i == "gene_name")     { gene = $(i+1); gsub(/"/, "", gene); gsub(/;/, "", gene) }
        }
        if (tx == "" || gene == "" || !(tx in cds_min)) next
        is_3utr = 0
        if (tx_strand[tx] == "+" && $4 > cds_max[tx]) is_3utr = 1
        if (tx_strand[tx] == "-" && $5 < cds_min[tx]) is_3utr = 1
        if (is_3utr)
            print $1 "\t" ($4 - 1) "\t" $5 "\t" gene "\t.\t" strand
    }
' "$TMP/annotation.gtf" | sort -k1,1 -k2,2n > "$TMP/utrs_3prime.bed"
echo "  $(wc -l < "$TMP/utrs_3prime.bed") 3' UTR intervals extracted"

# ─── Step 5: Run bedtools nuc to compute GC content ──────────────────────────
echo "[5/5] Running bedtools nuc..."
bedtools nuc -fi "$FASTA" -bed "$TMP/genes.bed"       > "$OUTPUT_DIR/genes_gc.bed"
bedtools nuc -fi "$FASTA" -bed "$TMP/utrs_3prime.bed" > "$OUTPUT_DIR/utrs_gc.bed"

echo "  Saved: $OUTPUT_DIR/genes_gc.bed"
echo "  Saved: $OUTPUT_DIR/utrs_gc.bed"
echo "Done."
