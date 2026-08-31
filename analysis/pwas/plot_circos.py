r"""
Plot genome-wide PWAS signal against published LUAD GWAS loci as a circos plot.

Draws, for each of chr1-22/X: an outer ring marking McKay et al. (2017) LUAD/lung
GWAS loci confirmed genome-wide significant (p<5e-8) in a given GWAS VCF, then
three concentric ring-pairs (1M/500K/250K cis-windows, outer to inner), each pair
showing PASTL and Enformer PWAS -log10(p) as a 2Mb-binned heatmap with
significant hits marked (diamond=PASTL, circle=Enformer). Genes significant in
at least one of the 6 (model x window) conditions are labelled outside the ring
with a leader line back to their position; genes that are both a McKay GWAS lead
gene and a recurring PWAS hit get a single combined "Gene*" label.

McKay loci are cross-checked against the GWAS VCF because the published table
(mckay2017_table2.csv) has no chrom/pos columns and no p-value for this specific
cohort's VCF -- coordinates and confirmation both come from matching each locus's
lead rsid against the VCF's FORMAT field (ES:SE:LP:AF:ID, LP=-log10(p)).

Inputs:
  {results_dir}/{model}_{window}/pwas_results.tsv   per-gene PWAS results (pastl/enformer x 250K/500K/1M)
  {bed}                                             sequences_protein.bed, for gene_symbol -> chrom/pos
  {mckay_csv}                                       mckay2017_table2.csv, McKay et al. 2017 Table 2 (Strata,
                                                    Locus, rs number, Gene, ...)
  {gwas_vcf}                                        GWAS summary-stats VCF (e.g. ebi-a-GCST004744.vcf.gz)

Output:
  {output_file}   circos.png (300 DPI)

Usage:
  python plot_circos.py \
    --results_dir out/analysis/s5fo3aak_best/pwas/results \
    --bed data/sequences_protein.bed \
    --mckay_csv /path/to/data/gwas/mckay2017_table2.csv \
    --gwas_vcf /path/to/data/gwas/ebi-a-GCST004744.vcf.gz \
    --output_file out/analysis/s5fo3aak_best/pwas/results/figures/pwas_circos.png
"""

import argparse
import gzip
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pycirclize import Circos

# Import project palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.gwas.gwas_config import PASTL_FILL, PASTL_INK, ENFORMER_FILL, ENFORMER_INK

PLUM = "#6D597A"
GAIN = "#EDF6F9"
MCKAY_RED = "#D62828"

HG38_CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895
}

# Gradient colormaps: dark saturated anchor (compressed to ~first 0.2 of range,
# background/near-zero bins) -> peach/gold at the high end (significant signal pops).
# Assignment: 250K=pink, 500K=purple, 1M=blue.
def _make_cmap(name, stops):
    return LinearSegmentedColormap.from_list(name, stops)

WINDOW_CMAPS = {
    "250K": _make_cmap("cmap_250K_pink",   [(0.00, "#6B1240"), (0.10, "#B02E6A"), (0.20, "#E0629C"), (0.55, "#ECA472"), (1.00, "#F8D76E")]),
    "500K": _make_cmap("cmap_500K_purple", [(0.00, "#3D2568"), (0.10, "#7A3B8C"), (0.20, "#CD6C9C"), (0.55, "#ECA472"), (1.00, "#F8D76E")]),
    "1M":   _make_cmap("cmap_1M_blue",     [(0.00, "#0A1152"), (0.10, "#2A3E8C"), (0.20, "#5D6FA6"), (0.55, "#ECA472"), (1.00, "#F8D76E")]),
}

# Ring order outer -> inner
WINDOWS_OUTER_TO_INNER = ["1M", "500K", "250K"]

# GWAS-locus mark width exaggerated for visibility at genome scale (real loci are
# far narrower than this at whole-chromosome zoom; disclosed in the figure caption)
GWAS_MARK_WIDTH_BP = 2_500_000

# Label genes significant in at least this many of the 6 (model x window) conditions
MIN_CONDITIONS_FOR_LABEL = 1


def load_gwas_vcf_significant(vcf_path: Path, pval_threshold: float = 5e-8) -> pd.DataFrame:
    """Parse the GWAS summary-stats VCF and return genome-wide significant variants
    as a DataFrame with columns: rsid, chrom, pos, neg_log10p.
    Expects FORMAT field ES:SE:LP:AF:ID (LP = -log10(p))."""
    lp_threshold = -np.log10(pval_threshold)
    rows = []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos, rsid = fields[0], fields[1], fields[2]
            sample = fields[9]
            lp = float(sample.split(":")[2])
            if lp > lp_threshold:
                rows.append((rsid, f"chr{chrom}", int(pos), lp))
    return pd.DataFrame(rows, columns=["rsid", "chrom", "pos", "neg_log10p"])


def load_mckay_confirmed_loci(mckay_csv: Path, vcf_path: Path) -> pd.DataFrame:
    """Load the McKay et al. 2017 Table 2 locus list and keep only the loci whose
    lead rsid is genome-wide significant (p<5e-8) in the given GWAS VCF. Coordinates
    for confirmed loci are taken from the VCF (the CSV has no chrom/pos columns)."""
    mckay = pd.read_csv(mckay_csv)
    mckay = mckay.rename(columns={"rs number": "rsid", "Gene": "gene", "Locus": "cytoband"})

    sig_variants = load_gwas_vcf_significant(vcf_path)

    confirmed = mckay.merge(sig_variants, on="rsid", how="inner")
    confirmed = confirmed[["cytoband", "rsid", "gene", "Strata", "chrom", "pos", "neg_log10p"]]
    confirmed = confirmed[confirmed["chrom"].isin(HG38_CHROM_SIZES.keys())]
    return confirmed.sort_values(["chrom", "pos"]).reset_index(drop=True)


def load_pwas_results(results_dir: Path, coord_map: dict) -> dict:
    """Load all 6 (model, window) PWAS result files and annotate with genomic coordinates."""
    files = {
        ("pastl", "250K"): "pastl_250K/pwas_results.tsv",
        ("pastl", "500K"): "pastl_500K/pwas_results.tsv",
        ("pastl", "1M"):   "pastl_1M/pwas_results.tsv",
        ("enformer", "250K"): "enformer_250K/pwas_results.tsv",
        ("enformer", "500K"): "enformer_500K/pwas_results.tsv",
        ("enformer", "1M"):   "enformer_1M/pwas_results.tsv",
    }
    data = {}
    for (model, window), relpath in files.items():
        df = pd.read_csv(results_dir / relpath, sep="\t").dropna(subset=["p_value"])
        df["chrom"] = df["gene_symbol"].map(lambda g: coord_map.get(g, {}).get("chrom"))
        df["pos"] = df["gene_symbol"].map(lambda g: coord_map.get(g, {}).get("center"))
        df = df.dropna(subset=["chrom", "pos"])
        df["-log10p"] = -np.log10(df["p_value"].clip(lower=1e-300))
        data[(model, window)] = df
    return data


def get_recurring_hit_genes(data: dict, min_conditions: int) -> set:
    sig_frames = [df[df["significant"] == True]["gene_symbol"] for df in data.values()]
    counts = pd.concat(sig_frames).value_counts()
    return set(counts[counts >= min_conditions].index)


def main():
    parser = argparse.ArgumentParser(description="Generate a Circos plot of PWAS hits vs McKay et al. GWAS loci")
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to pwas/results/")
    parser.add_argument("--bed", type=Path, required=True, help="Path to sequences_protein.bed")
    parser.add_argument("--mckay_csv", type=Path, required=True, help="Path to mckay2017_table2.csv")
    parser.add_argument("--gwas_vcf", type=Path, required=True, help="Path to GWAS summary-stats VCF (e.g. ebi-a-GCST004744.vcf.gz)")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save circos.png")
    args = parser.parse_args()

    # 1. Load gene coordinates
    print("Loading gene coordinates...")
    bed = pd.read_csv(args.bed, sep="\t")
    bed["start"] = pd.to_numeric(bed["start"], errors="coerce")
    bed["end"] = pd.to_numeric(bed["end"], errors="coerce")
    bed = bed.dropna(subset=["start", "end", "chrom"])
    bed["center"] = (bed["start"] + bed["end"]) / 2
    coord_map = bed.set_index("symbol")[["chrom", "center"]].to_dict(orient="index")

    # 2. Load PWAS results for all 6 (model, window) conditions
    print("Loading PWAS results...")
    data = load_pwas_results(args.results_dir, coord_map)
    recurring_hits = get_recurring_hit_genes(data, MIN_CONDITIONS_FOR_LABEL)

    # Build a single gene->coordinate lookup spanning all 6 conditions, so a gene's
    # label position doesn't depend on which specific (model, window) it happened to
    # be significant in (e.g. a gene significant only in Enformer must still be
    # labelled using coordinates from any dataframe it appears in).
    gene_positions = (
        pd.concat([df[["gene_symbol", "chrom", "pos"]] for df in data.values()])
        .drop_duplicates(subset="gene_symbol")
        .reset_index(drop=True)
    )

    # 3. Load McKay et al. 2017 GWAS loci, keeping only those confirmed genome-wide
    #    significant (p<5e-8) in the provided GWAS VCF; coordinates come from the VCF.
    print("Loading and cross-checking McKay et al. 2017 GWAS loci against VCF...")
    mckay = load_mckay_confirmed_loci(args.mckay_csv, args.gwas_vcf)
    n_total_loci = len(pd.read_csv(args.mckay_csv))
    print(f"  {len(mckay)}/{n_total_loci} McKay loci confirmed genome-wide significant in VCF")
    mckay["plot_start"] = mckay["pos"] - GWAS_MARK_WIDTH_BP / 2
    mckay["plot_end"] = mckay["pos"] + GWAS_MARK_WIDTH_BP / 2

    # 4. Compute radius bands: GWAS ring, then 1M/500K/250K pairs (outer->inner), no gap ring at centre
    band_height = 9.0
    gap = 1.3
    r_gwas_top, r_gwas_bot = 96, 92.5
    r_cursor = r_gwas_bot - gap

    band_positions = {}
    for w in WINDOWS_OUTER_TO_INNER:
        tl_top = r_cursor
        tl_bot = tl_top - band_height
        enf_top = tl_bot - 0.8
        enf_bot = enf_top - band_height
        band_positions[w] = ((tl_bot, tl_top), (enf_bot, enf_top))
        r_cursor = enf_bot - gap

    # 5. Build the circos plot
    print("Drawing Circos plot...")
    circos = Circos(sectors=HG38_CHROM_SIZES, space=2)

    for sector in circos.sectors:
        c_name = sector.name
        chrom_size = HG38_CHROM_SIZES[c_name]
        bins = np.arange(0, chrom_size + 2e6, 2e6)

        # Chromosome ideogram ring (also used as the anchor track for outer labels)
        track_chr = sector.add_track((97, 100), name="chrom_ideogram")
        track_chr.axis(fc=GAIN, ec="black", lw=0.5)
        track_chr.text(c_name.replace("chr", ""), color="black", size=9, weight="bold")

        # McKay GWAS locus ring
        track_gwas = sector.add_track((r_gwas_bot, r_gwas_top))
        track_gwas.axis(fc="white", ec="black", lw=0.3)
        g_sub = mckay[mckay["chrom"] == c_name]
        for _, r in g_sub.iterrows():
            track_gwas.rect(r["plot_start"], r["plot_end"], fc=MCKAY_RED, ec="none", alpha=1.0, zorder=6)

        # PWAS heatmap + significant-hit marker rings, one pair per window
        for w in WINDOWS_OUTER_TO_INNER:
            (tl_r, enf_r) = band_positions[w]
            cmap = WINDOW_CMAPS[w]
            for model, (r_bot, r_top) in [("pastl", tl_r), ("enformer", enf_r)]:
                df = data[(model, w)]
                sub = df[df["chrom"] == c_name].copy()
                track = sector.add_track((r_bot, r_top))
                track.axis(fc="black", ec="black", lw=0.3)
                if not sub.empty:
                    sub["bin"] = pd.cut(sub["pos"], bins=bins, labels=False)
                    agg = sub.groupby("bin")["-log10p"].max().reindex(range(len(bins) - 1), fill_value=0)
                    v = np.clip(agg.values, 0, 6).reshape(1, -1)
                    track.heatmap(v, cmap=cmap, vmin=0, vmax=6)

                sig_sub = sub[sub["significant"] == True]
                if not sig_sub.empty:
                    # PASTL = diamond, Enformer = circle; white fill, black edge
                    marker = "D" if model == "pastl" else "o"
                    track.scatter(sig_sub["pos"].to_numpy(),
                                   np.clip(sig_sub["-log10p"].to_numpy(), 0, 6),
                                   vmin=0, vmax=6, color="white", edgecolor="black",
                                   linewidth=1.0, s=50, zorder=5, marker=marker)

    # 6. Outer annotation labels with leader lines (McKay loci in red, PWAS hits in
    #    black; genes that are both get one combined "Gene*" label at the GWAS position)
    for sector in circos.sectors:
        c_name = sector.name
        track_chr = sector.get_track("chrom_ideogram")

        g_sub = mckay[mckay["chrom"] == c_name].copy()
        ref_sub = gene_positions[(gene_positions["chrom"] == c_name) &
                                  (gene_positions["gene_symbol"].isin(recurring_hits))].copy()

        gwas_genes_here = set(g_sub["gene"])
        hit_genes_here = set(ref_sub["gene_symbol"])
        overlap_genes = gwas_genes_here & hit_genes_here

        for _, r in g_sub[~g_sub["gene"].isin(overlap_genes)].iterrows():
            track_chr.annotate(r["pos"], r["gene"], min_r=101, max_r=112, label_size=8.5,
                                line_kws=dict(color=MCKAY_RED, lw=0.8),
                                text_kws=dict(color=MCKAY_RED, weight="bold", style="italic"))

        for _, r in ref_sub[~ref_sub["gene_symbol"].isin(overlap_genes)].iterrows():
            track_chr.annotate(r["pos"], r["gene_symbol"], min_r=101, max_r=118, label_size=9.5,
                                line_kws=dict(color="black", lw=0.8),
                                text_kws=dict(color="black", weight="bold"))

        for gene in overlap_genes:
            pos = g_sub[g_sub["gene"] == gene].iloc[0]["pos"]
            track_chr.annotate(pos, f"{gene}*", min_r=101, max_r=118, label_size=9.5,
                                line_kws=dict(color="black", lw=0.9),
                                text_kws=dict(color="black", weight="bold"))

    # 7. Render figure with colour bars, legend, title, caption (all positioned clear of the ring)
    fig = circos.plotfig(figsize=(13, 13))

    sm_1m = plt.cm.ScalarMappable(cmap=WINDOW_CMAPS["1M"], norm=plt.Normalize(vmin=0, vmax=6))
    sm_500 = plt.cm.ScalarMappable(cmap=WINDOW_CMAPS["500K"], norm=plt.Normalize(vmin=0, vmax=6))
    sm_250 = plt.cm.ScalarMappable(cmap=WINDOW_CMAPS["250K"], norm=plt.Normalize(vmin=0, vmax=6))
    for sm in (sm_1m, sm_500, sm_250):
        sm.set_array([])

    cax1 = fig.add_axes([1.10, 0.60, 0.014, 0.12])
    cb1 = fig.colorbar(sm_1m, cax=cax1); cb1.set_label("1M", fontsize=9); cb1.ax.tick_params(labelsize=8)
    cax2 = fig.add_axes([1.10, 0.44, 0.014, 0.12])
    cb2 = fig.colorbar(sm_500, cax=cax2); cb2.set_label("500K", fontsize=9); cb2.ax.tick_params(labelsize=8)
    cax3 = fig.add_axes([1.10, 0.28, 0.014, 0.12])
    cb3 = fig.colorbar(sm_250, cax=cax3); cb3.set_label("250K", fontsize=9); cb3.ax.tick_params(labelsize=8)
    fig.text(1.12, 0.745, "max −log10(p)\nper 2Mb bin", fontsize=8.5, ha="left")

    legend_elems = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor="white", markeredgecolor='black',
               markersize=8, label='PASTL (significant)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="white", markeredgecolor='black',
               markersize=8, label='Enformer (significant)'),
        Patch(facecolor=MCKAY_RED, edgecolor='none', label='McKay et al. 2017 LUAD/lung GWAS locus'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False, fontsize=11)

    fig.suptitle("Genome-wide PWAS signal vs published LUAD GWAS loci: PASTL vs Enformer",
                  fontsize=18, fontweight="normal", y=1.05)
    fig.text(0.5, -0.11,
             f"Rings (outer→inner): McKay et al. (2017, Nat Genet) LUAD/lung GWAS loci ({len(mckay)} of {n_total_loci}\n"
             "genome-wide significant in this cohort, mark width exaggerated for visibility); then 1M, 500K, 250K\n"
             "PWAS cis-windows, each pair = PASTL (diamond)/Enformer (circle). Labelled genes are significant in\n"
             f"\u22651 of 6 PWAS conditions. Gene* denotes a gene that is both a published McKay GWAS lead gene\n"
             "and a recurring PWAS hit.",
             ha="center", fontsize=10, style="italic")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=300, bbox_inches="tight")
    print(f"Circos plot saved to {args.output_file}")


if __name__ == "__main__":
    main()