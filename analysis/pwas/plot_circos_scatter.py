r"""
Three circos scatter plots (one per priority threshold: 250K, 500K, 1M) showing
per-gene PWAS -log10(p) for PASTL and Enformer on the same scatter ring, styled
as a Manhattan-on-a-circle.

Ring structure (outermost → innermost):
  1. SCATTER RING   – hollow circles per gene; both models on the same ring;
                      dotted Bonferroni threshold line; gene labels with leader lines
  2. McKAY GWAS RING – confirmed LUAD loci (McKay et al. 2017), red bars
  3. CHROMOSOME IDEOGRAM – alternating subtle grey fills, chr1–22 (no chrX)

Usage:
  python plot_circos_scatter.py \
      --results_dir out/analysis/s5fo3aak_best/pwas/results \
      --bed data/sequences_protein.bed \
      --mckay_csv /path/to/data/gwas/mckay2017_table2.csv \
      --gwas_vcf /path/to/data/gwas/ebi-a-GCST004744.vcf.gz \
      --output_dir out/analysis/s5fo3aak_best/pwas/figures/
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pycirclize import Circos

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import palette

# ── Colours ──────────────────────────────────────────────────────────────────

PASTL_COLOR   = palette.MODEL_FILL["pastl"]    # teal
ENFORMER_COLOR = palette.MODEL_FILL["enformer"]  # brick red
MCKAY_RED      = "#D62828"

# ── Genome ───────────────────────────────────────────────────────────────────

HG38_CHROM_SIZES_22 = {
    "chr1":  248956422, "chr2":  242193529, "chr3":  198295559,
    "chr4":  190214555, "chr5":  181538259, "chr6":  170805979,
    "chr7":  159345973, "chr8":  145138636, "chr9":  138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16":  90338345, "chr17":  83257441, "chr18":  80373285,
    "chr19":  58617616, "chr20":  64444167, "chr21":  46709983,
    "chr22":  50818468,
}

CHR_BG = ["#EBEBEB", "#F5F5F5"]  # alternating subtle grays

GWAS_MARK_WIDTH_BP = 2_500_000  # exaggerated for visibility at genome scale

# ── Radial layout (innermost → outermost) ────────────────────────────────────

R_CHR     = (49, 54)  # chromosome ideogram  (innermost of three, still near outside)
R_GWAS    = (57, 60)  # McKay GWAS ring      (shorter bars)
R_SCATTER = (63, 97)  # scatter ring         (outermost; centre of circle left empty)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gwas_vcf_significant(vcf_path: Path, pval_threshold: float = 5e-8) -> pd.DataFrame:
    lp_threshold = -np.log10(pval_threshold)
    rows = []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos, rsid = fields[0], fields[1], fields[2]
            lp = float(fields[9].split(":")[2])
            if lp > lp_threshold:
                rows.append((rsid, f"chr{chrom}", int(pos), lp))
    return pd.DataFrame(rows, columns=["rsid", "chrom", "pos", "neg_log10p"])


def load_mckay_confirmed_loci(mckay_csv: Path, vcf_path: Path) -> pd.DataFrame:
    mckay = pd.read_csv(mckay_csv).rename(
        columns={"rs number": "rsid", "Gene": "gene", "Locus": "cytoband"}
    )
    sig = load_gwas_vcf_significant(vcf_path)
    confirmed = mckay.merge(sig, on="rsid", how="inner")
    confirmed = confirmed[confirmed["chrom"].isin(HG38_CHROM_SIZES_22)]
    return confirmed.sort_values(["chrom", "pos"]).reset_index(drop=True)


def load_threshold_data(results_dir: Path, threshold: str, coord_map: dict) -> dict:
    dfs = {}
    for model in ("pastl", "enformer"):
        path = results_dir / f"{model}_{threshold}" / "pwas_results.tsv"
        df = pd.read_csv(path, sep="\t").dropna(subset=["p_value"])
        df["chrom"]  = df["gene_symbol"].map(lambda g: coord_map.get(g, {}).get("chrom"))
        df["pos"]    = df["gene_symbol"].map(lambda g: coord_map.get(g, {}).get("center"))
        df = df.dropna(subset=["chrom", "pos"])
        df = df[df["chrom"].isin(HG38_CHROM_SIZES_22)]
        df["-log10p"] = -np.log10(df["p_value"].clip(lower=1e-300))
        dfs[model] = df
    return dfs


# ── Plot ──────────────────────────────────────────────────────────────────────

def draw_scatter_circos(
    tl_df: pd.DataFrame,
    enf_df: pd.DataFrame,
    mckay: pd.DataFrame,
    threshold: str,
    output_file: Path,
    sig_type: str = "bonferroni",
):
    # Significance thresholds
    n_tested = max(tl_df["p_value"].notna().sum(), enf_df["p_value"].notna().sum())
    bonf_p   = 0.05 / n_tested
    bonf_log = -np.log10(bonf_p)

    if sig_type == "fdr":
        fdr_hits = pd.concat([
            tl_df[tl_df["sig_fdr"] == True]["p_value"],
            enf_df[enf_df["sig_fdr"] == True]["p_value"],
        ])
        thresh_p   = float(fdr_hits.max()) if not fdr_hits.empty else bonf_p
        thresh_log = -np.log10(thresh_p)
        thresh_label = f"FDR threshold (q < 0.05, p = {thresh_p:.2e})"
        sig_col      = "sig_fdr"
        title_sfx    = "FDR q < 0.05"
    else:
        thresh_p     = bonf_p
        thresh_log   = bonf_log
        thresh_label = f"Bonferroni threshold (p = {bonf_p:.2e})"
        sig_col      = "sig_bonferroni"
        title_sfx    = "Bonferroni p < 0.05"

    all_vals = pd.concat([tl_df["-log10p"], enf_df["-log10p"]])
    vmin = 0.0
    vmax = float(max(all_vals.max(), bonf_log)) + 0.8

    # Genes to label: -log10(p) >= 2.5 in either model
    label_genes: set[str] = set()
    for df in (tl_df, enf_df):
        label_genes |= set(df[df["-log10p"] >= 2.5]["gene_symbol"])

    # Genes significant under chosen threshold in either model → bold label
    sig_genes: set[str] = set()
    for df in (tl_df, enf_df):
        sig_genes |= set(df[df[sig_col] == True]["gene_symbol"])

    # Position lookup for labelled genes (any dataframe that has the gene)
    pos_lookup = (
        pd.concat([tl_df[["gene_symbol", "chrom", "pos"]],
                   enf_df[["gene_symbol", "chrom", "pos"]]])
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")
    )

    # ── Build circos ──────────────────────────────────────────────────────────
    circos = Circos(sectors=HG38_CHROM_SIZES_22, space=2)

    for i, sector in enumerate(circos.sectors):
        c = sector.name

        # 3. Chromosome ideogram (innermost)
        track_chr = sector.add_track(R_CHR, name="chrom_ideogram")
        track_chr.axis(fc=CHR_BG[i % 2], ec="black", lw=0.4)
        track_chr.text(c.replace("chr", ""), color="#333333", size=12)

        # 2. McKay GWAS ring
        track_gwas = sector.add_track(R_GWAS)
        track_gwas.axis(fc="white", ec="black", lw=0.4)
        for _, row in mckay[mckay["chrom"] == c].iterrows():
            s = max(0, row["pos"] - GWAS_MARK_WIDTH_BP / 2)
            e = min(sector.size, row["pos"] + GWAS_MARK_WIDTH_BP / 2)
            track_gwas.rect(s, e, fc=MCKAY_RED, ec="none")

        # 1. Scatter ring (outermost)
        track_sc = sector.add_track(R_SCATTER, name="scatter")
        track_sc.axis(fc="white", ec="none", lw=0.2)

        # Significance threshold line
        track_sc.line(
            [0, sector.size], [thresh_log, thresh_log],
            vmin=vmin, vmax=vmax, lw=0.6, ls=(0, (10, 4)), color="black", zorder=2,
        )

        # PASTL dots
        sub_tl = tl_df[tl_df["chrom"] == c]
        if not sub_tl.empty:
            track_sc.scatter(
                sub_tl["pos"].values, sub_tl["-log10p"].values,
                s=28, color="none", edgecolors=PASTL_COLOR,
                lw=0.8, vmin=vmin, vmax=vmax, zorder=3,
            )

        # Enformer dots
        sub_enf = enf_df[enf_df["chrom"] == c]
        if not sub_enf.empty:
            track_sc.scatter(
                sub_enf["pos"].values, sub_enf["-log10p"].values,
                s=28, color="none", edgecolors=ENFORMER_COLOR,
                lw=0.8, vmin=vmin, vmax=vmax, zorder=3,
            )

    # ── Render (must happen before manual annotation) ─────────────────────────
    tl_sig  = tl_df.set_index("gene_symbol")["-log10p"].to_dict()
    enf_sig = enf_df.set_index("gene_symbol")["-log10p"].to_dict()

    def to_r(v):
        n = max(0.0, min(1.0, (v - vmin) / (vmax - vmin))) if vmax > vmin else 0.0
        return R_SCATTER[0] + n * (R_SCATTER[1] - R_SCATTER[0])

    n_avg = (tl_df["p_value"].notna().sum() + enf_df["p_value"].notna().sum()) // 2
    fig = circos.plotfig(figsize=(14, 14))

    # ── Gene labels — manual polar drawing with overlap correction ───────────
    polar_ax = next(ax for ax in fig.axes if hasattr(ax, "set_theta_offset"))
    CCW_OFFSET  = 0.02   # initial counter-clockwise offset (radians)
    LABEL_PAD   = 2.0    # radial gap from line anchor to text
    LINE_PAD    = 7.0    # radial gap from furthest dot to line anchor
    MIN_ANG_SEP = 0.07   # minimum angular separation between adjacent labels

    # Build annotation list
    anns = []
    for sector in circos.sectors:
        c = sector.name
        for gene in [g for g in label_genes
                     if g in pos_lookup.index and pos_lookup.loc[g, "chrom"] == c]:
            pos     = float(pos_lookup.loc[gene, "pos"])
            theta   = sector.x_to_rad(pos)
            r_tl    = to_r(tl_sig.get(gene, 0.0))
            r_enf   = to_r(enf_sig.get(gene, 0.0))
            r_label = max(r_tl, r_enf) + LINE_PAD
            anns.append({"gene": gene, "theta": theta,
                         "theta_l": theta - CCW_OFFSET,
                         "r_tl": r_tl, "r_enf": r_enf,
                         "r_label": r_label, "bold": gene in sig_genes})

    # Sort and iteratively separate overlapping labels in theta
    anns.sort(key=lambda a: a["theta_l"])
    for _ in range(30):
        changed = False
        for i in range(1, len(anns)):
            gap = anns[i]["theta_l"] - anns[i - 1]["theta_l"]
            if gap < MIN_ANG_SEP:
                shift = (MIN_ANG_SEP - gap) / 2
                anns[i - 1]["theta_l"] -= shift
                anns[i]["theta_l"]     += shift
                changed = True
        if not changed:
            break

    max_r_label = max((a["r_label"] for a in anns), default=R_SCATTER[1] + 5)
    polar_ax.set_ylim(0, max_r_label + 16)

    for ann in anns:
        theta_l = ann["theta_l"]
        theta   = ann["theta"]
        r_tl    = ann["r_tl"]
        r_enf   = ann["r_enf"]
        r_label = ann["r_label"]

        # Leader lines from label anchor → each model's dot
        polar_ax.plot([theta_l, theta], [r_label, r_tl],
                      color=PASTL_COLOR, lw=0.8, zorder=5)
        polar_ax.plot([theta_l, theta], [r_label, r_enf],
                      color=ENFORMER_COLOR, lw=0.8, zorder=5)

        # Text rotation to read outward radially
        theta_deg = np.degrees(theta_l)
        rotation  = 90 - theta_deg
        if rotation < -90:
            rotation += 180
        ha = "left" if np.sin(theta_l) >= 0 else "right"

        polar_ax.text(
            theta_l, r_label + LABEL_PAD, ann["gene"],
            ha=ha, va="center",
            fontsize=16, color="#222222",
            fontweight="bold" if ann["bold"] else "normal",
            rotation=rotation, rotation_mode="anchor",
            zorder=6,
        )

    legend_elems = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor=PASTL_COLOR,
               markeredgewidth=1.2, markersize=9, label="PASTL"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor=ENFORMER_COLOR,
               markeredgewidth=1.2, markersize=9, label="Enformer"),
        Line2D([0], [0], color="black", lw=1.0, ls=(0, (10, 4)),
               label=thresh_label),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=16)

    fig.suptitle(
        f"PASTL vs Enformer PWAS ({threshold})\n"
        f"{n_avg:,} proteins tested; {title_sfx}",
        fontsize=16, fontweight="normal", y=0.97,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Path to pwas/results/ (contains pastl_*/pwas_results.tsv etc.)")
    parser.add_argument("--bed",         type=Path, required=True,
                        help="sequences_protein.bed (gene_symbol → chrom/pos)")
    parser.add_argument("--mckay_csv",   type=Path, required=True,
                        help="mckay2017_table2.csv")
    parser.add_argument("--gwas_vcf",    type=Path, required=True,
                        help="GWAS summary-stats VCF (ebi-a-GCST004744.vcf.gz)")
    parser.add_argument("--output_dir",  type=Path, required=True,
                        help="Directory to save the three PNG files")
    args = parser.parse_args()

    # Gene coordinates
    print("Loading gene coordinates...")
    bed = pd.read_csv(args.bed, sep="\t")
    bed["start"]  = pd.to_numeric(bed["start"],  errors="coerce")
    bed["end"]    = pd.to_numeric(bed["end"],    errors="coerce")
    bed = bed.dropna(subset=["start", "end", "chrom"])
    bed["center"] = (bed["start"] + bed["end"]) / 2
    coord_map = bed.set_index("symbol")[["chrom", "center"]].to_dict(orient="index")

    # McKay GWAS loci
    print("Loading McKay et al. 2017 GWAS loci...")
    mckay = load_mckay_confirmed_loci(args.mckay_csv, args.gwas_vcf)
    n_total = len(pd.read_csv(args.mckay_csv))
    print(f"  {len(mckay)}/{n_total} loci confirmed genome-wide significant in VCF")

    # Two circos per threshold: Bonferroni and FDR
    for threshold in ("250K", "500K", "1M"):
        print(f"\nGenerating {threshold} circos scatter...")
        data = load_threshold_data(args.results_dir, threshold, coord_map)
        for sig_type in ("bonferroni", "fdr"):
            out = args.output_dir / f"pwas_circos_scatter_{threshold}_{sig_type}.png"
            draw_scatter_circos(data["pastl"], data["enformer"], mckay,
                                threshold, out, sig_type=sig_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
