r"""
External validation of PWAS hits against curated gene resources.

Loads per-condition PWAS results (pastl/enformer x 250K/500K/1M plus the
unfiltered baseline), takes the genes significant under the chosen threshold
(--sig_type bonferroni or fdr), and cross-references them against optional
external gene sets: DepMap essential genes, DrugBank/ChEMBL drug targets, the
GWAS Catalog, and DisGeNET disease genes. Emits a hit-grid figure and summary
tables.

Usage:
  python compare_and_validate.py \
      --results_dir out/analysis/s5fo3aak_best/pwas/results \
      --depmap /path/to/CRISPR_gene_effect.csv \
      --drugbank /path/to/drug_targets.tsv \
      --gwas_catalog /path/to/gwas_catalog.tsv \
      --disgenet /path/to/disgenet.tsv \
      --output_dir out/analysis/s5fo3aak_best/pwas/validation
"""
import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib_venn import venn2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import palette

def parse_depmap(path: str, sample_info_path: str = None) -> set:
    """
    Parse DepMap CRISPR_gene_effect.csv.
    Returns a set of gene symbols with median CERES < -0.5.
    If sample_info_path is provided, restricts to LUAD cell lines.
    """
    print(f"Loading DepMap from {path} ...")
    df = pd.read_csv(path, index_col=0)
    
    if sample_info_path and Path(sample_info_path).exists():
        info = pd.read_csv(sample_info_path)
        # Filter to LUAD (Lineage=lung, Subtype=adenocarcinoma)
        # DepMap sample_info uses 'OncotreeSubtype' or 'primary_disease'
        luad_samples = info[info['OncotreeSubtype'].str.contains('Adenocarcinoma', na=False, case=False) & 
                            info['primary_disease'].str.contains('Lung', na=False, case=False)]['DepMap_ID']
        common = df.index.intersection(luad_samples)
        if len(common) > 0:
            df = df.loc[common]
            print(f"  Filtered to {len(common)} LUAD cell lines.")
    
    medians = df.median(axis=0)
    essential = medians[medians < -0.5]
    
    # Extract gene symbols from "GENE (ID)"
    genes = set()
    for col in essential.index:
        match = re.match(r"^([^\s]+)", col)
        if match:
            genes.add(match.group(1))
            
    print(f"  Found {len(genes)} essential genes (median CERES < -0.5).")
    return genes

def parse_drugbank(path: str) -> set:
    """
    Parse DrugBank / ChEMBL drug targets.
    Expects a CSV/TSV with 'Gene Name', 'Hugo_Symbol', or 'gene_symbol'.
    """
    print(f"Loading Drug targets from {path} ...")
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
        
    genes = set()
    # Try common column names
    for col in ['Gene Name', 'Hugo_Symbol', 'gene_symbol', 'symbol', 'Gene']:
        if col in df.columns:
            genes.update(df[col].dropna().astype(str).unique())
            break
            
    if not genes:
        # Fallback: scan all columns for recognizable gene symbols?
        print("  [WARN] Could not find a standard gene symbol column in drug file.")
    else:
        print(f"  Found {len(genes)} drug target genes.")
    return set(genes)

def parse_gwas_catalog(path: str) -> set:
    """
    Parse GWAS Catalog full associations TSV.
    Filters to lung cancer/carcinoma traits via MAPPED_TRAIT and extracts
    gene symbols from MAPPED_GENE.
    """
    print(f"Loading GWAS Catalog from {path} ...")
    df = pd.read_csv(path, sep="\t", low_memory=False)

    lung_terms = ["lung cancer", "lung carcinoma", "lung adenocarcinoma",
                  "non-small cell lung", "squamous cell lung", "small cell lung"]
    mask = df["MAPPED_TRAIT"].fillna("").str.lower().apply(
        lambda t: any(term in t for term in lung_terms)
    )
    lung_df = df[mask]
    print(f"  {mask.sum()} associations matched lung cancer traits.")

    genes = set()
    for val in lung_df["MAPPED_GENE"].dropna():
        for part in re.split(r"[,\-]", val):
            g = part.strip()
            if g and g.lower() != "intergenic":
                genes.add(g)

    print(f"  Found {len(genes)} unique genes.")
    return genes

def parse_disgenet(path: str, cui: str = "C0152013") -> set:
    """
    Parse DisGeNET gene-disease associations.
    Filters to the provided CUI (default Lung adenocarcinoma C0152013).
    """
    print(f"Loading DisGeNET from {path} ...")
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
        
    # Standard DisGeNET columns: diseaseId, geneSymbol
    if "diseaseId" in df.columns and "geneSymbol" in df.columns:
        luad = df[df["diseaseId"] == cui]
        genes = set(luad["geneSymbol"].unique())
    else:
        # If it's already pre-filtered or columns differ
        genes = set()
        for col in ['geneSymbol', 'Gene', 'symbol']:
            if col in df.columns:
                genes.update(df[col].dropna().astype(str).unique())
                break
                
    print(f"  Found {len(genes)} genes associated with {cui}.")
    return genes

def plot_hit_grid(all_results, output_file,
                  sig_type="bonferroni",
                  radius_min=0.18, radius_max=0.40,
                  alpha_min=0.55, alpha_max=1.0):
    """Correlogram-style significance grid with continuous size+alpha encoding strength.

    sig_type: "bonferroni" — show only Bonferroni-significant genes (red colormap)
              "fdr"        — show all FDR-significant genes (blue colormap)
    """
    from matplotlib.colors import LinearSegmentedColormap
    palette.use_style()

    CONDITIONS = ["all", "pastl_250K", "pastl_500K", "pastl_1M",
                  "enformer_250K", "enformer_500K", "enformer_1M"]
    ROW_LABELS = ["Baseline (all)", "PASTL 250K", "PASTL 500K", "PASTL 1M",
                  "Enformer 250K", "Enformer 500K", "Enformer 1M"]

    # n=0 → light/weak end, n=1 → dark/strong end (both cmaps reversed so high values = dark)
    BONF_CMAP = LinearSegmentedColormap.from_list(
        "red_to_yellow",
        ["#951d29", "#c63f31", "#e77550", "#f1b36f", "#fae29b"]
    ).reversed()
    FDR_CMAP = LinearSegmentedColormap.from_list(
        "dark_to_light_blue",
        ["#31358f", "#5274b2", "#80abcf", "#b6d8e9", "#e3f2f8"]
    ).reversed()

    if sig_type == "bonferroni":
        cmap       = BONF_CMAP
        p_col      = "p_bonferroni"
        sig_col    = "sig_bonferroni"
        legend_lbl = "Bonferroni"
        title_sfx  = "(Bonferroni p < 0.05)"
    else:
        cmap       = FDR_CMAP
        p_col      = "q_fdr"
        sig_col    = "sig_fdr"
        legend_lbl = "FDR"
        title_sfx  = "(BH-FDR q < 0.05)"

    # Build hit sets and strength values
    hit_sets = {}
    strength = {}
    raw_p    = {}
    for m in CONDITIONS:
        df = all_results.get(m)
        if df is None:
            hit_sets[m] = set(); continue
        df = df.set_index("gene_symbol")
        hit_sets[m] = set(df[df[sig_col] == True].index)
        for gene in hit_sets[m]:
            p = float(df.loc[gene, p_col])
            strength[(m, gene)] = -np.log10(max(p, 1e-300))
            raw_p[(m, gene)]    = p

    all_sig = set()
    for m in CONDITIONS:
        all_sig |= hit_sets[m]

    if not all_sig:
        print(f"No {sig_type} significant genes — skipping hit grid.")
        return

    strengths = list(strength.values())

    def norm_val(v):
        lo, hi = min(strengths), max(strengths)
        return 1.0 if lo == hi else (v - lo) / (hi - lo)

    def to_radius(n): return radius_min + (radius_max - radius_min) * n

    gene_counts = {g: sum(1 for m in CONDITIONS if g in hit_sets[m]) for g in all_sig}
    genes = sorted(all_sig, key=lambda g: (-gene_counts[g], g))

    n_rows = len(CONDITIONS)
    n_cols = len(genes)

    cell = 0.20
    fig, ax = plt.subplots(figsize=(n_cols * cell + 2.5, n_rows * cell + 2.5))

    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(mpatches.Rectangle(
                (c, r), 1, 1, linewidth=0.15, edgecolor="black",
                facecolor="white", zorder=1))

    for r, m in enumerate(CONDITIONS):
        y = n_rows - r - 0.5
        for c, gene in enumerate(genes):
            x = c + 0.5
            if gene in hit_sets[m]:
                s = strength.get((m, gene), 0)
                n = norm_val(s)
                ax.add_patch(mpatches.Circle(
                    (x, y), to_radius(n), ec="none",
                    color=cmap(n), zorder=3))

    ax.axhline(y=n_rows - 1, color="black", lw=0.7, zorder=4)
    ax.axhline(y=n_rows - 4, color="black", lw=0.7, zorder=4)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")

    ax.set_yticks([n_rows - r - 0.5 for r in range(n_rows)])
    ax.set_yticklabels(ROW_LABELS, fontsize=6)
    ax.tick_params(length=0)

    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels(genes, rotation=90, ha="center", va="bottom", fontsize=6)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.15)
        spine.set_color("black")

    ax.set_title(f"PWAS-significant genes by model and variant priority threshold {title_sfx}", pad=60)

    # --- Gradient legend bar ---
    def fmt_p(p):
        return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

    grad_v = np.linspace(0, 1, 256).reshape(-1, 1)
    # Fix bar to ~0.15 inches wide regardless of how many genes are plotted
    bar_w = 0.05 / (n_cols * cell)
    bar_h = 0.55
    bar_x = 1.06
    y0    = 0.5 - bar_h / 2
    ax_bar = ax.inset_axes([bar_x, y0, bar_w, bar_h])
    ax_bar.imshow(grad_v, aspect="auto", cmap=cmap,
                  extent=[0, 1, 0, 1], origin="lower")

    for sp in ax_bar.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("black")

    cat_raw = list(raw_p.values())
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels([fmt_p(max(cat_raw)), fmt_p(min(cat_raw))],
                           fontsize=5, rotation=0)
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_tick_params(length=2, width=0.5)
    ax_bar.set_xticks([])
    ax_bar.set_title(legend_lbl, fontsize=6, pad=10, loc="left")

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hit grid to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--depmap", default=None)
    parser.add_argument("--drugbank", default=None)
    parser.add_argument("--disgenet", default=None)
    parser.add_argument("--gwas_catalog", default=None)
    parser.add_argument("--sig_type", default="bonferroni", choices=["bonferroni", "fdr"],
                        help="Which significance threshold to use for validation and count plots")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_file", default=None,
                        help="Path for hit grid PNG (default: figures/pwas_hit_grid.png)")
    parser.add_argument("--radius_min", type=float, default=0.18)
    parser.add_argument("--radius_max", type=float, default=0.40)
    parser.add_argument("--alpha_min",  type=float, default=0.55)
    parser.add_argument("--alpha_max",  type=float, default=1.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    models = ["pastl_250K", "pastl_500K", "pastl_1M", 
              "enformer_250K", "enformer_500K", "enformer_1M", "all"]

    all_results = {}
    sig_genes = {}

    summary_rows = []
    
    for m in models:
        p = results_dir / m / "pwas_results.tsv"
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            all_results[m] = df
            
            # Extract significant
            if args.sig_type == "fdr" and "sig_fdr" in df.columns:
                sig = df[df["sig_fdr"] == True]["gene_symbol"].tolist()
            elif "sig_bonferroni" in df.columns:
                sig = df[df["sig_bonferroni"] == True]["gene_symbol"].tolist()
            elif "significant" in df.columns:
                sig = df[df["significant"] == True]["gene_symbol"].tolist()
            else:
                sig = df[df["p_bonferroni"] < 0.05]["gene_symbol"].tolist()
                
            sig_genes[m] = set(sig)
            
            n_tested = len(df.dropna(subset=["p_value"]))
            summary_rows.append({
                "model_threshold": m,
                "n_genes_tested": n_tested,
                "n_genes_significant": len(sig),
            })
            
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "tables" / "pwas_summary.tsv", sep="\t", index=False)
    print(f"Saved summary table with {len(summary_df)} models.")

    hit_grid_base = args.output_file or str(out_dir / "figures" / "pwas_hit_grid.png")
    bonf_path = hit_grid_base.replace(".png", "_bonferroni.png")
    fdr_path  = hit_grid_base.replace(".png", "_fdr.png")
    plot_hit_grid(all_results, bonf_path, sig_type="bonferroni",
                  radius_min=args.radius_min, radius_max=args.radius_max,
                  alpha_min=args.alpha_min,   alpha_max=args.alpha_max)
    plot_hit_grid(all_results, fdr_path,  sig_type="fdr",
                  radius_min=args.radius_min, radius_max=args.radius_max,
                  alpha_min=args.alpha_min,   alpha_max=args.alpha_max)

    # ---------------------------------------------------------
    # Plots: Grouped Bar Chart
    # ---------------------------------------------------------
    palette.use_style()
    thresholds = ["250K", "500K", "1M"]
    x = np.arange(len(thresholds))
    offsets, bw = palette.grouped_bar_positions(len(thresholds), 2)

    tl_counts = [len(sig_genes.get(f"pastl_{t}", [])) for t in thresholds]
    enf_counts = [len(sig_genes.get(f"enformer_{t}", [])) for t in thresholds]

    fig, ax = plt.subplots()
    ax.bar(x + offsets[0], tl_counts, bw, label='PASTL', color=palette.MODEL_FILL["pastl"])
    ax.bar(x + offsets[1], enf_counts, bw, label='Enformer', color=palette.MODEL_FILL["enformer"])

    baseline_count = len(sig_genes.get("all", []))
    if baseline_count > 0:
        ax.axhline(y=baseline_count, color=palette.FILL["other"], linestyle="--", lw=0.6, label="Baseline (All SNPs)")

    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlim(*palette.bar_limits(len(thresholds)))
    ax.set_ylabel("Number of Significant Genes")
    ax.set_xlabel("Priority Threshold")
    ax.set_title("PWAS Significant Genes by Model and Threshold", pad=15)
    palette.square_legend(ax, ['PASTL', 'Enformer'], [palette.MODEL_FILL["pastl"], palette.MODEL_FILL["enformer"]], frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "significant_genes_bar.pdf", dpi=300)
    plt.savefig(out_dir / "figures" / "significant_genes_bar.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plots: Venn Diagrams
    # ---------------------------------------------------------
    for t in thresholds:
        tl_set = sig_genes.get(f"pastl_{t}", set())
        enf_set = sig_genes.get(f"enformer_{t}", set())
        if not tl_set and not enf_set: continue
        
        fig, ax = plt.subplots(figsize=(4, 4))
        v = venn2([tl_set, enf_set], set_labels=('PASTL', 'Enformer'), ax=ax)
        if v.get_patch_by_id('10'):
            v.get_patch_by_id('10').set_color(palette.MODEL_FILL["pastl"])
            v.get_patch_by_id('10').set_alpha(0.45)
            v.get_patch_by_id('10').set_edgecolor(palette.MODEL_FILL["pastl"])
            v.get_patch_by_id('10').set_linewidth(0.5)
        if v.get_patch_by_id('01'):
            v.get_patch_by_id('01').set_color(palette.MODEL_FILL["enformer"])
            v.get_patch_by_id('01').set_alpha(0.45)
            v.get_patch_by_id('01').set_edgecolor(palette.MODEL_FILL["enformer"])
            v.get_patch_by_id('01').set_linewidth(0.5)
        if v.get_patch_by_id('11'):
            v.get_patch_by_id('11').set_color(palette.FILL["other"])
            v.get_patch_by_id('11').set_alpha(0.45)
        ax.set_title(f"PWAS Significant Overlap ({t})", pad=15)
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / f"venn_{t}.pdf", dpi=300)
        plt.savefig(out_dir / "figures" / f"venn_{t}.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    val_sets = {}
    if args.depmap and Path(args.depmap).exists():
        val_sets["DepMap (Essential)"] = parse_depmap(args.depmap)
    if args.drugbank and Path(args.drugbank).exists():
        val_sets["Drug Target"] = parse_drugbank(args.drugbank)
    if args.disgenet and Path(args.disgenet).exists():
        val_sets["DisGeNET (LUAD)"] = parse_disgenet(args.disgenet)
    if args.gwas_catalog and Path(args.gwas_catalog).exists():
        val_sets["GWAS Catalog (Lung)"] = parse_gwas_catalog(args.gwas_catalog)

    if val_sets:
        val_results = []
        for m, s_genes in sig_genes.items():
            if not s_genes: continue
            
            row = {"model_threshold": m, "n_significant": len(s_genes)}
            for v_name, v_set in val_sets.items():
                overlap = len(s_genes.intersection(v_set))
                frac = overlap / len(s_genes) if len(s_genes) > 0 else 0
                row[f"{v_name}_overlap"] = overlap
                row[f"{v_name}_fraction"] = frac
            val_results.append(row)
            
        val_df = pd.DataFrame(val_results)
        val_df.to_csv(out_dir / "tables" / "validation_summary.tsv", sep="\t", index=False)
        print("Saved validation summary table.")
        
        # Plot validation rates
        for v_name in val_sets.keys():
            fig, ax = plt.subplots()
            tl_frac = [val_df.loc[val_df["model_threshold"]==f"pastl_{t}", f"{v_name}_fraction"].values[0]
                       if f"pastl_{t}" in val_df["model_threshold"].values else 0 for t in thresholds]
            enf_frac = [val_df.loc[val_df["model_threshold"]==f"enformer_{t}", f"{v_name}_fraction"].values[0]
                        if f"enformer_{t}" in val_df["model_threshold"].values else 0 for t in thresholds]

            ax.bar(x + offsets[0], tl_frac, bw, label='PASTL', color=palette.MODEL_FILL["pastl"])
            ax.bar(x + offsets[1], enf_frac, bw, label='Enformer', color=palette.MODEL_FILL["enformer"])

            if "all" in val_df["model_threshold"].values:
                baseline_frac = val_df.loc[val_df["model_threshold"]=="all", f"{v_name}_fraction"].values[0]
                ax.axhline(y=baseline_frac, color=palette.FILL["other"], linestyle="--", lw=0.6, label="Baseline (All SNPs)")

            ax.set_xticks(x)
            ax.set_xticklabels(thresholds)
            ax.set_xlim(*palette.bar_limits(len(thresholds)))
            ax.set_ylabel(f"Fraction of Genes in {v_name}")
            ax.set_xlabel("Priority Threshold")
            ax.set_title(f"Validation Enrichment: {v_name}", pad=15)
            palette.square_legend(ax, ['PASTL', 'Enformer'], [palette.MODEL_FILL["pastl"], palette.MODEL_FILL["enformer"]], frameon=False)
            plt.tight_layout()
            safe_name = v_name.split()[0].lower()
            plt.savefig(out_dir / "figures" / f"validation_bar_{safe_name}.pdf", dpi=300)
            plt.savefig(out_dir / "figures" / f"validation_bar_{safe_name}.png", dpi=300)
            plt.close()
            
if __name__ == "__main__":
    main()
