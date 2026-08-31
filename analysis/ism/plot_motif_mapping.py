r"""
Plot motif-mapping results for the ISM pipeline.

Loads the motif enrichment and hits tables (from intersect_pastl_dominant.py),
the ISM regions table, and the per-gene ISM score .h5 files, then produces a
model-divergence boxplot, a top-driver barplot, and per-gene locus tracks
(standard and lollipop) of PASTL vs Enformer ISM z-scores over mapped motifs.
Candidate genes for the locus tracks are selected automatically unless --gene
is given.

Usage:
  python plot_motif_mapping.py \
      --wandb_run_id s5fo3aak --checkpoint best.pt \
      --output_dir out/analysis/s5fo3aak_best/ism/figures
"""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import palette as _pal

_BOLD  = _pal.PALETTES["bold"]
_MUTED = _pal.PALETTES["muted"]

# Model line/fill colours — fixed bold pairing.
MODEL_FILL = {"PASTL": _pal.MODEL_FILL["pastl"], "Enformer": _pal.MODEL_FILL["enformer"]}
MODEL_INK  = MODEL_FILL

# Region colours used in plot_locus_track (bold palette, standard track layout).
REGION_FILL = {
    "tss":       _BOLD["cat3"],
    "utr5":      _BOLD["cat4"],
    "utr3":      _BOLD["cat5"],
    "gene_body": _BOLD["cat1"],
}
REGION_INK = REGION_FILL  # locus_track uses both; keep consistent

# Annotation bar colours for plot_locus_track_lollipop — muted palette.
# tss=blue(cat3), utr5=purple(cat4), utr3=gold(cat5), gene_body=grey(other)
ANNO_FILL = {
    "tss":       _MUTED["cat3"],   # navy/blue
    "utr5":      _MUTED["cat4"],   # mauve/purple
    "utr3":      _MUTED["cat5"],   # gold
    "gene_body": _MUTED["other"],  # grey
}

MOTIF_FILLS = [_BOLD["cat1"], _BOLD["cat2"], _BOLD["cat3"], _BOLD["cat4"], _BOLD["cat5"], _BOLD["other"]]
MOTIF_INKS  = MOTIF_FILLS

def plot_divergence_boxplot(hits_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = hits_df.groupby("motif_type")["z_diff"]
    medians = grouped.median().sort_values(ascending=False)

    data = [grouped.get_group(m).values for m in medians.index]

    bplot = ax.boxplot(data, patch_artist=True, showfliers=False)

    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(MOTIF_FILLS[i % len(MOTIF_FILLS)])
        patch.set_edgecolor(MOTIF_INKS[i % len(MOTIF_INKS)])
        patch.set_alpha(0.85)

    for i, median in enumerate(bplot['medians']):
        median.set_color(MOTIF_INKS[i % len(MOTIF_INKS)])
        median.set_linewidth(1.5)

    ax.set_xticks(np.arange(1, len(medians.index) + 1))
    ax.set_xticklabels(medians.index, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Disruption Divergence (z_diff)", fontsize=8)
    ax.set_title("Magnitude of Model Divergence by Motif Type", fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:g}'))

    fig.tight_layout()
    out = Path(out_dir) / "divergence_boxplot.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

def plot_top_drivers_barplot(hits_df, out_dir):
    motif_stats = hits_df.groupby(["motif_name", "motif_type"]).agg(
        mean_z=("z_diff", "mean"),
        count=("z_diff", "count")
    ).reset_index()

    robust = motif_stats[motif_stats["count"] >= 5]

    # Top 10 per category + all polyA signals
    TOP_N = 10
    CATEGORIES = ["miRNA_targetscan", "RBP_fimo"]
    parts = []
    for cat in CATEGORIES:
        top = (robust[robust["motif_type"] == cat]
               .sort_values("mean_z", ascending=False)
               .head(TOP_N))
        parts.append(top)
    polya = robust[robust["motif_type"] == "polyA_signal"]
    parts.append(polya)
    combined = pd.concat(parts, ignore_index=True).sort_values("mean_z", ascending=False)

    # Match track plot colour assignment: tab10 keyed by sorted motif_type
    all_mtypes = sorted(hits_df["motif_type"].unique())
    try:
        cmap = plt.colormaps["tab10"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    global_mtype_color = {m: cmap(i % 10) for i, m in enumerate(all_mtypes)}
    type_to_color_fill = global_mtype_color
    type_to_color_ink  = global_mtype_color

    n = len(combined)
    bar_height = 0.22
    fig_h = max(4, n * (bar_height + 0.005) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = np.arange(n)
    bar_colors  = [type_to_color_fill[t] for t in combined["motif_type"]]
    edge_colors = [type_to_color_ink[t]  for t in combined["motif_type"]]

    ax.barh(y_pos, combined["mean_z"], height=bar_height,
            color=bar_colors, edgecolor=edge_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined["motif_name"], fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=8)

    ax.set_xlabel("Average Disruption Divergence (z_diff)", fontsize=8)
    ax.set_title(f"Top {TOP_N} motif drivers per category (min 5 hits)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:g}'))

    legend_patches = [
        mpatches.Patch(facecolor=global_mtype_color[t], edgecolor=global_mtype_color[t], label=t)
        for t in all_mtypes if t in combined["motif_type"].values
    ]
    ax.legend(handles=legend_patches, title="Motif Category",
              bbox_to_anchor=(1.05, 1), loc="upper left",
              frameon=False, fontsize=8, title_fontsize=8)

    fig.tight_layout()
    out = Path(out_dir) / "top_motif_drivers.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

def select_candidate_genes(hits_df, enrichment_df, scores_dir, n_examples, tl_params, enf_params):
    target_hits = hits_df[hits_df["motif_type"].isin(["miRNA_targetscan", "uORF"])].copy()

    tl_mean, tl_std = tl_params
    enf_mean, enf_std = enf_params

    target_hits["local_prominence"] = -np.inf

    for gene, group in target_hits.groupby("symbol"):
        h5_path = Path(scores_dir) / f"{gene}.h5"
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            pos = f["position"][:]
            tl = f["pastl_ism"][:]

        tl_z = (tl - tl_mean) / tl_std

        for idx, row in group.iterrows():
            start, end = row["start"], row["end"]
            center = (start + end) // 2
            motif_mask = (pos >= start) & (pos <= end)
            win_mask = (pos >= center - 200) & (pos <= center + 200)

            if motif_mask.any() and win_mask.any():
                motif_max = tl_z[motif_mask].max()
                win_std = np.std(tl_z[win_mask])

                if win_std == 0:
                    win_std = 1e-6

                target_hits.at[idx, "local_prominence"] = motif_max / win_std

    candidates = []

    mirna_hits = target_hits[target_hits["motif_type"] == "miRNA_targetscan"]
    mirna_symbols = mirna_hits.sort_values("local_prominence", ascending=False)["symbol"].unique()
    for sym in mirna_symbols[:n_examples]:
        candidates.append((sym, "miRNA_targetscan"))

    uorf_hits = target_hits[target_hits["motif_type"] == "uORF"]
    uorf_symbols = uorf_hits.sort_values("local_prominence", ascending=False)["symbol"].unique()
    for sym in uorf_symbols[:n_examples]:
        candidates.append((sym, "uORF"))

    return candidates

def plot_locus_track(gene, regions_df, hits_df, scores_dir, out_dir, tl_params, enf_params, target_motif_type=None):
    h5_path = Path(scores_dir) / f"{gene}.h5"
    if not h5_path.exists():
        print(f"Error: Scores file {h5_path} not found. Skipping locus track for {gene}.")
        return

    with h5py.File(h5_path, "r") as f:
        pos = f["position"][:]
        pastl_ism = f["pastl_ism"][:]
        enformer_ism = f["enformer_ism"][:]
        tss = int(f.attrs.get("tss", 0))
        strand = str(f.attrs.get("strand", "+"))
        if isinstance(strand, bytes):
            strand = strand.decode()

    tl_mean, tl_std = tl_params
    enf_mean, enf_std = enf_params

    pastl_ism_raw = (pastl_ism - tl_mean) / tl_std
    enformer_ism_raw = (enformer_ism - enf_mean) / enf_std

    gene_hits = hits_df[hits_df["symbol"] == gene].copy()
    prominences = []
    for _, row in gene_hits.iterrows():
        start, end = row["start"], row["end"]
        center = (start + end) // 2
        motif_mask = (pos >= start) & (pos <= end)
        win_mask = (pos >= center - 200) & (pos <= center + 200)
        if motif_mask.any() and win_mask.any():
            win_std = np.std(pastl_ism_raw[win_mask])
            if win_std == 0:
                win_std = 1e-6
            prominences.append(pastl_ism_raw[motif_mask].max() / win_std)
        else:
            prominences.append(-np.inf)
    gene_hits["local_prominence"] = prominences

    pastl_ism = pastl_ism_raw
    enformer_ism = enformer_ism_raw

    if strand == "+":
        pos_rel = pos - tss
    else:
        pos_rel = tss - pos

    sort_idx = np.argsort(pos_rel)
    pos_rel = pos_rel[sort_idx].astype(float)
    pastl_ism = pastl_ism[sort_idx]
    enformer_ism = enformer_ism[sort_idx]

    gap_idx = np.where(np.diff(pos_rel) > 1)[0]
    pos_rel = np.insert(pos_rel, gap_idx + 1, np.nan)
    pastl_ism = np.insert(pastl_ism, gap_idx + 1, np.nan)
    enformer_ism = np.insert(enformer_ism, gap_idx + 1, np.nan)

    # Establish global colour mapping for all possible motif types in hits_df
    all_mtypes = sorted(hits_df["motif_type"].unique())
    try:
        colors = plt.colormaps["tab10"]
    except AttributeError:
        colors = plt.cm.get_cmap("tab10")
    global_mtype_color = {m: colors(i % 10) for i, m in enumerate(all_mtypes)}

    # Automatic 400bp Zoom Centred on the Highest Prominence TARGET Motif Hit
    if not gene_hits.empty:
        if target_motif_type:
            target_subset = gene_hits[gene_hits["motif_type"] == target_motif_type]
            if not target_subset.empty:
                top_hit = target_subset.loc[target_subset["local_prominence"].idxmax()]
            else:
                top_hit = gene_hits.loc[gene_hits["local_prominence"].idxmax()]
        else:
            top_hit = gene_hits.loc[gene_hits["local_prominence"].idxmax()]

        if strand == "+":
            center_rel = top_hit["start"] - tss
        else:
            center_rel = tss - top_hit["end"]

        # Calculate strict x/y boundaries prior to plotting motif bands
        visible_mask = (pos_rel >= center_rel - 150) & (pos_rel <= center_rel + 150)
        if visible_mask.any():
            max_y = np.nanmax([np.nanmax(pastl_ism[visible_mask]), np.nanmax(enformer_ism[visible_mask])])
            min_y = np.nanmin([np.nanmin(pastl_ism[visible_mask]), np.nanmin(enformer_ism[visible_mask])])
            padding = (max_y - min_y) * 0.1 if max_y > min_y else 0.5

            shared_min = min_y - padding
            shared_max = max_y + padding
        else:
            shared_min, shared_max = -0.5, 1.5
    else:
        center_rel = 0
        shared_min, shared_max = -0.5, 1.5

    # Subplots layout 3x1, uniform height, compressed vertical spacing
    fig, (ax_tl, ax_enf, ax_anno) = plt.subplots(
        nrows=3, ncols=1, figsize=(13, 4.0),
        sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.4], 'hspace': 0.15}
    )

    gene_mtypes = list(gene_hits["motif_type"].unique())

    # Panel A: PASTL Disruption Score
    ax_tl.plot(pos_rel, pastl_ism, color=MODEL_INK.get("PASTL", "green"), linewidth=0.9, alpha=0.9, zorder=2)
    ax_tl.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax_tl.set_ylabel("PASTL\n(z-ISM)", fontsize=8)
    ax_tl.tick_params(axis='both', which='major', labelsize=8)
    ax_tl.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_tl.spines['bottom'].set_visible(False)
    ax_tl.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:g}'))

    # Panel B: Enformer Disruption Score
    ax_enf.plot(pos_rel, enformer_ism, color=MODEL_INK.get("Enformer", "red"), linewidth=0.9, alpha=0.9, zorder=2)
    ax_enf.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax_enf.set_ylabel("Enformer\n(z-ISM)", fontsize=8)
    ax_enf.tick_params(axis='both', which='major', labelsize=8)
    ax_enf.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_enf.spines['bottom'].set_visible(False)
    ax_enf.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:g}'))

    # Panel C: Gene Annotation and Motif Hits
    gene_regions = regions_df[regions_df["symbol"] == gene]
    for _, row in gene_regions.iterrows():
        s, e = row["start"], row["end"]
        if strand == "+":
            s_rel, e_rel = s - tss, e - tss
        else:
            s_rel, e_rel = tss - e, tss - s
        rtype = row["region_type"]
        color = ANNO_FILL.get(rtype, "#cccccc")
        zorder = 1
        if rtype == "tss":
            zorder = 2
        elif rtype in ("utr5", "utr3"):
            zorder = 3

        ax_anno.add_patch(Rectangle((s_rel, 0.25), e_rel - s_rel, 0.3, facecolor=color, edgecolor="none", label=rtype, zorder=zorder))

    # Draw motif hits in a single band
    if len(gene_mtypes) > 0:
        y_start = 0.60
        band_height = 0.3
        for _, row in gene_hits.iterrows():
            s, e = row["start"], row["end"]
            if strand == "+":
                s_rel, e_rel = s - tss, e - tss
            else:
                s_rel, e_rel = tss - e, tss - s
            mtype = row["motif_type"]
            color = global_mtype_color[mtype]
            # More saturated (higher alpha) than before, all on one track
            ax_anno.add_patch(Rectangle((s_rel, y_start), max(e_rel - s_rel, 1), band_height, facecolor=color, alpha=0.4, lw=0, zorder=4))

    ax_anno.set_ylim(0, 1)
    ax_anno.set_yticks([])
    ax_anno.set_ylabel("Annotation\n& Motifs", fontsize=8)
    ax_anno.tick_params(axis='x', which='major', labelsize=8)
    ax_anno.set_xlabel("TSS-Relative Position", fontsize=8)

    handles, labels = ax_anno.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for m in gene_mtypes:
        by_label[m] = mpatches.Patch(facecolor=global_mtype_color[m], alpha=0.5, label=m)

    if by_label:
        ax_anno.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), 
                       loc="upper left", fontsize=8, frameon=False, 
                       handlelength=0.8, handleheight=0.8)

    # Apply strict limits configured prior to rendering
    ax_tl.set_xlim(center_rel - 150, center_rel + 150)
    ax_tl.set_ylim(shared_min, shared_max)
    ax_enf.set_ylim(shared_min, shared_max)

    fig.suptitle(f"Locus-Specific ISM Track: {gene}", y=0.95, fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    out = Path(out_dir) / f"{gene}_locus_track.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")

def plot_locus_track_lollipop(gene, regions_df, hits_df, scores_dir, out_dir, tl_params, enf_params, start_rel, end_rel):
    """Lollipop version of locus track: sequence logo + per-alt-base lollipops + annotation track."""
    h5_path = Path(scores_dir) / f"{gene}.h5"
    if not h5_path.exists():
        print(f"Error: Scores file {h5_path} not found. Skipping lollipop track for {gene}.")
        return

    with h5py.File(h5_path, "r") as f:
        pos     = f["position"][:]
        tl_raw  = f["pastl_ism"][:]
        enf_raw = f["enformer_ism"][:]
        ref_raw = f["ref_base"][:]
        alt_raw = f["alt_base"][:]
        tss     = int(f.attrs.get("tss", 0))
        strand  = str(f.attrs.get("strand", "+"))
        if isinstance(strand, bytes):
            strand = strand.decode()

    ref_base = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in ref_raw])
    alt_base = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in alt_raw])

    COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    if strand == "-":
        ref_base = np.array([COMP.get(b, b) for b in ref_base])
        alt_base = np.array([COMP.get(b, b) for b in alt_base])

    tl_mean, tl_std   = tl_params
    enf_mean, enf_std = enf_params
    tl_z  = (tl_raw  - tl_mean)  / tl_std
    enf_z = (enf_raw - enf_mean) / enf_std

    pos_rel = (pos - tss).astype(float) if strand == "+" else (tss - pos).astype(float)

    sort_idx = np.argsort(pos_rel)
    pos_rel  = pos_rel[sort_idx]
    tl_z     = tl_z[sort_idx]
    enf_z    = enf_z[sort_idx]
    ref_base = ref_base[sort_idx]
    alt_base = alt_base[sort_idx]

    # Per-position mean z-scores (for prominence and logo)
    tmp = pd.DataFrame({"pos": pos_rel, "tl": tl_z, "enf": enf_z})
    agg = tmp.groupby("pos").agg({"tl": "mean", "enf": "mean"})
    uniq_pos   = agg.index.values
    mean_tl_z  = agg["tl"].values

    tmp_ref    = pd.DataFrame({"pos": pos_rel, "ref": ref_base}).drop_duplicates("pos").set_index("pos")
    ref_by_pos = tmp_ref["ref"]

    gene_hits   = hits_df[hits_df["symbol"] == gene].copy()

    # Visible window
    vis_mask     = (pos_rel >= start_rel) & (pos_rel <= end_rel)
    vis_pos      = pos_rel[vis_mask]
    vis_tl       = tl_z[vis_mask]
    vis_enf      = enf_z[vis_mask]
    vis_ref      = ref_base[vis_mask]
    vis_alt      = alt_base[vis_mask]
    win_uniq     = (uniq_pos >= start_rel) & (uniq_pos <= end_rel)
    vis_uniq_pos = uniq_pos[win_uniq]
    vis_mean_tl  = mean_tl_z[win_uniq]

    shared_ymax = max(np.nanmax(np.abs(vis_tl)), np.nanmax(np.abs(vis_enf)), 0.5) * 1.2 if vis_mask.any() else 2.0
    shared_ymin = min(np.nanmin(vis_tl), np.nanmin(vis_enf), 0.0) * 1.2 if vis_mask.any() else -0.5
    
    tl_ylim = enf_ylim = shared_ymax
    tl_ymin = enf_ymin = shared_ymin

    BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "T": "#d62728", "G": "#ff7f0e"}
    all_mtypes  = sorted(hits_df["motif_type"].unique())
    try:
        cmap = plt.colormaps["tab10"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    global_mtype_color = {m: cmap(i % 10) for i, m in enumerate(all_mtypes)}
    gene_mtypes = list(gene_hits["motif_type"].unique())

    # --- Figure: logo + tl lollipop + enf lollipop + annotation ---
    fig, (ax_logo, ax_tl, ax_enf, ax_anno) = plt.subplots(
        nrows=4, ncols=1, figsize=(18, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [0.8, 0.8, 0.8, 0.4], "hspace": 0.15}
    )

    # ---- Panel A: Sequence Logo (logomaker or bar fallback) ----
    logo_rows = []
    for p, m in zip(vis_uniq_pos, vis_mean_tl):
        ref = ref_by_pos.get(p, "N")
        row = {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0}
        if ref in row:
            row[ref] = max(float(m), 0.0)
        logo_rows.append(row)
    logo_df = pd.DataFrame(logo_rows, index=vis_uniq_pos.astype(int))

    try:
        import logomaker
        logomaker.Logo(logo_df, ax=ax_logo,
                       color_scheme={b: BASE_COLORS[b] for b in "ACGT"})
    except (ImportError, Exception) as logo_err:
        print(f"  logomaker unavailable ({logo_err}); using bar fallback for logo panel.")
        for p, row_d in zip(vis_uniq_pos, logo_rows):
            ref = next((b for b in "ACGT" if row_d[b] > 0), "N")
            ax_logo.bar(p, row_d.get(ref, 0), color=BASE_COLORS.get(ref, "gray"), width=0.8, alpha=0.85)

    max_peak = logo_df.values.max() if not logo_df.empty else 1.0
    if pd.isna(max_peak) or max_peak <= 0:
        max_peak = 1.0
    ax_logo.set_ylim(-0.1 * max_peak, 1.2 * max_peak)

    ax_logo.set_ylabel("Mean ISM\n(ref base)", fontsize=8)
    ax_logo.tick_params(axis="both", which="major", labelsize=8)
    ax_logo.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_logo.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_logo.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x >= 0 else ""))
    for sp in ["top", "right", "bottom"]:
        ax_logo.spines[sp].set_visible(False)

    # ---- Panels B & C: Lollipops ----
    BASES        = ["A", "C", "G", "T"]
    base_offsets = {"A": -0.45, "C": -0.15, "G": 0.15, "T": 0.45}

    for base in BASES:
        mask = vis_alt == base
        if not mask.any():
            continue
        xvals = vis_pos[mask] + base_offsets[base]
        bc    = BASE_COLORS[base]
        ax_tl.vlines(xvals, 0, vis_tl[mask], colors=MODEL_FILL["PASTL"], linewidth=0.8, alpha=0.7, zorder=1)
        ax_tl.scatter(xvals, vis_tl[mask], c=bc, marker="s", s=12, edgecolors="none", zorder=2, label=base)
        ax_enf.vlines(xvals, 0, vis_enf[mask], colors=MODEL_FILL["Enformer"], linewidth=0.8, alpha=0.7, zorder=1)
        ax_enf.scatter(xvals, vis_enf[mask], c=bc, marker="s", s=12, edgecolors="none", zorder=2)

    # Annotate top-5 lollipops per panel
    for ax, vals in [(ax_tl, vis_tl), (ax_enf, vis_enf)]:
        if len(vals) == 0:
            continue
        top_idx = np.argsort(np.abs(vals))[-min(5, len(vals)):]
        for i in top_idx:
            x = float(vis_pos[i]) + base_offsets.get(vis_alt[i], 0)
            y = float(vals[i])
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 4 if y >= 0 else -8), fontsize=6,
                        ha="center", va="bottom" if y >= 0 else "top")

    for ax, ymin, ymax, label in [
        (ax_tl,  tl_ymin,  tl_ylim,  "PASTL\n(z-ISM)"),
        (ax_enf, enf_ymin, enf_ylim, "Enformer\n(z-ISM)"),
    ]:
        ax.axhline(0, color="black", linewidth=0.8, zorder=0)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_ylim(ymin, ymax)

    base_patches = [mpatches.Patch(color=BASE_COLORS[b], label=b) for b in BASES]
    ax_tl.legend(handles=base_patches, bbox_to_anchor=(1.02, 1), loc="upper left",
                 fontsize=8, frameon=False, title="Alt Base", title_fontsize=8,
                 handlelength=0.8, handleheight=0.8)

    # ---- Panel D: Annotation (identical layout to plot_locus_track) ----
    gene_regions = regions_df[regions_df["symbol"] == gene]
    for _, row in gene_regions.iterrows():
        s, e      = row["start"], row["end"]
        s_rel     = (s - tss) if strand == "+" else (tss - e)
        e_rel     = (e - tss) if strand == "+" else (tss - s)
        rtype     = row["region_type"]
        color  = ANNO_FILL.get(rtype, "#cccccc")
        zorder = {"tss": 2, "utr5": 3, "utr3": 3}.get(rtype, 1)
        ax_anno.add_patch(Rectangle((s_rel, 0.25), e_rel - s_rel, 0.3,
                                    facecolor=color, edgecolor="none",
                                    label=rtype, zorder=zorder))

    if gene_mtypes:
        for _, row in gene_hits.iterrows():
            s, e  = row["start"], row["end"]
            s_rel = (s - tss) if strand == "+" else (tss - e)
            e_rel = (e - tss) if strand == "+" else (tss - s)
            mtype = row["motif_type"]
            color = global_mtype_color[mtype]
            ax_anno.add_patch(Rectangle((s_rel, 0.60), max(e_rel - s_rel, 1), 0.3,
                                        facecolor=color, edgecolor="none", alpha=0.4, zorder=4))

    ax_anno.set_ylim(0, 1)
    ax_anno.set_yticks([])
    ax_anno.set_ylabel("Annotation\n& Motifs", fontsize=8)
    ax_anno.tick_params(axis="x", which="major", labelsize=8)
    ax_anno.set_xlabel("TSS-Relative Position", fontsize=8)

    handles, labels = ax_anno.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for m in gene_mtypes:
        by_label[m] = mpatches.Patch(facecolor=global_mtype_color[m], alpha=0.4, label=m)
    if by_label:
        ax_anno.legend(by_label.values(), by_label.keys(),
                       bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False,
                       handlelength=0.8, handleheight=0.8)

    ax_tl.set_xlim(start_rel, end_rel)

    fig.suptitle(f"Locus-Specific ISM Track (Lollipop): {gene}", y=0.98, fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.85, 0.97])
    out_path = Path(out_dir) / f"{gene}_locus_track_lollipop.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize ISM motif mapping results.")
    parser.add_argument("--wandb_run_id", required=True,
                        help="W&B run ID identifying the model output directory.")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename used to build the run slug (default: best.pt).")
    parser.add_argument("--enrichment_tsv", default=None,
                        help="motif_enrichment.tsv (default: <run>/ism/motifs/motif_enrichment.tsv)")
    parser.add_argument("--hits_tsv", default=None,
                        help="motif_hits.tsv (default: <run>/ism/motifs/motif_hits.tsv)")
    parser.add_argument("--regions_tsv", default=None,
                        help="ism_regions.tsv (default: <run>/ism/ism_regions.tsv)")
    parser.add_argument("--scores_dir", default=None,
                        help="Directory of per-gene .h5 files (default: <run>/ism/scores)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for plots (default: <run>/ism/figures)")
    parser.add_argument("--gene",
                        help="Gene symbol for locus-specific track plot. If omitted, genes are auto-selected.")
    parser.add_argument("--n_examples", type=int, default=10,
                        help="Number of automated locus track examples to plot if --gene is not specified (default: 10)")

    args = parser.parse_args()

    slug = f"{args.wandb_run_id}_{Path(args.checkpoint).stem}"
    run_dir = Path("out/analysis") / slug / "ism"
    if args.enrichment_tsv is None:
        args.enrichment_tsv = str(run_dir / "motifs" / "motif_enrichment.tsv")
    if args.hits_tsv is None:
        args.hits_tsv = str(run_dir / "motifs" / "motif_hits.tsv")
    if args.regions_tsv is None:
        args.regions_tsv = str(run_dir / "ism_regions.tsv")
    if args.scores_dir is None:
        args.scores_dir = str(run_dir / "scores")
    if args.output_dir is None:
        args.output_dir = str(run_dir / "figures")

    style_path = Path(__file__).parent.parent.parent / "pastl.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    try:
        enrichment_df = pd.read_csv(args.enrichment_tsv, sep="\t")
        hits_df = pd.read_csv(args.hits_tsv, sep="\t")
        regions_df = pd.read_csv(args.regions_tsv, sep="\t")
    except Exception as e:
        print(f"Error reading input TSVs: {e}")
        return

    print("Computing global Z-score parameters...")
    tl_all, enf_all = [], []
    for h5 in Path(args.scores_dir).glob("*.h5"):
        with h5py.File(h5, "r") as f:
            tl_all.append(f["pastl_ism"][:])
            enf_all.append(f["enformer_ism"][:])
    if not tl_all:
        print("No .h5 files found in scores_dir.")
        return
    tl_concat = np.concatenate(tl_all)
    enf_concat = np.concatenate(enf_all)
    tl_params = (tl_concat.mean(), tl_concat.std())
    enf_params = (enf_concat.mean(), enf_concat.std())
    del tl_concat, enf_concat, tl_all, enf_all

    print("Generating Divergence Boxplot...")
    plot_divergence_boxplot(hits_df, args.output_dir)

    print("Generating Top Drivers Barplot...")
    plot_top_drivers_barplot(hits_df, args.output_dir)

    if args.gene:
        print(f"Generating Locus-Specific ISM Track for {args.gene}...")
        plot_locus_track(args.gene, regions_df, hits_df, args.scores_dir, args.output_dir, tl_params, enf_params)
    else:
        print(f"Automatically selecting candidate genes for locus tracks...")
        candidate_genes = select_candidate_genes(hits_df, enrichment_df, args.scores_dir, args.n_examples, tl_params, enf_params)
        if not candidate_genes:
            print("No suitable candidate genes found.")
        for gene, motif_type in candidate_genes:
            print(f"Generating Locus-Specific ISM Track for {gene} (Targeting {motif_type})...")
            plot_locus_track(gene, regions_df, hits_df, args.scores_dir, args.output_dir, tl_params, enf_params, target_motif_type=motif_type)

        print("\nGenerating Lollipop ISM Tracks for specific genes/intervals...")
        lollipop_targets = {
            "CELF1": (22550, 22800),
            "CHST7": (24100, 24350),
            "LYPD3": (4570, 4820),
            "TRMT61A": (5640, 5890),
        }
        for g, (start_rel, end_rel) in lollipop_targets.items():
            print(f"Generating custom Lollipop ISM Track for {g} ({start_rel}-{end_rel})...")
            plot_locus_track_lollipop(g, regions_df, hits_df, args.scores_dir, args.output_dir, tl_params, enf_params, start_rel, end_rel)

if __name__ == "__main__":
    main()