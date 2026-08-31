"""
Shared configuration for the GWAS LUAD variant-effect analysis: VEP consequence
grouping, GWAS significance threshold, palette bindings, and axis styling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import palette

GWAS_SIG_THRESHOLD = 5e-8

PASTL_FILL   = palette.MODEL_FILL["pastl"]
PASTL_INK    = palette.MODEL_FILL["pastl"]
ENFORMER_FILL = palette.MODEL_FILL["enformer"]
ENFORMER_INK  = palette.MODEL_FILL["enformer"]


# =============================================================================
# Consequence grouping definitions (VEP SO term -> coarse class)
# =============================================================================

CONSEQUENCE_MAP = {
    "stop_gained":                         "stop/frameshift",
    "stop_lost":                           "stop/frameshift",
    "frameshift_variant":                  "stop/frameshift",
    "start_lost":                          "stop/frameshift",
    "transcript_ablation":                 "stop/frameshift",
    "splice_acceptor_variant":             "splice",
    "splice_donor_variant":                "splice",
    "splice_region_variant":               "splice",
    "splice_donor_region_variant":         "splice",
    "splice_polypyrimidine_tract_variant": "splice",
    "splice_donor_5th_base_variant":       "splice",
    "missense_variant":                    "missense",
    "protein_altering_variant":            "missense",
    "inframe_insertion":                   "inframe_indel",
    "inframe_deletion":                    "inframe_indel",
    "synonymous_variant":                  "synonymous",
    "stop_retained_variant":               "synonymous",
    "start_retained_variant":              "synonymous",
    "5_prime_UTR_variant":                 "5'UTR",
    "3_prime_UTR_variant":                 "3'UTR",
    "upstream_gene_variant":               "upstream/promoter",
    "intron_variant":                      "intronic",
    "non_coding_transcript_exon_variant":  "intronic",
    "non_coding_transcript_variant":       "intronic",
    "downstream_gene_variant":             "downstream",
    "intergenic_variant":                  "intergenic",
    "regulatory_region_variant":           "regulatory",
    "TF_binding_site_variant":             "regulatory",
    "TFBS_ablation":                       "regulatory",
    "TFBS_amplification":                  "regulatory",
}

SEVERITY_ORDER = list(CONSEQUENCE_MAP.keys())

CLASS_ORDER = [
    "stop/frameshift", "splice", "missense", "inframe_indel", "5'UTR",
    "upstream/promoter", "regulatory", "synonymous", "intronic", "3'UTR",
    "downstream", "intergenic", "other", "unannotated",
]

CLASS_PALETTE = {}
_cycle_classes = [c for c in CLASS_ORDER if c not in ("other", "unannotated")]
for i, cls in enumerate(_cycle_classes):
    CLASS_PALETTE[cls] = palette.GROUP_FILLS[i % len(palette.GROUP_FILLS)]
CLASS_PALETTE["other"]       = palette.FILL["other"]
CLASS_PALETTE["unannotated"] = palette.FILL["other"]

_BOLD = palette.PALETTES["bold"]
DISTANCE_BINS   = [0, 2_000, 10_000, 50_000, 100_000]
DISTANCE_LABELS = ["0–2 kb", "2–10 kb", "10–50 kb", "50–100 kb"]
DISTANCE_INKS   = [_BOLD["cat1"], _BOLD["cat2"], _BOLD["cat3"], _BOLD["cat4"]]


def heatmap_cmap():
    return palette.get_heatmap_cmap()


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def style_axes(ax):
    pass  # handled by palette.use_style() / pastl.mplstyle
