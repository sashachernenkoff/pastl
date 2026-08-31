"""PASTL figure palette — single source of truth for all plot colours.

Two switchable palettes:
  "bold"  → Bold Controlled  (saturated)
  "muted" → Muted Academic   (desaturated)

MODEL_FILL / MODEL_INK: fixed PASTL (blue) vs. Enformer (rose) pairing, used
across all model-comparison figures regardless of which palette is active.

HEATMAP_HEX / get_heatmap_cmap(): continuous colormap for heatmaps (GWAS
consequence-class, PWAS), separate from the categorical palettes.
"""
from pathlib import Path

# Fixed PASTL vs. Enformer pairing - same across both palettes below.
MODEL_FILL = {
    "pastl":   "#008080",  # teal
    "enformer": "#C0392B",  # dusty rose
}
MODEL_INK = MODEL_FILL  # single-tone bars: fill and outline/ink are the same colour here

PALETTES = {
    "bold": {
        "cat1": "#008080",   # teal          - primary / PASTL
        "cat2": "#C0392B",   # brick red     - secondary / Enformer
        "cat3": "#1B4F72",   # deep blue     - tertiary
        "cat4": "#7D3C98",   # purple        - quaternary
        "cat5": "#B9770E",   # amber/ochre   - fifth
        "other": "#5D6D7E",  # slate grey    - background / other
    },
    "muted": {
        "cat1": "#5A8585",   # muted teal
        "cat2": "#B5654A",   # brick
        "cat3": "#3B4E6E",   # navy
        "cat4": "#8C7A9B",   # mauve
        "cat5": "#C9A24B",   # gold
        "other": "#A8A8A8",  # grey
    },
}
PALETTE_ORDER = ["cat1", "cat2", "cat3", "cat4", "cat5", "other"]

# Continuous colormap for heatmaps (GWAS consequence-class, PWAS, etc.).
# Kept separate from the categorical palettes above.
HEATMAP_HEX = [
    "#F8D76E",  # Pale Yellow/Gold
    "#ECA472",  # Peach/Orange
    "#CD6C9C",  # Pink
    "#984EA3",  # Magenta-Purple
    "#593895",  # Deep Purple
]

# Active palette (default: bold). Call set_palette() to switch.
_active = "bold"
FILL = dict(PALETTES[_active])
FILL["pastl"] = MODEL_FILL["pastl"]
FILL["enformer"] = MODEL_FILL["enformer"]
GROUP_FILLS = [FILL[k] for k in PALETTE_ORDER]


def set_palette(name):
    """
    Switch active palette: 'bold' or 'muted'. Call before use_style().

    IMPORTANT: this reassigns the module-level FILL/GROUP_FILLS names.
    If you did `from palette import FILL`, your local `FILL` will NOT
    update when you call this - `from x import y` snapshots the
    reference at import time. Either `import palette as pal` and use
    `pal.FILL` / `pal.GROUP_FILLS`, or re-import after switching.
    """
    global _active, FILL, GROUP_FILLS
    assert name in PALETTES, f"Unknown palette '{name}', choose from {list(PALETTES)}"
    _active = name
    FILL = dict(PALETTES[_active])
    FILL["pastl"] = MODEL_FILL["pastl"]
    FILL["enformer"] = MODEL_FILL["enformer"]
    GROUP_FILLS = [FILL[k] for k in PALETTE_ORDER]


def get_heatmap_cmap(reverse=False):
    """Build the continuous colormap from HEATMAP_HEX (gold -> deep purple)."""
    from matplotlib.colors import LinearSegmentedColormap
    hexes = list(reversed(HEATMAP_HEX)) if reverse else HEATMAP_HEX
    return LinearSegmentedColormap.from_list("pastl_heatmap", hexes)


BAR_WIDTH = 0.10  # thesis-wide default bar width / height for all bar plots


def bar_limits(n):
    """
    Equal-padding axis limits for a bar chart with n bars at integer positions.

    Edge padding = half the inter-bar gap, giving the pattern:
        8.33% | bar | 16.67% | bar | ... | bar | 8.33%   (for n = 6)

    Usage:
        ax.set_xlim(*palette.bar_limits(n))   # vertical bars
        ax.set_ylim(*palette.bar_limits(n))   # horizontal bars
    """
    return (-0.5, n - 0.5)


def grouped_bar_positions(n_groups, n_bars_per_group, bar_width=BAR_WIDTH, gap=0.03):
    """
    Compute x-offsets for grouped/comparison bar charts with SKINNY bars and
    a small gap between bars within each group.
    Returns (offsets, bar_width): offsets is a list to add to np.arange(n_groups).
    """
    import numpy as np
    total_width = n_bars_per_group * bar_width + (n_bars_per_group - 1) * gap
    start = -total_width / 2 + bar_width / 2
    offsets = [start + i * (bar_width + gap) for i in range(n_bars_per_group)]
    return offsets, bar_width


def square_legend(ax, labels, colors, **kwargs):
    """
    Draw a legend with guaranteed square colour swatches. Use this instead of
    ax.legend() for any bar plot - bar-patch legend handles inherit the
    bar's own (skinny) aspect ratio and will not become square from
    legend.handleheight in the style sheet alone, since that only rescales
    the handle box, not the shape drawn inside it.
    """
    import matplotlib.patches as mpatches
    handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors]
    kwargs.setdefault("handlelength", 1.0)
    kwargs.setdefault("handleheight", 1.0)
    return ax.legend(handles, labels, **kwargs)


def use_style():
    """
    Apply the PASTL matplotlib style sheet (spines/ticks/fonts/sizing).
    Does NOT set colours - call set_palette() separately, then apply the
    category cycle yourself via GROUP_FILLS/FILL, since colour cycling
    depends on which palette is active at call time.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    plt.style.use(Path(__file__).parent / "pastl.mplstyle")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=GROUP_FILLS)
