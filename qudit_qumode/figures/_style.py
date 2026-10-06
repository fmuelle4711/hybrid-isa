"""Shared Nature-style figure defaults (matches figures/fig6_three_way.py).

Import at the top of every paper figure script:

    from figures._style import (
        apply_style, CHROMO_COLORS, SOLVER_STYLES,
        open_frame, panel_header,
    )
    apply_style()

Keeps fig 4 (fig6_three_way) and figs 5-8 visually consistent in font,
palette, spines, and grid without duplicating rcParams everywhere.
"""

import matplotlib.pyplot as plt


# Muted Okabe-Ito palette, chromophore-color-coded
CHROMO_COLORS = {
    "A": "#3B6DA5",  # muted blue
    "B": "#C86A2A",  # muted vermillion
    "C": "#2E8A66",  # muted bluish green
}
CHROMO_MARKERS = {"A": "o", "B": "s", "C": "^"}

# Consistent marker + line sizes across paper figs 5-8.
LINEWIDTH  = 0.9    # data-curve linewidth
MARKERSIZE = 4.0    # scatter/plot marker size
MARKEREDGE = 0.6    # marker edge width

# Marker downsampling / staggering strategy for figs 5-7 (overlaid Vu +
# Method A + Method B curves).  Each solver uses `markevery=(offset,
# MARKER_STRIDE)`.  The stride is chosen large enough that the ~200-point
# sampled data grid renders roughly a dozen markers per curve rather
# than one per sample; offsets 0, MARKER_STRIDE//3, 2*MARKER_STRIDE//3
# stagger the three methods' markers by 1/3 of a stride so no marker
# type is systematically hidden behind another after staggering.
MARKER_STRIDE = 15
_S = MARKER_STRIDE
SOLVER_STYLES = {
    "vu":    {"ls": "--", "marker": "o", "lw": LINEWIDTH,
              "markevery": (0,          _S), "label": "Vu published"},
    "qudit": {"ls": "-",  "marker": "^", "lw": LINEWIDTH,
              "markevery": (_S // 3,    _S), "label": "Method A (qudit)"},
    "qubit": {"ls": ":",  "marker": "s", "lw": LINEWIDTH,
              "markevery": (2*_S // 3,  _S), "label": "Method B (qubit)"},
}

# Two encodings (used in fig 8 convergence)
ENCODING_COLORS = {"qudit": "#3B6DA5", "qubit": "#C86A2A"}
ENCODING_MARKERS = {"qudit": "^", "qubit": "s"}


def apply_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.titlesize":    8,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "grid.linewidth":    0.3,
        "grid.color":        "#B8B8B8",
        "lines.linewidth":   LINEWIDTH,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def open_frame(ax, xlim=None, ylim=(-0.03, 1.03),
               yticks=(0.0, 0.25, 0.5, 0.75, 1.0),
               xticks=None,
               ygrid=True):
    """Remove top/right spines; thin the rest; faint dotted y-grid only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if xticks is not None:
        ax.set_xticks(list(xticks))
    ax.tick_params(axis="both", direction="out")
    if ygrid:
        ax.yaxis.grid(True, alpha=0.4, linestyle=":")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def panel_header(ax, letter, subtitle=""):
    """Bold panel letter (outside upper-left) + optional bold subtitle."""
    ax.text(-0.02, 1.12, letter,
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            ha="right", va="bottom")
    if subtitle:
        ax.set_title(subtitle, loc="left", pad=4, fontweight="bold")
