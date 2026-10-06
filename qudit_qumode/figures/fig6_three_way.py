"""
fig6_three_way.py  --  Three-panel closed-system comparison (Nature-style).

Same data as before -- no physics changes. Layout compressed to a single
1x3 row:

  a  Published reference        (Vu H_cqed Trotter, idealSim_Fock8_10k.out)
  b  Qubit compilation          (Method B, one-hot 3-qubit encoding of H_cqed)
  c  Qudit compilation          (Method A, 4-level qudit encoding of H_lab)

The previous "Method A (reduced)" (fused-QPS) panel has been dropped --
it is algebraically identical to Method A and added no information.

Style: sans-serif Arial-fallback ~7-8 pt, open frame (no top/right
spines), muted Okabe-Ito palette, thin curves + small unfilled markers,
faint dotted y-grid only, single horizontal legend at the bottom.
Two-column revtex4-2 width, vector PDF.

Run:
    python -m figures.fig6_three_way
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
FIG_DIR    = os.path.join(ROOT_DIR, "figures")


def dp(f):
    return os.path.join(DATA_DIR, f)


# ------------------------------------------------------------------ #
# Palette -- muted Okabe-Ito                                          #
# ------------------------------------------------------------------ #
cols = {
    "A": "#3B6DA5",   # muted blue
    "B": "#C86A2A",   # muted vermillion
    "C": "#2E8A66",   # muted bluish green
}
mk = {"A": "o", "B": "s", "C": "^"}


# ------------------------------------------------------------------ #
# LOAD DATA                                                           #
# ------------------------------------------------------------------ #

vu_path = dp("vu_reference/idealSim_Fock8_10k.out")
print(f"Loading Vu Trotter: {vu_path}")
with open(vu_path) as f:
    lines = f.read().strip().split("\n")


def parse_line(line):
    return np.array(
        [complex(m.group()).real for m in re.finditer(r"\([^)]+\)", line)]
    )


t_vu = parse_line(lines[0])
P_vu = {ch: parse_line(lines[i + 1]) for i, ch in enumerate("ABC")}
if t_vu[-1] < 1e-6:
    t_vu *= 1e12
print(f"  {len(t_vu)} points")


def _load_method(npz_name):
    path = dp(npz_name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    d = np.load(path)
    return {
        "t":    d["t"],
        "t_ex": d["t_exact"],
        "P":    {"A": d["PA"], "B": d["PB"], "C": d["PC"]},
        "P_ex": {"A": d["PA_exact"], "B": d["PB_exact"], "C": d["PC_exact"]},
    }


bq = _load_method("trotter_method_b_qubit_fock8.npz")
aq = _load_method("trotter_method_a_fock8.npz")


# ------------------------------------------------------------------ #
# STYLE                                                               #
# ------------------------------------------------------------------ #

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.titlesize":    7,
    "axes.labelsize":    7.5,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6.5,
    "legend.fontsize":   7,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "grid.linewidth":    0.3,
    "grid.color":        "#B8B8B8",
    "lines.linewidth":   0.9,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Two-column width; height keeps panels roughly square.
FIG_W = 7.0
FIG_H = 2.7

fig, axes = plt.subplots(
    1, 3,
    figsize=(FIG_W, FIG_H),
    sharex=True, sharey=True,
    gridspec_kw=dict(wspace=0.10),
)


def _open_frame(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="both", direction="out")
    ax.yaxis.grid(True, alpha=0.4, linestyle=":")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _add_panel_header(ax, letter, method_name):
    """Bold panel letter outside the axes, top-left; short bold method
    name as a left-aligned title just above the axes."""
    ax.text(-0.02, 1.12, letter,
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            ha="right", va="bottom")
    ax.set_title(method_name, loc="left", pad=4, fontweight="bold")


def _plot_curves(ax, t_ex, P_ex, t_tr, P_tr):
    sp = max(1, len(t_tr) // 32)
    for ch in "ABC":
        ax.plot(t_ex, P_ex[ch],
                color=cols[ch], lw=0.85, alpha=0.95)
        ax.scatter(t_tr[::sp], P_tr[ch][::sp],
                   facecolors="none", edgecolors=cols[ch],
                   marker=mk[ch], s=6.5, lw=0.55)


def panel_missing(ax, letter, method_name, hint):
    ax.text(0.5, 0.5, f"data missing\n{hint}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=6, color="#999999")
    _open_frame(ax)
    _add_panel_header(ax, letter, method_name)


# Panel a: Published reference (Vu H_cqed Trotter)
ax = axes[0]
if bq is not None:
    for ch in "ABC":
        ax.plot(bq["t_ex"], bq["P_ex"][ch],
                color=cols[ch], lw=0.85, alpha=0.95)
sp_vu = max(1, len(t_vu) // 32)
for ch in "ABC":
    ax.scatter(t_vu[::sp_vu], P_vu[ch][::sp_vu],
               facecolors="none", edgecolors=cols[ch],
               marker=mk[ch], s=6.5, lw=0.55)
_open_frame(ax)
_add_panel_header(ax, "a", "Published reference")

# Panel b: Qubit compilation (Method B)
ax = axes[1]
if bq is not None:
    _plot_curves(ax, bq["t_ex"], bq["P_ex"], bq["t"], bq["P"])
    _open_frame(ax)
    _add_panel_header(ax, "b", "Qubit compilation")
else:
    panel_missing(ax, "b", "Qubit compilation", "run trotter_method_b_qubit")

# Panel c: Qudit compilation (Method A)
ax = axes[2]
if aq is not None:
    _plot_curves(ax, aq["t_ex"], aq["P_ex"], aq["t"], aq["P"])
    _open_frame(ax)
    _add_panel_header(ax, "c", "Qudit compilation")
else:
    panel_missing(ax, "c", "Qudit compilation", "run trotter_method_a")

# Y-axis label on leftmost panel only
axes[0].set_ylabel("Population")

# X-axis label under the center panel (visually centered under the row)
axes[1].set_xlabel("Time (ps)")

# Shared legend at the bottom, horizontal, no box
legend_handles = [
    Line2D([0], [0], color=cols["A"], lw=0.85, marker="o",
           markerfacecolor="none", markeredgecolor=cols["A"],
           markersize=3.5, label="chromophore A"),
    Line2D([0], [0], color=cols["B"], lw=0.85, marker="s",
           markerfacecolor="none", markeredgecolor=cols["B"],
           markersize=3.5, label="chromophore B"),
    Line2D([0], [0], color=cols["C"], lw=0.85, marker="^",
           markerfacecolor="none", markeredgecolor=cols["C"],
           markersize=3.5, label="chromophore C"),
    Line2D([0], [0], color="black", lw=0.85, label="exact"),
    Line2D([0], [0], color="black", lw=0, marker="o",
           markerfacecolor="none", markeredgecolor="black",
           markersize=3.5, label="Trotter"),
]

# Reserve room at the bottom for the legend; leave the axes untouched
# by tight_layout so the xlabel + legend can coexist cleanly.
plt.subplots_adjust(left=0.06, right=0.995, top=0.87,
                    bottom=0.28, wspace=0.10)

fig.legend(handles=legend_handles,
           loc="lower center",
           bbox_to_anchor=(0.5, 0.02),
           ncol=5, frameon=False,
           handletextpad=0.4, columnspacing=1.6)

for ext in ["png", "pdf"]:
    out = os.path.join(FIG_DIR, f"fig6_three_way.{ext}")
    plt.savefig(out, dpi=400, bbox_inches="tight")
    print(f"  Saved: {out}")
    out2 = dp(f"fig6_three_way.{ext}")
    plt.savefig(out2, dpi=400, bbox_inches="tight")

plt.close(fig)
print("Done.")
