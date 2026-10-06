"""analyze_n4_results.py -- end-to-end validation report for the 12
fig*_hqml_*_N4_10000shots.npz combos against the Vu published reference.

For each (fig, method) in {6, 8, 9, 10} x {a_qudit, a_qudit_d, b_qubit}:

  * loads ``data/fig{N}_hqml_{method}_N4_10000shots.npz``;
  * loads the matching Vu .out file(s) from ``data/vu_reference/``
    (mapping from :mod:`figures._load_helper.VU_OUT_FILE`, plus
    ``idealSim_Fock8_10k.out`` for fig 6);
  * interpolates Vu onto our timepoint grid;
  * computes per-channel (P_A, P_B, P_C) RMSE and max-deviation against
    Vu, and an aggregate RMSE = sqrt((RMSE_A^2 + RMSE_B^2 + RMSE_C^2)/3).

Outputs (written under qudit_qumode/):
  * ``scripts/N4_validation_report.txt`` -- human-readable table + per-method
    aggregates and a "which method is closest to Vu" summary.
  * ``scripts/N4_validation_report.csv`` -- raw table for spreadsheet use.
  * ``figures/N4_method_comparison_fig8_top.png`` -- A/B/C populations on the
    fig8_top panel for all three methods overlaid with the Vu reference.

Run:
    python scripts/analyze_n4_results.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
VU_DIR   = os.path.join(ROOT, "data", "vu_reference")
FIG_DIR  = os.path.join(ROOT, "figures")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT_TXT  = os.path.join(SCRIPT_DIR, "N4_validation_report.txt")
REPORT_CSV  = os.path.join(SCRIPT_DIR, "N4_validation_report.csv")
SUMMARY_MD  = os.path.join(FIG_DIR, "N4_SUMMARY.md")
# Per-panel comparison plots: figures/N4_comparison_{fig}_{panel}.png
def panel_png(fig: int, panel: str) -> str:
    return os.path.join(FIG_DIR, f"N4_comparison_{fig}_{panel}.png")

METHODS = ("a_qudit", "a_qudit_d", "b_qubit")
FIGS    = (6, 8, 9, 10)
FIG_PANELS = {
    6:  ("fig6_closed",),
    8:  ("fig8_top", "fig8_middle", "fig8_bottom"),
    9:  ("fig9_top", "fig9_middle", "fig9_bottom"),
    10: ("fig10_combined", "fig10_damp_only"),
}
# Panel -> Vu .out reference (fig6 uses the universal closed-system Fock-8 idealSim)
VU_FOR_PANEL = {
    "fig6_closed":     "idealSim_Fock8_10k.out",
    "fig8_top":        "amp_3.15-3.15-3.15_Fock8_10k.out",
    "fig8_middle":     "amp_3.15-9.45-3.15_Fock8_10k.out",
    "fig8_bottom":     "amp_3.15-1.05-3.15_Fock8_10k.out",
    "fig9_top":        "dep_0.9-0.9-0.9_Fock8_10k.out",
    "fig9_middle":     "dep_0.9-2.7-0.9_Fock8_10k.out",
    "fig9_bottom":     "dep_0.9-0.3-0.9_Fock8_10k.out",
    "fig10_combined":  "amp_all3.15_dep_all0.9_Fock8_10k.out",
    "fig10_damp_only": "amp_3.15-3.15-3.15_Fock8_10k.out",
}
PRETTY_METHOD = {
    "a_qudit":   "A native qudit (qnode)",
    "a_qudit_d": "A QPS-fused qudit-d (qnode)",
    "b_qubit":   "B 3-qubit (qnode)",
}
# Method-comparison plot uses channel colour + per-source linestyle so all
# three populations from all three methods + Vu overlay on a single axes.
CHANNEL_COLORS = {"A": "#C0392B", "B": "#27AE60", "C": "#2980B9"}
METHOD_STYLE = {
    # method -> (linestyle, marker)
    "vu":        ("-",  None),   # Vu reference: thick solid, no marker
    "a_qudit":   ("--", "o"),
    "a_qudit_d": (":",  "s"),
    "b_qubit":   ("-.", "^"),
}


# ---------------------------------------------------------------- #
# Loaders                                                          #
# ---------------------------------------------------------------- #

def _parse_vu(path):
    """Parse Vu's complex-pair .out (rows: t, PA, PB, PC). Real parts only."""
    with open(path) as f:
        lines = f.read().strip().split("\n")
    def parse(line):
        return np.array([complex(m.group()).real for m in re.finditer(r"\([^)]+\)", line)])
    t  = parse(lines[0])
    PA = parse(lines[1]); PB = parse(lines[2]); PC = parse(lines[3])
    if t[-1] < 1e-6:                       # convert seconds -> ps if needed
        t *= 1e12
    return t, PA, PB, PC


def load_panel(fig: int, method: str, panel: str):
    """Return (t_ps, PA, PB, PC) for one (fig, method, panel) from the
    N4 10000-shot npz, or None if missing.
    Fig 6 npz has one trace (no panel prefix); dissipative npz has the
    {panel}_t/_PA/_PB/_PC schema.
    """
    npz_path = os.path.join(DATA_DIR, f"fig{fig}_hqml_{method}_N4_10000shots.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    if fig == 6:
        return d["t"], d["PA"], d["PB"], d["PC"]
    key_t = f"{panel}_t"
    if key_t not in d.files:
        return None
    return d[key_t], d[f"{panel}_PA"], d[f"{panel}_PB"], d[f"{panel}_PC"]


def load_vu_at_grid(panel: str, t_grid):
    """Load Vu reference + interp onto t_grid (ps). Returns (PA, PB, PC) or None."""
    vu_path = os.path.join(VU_DIR, VU_FOR_PANEL[panel])
    if not os.path.exists(vu_path):
        return None
    t_vu, PA, PB, PC = _parse_vu(vu_path)
    interp = lambda y: interp1d(t_vu, y, bounds_error=False, fill_value="extrapolate")(t_grid)
    return interp(PA), interp(PB), interp(PC)


# ---------------------------------------------------------------- #
# Metrics                                                          #
# ---------------------------------------------------------------- #

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))


def max_dev(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def aggregate(rA, rB, rC):
    return float(np.sqrt((rA**2 + rB**2 + rC**2) / 3))


# ---------------------------------------------------------------- #
# Walk the 12 combos                                               #
# ---------------------------------------------------------------- #

def build_table():
    """Return a list of dict rows, one per (fig, method, panel)."""
    rows = []
    for fig in FIGS:
        panels = FIG_PANELS[fig]
        for method in METHODS:
            for panel in panels:
                row = {"fig": fig, "method": method, "panel": panel,
                       "RMSE_A": None, "RMSE_B": None, "RMSE_C": None,
                       "agg":   None,
                       "max_A": None, "max_B": None, "max_C": None,
                       "n_pts": 0, "status": "ok"}
                rec = load_panel(fig, method, panel)
                if rec is None:
                    row["status"] = "npz_missing"
                    rows.append(row); continue
                t_ps, PA, PB, PC = rec
                vu = load_vu_at_grid(panel, t_ps)
                if vu is None:
                    row["status"] = "vu_missing"
                    rows.append(row); continue
                PA_vu, PB_vu, PC_vu = vu
                row["RMSE_A"] = rmse(PA, PA_vu)
                row["RMSE_B"] = rmse(PB, PB_vu)
                row["RMSE_C"] = rmse(PC, PC_vu)
                row["max_A"]  = max_dev(PA, PA_vu)
                row["max_B"]  = max_dev(PB, PB_vu)
                row["max_C"]  = max_dev(PC, PC_vu)
                row["agg"]    = aggregate(row["RMSE_A"], row["RMSE_B"], row["RMSE_C"])
                row["n_pts"]  = len(t_ps)
                rows.append(row)
    return rows


# ---------------------------------------------------------------- #
# Reporters                                                        #
# ---------------------------------------------------------------- #

def write_csv(rows, path):
    cols = ["fig", "method", "panel", "n_pts",
            "RMSE_A", "RMSE_B", "RMSE_C", "agg",
            "max_A", "max_B", "max_C", "status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})


def format_report(rows, shot_noise_floor: float, fock_floor: float):
    lines = []
    lines.append("=" * 96)
    lines.append("N=4, shots=10000  --  fig*_hqml_*_N4_10000shots.npz   vs   Vu Fock-8 10k reference")
    lines.append("=" * 96)
    lines.append(f"  shot-noise floor at 10000 shots ~ 1/sqrt(10000) = {shot_noise_floor:.4f}")
    lines.append(f"  Fock-truncation floor (N=4 vs N=8) ~ {fock_floor:.3f} (panel-dependent)")
    lines.append("")
    hdr = (f"{'fig':>3s} {'method':>10s} {'panel':>16s} {'n_pts':>5s}  "
           f"{'RMSE_A':>7s} {'RMSE_B':>7s} {'RMSE_C':>7s} {'agg':>7s}   "
           f"{'max_A':>7s} {'max_B':>7s} {'max_C':>7s}   status")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['fig']:>3d} {r['method']:>10s} {r['panel']:>16s} {r['n_pts']:>5d}  "
                         f"{'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s}   "
                         f"{'-':>7s} {'-':>7s} {'-':>7s}   {r['status']}")
        else:
            lines.append(f"{r['fig']:>3d} {r['method']:>10s} {r['panel']:>16s} {r['n_pts']:>5d}  "
                         f"{r['RMSE_A']:>7.4f} {r['RMSE_B']:>7.4f} {r['RMSE_C']:>7.4f} {r['agg']:>7.4f}   "
                         f"{r['max_A']:>7.4f} {r['max_B']:>7.4f} {r['max_C']:>7.4f}   ok")

    # ---- aggregates per (fig, method) ----
    lines.append("")
    lines.append("Per-(fig, method) aggregate RMSE (mean across panels):")
    fm_agg = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            fm_agg[(r["fig"], r["method"])].append(r["agg"])
    for (fig, method) in sorted(fm_agg):
        mean = float(np.mean(fm_agg[(fig, method)]))
        lines.append(f"  fig {fig:>2d}  {method:>10s}  mean(agg) = {mean:.4f}  "
                     f"(n_panels = {len(fm_agg[(fig, method)])})")

    # ---- aggregates per method, across all figs ----
    lines.append("")
    lines.append("Per-method overall aggregate RMSE (mean of all panel-aggregates):")
    method_agg = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            method_agg[r["method"]].append(r["agg"])
    best = None
    for method in METHODS:
        if method not in method_agg:
            continue
        m = float(np.mean(method_agg[method]))
        lines.append(f"  {method:>10s}  {PRETTY_METHOD[method]:<32s}  overall mean(agg) = {m:.4f}  "
                     f"(n_panels = {len(method_agg[method])})")
        if best is None or m < best[1]:
            best = (method, m)
    if best is not None:
        lines.append("")
        lines.append(f"==> Closest method to Vu (lowest mean aggregate RMSE across all panels):")
        lines.append(f"        {best[0]}  ({PRETTY_METHOD[best[0]]})  overall mean(agg) = {best[1]:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- #
# Method-comparison plot for fig8_top                              #
# ---------------------------------------------------------------- #

def plot_panel_overlay(fig_id: int, panel: str) -> str | None:
    """One panel, single axes: A/B/C populations from Vu + all three
    Hybridlane methods.

    Colour encodes channel (A red / B green / C blue, matching Vu's
    convention); line style + marker encodes source (Vu thick solid /
    a_qudit dashed-o / a_qudit_d dotted-s / b_qubit dashdot-^).

    Returns the relative-to-FIG_DIR path of the saved PNG (or None if
    no traces could be loaded for the panel).
    """
    out_path = panel_png(fig_id, panel)
    fig, ax = plt.subplots(figsize=(9, 6))

    drew_any = False

    # Vu reference (thick solid, dense native Vu grid)
    if panel in VU_FOR_PANEL:
        vu_path = os.path.join(VU_DIR, VU_FOR_PANEL[panel])
        if os.path.exists(vu_path):
            t_vu, PA_vu, PB_vu, PC_vu = _parse_vu(vu_path)
            ls_vu, _ = METHOD_STYLE["vu"]
            for ch, y in zip("ABC", (PA_vu, PB_vu, PC_vu)):
                ax.plot(t_vu, y, ls_vu, color=CHANNEL_COLORS[ch], lw=2.2,
                        label=f"Vu Fock-8 10k  P_{ch}", zorder=5)
            drew_any = True

    # Three Hybridlane qnode methods
    for method in METHODS:
        rec = load_panel(fig_id, method, panel)
        if rec is None:
            continue
        t_ps, PA, PB, PC = rec
        ls, mk = METHOD_STYLE[method]
        markevery = max(1, len(t_ps) // 20)
        for ch, y in zip("ABC", (PA, PB, PC)):
            ax.plot(t_ps, y, ls, color=CHANNEL_COLORS[ch], lw=1.4,
                    marker=mk, ms=4.5, markevery=markevery, alpha=0.85,
                    label=f"{PRETTY_METHOD[method]}  P_{ch}")
        drew_any = True

    if not drew_any:
        plt.close(fig)
        return None

    ax.set_xlim(0, 2.05); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Population")
    ax.set_title(f"Fig {fig_id} {panel}: A/B/C overlayed, all methods\n"
                 f"(N=4, shots=10000)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_panel_plots() -> list[tuple[int, str, str]]:
    """Render one comparison PNG per (fig, panel) combo. Returns
    [(fig_id, panel, png_path_or_None), ...]."""
    out = []
    for fig_id in FIGS:
        for panel in FIG_PANELS[fig_id]:
            png = plot_panel_overlay(fig_id, panel)
            out.append((fig_id, panel, png))
    return out


def write_summary_md(panel_plots: list[tuple[int, str, str]],
                     rows: list[dict],
                     report_text: str) -> None:
    """Embed every panel PNG (relative to figures/) into a single markdown
    summary alongside the per-panel RMSE table and the per-method
    aggregate ranking."""
    lines = []
    lines.append("# N=4 / 10 000-shot validation summary")
    lines.append("")
    lines.append("All 9 dissipative + closed-system panels at Fock cutoff "
                 "N=4 with 10 000 shots, three Hybridlane qnode methods "
                 "compared against Vu's published Fock-8 / 10 000-shot "
                 "reference. Per-panel comparison plots overlay populations "
                 "P_A (red), P_B (green) and P_C (blue) from all four "
                 "sources on a single axes.")
    lines.append("")
    lines.append("**Source legend (encoded as line style + marker):**")
    lines.append("")
    lines.append("| Source | Line | Marker |")
    lines.append("|---|---|---|")
    lines.append("| Vu Fock-8 10k reference | solid (thick) | — |")
    lines.append("| Method A native qudit (qnode) | dashed | `o` |")
    lines.append("| Method A QPS-fused qudit-d (qnode) | dotted | `s` |")
    lines.append("| Method B 3-qubit (qnode) | dash-dot | `^` |")
    lines.append("")

    # ---- per-panel plots ----
    lines.append("## Per-panel plots")
    lines.append("")
    for fig_id, panel, png in panel_plots:
        if png is None:
            lines.append(f"### Fig {fig_id} – {panel}")
            lines.append("")
            lines.append("_No data available for this panel._")
            lines.append("")
            continue
        rel = os.path.relpath(png, FIG_DIR).replace(os.sep, "/")
        lines.append(f"### Fig {fig_id} – {panel}")
        lines.append("")
        lines.append(f"![{panel}]({rel})")
        lines.append("")

    # ---- RMSE table ----
    lines.append("## Per-panel RMSE vs Vu")
    lines.append("")
    lines.append("Aggregate is `sqrt((RMSE_A^2 + RMSE_B^2 + RMSE_C^2) / 3)`. "
                 "Shot-noise floor at 10 000 shots is `1/sqrt(10000) ≈ 0.01`. "
                 "The Fock-truncation floor (N=4 vs Vu's N=8) is panel-"
                 "dependent — most panels sit within a few %.")
    lines.append("")
    hdr = ("| fig | method | panel | RMSE_A | RMSE_B | RMSE_C | "
           "**agg** | max_A | max_B | max_C | status |")
    sep = "|---" * 11 + "|"
    lines.append(hdr); lines.append(sep)
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['fig']} | {r['method']} | {r['panel']} | "
                         f"– | – | – | – | – | – | – | {r['status']} |")
        else:
            lines.append(
                f"| {r['fig']} | {r['method']} | {r['panel']} | "
                f"{r['RMSE_A']:.4f} | {r['RMSE_B']:.4f} | {r['RMSE_C']:.4f} | "
                f"**{r['agg']:.4f}** | "
                f"{r['max_A']:.4f} | {r['max_B']:.4f} | {r['max_C']:.4f} | ok |")
    lines.append("")

    # ---- per-method overall ranking ----
    lines.append("## Per-method overall aggregate (mean across all 9 panels)")
    lines.append("")
    method_agg = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            method_agg[r["method"]].append(r["agg"])
    lines.append("| Method | Description | mean(agg) | n_panels |")
    lines.append("|---|---|---|---|")
    ranking = []
    for method in METHODS:
        if method not in method_agg:
            continue
        m = float(np.mean(method_agg[method]))
        ranking.append((method, m))
        lines.append(f"| `{method}` | {PRETTY_METHOD[method]} | "
                     f"{m:.4f} | {len(method_agg[method])} |")
    if ranking:
        best = min(ranking, key=lambda x: x[1])
        lines.append("")
        lines.append(f"**Closest to Vu overall:** `{best[0]}` "
                     f"({PRETTY_METHOD[best[0]]}) at mean(agg) = "
                     f"`{best[1]:.4f}`.")
    lines.append("")

    # ---- raw report appendix ----
    lines.append("## Raw text report")
    lines.append("")
    lines.append("```text")
    lines.append(report_text)
    lines.append("```")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- #
# Driver                                                           #
# ---------------------------------------------------------------- #

def main():
    rows = build_table()
    write_csv(rows, REPORT_CSV)
    report = format_report(rows, shot_noise_floor=1.0 / np.sqrt(10000),
                           fock_floor=0.02)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")

    panel_plots = generate_all_panel_plots()
    write_summary_md(panel_plots, rows, report)

    print(report)
    print()
    print(f"Wrote: {REPORT_TXT}")
    print(f"Wrote: {REPORT_CSV}")
    for fig_id, panel, png in panel_plots:
        marker = png if png else "(no data)"
        print(f"Wrote: {marker}")
    print(f"Wrote: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
