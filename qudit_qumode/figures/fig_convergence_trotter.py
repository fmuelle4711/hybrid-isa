"""fig_convergence_trotter.py -- Second-order Trotter convergence (fig 8).

Nature-style, matches figures/fig6_three_way.py (fig 4 in the paper).
Single log-log panel: mean RMSE vs Trotter step for both encodings.

Output: figures/fig_convergence_trotter.{png,pdf}
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG  = os.path.join(ROOT, "figures")

from figures._style import (
    apply_style, ENCODING_COLORS, ENCODING_MARKERS,
    MARKERSIZE, MARKEREDGE, LINEWIDTH,
)

ENCODINGS = ["qubit", "qudit"]


def load_runs(encoding, cutoff):
    pat = os.path.join(DATA, f"convergence_trotter_{encoding}_tau*fs_N{cutoff}.npz")
    rows = []
    for path in glob.glob(pat):
        m = re.search(r"_tau(\d+)fs_", path)
        if not m:
            continue
        tau_fs = int(m.group(1))
        d = np.load(path)
        rmse_mean = float(np.mean([d["rmse_A"], d["rmse_B"], d["rmse_C"]]))
        rows.append((tau_fs, rmse_mean,
                     float(d["rmse_A"]), float(d["rmse_B"]), float(d["rmse_C"])))
    rows.sort()
    return rows


def fit_loglog(taus, rmses):
    if len(taus) < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.log10(np.asarray(taus, float))
    y = np.log10(np.asarray(rmses, float))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def _open_frame_loglog(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", direction="out")
    ax.grid(True, which="both", alpha=0.35, linestyle=":")
    ax.set_axisbelow(True)


def main():
    cutoff = 8
    apply_style()

    # Two-column revtex4-2 width; single-panel plot.
    fig, ax = plt.subplots(figsize=(3.8, 3.0))

    fits = {}
    for enc in ENCODINGS:
        rows = load_runs(enc, cutoff)
        if not rows:
            print(f"  WARN: no data for encoding={enc}"); continue
        taus = [r[0] for r in rows]
        rmses = [r[1] for r in rows]
        slope_full, b_full, r2_full = fit_loglog(taus, rmses)
        upper = [(t, r) for t, r in zip(taus, rmses) if t >= 5]
        if len(upper) >= 2:
            slope_asy, b_asy, r2_asy = fit_loglog(
                [u[0] for u in upper], [u[1] for u in upper])
        else:
            slope_asy = b_asy = r2_asy = float("nan")
        fits[enc] = dict(taus=taus, rmses=rmses,
                         slope_full=slope_full, intercept_full=b_full, r2_full=r2_full,
                         slope_asy=slope_asy, intercept_asy=b_asy, r2_asy=r2_asy)

        ax.loglog(taus, rmses,
                  color=ENCODING_COLORS[enc],
                  marker=ENCODING_MARKERS[enc],
                  lw=LINEWIDTH, ms=MARKERSIZE,
                  markerfacecolor="none", markeredgewidth=MARKEREDGE,
                  label=(rf"{enc}   slope = {slope_asy:.2f}"))
        if not np.isnan(slope_asy):
            x_fit = np.array([5, max(taus)], float)
            y_fit = 10 ** (slope_asy * np.log10(x_fit) + b_asy)
            ax.loglog(x_fit, y_fit, ls=":", color=ENCODING_COLORS[enc],
                      lw=0.7, alpha=0.7)

    # Reference slope-2 line
    if fits:
        ref_tau = np.array([5, 30], float)
        any_enc = next(iter(fits.values()))
        anchor_r = any_enc["rmses"][-1]; anchor_t = any_enc["taus"][-1]
        ref_y = anchor_r * (ref_tau / anchor_t) ** 2
        ax.loglog(ref_tau, ref_y, ls="--", color="#7A7A7A",
                  lw=LINEWIDTH, alpha=0.8, label="reference  slope = 2")

    # The "sesolve integration floor" band that used to be drawn here at
    # 5e-5 is REMOVED.  It marked an artefact of the default integrator
    # tolerances (atol 1e-8 / rtol 1e-6), which put a floor at ~3.5e-5 on the
    # measured RMSE and biased the fitted slopes down to 1.80 / 1.53.  With
    # atol 1e-12 / rtol 1e-10 on both the Trotter steps and the exact
    # reference the floor is gone -- the tau = 1 fs point drops by ~16x and
    # the data is monotone and straight through that region -- so shading it
    # would now mislead.  See results/convergence_figure_audit.txt.

    ax.set_xlabel(r"Trotter step $\tau$  (fs)")
    ax.set_ylabel("mean RMSE  (Trotter vs exact)")
    _open_frame_loglog(ax)
    ax.legend(loc="lower right", frameon=False,
              handletextpad=0.4, borderaxespad=0.3)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out = os.path.join(FIG, f"fig_convergence_trotter.{ext}")
        plt.savefig(out, dpi=400, bbox_inches="tight")
        print(f"  Saved: {out}")
        out2 = os.path.join(DATA, f"fig_convergence_trotter.{ext}")
        plt.savefig(out2, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=== Trotter tau convergence summary ===")
    print(f"{'enc':<6s} {'tau_fs':<8s} {'mean RMSE':<14s} "
          f"{'rmse_A':<11s} {'rmse_B':<11s} {'rmse_C':<11s}")
    for enc in ENCODINGS:
        rows = load_runs(enc, cutoff)
        for tau_fs, mean_rmse, rA, rB, rC in rows:
            print(f"{enc:<6s} {tau_fs:<8d} {mean_rmse:<14.3e} "
                  f"{rA:<11.3e} {rB:<11.3e} {rC:<11.3e}")
    print()
    print("=== fit summary ===")
    print(f"{'enc':<6s} {'range':<14s} {'slope':<8s} {'prefactor':<12s} {'R^2':<8s}")
    for enc, f in fits.items():
        print(f"{enc:<6s} {'full':<14s} {f['slope_full']:<8.3f} "
              f"{10**f['intercept_full']:<12.2e} {f['r2_full']:<8.4f}")
        print(f"{enc:<6s} {'tau >= 5 fs':<14s} {f['slope_asy']:<8.3f} "
              f"{10**f['intercept_asy']:<12.2e} {f['r2_asy']:<8.4f}")


if __name__ == "__main__":
    main()
