"""circuit_diagrams_28.py -- New generalised 28-op qudit compilation.

Produces:
    figures/circuit_qudit_generalized_28_one_step.{png,pdf,svg}
    data/circuit_qudit_generalized_28_one_step.json

This file DOES NOT modify the existing 34-op path (circuit_diagrams.py); it is
a parallel implementation for validation.  It uses two new ISA symbols:

    QPS_gen  --  arbitrary-diagonal virtual-Z on the qudit (rank-3 here):
                 exp[-i (theta_A Pi_A + theta_B Pi_B + theta_C Pi_C)]
                 One AWG phase-register write covering three levels; 0 s
                 laser time.  Algebraically = product of the three rank-1
                 QPS gates (Section B of the 28-audit).

    QT_multi --  simultaneous multi-transition QT on the qudit:
                 U_seq = exp(-i theta_AB X_AB) exp(-i theta_AC X_AC)
                 EXACTLY reproduced by ONE scheduled multitone control
                 block (Section C1 of the 28-audit).  Matrix identical
                 to the sequential product; only the compilation label
                 changes.

Per half-step count:
    4 R  +  3 QCR  +  4 QCD  +  1 QPS_gen  +  1 QT_multi  =  13
Center block: 2 QT-SB.  Backward half-step: symmetric reverse.
Total: 13 + 2 + 13 = 28.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import params as P

FIG_DIR = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data")

TAU = 0.005e-12                                              # 5 fs
TAU_H = TAU / 2

_G, _A, _B, _C = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Abstract gate list
# ---------------------------------------------------------------------------
def make_qudit_generalized_28() -> tuple:
    """Return (fwd, center, bwd, gate_list_json, count).

    Each half-step:
        4 Rotation + 3 QCR + 4 QCD + 1 QPS_gen + 1 QT_multi = 13.
    Center 2 QT-SB.  Backward = reverse of forward.  Total = 28.
    """
    # Aggregated diagonal QPS coefficients (the Pi_A fused already includes
    # both omega_ea*S_a and omega_l*S_l as in the original make_qudit_d).
    theta_A_qps = (P.omega_ea * P.S_a + P.omega_l * P.S_l) * TAU_H
    theta_B_qps = P.omega_eb * P.S_b * TAU_H
    theta_C_qps = P.omega_ec * P.S_c * TAU_H

    theta_AB_qt = P.J_AB0 * TAU_H
    theta_AC_qt = P.J_AC0 * TAU_H

    fwd = [
        # 4 Rotation
        ("Rotation", ["a"], [P.omega_ga * TAU_H], "H_free"),
        ("Rotation", ["b"], [P.omega_gb * TAU_H], "H_free"),
        ("Rotation", ["c"], [P.omega_gc * TAU_H], "H_free"),
        ("Rotation", ["l"], [P.omega_l  * TAU_H], "H_free"),
        # 3 QCR (per chromophore-local mode)
        ("QCR", ["q", "a"], [-2 * P.chi_a * TAU_H, "level=A"], "H_QCR"),
        ("QCR", ["q", "b"], [-2 * P.chi_b * TAU_H, "level=B"], "H_QCR"),
        ("QCR", ["q", "c"], [-2 * P.chi_c * TAU_H, "level=C"], "H_QCR"),
        # 4 QCD (3 local + 1 mode-l on Pi_A)
        ("QCD", ["q", "a"], [P.omega_ea * float(np.sqrt(P.S_a)) * TAU_H,
                              "phi=pi/2", "level=A"], "H_QCD"),
        ("QCD", ["q", "b"], [P.omega_eb * float(np.sqrt(P.S_b)) * TAU_H,
                              "phi=pi/2", "level=B"], "H_QCD"),
        ("QCD", ["q", "c"], [P.omega_ec * float(np.sqrt(P.S_c)) * TAU_H,
                              "phi=pi/2", "level=C"], "H_QCD"),
        ("QCD", ["q", "l"], [P.omega_l  * float(np.sqrt(P.S_l)) * TAU_H,
                              "phi=pi/2", "level=A"], "H_QCD (mode-l on Pi_A)"),
        # 1 generalised diagonal QPS (replaces 3 rank-1 QPS)
        ("QPS_gen", ["q"], [
            theta_A_qps, theta_B_qps, theta_C_qps,
            "diagonal virtual-Z: exp[-i(theta_A Pi_A + theta_B Pi_B + theta_C Pi_C)]"
        ], "H_QPS diagonal (fusion of Pi_A, Pi_B, Pi_C)"),
        # 1 multi-transition QT (replaces 2 sequential QT)
        ("QT_multi", ["q"], [
            theta_AB_qt, theta_AC_qt,
            "C1: multitone that reproduces U_seq = exp(-i th_AB X_AB) exp(-i th_AC X_AC)"
        ], "H_QT multi (simultaneous A-B, A-C tones)"),
    ]
    center = [
        ("QT-SB", ["q", "l"], [P.J_AB0 * P.eta_AB * TAU,
                               "level_i=A,level_j=B,sideband"], "H_QT-SB"),
        ("QT-SB", ["q", "l"], [P.J_AC0 * P.eta_AC * TAU,
                               "level_i=A,level_j=C,sideband"], "H_QT-SB"),
    ]
    bwd = list(reversed(fwd))
    seq = fwd + center + bwd
    assert len(seq) == 28, f"expected 28 got {len(seq)}"

    gate_list = []
    for g, w, p, s in seq:
        gate_list.append(dict(
            gate=g, wires=w,
            params=[str(x) for x in p],
            source_term=s,
        ))
    return fwd, center, bwd, gate_list, len(seq)


# ---------------------------------------------------------------------------
# Custom matplotlib circuit renderer (sketch-style)
# ---------------------------------------------------------------------------
def _draw_wire(ax, y, x0, x1, name, color="#000"):
    ax.hlines(y, x0, x1, color=color, lw=1.2)
    ax.text(x0 - 0.2, y, name, ha="right", va="center", fontsize=11,
             weight="bold")


def _draw_gate_box(ax, x_c, y_c, w, h, label, colour="#FDE8CE",
                    edge="#7B4A17"):
    box = FancyBboxPatch((x_c - w / 2, y_c - h / 2), w, h,
                          boxstyle="round,pad=0.03",
                          fc=colour, ec=edge, lw=1.1)
    ax.add_patch(box)
    ax.text(x_c, y_c, label, ha="center", va="center", fontsize=8.5)


def _draw_two_wire_gate(ax, x_c, y_top, y_bot, w, label,
                         colour="#FDE8CE", edge="#7B4A17"):
    h = (y_top - y_bot) + 0.9
    y_c = (y_top + y_bot) / 2
    box = FancyBboxPatch((x_c - w / 2, y_c - h / 2), w, h,
                          boxstyle="round,pad=0.03",
                          fc=colour, ec=edge, lw=1.1)
    ax.add_patch(box)
    ax.text(x_c, y_c, label, ha="center", va="center", fontsize=8.5,
             rotation=90)


def _measure_symbol(ax, x_c, y_c, r=0.28):
    circ = Circle((x_c, y_c), r, fc="#FDE8CE", ec="#7B4A17", lw=1.1)
    ax.add_patch(circ)
    ax.plot([x_c - 0.15, x_c + 0.15], [y_c - 0.05, y_c + 0.15],
             color="#7B4A17", lw=1.1)


def draw_circuit(fwd, center, bwd, save_base):
    """Render the 28-op circuit with our own matplotlib primitives."""
    seq = fwd + center + bwd
    wires = ["q", "a", "b", "c", "l"]
    y_of = {w: len(wires) - 1 - i for i, w in enumerate(wires)}     # top->bottom
    n_gates = len(seq)

    # Compact width per gate
    w_gate = 0.9
    x_spacing = 1.1
    x0 = 1.6
    fig_w = x0 + x_spacing * (n_gates + 2)
    fig_h = 5.0

    fig, ax = plt.subplots(figsize=(fig_w * 0.55, fig_h), constrained_layout=True)
    ax.set_xlim(-0.5, fig_w)
    ax.set_ylim(-0.7, len(wires) - 0.3)
    ax.axis("off")

    # Draw wires
    for w in wires:
        _draw_wire(ax, y_of[w], 1.0, fig_w - 1.0, w)
    # Qudit shape marker on the LHS (a circle for the qudit q)
    _measure_symbol(ax, 0.5, y_of["q"])
    for w in ("a", "b", "c", "l"):
        ax.plot(0.5, y_of[w], marker="v", ms=13, color="#B8CFB8",
                 mec="#5A805A", mew=0.9)

    # Colour scheme by gate type
    palette = {
        "Rotation": ("#F0E5D8", "#7B7B7B"),
        "QCR":      ("#FDE8CE", "#7B4A17"),
        "QCD":      ("#FDE8CE", "#7B4A17"),
        "QPS_gen":  ("#FCE0A6", "#7B4A17"),
        "QT_multi": ("#F7CDBE", "#7B4A17"),
        "QT-SB":    ("#F1D1E6", "#7B4A17"),
    }

    # (label handling inline in the loop below)
    for k, (g, wr, params, _src) in enumerate(seq):
        x_c = x0 + k * x_spacing
        fc, ec = palette.get(g, ("#EEE", "#333"))
        # Build label
        if g == "Rotation":
            label = "R"
        elif g == "QCR":
            lv = [p for p in params if isinstance(p, str) and p.startswith("level=")][0][-1]
            label = f"QCR$_{{{lv}}}$"
        elif g == "QCD":
            lv = [p for p in params if isinstance(p, str) and p.startswith("level=")][0][-1]
            label = f"QCD$_{{{lv}}}$"
        elif g == "QPS_gen":
            label = "QPS$_{gen}$"
        elif g == "QT_multi":
            label = "QT$_{multi}$"
        elif g == "QT-SB":
            pair = "1,2" if "level_j=B" in params[-1] else "1,3"
            label = f"QTsb$_{{{pair}}}$"
        else:
            label = g

        if len(wr) == 1:
            _draw_gate_box(ax, x_c, y_of[wr[0]], w_gate, 0.7, label,
                            colour=fc, edge=ec)
        else:
            y_top = max(y_of[w] for w in wr)
            y_bot = min(y_of[w] for w in wr)
            _draw_two_wire_gate(ax, x_c, y_top, y_bot, w_gate, label,
                                 colour=fc, edge=ec)

    # Measurement icons on the right
    for w in wires:
        _measure_symbol(ax, fig_w - 0.7, y_of[w])

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{save_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    _ = ap.parse_args()
    print("Building the 28-op generalised qudit compilation...")
    fwd, center, bwd, gate_list, count = make_qudit_generalized_28()
    print(f"  gate count: {count}  (fwd={len(fwd)}, center={len(center)}, "
          f"bwd={len(bwd)})")
    by_kind = {}
    for g in gate_list:
        by_kind[g["gate"]] = by_kind.get(g["gate"], 0) + 1
    print(f"  by primitive: {dict(sorted(by_kind.items()))}")

    fig_path = os.path.join(FIG_DIR, "circuit_qudit_generalized_28_one_step")
    draw_circuit(fwd, center, bwd, fig_path)
    print(f"  saved: {fig_path}.{{png,pdf,svg}}")

    data_path = os.path.join(DATA_DIR, "circuit_qudit_generalized_28_one_step.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(dict(implementation="qudit_generalized_28",
                        gate_count_total=count,
                        gate_count_actual=len(gate_list),
                        by_primitive=by_kind,
                        gates=gate_list), f, indent=2, default=str)
    print(f"  saved: {data_path}")


if __name__ == "__main__":
    main()
