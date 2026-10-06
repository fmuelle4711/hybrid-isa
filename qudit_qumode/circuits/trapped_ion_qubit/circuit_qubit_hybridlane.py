"""One-hot QUBIT-encoding Trotter step, drawn with HybridLane / PennyLane.

The existing figures/circuit_qubit_one_step.* are hand-drawn matplotlib. This
script builds the same circuit as a real hqml QNode and renders it with
PennyLane's own drawer (qml.draw_mpl / qml.draw), so the picture is produced by
the framework rather than by bespoke plotting code.

Circuit is the one-hot Method-B Trotter step from
simulations/vibronic_scaling.py::build_one_hot_trotter_step, gate for gate:

    det_fwd, chi_fwd, cd_fwd, j_fwd, jl_mid, j_rev, cd_rev, chi_rev, det_rev

with counts  chi 2Nc | cd 2(Nc+1) | Rotation 2(Nc+1) | rz 2(Nc-1)
             J 2(Nc-1) | Jl (Nc-1)          total = 11 Nc - 1.

VARIANTS -- REMOVED 2026-09-09.  One figure is emitted:
  circuit_qubit_hybridlane_one_step.*           full step, all gates drawn

  Three reduced variants used to be produced alongside it (_norz with the
  virtual-Z gates dropped, _noh with the Hadamards dropped, _norz_noh with
  both), together with two diagnostics -- rz_frame_check and h_basis_check --
  whose only job was to bound what those reductions did and did not claim.
  All of it is gone, for two reasons:

    stale     every reduced figure on disk was rendered from the pre-b1f8ee6
              jl_block, which emitted two Jl sandwiches instead of four and so
              drew XX (x) x_l where the Hamiltonian has (XX + YY) (x) x_l.
              They were wrong in the same way the main figure was, and nothing
              referenced them -- no .tex, no .md, no results file.
    unsafe    two of the three were labelled SCHEMATIC ONLY in this docstring
              because h_basis_check measured the Jl qubit-side generator
              changing from +1.000*XX to +1.000*ZZ when the Hadamards are
              removed.  A figure that does not implement H_Jl, sitting in the
              figures/ directory next to one that does, is a trap.

  If a reduced view is ever wanted again, regenerate it from the CURRENT
  jl_block rather than restoring these files, and note that `include_rz` only
  ever guarded the per-chromophore detuning RZ gates -- the RZ(-+pi/2) basis
  rotations that turn sigma_x into sigma_y inside the Jl sandwiches were never
  optional and were never dropped by any variant.

GATE ORDER -- det_block moved to the front (2026-09-01)
  The mode-free-evolution R gates (hqml.Rotation) used to sit third, in the
  middle of the forward half-step.  They are now first, so every R gate is at
  the start of the drawing.  This is legitimate, not cosmetic licence:

    * A symmetric second-order Suzuki step is second-order accurate for ANY
      ordering of the sub-Hamiltonians in the forward half, provided the second
      half is the EXACT reverse of the first.  Forward is now
      (det, chi, cd, j) and the reverse half is (j, cd, chi, det), so the
      mirror symmetry -- the thing the order actually has to preserve -- holds.
      The two orderings differ at O(tau^3), the same order as the Trotter error
      already being made.
    * Bonus: with det first and its mirror last, the trailing det(tau/2) of
      step k and the leading det(tau/2) of step k+1 become adjacent and merge
      into a single det(tau) across a multi-step sequence.

  Note the R gates still appear at BOTH ends: the reversed half-step must end
  with det_rev or the step stops being symmetric.  Only the forward block moved.

  DIVERGENCE, stated rather than hidden: build_one_hot_trotter_step still emits
  the old order (chi, cd, det, j | jl | ...), so this drawing no longer matches
  it gate-for-gate in ORDER.  Gate COUNTS are unchanged (11Nc-1, and every
  per-type count above).  The builder was deliberately left alone because
  SymCircuit.depth() (vibronic_scaling.py:84-94) is a greedy layer-packer and
  is therefore order-sensitive, so re-ordering it could move published depth
  numbers -- that is a separate decision, not a figure tweak.

PRIMITIVE MAPPING (stated so the drawing is auditable)
  chi   sigma_z^R (x) n_R      -> hqml.ConditionalRotation      [native]
  cd    sigma_z^R (x) x_R      -> hqml.ConditionalDisplacement  [native]
  Rot   mode free evolution    -> hqml.Rotation                 [native]
  rz    qubit virtual-Z        -> qml.RZ                        [native, T=0]
  J     2-qubit XY exchange    -> qml.IsingXY                   [native]
  Jl    2-qubit exchange (x) x_l -> NO single native primitive.
        Decomposed as H-CNOT sandwich around hqml.ConditionalDisplacement on
        the link mode, which is the decomposition already used in this repo by
        simulations/circuit_fig89_10_hqml_qnode_b_qubit.py (lines ~137-146).
        Every gate of that decomposition is drawn; nothing is hidden in a box.

Angles come from params.py and the same TAU as simulations/circuit_diagrams.py.
Nothing is invented; the drawing is structural, so angle values only set labels.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pennylane as qml
import hybridlane as hqml

HERE = Path(__file__).resolve().parent
# Repo root = the directory holding params.py.  Found by walking up rather
# than by counting levels, so moving this file between subfolders (as on
# 2026-09-09, scripts/ -> circuits/ -> circuits/trapped_ion_qubit/) does not
# silently break the imports.
ROOT = next(p for p in HERE.parents if (p / "params.py").exists())
sys.path.insert(0, str(ROOT))
import params as P  # noqa: E402

# Outputs live beside this script, not in the repo-wide figures/ and results/
# directories (moved 2026-09-09).  The circuit drawings are a self-contained
# set -- two levels of the same step plus their equivalence proof -- and are
# not inputs to any table or panel.
OUTDIR = HERE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
TXTDIR = HERE / "results"
TXTDIR.mkdir(parents=True, exist_ok=True)

# same convention as simulations/circuit_diagrams.py
TAU = 0.005e-12          # 5 fs, Hamiltonian in Hz
TAU_H = TAU / 2

NC = 3                   # 3 chromophores -> 11*3 - 1 = 32 gates

QB = [f"q{r}" for r in range(NC)]
ML = [f"mL{r}" for r in range(NC)]
LINK = "link"
WIRE_ORDER = QB + ML + [LINK]

# per-chromophore couplings, reused from params (indices A,B,C)
CHI = [P.chi_a, P.chi_b, P.chi_c][:NC]
OMEGA_E = [P.omega_ea, P.omega_eb, P.omega_ec][:NC]
S_LOC = [P.S_a, P.S_b, P.S_c][:NC]
OMEGA_M = [P.omega_ga, P.omega_gb, P.omega_gc][:NC]
J0 = [P.J_AB0, P.J_AC0]
ETA_J = [P.eta_AB, P.eta_AC]


def one_hot_trotter_step():
    """Exactly build_one_hot_trotter_step, in hqml primitives.

    Every gate is drawn.  The include_rz / include_h switches that used to
    produce reduced variants were removed on 2026-09-09; see the VARIANTS
    note in the module docstring.
    """

    def chi_block(rev=False):
        rng = reversed(range(NC)) if rev else range(NC)
        for r in rng:
            hqml.ConditionalRotation(-2 * CHI[r] * TAU_H, wires=[QB[r], ML[r]])

    def cd_block(rev=False):
        if not rev:
            for r in range(NC):
                hqml.ConditionalDisplacement(
                    OMEGA_E[r] * np.sqrt(S_LOC[r]) * TAU_H, np.pi / 2,
                    wires=[QB[r], ML[r]])
            hqml.ConditionalDisplacement(
                P.omega_l * np.sqrt(P.S_l) * TAU_H, np.pi / 2,
                wires=[QB[0], LINK])
        else:
            hqml.ConditionalDisplacement(
                P.omega_l * np.sqrt(P.S_l) * TAU_H, np.pi / 2,
                wires=[QB[0], LINK])
            for r in reversed(range(NC)):
                hqml.ConditionalDisplacement(
                    OMEGA_E[r] * np.sqrt(S_LOC[r]) * TAU_H, np.pi / 2,
                    wires=[QB[r], ML[r]])

    def det_block(rev=False):
        modes = ML + [LINK]
        if not rev:
            for m, w in zip(modes, OMEGA_M + [P.omega_l]):
                hqml.Rotation(w * TAU_H, wires=m)
            for r in range(1, NC):
                qml.RZ(OMEGA_E[r] * S_LOC[r] * TAU_H, wires=QB[r])
        else:
            for r in reversed(range(1, NC)):
                qml.RZ(OMEGA_E[r] * S_LOC[r] * TAU_H, wires=QB[r])
            for m, w in zip(reversed(modes), reversed(OMEGA_M + [P.omega_l])):
                hqml.Rotation(w * TAU_H, wires=m)

    def j_block(rev=False):
        """J = (XX + YY)/2 per pair, drawn as TWO Molmer-Sorensen windows.

        One bichromatic MS gives sigma_phi (x) sigma_phi for a single phase
        setting, i.e. XX or YY, never their sum.  Since [XX, YY] = 0 the
        split into two windows is EXACT -- it is forced by the hardware, not
        by Trotterisation.  Previously drawn as one qml.IsingXY, which is a
        correct logical gate but not a pulse sequence, and which made the
        figure show 4 entangling boxes where the hardware needs 8.
        """
        rng = reversed(range(1, NC)) if rev else range(1, NC)
        for r in rng:
            axes = (qml.IsingYY, qml.IsingXX) if rev else (qml.IsingXX,
                                                           qml.IsingYY)
            for G in axes:
                G(J0[r - 1] * TAU_H, wires=[QB[0], QB[r]])

    def jl_block():
        """Jl: (|A><R| + h.c.) (x) x_link = (XX + YY)/2 (x) x_link.

        FOUR terms, not two.  g_abl and g_acl each appear once in H2XX and
        once in H2YY, and each needs its OWN H-CNOT sandwich around a
        ConditionalDisplacement on the link mode, because one sandwich
        rotates a single Pauli axis into sigma_z.  The YY sandwiches carry an
        extra RZ(-pi/2) before the Hadamard to turn sigma_x into sigma_y,
        matching _H2YY of circuit_fig89_10_hqml_qnode_b_qubit.py.

        CORRECTED.  An earlier version emitted only the two XX sandwiches, so
        the drawn circuit implemented XX (x) x_l in place of (XX + YY) (x)
        x_l -- half the Jl coupling was missing -- and the tape showed 44
        operations where Appendix B requires 58 timed plus 12 virtual.  The
        simulated trajectories were never affected: they come from the qnode
        module, which has all four sandwiches.
        """
        for axis in ("xx", "yy"):
            for r in range(1, NC):
                g = J0[r - 1] * ETA_J[r - 1] * TAU
                if axis == "yy":
                    qml.RZ(-np.pi / 2, wires=QB[0])
                    qml.RZ(-np.pi / 2, wires=QB[r])
                qml.Hadamard(wires=QB[0])
                qml.Hadamard(wires=QB[r])
                qml.CNOT(wires=[QB[r], QB[0]])
                hqml.ConditionalDisplacement(abs(g) / 2.0,
                                             float(np.angle(g + 0j)),
                                             wires=[QB[0], LINK])
                qml.CNOT(wires=[QB[r], QB[0]])
                qml.Hadamard(wires=QB[r])
                qml.Hadamard(wires=QB[0])
                if axis == "yy":
                    qml.RZ(np.pi / 2, wires=QB[r])
                    qml.RZ(np.pi / 2, wires=QB[0])

    # forward half: det first, so every R gate is at the start of the drawing
    det_block(); chi_block(); cd_block(); j_block()
    jl_block()
    # reversed half must be the exact mirror of the forward half
    j_block(rev=True); cd_block(rev=True); chi_block(rev=True); det_block(rev=True)
    return qml.state()


def emit(dev):
    node = qml.QNode(one_hot_trotter_step, dev)

    txt = qml.draw(node, wire_order=WIRE_ORDER, max_length=200)()
    out_txt = TXTDIR / "circuit_qubit_hybridlane.txt"
    hdr = [
        "One-hot QUBIT-encoding Trotter step, drawn by PennyLane/HybridLane",
        f"Nc = {NC}   wires = {WIRE_ORDER}",
        "gate counts: chi 2Nc | cd 2(Nc+1) | Rotation 2(Nc+1) | rz 2(Nc-1)"
        " | J 2(Nc-1) | Jl (Nc-1)   -> 11Nc-1",
        "Jl has no native primitive; drawn as its H-CNOT-CD-CNOT-H decomposition,",
        "so the rendered gate count exceeds 11Nc-1 by the decomposition overhead.",
        "J is drawn as TWO MS windows per pair (IsingXX then IsingYY): one",
        "bichromatic window gives a single Pauli axis, never XX+YY.  Jl likewise",
        "needs FOUR sandwiches -- one per (axis, chromophore) pair -- not two.",
        "order: det_fwd, chi_fwd, cd_fwd, j_fwd | jl_mid | j_rev, cd_rev, chi_rev, det_rev",
        "  det (the R gates) moved to the front; reverse half is its exact mirror, so",
        "  the step is still symmetric second-order Suzuki.  Counts unchanged.",
        "  build_one_hot_trotter_step still emits the OLD order -- counts match, order",
        "  does not; see this script's module docstring for why the builder was not",
        "  changed (SymCircuit.depth() is order-sensitive).",
        "",
    ]
    out_txt.write_text("\n".join(hdr) + txt + "\n", encoding="utf-8")
    print("\n".join(hdr))
    print(txt)
    print(f"\nwrote {out_txt}")

    # ---- mpl drawing, HybridLane's OWN drawer ----
    # NOTE: qml.draw_mpl (PennyLane's) fails here -- HybridLane's mpl hook calls
    # MPLDrawer.z_conditional, which PennyLane 0.44.1's MPLDrawer does not have.
    # hqml.draw_mpl uses HybridLane's own MPLDrawer subclass, which does.
    fig, ax = hqml.draw_mpl(node, wire_order=WIRE_ORDER,
                            show_wire_types=True, decimals=2)()
    fig.suptitle(
        f"One-hot qubit encoding, one Trotter step (Nc={NC})  "
        f"— drawn by HybridLane",
        fontsize=10)
    for ext in ("png", "pdf", "svg"):
        p = OUTDIR / f"circuit_qubit_hybridlane_one_step.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"wrote {p}")
    plt.close(fig)


def main():
    # The PennyLane text drawing uses box-drawing glyphs; a cp1252 console
    # raises UnicodeEncodeError on print (the .txt file is always utf-8).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    dev = qml.device("default.qubit", wires=WIRE_ORDER)
    emit(dev)


if __name__ == "__main__":
    main()
