"""Entangling-only gate counts per FULL symmetric Trotter step.

Check 2 of a two-part reviewer request.  Single-qubit and single-qudit gates
are excluded; a gate counts as entangling iff it acts on MORE THAN ONE WIRE.

WHY WIRE COUNT AND NOT THE BUILDER'S FLAG
  simulations/vibronic_scaling.py marks QT with entangling=True, but QT is
  c.append('QT', [q], ...) -- ONE wire.  It is a transition between two levels
  of the same qudit and entangles nothing.  Both classifications are printed so
  the discrepancy is visible rather than silently resolved.

SOURCES -- the same builders that feed the runtime table
  TI qubit logical : vibronic_scaling.build_one_hot_trotter_step(Nc)
  TI qubit native  : the logical -> TI-native map used by
                     scripts/plot_logical_vs_native.py, with the CORRECTED Jl
                     (an XX sandwich AND a YY sandwich; one sandwich realises
                     XX only -- measured +1.000*XX by h_basis_check)
  TI qudit (28)    : simulations/circuit_diagrams_28.py::make_qudit_generalized_28

REPORT ONLY.  Nothing is fixed.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulations.vibronic_scaling import build_one_hot_trotter_step  # noqa: E402
from simulations.circuit_diagrams_28 import make_qudit_generalized_28  # noqa: E402

NC = 3
OUT = ROOT / "results" / "entangling_only_counts.txt"
LINES = []


def W(s=""):
    LINES.append(s)
    print(s, flush=True)


# TI-native expansion, same map as plot_logical_vs_native.py.
# (name, n_wires) per emitted primitive.
SAND_XX = [("H", 1), ("H", 1), ("CNOT", 2), ("CD", 2), ("CNOT", 2),
           ("H", 1), ("H", 1)]
SAND_YY = [("Z", 1), ("H", 1), ("Z", 1), ("H", 1), ("CNOT", 2), ("CD", 2),
           ("CNOT", 2), ("H", 1), ("Z", 1), ("H", 1), ("Z", 1)]
NATIVE = {
    "Rotation": [("Rot", 1)],
    "rz":       [("Z", 1)],
    "chi":      [("CR", 2)],
    "cd":       [("CD", 2)],
    "J":        [("MS", 2), ("MS", 2)],
    "Jl":       SAND_XX + SAND_YY,
}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("=" * 104)
    W(f"ENTANGLING-ONLY GATE COUNTS -- full symmetric Trotter step, Nc = {NC}")
    W("=" * 104)
    W("  entangling := acts on more than one wire.  Single-qubit and")
    W("  single-qudit gates excluded.")
    W()

    # ---------------- TI qubit, logical ----------------
    c = build_one_hot_trotter_step(NC)
    W("1  TI QUBIT, LOGICAL  (vibronic_scaling.build_one_hot_trotter_step)")
    W(f"     {'gate':>10} {'count':>6} {'wires':>6} {'entangling?':>12}")
    ql_ent = 0
    ql_tot = 0
    per = Counter()
    wires_of = {}
    for g in c.gates:
        per[g.name] += 1
        wires_of[g.name] = len(g.wires)
    for name, n in sorted(per.items(), key=lambda kv: -kv[1]):
        w = wires_of[name]
        ent = w > 1
        ql_tot += n
        if ent:
            ql_ent += n
        W(f"     {name:>10} {n:>6} {w:>6} {('YES' if ent else 'no'):>12}")
    W(f"     TOTAL {ql_tot}   ENTANGLING-ONLY = {ql_ent}")
    W(f"     (builder's own n_entangling() flag says {c.n_entangling()})")
    W()

    # ---------------- TI qubit, native ----------------
    W("2  TI QUBIT, TI-NATIVE  (logical expanded; Jl = XX sandwich + YY sandwich)")
    nat = Counter()
    nat_w = {}
    for g in c.gates:
        for nm, w in NATIVE[g.name]:
            nat[nm] += 1
            nat_w[nm] = w
    qn_ent = sum(n for nm, n in nat.items() if nat_w[nm] > 1)
    qn_tot = sum(nat.values())
    qn_timed = sum(n for nm, n in nat.items() if nm != "Z")
    W(f"     {'primitive':>10} {'count':>6} {'wires':>6} {'entangling?':>12}")
    for nm, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        W(f"     {nm:>10} {n:>6} {nat_w[nm]:>6} "
          f"{('YES' if nat_w[nm] > 1 else 'no'):>12}")
    W(f"     TOTAL {qn_tot}  (timed, excluding virtual-Z: {qn_timed})")
    W(f"     ENTANGLING-ONLY = {qn_ent}")
    W()

    # ---------------- TI qudit, 28 ----------------
    fwd, center, bwd, gate_list, count = make_qudit_generalized_28()
    W(f"3  TI QUDIT, 28-GATE FUSED  (circuit_diagrams_28.make_qudit_generalized_28)")
    W(f"     builder asserts total = {count}")
    qd = Counter()
    qd_w = {}
    for g in gate_list:
        qd[g["gate"]] += 1
        qd_w[g["gate"]] = len(g["wires"])
    qd_ent = sum(n for nm, n in qd.items() if qd_w[nm] > 1)
    W(f"     {'gate':>12} {'count':>6} {'wires':>6} {'entangling?':>12}")
    for nm, n in sorted(qd.items(), key=lambda kv: -kv[1]):
        W(f"     {nm:>12} {n:>6} {qd_w[nm]:>6} "
          f"{('YES' if qd_w[nm] > 1 else 'no'):>12}")
    W(f"     TOTAL {sum(qd.values())}   ENTANGLING-ONLY = {qd_ent}")
    W()

    # ---------------- comparison ----------------
    W("=" * 104)
    W("COMPARISON -- entangling only, full symmetric step, Nc = 3")
    W("=" * 104)
    W(f"{'encoding':>34} {'all gates':>11} {'entangling':>12} {'frac':>7}")
    for lbl, tot, ent in (("TI qubit, logical", ql_tot, ql_ent),
                          ("TI qubit, TI-native (timed)", qn_timed, qn_ent),
                          ("TI qudit, 28-gate fused", sum(qd.values()), qd_ent)):
        W(f"{lbl:>34} {tot:>11} {ent:>12} {100*ent/tot:>6.0f}%")
    W()
    W(f"  qubit-native / qudit entangling ratio = {qn_ent}/{qd_ent} = "
      f"{qn_ent/qd_ent:.2f}x")
    W()
    W("  CAVEATS")
    W("    * QT is flagged entangling=True in vibronic_scaling.py but acts on a")
    W("      SINGLE wire -- it is an intra-qudit level transition and entangles")
    W("      nothing.  Counted here as non-entangling on wire count; the")
    W("      builder's flag disagrees and is noted above.")
    W("    * CNOT is counted as ONE entangling primitive.  On a trapped ion it")
    W("      is itself an MS plus local rotations, and no CNOT duration is")
    W("      anchored in this repo, so the qubit-native entangling count is a")
    W("      LOWER BOUND.")
    W("    * The qudit 28-gate builder fuses QPS and QT.  QPS fusion is exact")
    W("      (commuting projectors); QT fusion is NOT exact by commutation")
    W("      ([|A><B|,|A><C|] = 1.0, measured) and rests on a multi-tone")
    W("      hardware claim priced from step_qt_meth_aggcap.py.")
    W("    * The 28-gate qudit builder is NOT the same builder as")
    W("      build_qudit_trotter_step (which gives 36 unfused / 34 QPS-fused).")
    W("      Comparing 28 against the qubit numbers mixes builders; stated here,")
    W("      not resolved.")
    W()
    W("  REPORT ONLY.  Nothing fixed.")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
