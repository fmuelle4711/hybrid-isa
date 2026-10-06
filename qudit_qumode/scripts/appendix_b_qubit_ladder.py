"""Appendix B: the qubit reference, 32 logical operations -> 58 timed pulses.

Reads the builder (simulations/vibronic_scaling.py::build_one_hot_trotter_step)
and applies the TI-native expansion used by scripts/plot_logical_vs_native.py.
No reruns, no propagation.

The native map, with the reason for each expansion:
  Rotation -> 1 Rotation    mode free evolution, native single primitive
  rz       -> 0 timed       virtual-Z: software phase-frame update, T = 0
  chi (CR) -> 1 CR          sigma_z (x) n, native dispersive interaction
  cd (CD)  -> 1 CD          sigma_z (x) x, native sigma_z SDF
  J        -> 2 MS          H_J = g(XX+YY)/2.  ONE Molmer-Sorensen window
                            generates sigma_phi (x) sigma_phi only:
                              cos^2(phi) XX + sin^2(phi) YY
                                 + cos(phi)sin(phi)(XY+YX)
                            The cross terms vanish only at phi = 0 (XX alone)
                            or phi = pi/2 (YY alone), so NO single window gives
                            XX+YY.  Two windows are EXACT, not a Trotter
                            approximation, because [X(x)X, Y(x)Y] = 0.
                            (results/j_ms_convention.txt)
  Jl       -> 2 sandwiches  H_Jl = g(XX+YY)(x)x_l/2.  Needs an XX sandwich AND
                            a YY sandwich; the split is exact since
                            [XX(x)x, YY(x)x] = 0.  Each sandwich is
                              4 x carrier-pi/2 (H) + 2 x CNOT + 1 x CD
                            and the YY one additionally carries 4 virtual-Z
                            (the RZ(+-pi/2) conjugation that rotates XX -> YY).
                            ONE sandwich alone realises XX only -- measured
                            +1.000*XX by h_basis_check in
                            scripts/circuit_qubit_hybridlane.py.

CNOT PRICING, stated because it bounds the result:
  CNOT is counted here as ONE timed primitive.  On a trapped ion a CNOT is
  itself a Molmer-Sorensen plus local rotations, and NO CNOT duration is
  anchored anywhere in this repo.  So 58 is a LOWER BOUND on the pulse count,
  and the runtime table does not price the sandwich at all -- it holds H_Jl at
  a single MS window, which is the conservative choice against the qudit claim.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simulations.vibronic_scaling import build_one_hot_trotter_step  # noqa: E402

NC = 3
OUT = ROOT / "results" / "qubit_ladder_58.txt"
LINES = []


def W(s=""):
    LINES.append(s)
    print(s, flush=True)


SAND_XX = [("H", 1, "timed"), ("H", 1, "timed"), ("CNOT", 2, "timed"),
           ("CD", 2, "timed"), ("CNOT", 2, "timed"), ("H", 1, "timed"),
           ("H", 1, "timed")]
SAND_YY = [("Z", 1, "virtual"), ("H", 1, "timed"), ("Z", 1, "virtual"),
           ("H", 1, "timed"), ("CNOT", 2, "timed"), ("CD", 2, "timed"),
           ("CNOT", 2, "timed"), ("H", 1, "timed"), ("Z", 1, "virtual"),
           ("H", 1, "timed"), ("Z", 1, "virtual")]

NATIVE = {
    "Rotation": ([("Rot", 1, "timed")], "native mode free evolution"),
    "rz":       ([("Z", 1, "virtual")], "virtual-Z, T = 0"),
    "chi":      ([("CR", 2, "timed")], "native dispersive sigma_z (x) n"),
    "cd":       ([("CD", 2, "timed")], "native sigma_z SDF"),
    "J":        ([("MS", 2, "timed"), ("MS", 2, "timed")],
                 "XX+YY needs 2 MS windows; exact, [XX,YY]=0"),
    "Jl":       (SAND_XX + SAND_YY,
                 "XX sandwich + YY sandwich; no native 3-body primitive"),
}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    c = build_one_hot_trotter_step(NC)
    per = Counter(g.name for g in c.gates)

    W("=" * 100)
    W(f"APPENDIX B -- QUBIT REFERENCE LADDER, 32 LOGICAL -> 58 TIMED, Nc = {NC}")
    W("=" * 100)
    W("  source: simulations/vibronic_scaling.py::build_one_hot_trotter_step")
    W("  no reruns, no propagation")
    W()

    W("1  LOGICAL HISTOGRAM BY GATE TYPE")
    W("-" * 100)
    W(f"  {'gate':>10} {'count':>6} {'formula':>14}   {'what it is':<44}")
    forms = {"chi": "2Nc", "cd": "4Nc-4", "Rotation": "4Nc-4",
             "rz": "2(Nc-1)", "J": "2(Nc-1)", "Jl": "2(Nc-2)"}
    what = {"chi": "sigma_z (x) n, per-site dispersive (CR)",
            "cd": "sigma_z (x) x, per-site + link (CD)",
            "Rotation": "mode free evolution",
            "rz": "qubit detuning, virtual-Z",
            "J": "2-qubit XY exchange (XX+YY)/2",
            "Jl": "2-qubit exchange (x) x_l, 3-body"}
    tot = 0
    for name in ("chi", "cd", "Rotation", "rz", "J", "Jl"):
        n = per[name]; tot += n
        W(f"  {name:>10} {n:>6} {forms[name]:>14}   {what[name]:<44}")
    W(f"  {'TOTAL':>10} {tot:>6} {'16Nc-16':>14}")
    W("  Rotation and cd are counted 4Nc-4, audited against Vu Eq. 18 to")
    W("  21: one mode free evolution per mode per half step over the")
    W("  2Nc-2 modes, and per site one conditional displacement on its")
    W("  high mode and one on its low mode, boundary low modes dropped.")
    W("  J_l is counted 2(Nc-2), the chain link count of Vu Eq. 21 with")
    W("  Eq. 52's boundary rule, not Nc-1.  The two agree at Nc = 3, where")
    W("  both bonds are end bonds, so every number in this table is")
    W("  unchanged; the closed form differs for Nc > 3.")
    W()

    W("2  THE LADDER, LOGICAL GATE BY LOGICAL GATE")
    W("-" * 100)
    W(f"  {'logical':>9} {'x':>4} {'-> timed each':>14} {'timed tot':>10} "
      f"{'VZ each':>8} {'VZ tot':>7}   reason")
    T = V = 0
    detail = {}
    for name in ("chi", "cd", "Rotation", "rz", "J", "Jl"):
        kids, why = NATIVE[name]
        n = per[name]
        t_each = sum(1 for _, _, kind in kids if kind == "timed")
        v_each = sum(1 for _, _, kind in kids if kind == "virtual")
        T += n * t_each; V += n * v_each
        detail[name] = (n, t_each, v_each, kids)
        W(f"  {name:>9} {n:>4} {t_each:>14} {n*t_each:>10} "
          f"{v_each:>8} {n*v_each:>7}   {why}")
    W(f"  {'':>9} {'':>4} {'':>14} {T:>10} {'':>8} {V:>7}")
    W()
    W(f"  32 logical  ->  {T} timed pulses  +  {V} virtual-Z (T = 0)")
    W()

    W("3  WHERE THE +26 COMES FROM")
    W("-" * 100)
    W(f"  {'source':<46} {'timed':>7} {'delta vs logical':>18}")
    base = per["chi"] + per["cd"] + per["Rotation"]
    W(f"  {'CR + CD + Rotation, 1:1 with logical':<46} {base:>7} "
      f"{'+0':>18}")
    W(f"  {'rz -> virtual-Z, no pulse':<46} {0:>7} "
      f"{-per['rz']:>18}")
    W(f"  {'J -> 2 MS windows each':<46} {2*per['J']:>7} "
      f"{'+%d' % per['J']:>18}")
    jl_t = sum(1 for _, _, k in NATIVE['Jl'][0] if k == 'timed')
    W(f"  {'Jl -> XX sandwich + YY sandwich':<46} {jl_t*per['Jl']:>7} "
      f"{'+%d' % (jl_t*per['Jl'] - per['Jl']):>18}")
    W(f"  {'TOTAL':<46} {T:>7} {'+%d' % (T-32):>18}")
    W()
    W("  So essentially ALL the inflation is Jl (+26 of +26 net, before the")
    W("  -4 from rz becoming free and the +4 from J doubling).  26 of the 32")
    W("  logical operations are 1:1 with a native pulse; the two Jl gates carry")
    W("  the entire compilation cost.")
    W()

    W("4  THE Jl SANDWICH, GATE BY GATE")
    W("-" * 100)
    W("  per Jl coupling, XX half:")
    for g, w, k in SAND_XX:
        W(f"      {g:<6} {w}-wire  {k}")
    W("  per Jl coupling, YY half (RZ(+-pi/2) conjugation rotates XX -> YY):")
    for g, w, k in SAND_YY:
        W(f"      {g:<6} {w}-wire  {k}")
    nat = Counter()
    for g, w, k in NATIVE["Jl"][0]:
        nat[(g, k)] += 1
    W()
    W(f"  per Jl: " + ", ".join(f"{v}x {g} ({k})" for (g, k), v in
                                 sorted(nat.items())))
    W(f"  x {per['Jl']} couplings = "
      + ", ".join(f"{v*per['Jl']}x {g}" for (g, k), v in sorted(nat.items())))
    W()
    W("  ONE sandwich alone realises XX only, not XX+YY -- measured +1.000*XX")
    W("  by h_basis_check (scripts/circuit_qubit_hybridlane.py).  Both halves")
    W("  are required, and the split is exact because [XX(x)x, YY(x)x] = 0.")
    W()

    W("5  FULL NATIVE HISTOGRAM")
    W("-" * 100)
    allnat = Counter()
    for name in per:
        kids, _ = NATIVE[name]
        for g, w, k in kids:
            allnat[(g, k)] += per[name]
    W(f"  {'primitive':>10} {'count':>6} {'kind':>9}")
    for (g, k), v in sorted(allnat.items(), key=lambda kv: -kv[1]):
        W(f"  {g:>10} {v:>6} {k:>9}")
    W(f"  {'TIMED':>10} {T:>6}")
    W(f"  {'VIRTUAL':>10} {V:>6}")
    W()

    W("6  CNOT PRICING -- WHY 58 IS A LOWER BOUND")
    W("-" * 100)
    cn = allnat[("CNOT", "timed")]
    W(f"  {cn} CNOTs appear, each counted as ONE timed primitive.")
    W("  On a trapped ion a CNOT is itself a Molmer-Sorensen plus local")
    W("  rotations, and NO CNOT duration is anchored anywhere in this repo")
    W("  (searched; only gate-set mentions in docs/scaling_theorems.md).")
    W(f"  So {T} is a LOWER BOUND on the timed-pulse count.")
    W()
    W("  The runtime table does NOT price the sandwich at all: it holds H_Jl at")
    W("  a single MS window (200 us).  Pricing it from the decomposition instead")
    W("  gave 412.57 us per Jl -- 2.06x MORE -- but that rests on an ASSUMED")
    W("  CNOT time, so it was printed and NOT adopted (runtime_table.md).")
    W("  Holding Jl at one window is therefore the conservative choice AGAINST")
    W("  the qudit advantage, not for it.")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
