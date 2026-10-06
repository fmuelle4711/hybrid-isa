"""TIMED PULSE SEGMENTS per Trotter step, both encodings.

Backs the Timed pulses column of tab:gatecounts: 58 for the qubit
reference, 27 for the qudit reference, 25 for the multitone qudit.

A "segment" is one contiguous interval of drive a sequencer has to emit.
Rule set, the ONE-PULSE convention the manuscript uses:
  C_Pi R, CR  -> 1 segment each.  A single flat pulse on the detuned
                 sideband, one detuning period long (App. D rule R2).
                 An earlier version of this file counted 3, the
                 (T, pi, T) echo; that convention is superseded.
  C_Pi_z D, CD -> 1 segment each
  Rotation     -> 1
  Z_d, r_z     -> 0 (virtual, frame update, no pulse)
  X_E          -> 1 segment each; in the multitone schedule the X_E of
                  each half step fuse into one block, 2 blocks per step
  C_Pi_x D     -> the N_c - 1 central sideband terms are applied as ONE
                  summed exponential, the merged centre, so 1 segment
                  per step however many terms it holds
  J            -> 2 (two MS windows; XX and YY do not fuse)
  J_l          -> counted two ways:
                  (a) as its actual decomposition, the manuscript's
                      convention: 8 H + 4 CNOT + 2 CD = 14 per J_l
                  (b) as one MS window, an alternative kept for reference
Counts are read from pulses_meth/runtime_table.json, where QT_SB holds
the N_c - 1 sequential centre terms and QT_multi the 2(N_c - 1) X_E.

The result is cross-checked against results/cpir_one_pulse_regen.txt,
which reprices the same counts independently.  A mismatch exits nonzero.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "segment_counts.txt"
REF = ROOT / "results" / "cpir_one_pulse_regen.txt"
JL_DECOMP = 8 + 4 + 2           # H + CNOT + CD per J_l
L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def items(qd, qb):
    """(name, count, segments per gate) for each encoding and convention."""
    qudit_ref = [("Rotation", qd["Rotation"], 1),
                 ("C_Pi R", qd["QCR"], 1),
                 ("C_Pi_z D", qd["QCD"], 1),
                 ("X_E", qd["QT_multi"], 1),
                 ("C_Pi_x D centre, merged", 1, 1),
                 ("Z_d", qd["QPS"], 0)]
    qudit_mt = [("Rotation", qd["Rotation"], 1),
                ("C_Pi R", qd["QCR"], 1),
                ("C_Pi_z D", qd["QCD"], 1),
                ("X_E, multitone blocks", 2, 1),
                ("C_Pi_x D centre, merged", 1, 1),
                ("Z_d, generalized", 2, 0)]
    qubit_dec = [("Rotation", qb["Rotation"], 1),
                 ("CR", qb["QCR"], 1),
                 ("CD", qb["QCD"], 1),
                 ("r_z", qb["rz"], 0),
                 ("J", qb["MS_J"], 2),
                 ("J_l, decomposed", qb["MS_Jl"], JL_DECOMP)]
    qubit_1w = qubit_dec[:-1] + [("J_l, one MS window", qb["MS_Jl"], 1)]
    return qudit_ref, qudit_mt, qubit_dec, qubit_1w


def total(it):
    return sum(n * s for _, n, s in it)


def reference_rows():
    """Nc -> (qudit ref, qudit multitone, qubit) one-pulse targets."""
    txt = REF.read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r"^\s+(\d+)\s+\d+\s*->\s*(\d+)\s+\d+\s*->\s*(\d+)"
                         r"\s+\d+\s*->\s*(\d+)\s*$", txt, re.M):
        out[int(m.group(1))] = (int(m.group(2)), int(m.group(3)),
                                int(m.group(4)))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    d = json.loads((ROOT / "pulses_meth" / "runtime_table.json").read_text())

    W("=" * 104)
    W("TIMED PULSE SEGMENTS PER TROTTER STEP -- one-pulse convention")
    W("=" * 104)
    W("  C_Pi R and CR count 1 segment each (single pulse, App. D rule R2).")
    W("  The N_c - 1 central C_Pi_x D terms form one merged block.")
    W("  C_Pi_z D and CD count 1.  Virtual gates count 0.")
    W("  Qubit: J_l decomposed into 8 H + 4 CNOT + 2 CD, the manuscript's")
    W("  convention; the one-window alternative is listed for reference.")
    W()

    summary = {}
    for row in d["rows"]:
        Nc = row["Nc"]
        qref, qmt, bdec, b1w = items(row["qudit_counts"], row["qubit_counts"])
        summary[Nc] = (total(qref), total(qmt), total(bdec), total(b1w))
        W("-" * 104)
        W(f"  Nc = {Nc}")
        for tag, it in (("QUDIT, reference", qref),
                        ("QUDIT, multitone", qmt),
                        ("QUBIT, J_l decomposed (manuscript)", bdec),
                        ("QUBIT, J_l as one MS window (alternative)", b1w)):
            W(f"    {tag}")
            W(f"      {'gate':<28} {'count':>6} {'seg/gate':>9} "
              f"{'segments':>9}")
            for nm, n, s in it:
                W(f"      {nm:<28} {n:>6} {s:>9} {n * s:>9}")
            W(f"      {'TOTAL SEGMENTS':<28} {'':>6} {'':>9} {total(it):>9}")
        W()

    W("=" * 104)
    W("SUMMARY -- segments per step")
    W("=" * 104)
    W(f"{'Nc':>4} {'qudit ref':>10} {'qudit mt':>9} {'qubit':>7} "
      f"{'qubit (J_l=1 win)':>18}")
    for Nc, (r, m, b, b1) in summary.items():
        W(f"{Nc:>4} {r:>10} {m:>9} {b:>7} {b1:>18}")
    W()

    # ---- cross-check against the independent one-pulse repricing ----------
    W("=" * 104)
    W("CROSS-CHECK -- against results/cpir_one_pulse_regen.txt")
    W("=" * 104)
    ref = reference_rows()
    ok = bool(ref)
    if not ref:
        W("  could not read the reference rows -- FAIL")
    for Nc, (r, m, b, _) in summary.items():
        if Nc not in ref:
            W(f"  Nc={Nc:<3} no reference row -- FAIL")
            ok = False
            continue
        rr, rm, rb = ref[Nc]
        good = (r, m, b) == (rr, rm, rb)
        ok = ok and good
        W(f"  Nc={Nc:<3} here {r}/{m}/{b}   reference {rr}/{rm}/{rb}   "
          f"{'match' if good else 'MISMATCH'}")
    W()
    r3, m3, b3, _ = summary[3]
    W(f"  N_c = 3, the manuscript's tab:gatecounts: qubit {b3}, qudit "
      f"reference {r3}, qudit multitone {m3}.")
    W(f"  Expected 58 / 27 / 25: "
      f"{'PASS' if (b3, r3, m3) == (58, 27, 25) else 'FAIL'}")
    ok = ok and (b3, r3, m3) == (58, 27, 25)
    W()
    W("  Segment count and runtime are different axes: the qubit's extra")
    W("  cost sits in a few long MS windows, not in many segments.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
