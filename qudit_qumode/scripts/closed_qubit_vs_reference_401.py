"""Closed-system qubit reference vs Vu's published trajectory, 401 points.

WHY.  Sec.~results_closed and the fig:closed caption both state that the
qubit compilation reproduces the published reference to 0.16%.  That value
appears only as prose in writeup/results_closed.tex; no generated file in the
repo carries it, and the file that sentence refers to
(_qubit_vustack_N8.npz) is not in data/.  This script recomputes the
comparison from the files that ARE here, so the manuscript can quote a
number with a source.

WHAT IS COMPARED.
  ours  data/trotter_method_b_qubit_fock8.npz
        401 time points over 0-2 ps, Fock 8, gate_count 32.
        PA/PB/PC  = the Trotterized qubit circuit  <- this is "the qubit
                    compilation" the sentence is about
        PA_exact  = direct integration of H_cqed on a finer grid (500 pts)
  ref   data/vu_reference/idealSim_Fock8_10k.out
        201 time points over 0-2 ps.  Despite the "10k" in the filename the
        values are NOT shot-quantised (they are not multiples of 1e-4), so
        this is a statevector trajectory, not a sampled one.

GRID.  The reference's 201 points coincide exactly with every second point
of our 401, so the primary comparison needs no interpolation.  The
interpolated variant is printed alongside to show it changes nothing.

Output: results/closed_qubit_vs_reference_401.txt
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OURS = ROOT / "data" / "trotter_method_b_qubit_fock8.npz"
REF = ROOT / "data" / "vu_reference" / "idealSim_Fock8_10k.out"
QUDIT = ROOT / "data" / "trotter_method_a_fock8.npz"

L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def commit():
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def rmse_pct(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) * 100)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    d = np.load(OURS)
    ref = np.loadtxt(REF, dtype=complex).real.T          # (201, 4): t, PA, PB, PC
    t_ref, R = ref[:, 0], {k: ref[:, i + 1] for i, k in enumerate("ABC")}
    t = d["t"]

    W("=" * 104)
    W("CLOSED-SYSTEM QUBIT REFERENCE vs VU'S PUBLISHED TRAJECTORY")
    W("=" * 104)
    W(f"  commit   {commit()}")
    W(f"  ours     {OURS.relative_to(ROOT)}")
    W(f"           {len(t)} points, t = {t[0]:.3f} to {t[-1]:.3f} ps, "
      f"gate_count = {int(d['gate_count'])}")
    W(f"  ref      {REF.relative_to(ROOT)}")
    W(f"           {len(t_ref)} points, t = {t_ref[0]:.3f} to "
      f"{t_ref[-1]:.3f} ps")
    W()

    # --- is the reference shot-sampled? -------------------------------------
    scaled = R["A"][1:8] * 10000
    W("  IS THE REFERENCE SHOT-SAMPLED?")
    W(f"    PA x 10000 = {np.array2string(scaled, precision=2)}")
    W("    Not integers, so the published trajectory is a statevector result,")
    W("    not a 10,000-shot sample.  A 'shot-noise floor' argument does not")
    W("    apply to this comparison.")
    W()

    # --- grid alignment -----------------------------------------------------
    align = float(np.abs(t_ref - t[::2]).max())
    W("  GRID ALIGNMENT")
    W(f"    max |t_ref - t_ours[::2]| = {align:.3e} ps")
    W(f"    exact coincidence: {align < 1e-12}  -> primary comparison uses "
      "the 201 shared points, no interpolation")
    W()

    W("=" * 104)
    W("RESULT")
    W("=" * 104)
    W(f"  {'comparison':<46} {'A':>9} {'B':>9} {'C':>9} {'mean':>9}")
    rows = []

    def emit(label, get):
        r = [rmse_pct(*get(k)) for k in "ABC"]
        m = float(np.mean(r))
        rows.append((label, r, m))
        W(f"  {label:<46} {r[0]:>9.4f} {r[1]:>9.4f} {r[2]:>9.4f} {m:>9.4f}")

    emit("Trotterized qubit circuit vs ref (201 pts)",
         lambda k: (d["P" + k][::2], R[k]))
    emit("Trotterized qubit circuit vs ref (interp 401)",
         lambda k: (d["P" + k], np.interp(t, t_ref, R[k])))
    emit("exact H_cqed vs ref (interp)",
         lambda k: (np.interp(t_ref, d["t_exact"], d["P" + k + "_exact"]),
                    R[k]))
    emit("Trotterized qubit circuit vs its own exact",
         lambda k: (d["P" + k],
                    np.interp(t, d["t_exact"], d["P" + k + "_exact"])))

    if QUDIT.exists():
        q = np.load(QUDIT)
        emit("qudit circuit vs ref (201 pts)",
             lambda k: (q["P" + k][::2], R[k]))
        emit("qudit circuit vs qubit circuit (401 pts)",
             lambda k: (q["P" + k], d["P" + k]))

    W()
    W("  All values are percentage points of population.")
    W()
    W("=" * 104)
    W("BEARING ON THE MANUSCRIPT'S 0.16%")
    W("=" * 104)
    headline = rows[0][2]
    W(f"  The 401-point qubit compilation agrees with the published")
    W(f"  trajectory to {headline:.3f}% mean RMSE on the shared grid.")
    W()
    W("  The manuscript says 0.16%.  This run does NOT reproduce it, and")
    W("  neither does the 41-point 10^4-shot file: "
      "data/fig6_hqml_b_qubit_N8_10000shots.npz")
    W("  gives 0.43% against the same reference.  The three numbers come")
    W("  from three different objects:")
    W("      this run   deterministic 401-point Trotter circuit")
    W("      0.43%      10^4-shot sampled circuit, 41 points")
    W("      0.16%      writeup/results_closed.tex prose, referring to")
    W("                 _qubit_vustack_N8.npz, which is not in data/")
    W()
    W("  RECOMMENDATION.  Quote this run, with this file as the source, or")
    W("  drop the number.  Do not keep 0.16% -- it cannot be traced.")
    W()
    W("  NOTE ON THE ACCOMPANYING CLAIM.  The manuscript's struck text said")
    W("  the agreement was 'at the level of its statistical uncertainty'.")
    W("  That strike is correct and should stay struck: the reference is not")
    W("  sampled, so there is no statistical uncertainty to be at the level")
    W("  of.")
    W()
    W("  NO ADOPTION.  Report only.")

    out = ROOT / "results" / "closed_qubit_vs_reference_401.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
