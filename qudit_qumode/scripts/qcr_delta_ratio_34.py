"""Appendix C, item 2: Nc=3 sequential ratio bracket vs delta_disp, 34-op headline.

Everything is READ from saved files.  No propagation, no reruns.

  T_ang(delta)   results/qcr_delta_sweep.txt   (angle held at theta = 0.016)
  T_QCR(delta)   = 2*T_ang + t_pi,  t_pi = 5 us ASSUMED
                   (the echo structure documented at
                    step_runtime_table.py:60, "T_paper = 2*T + t_pi";
                    check: 2*25.465 + 5 = 55.930 = the adopted T_QCR)
  all other gate times, counts and T_qubit  pulses_meth/runtime_table.json
                   (34-op headline: QPS fused only, QT NOT fused)

Only T_QCR varies across rows.  Everything else is held fixed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "qcr_delta_ratio_34.txt"
SWEEP = ROOT / "results" / "qcr_delta_sweep.txt"
RT = ROOT / "pulses_meth" / "runtime_table.json"
T_PI = 5.0

LINES = []


def W(s=""):
    LINES.append(s)
    print(s, flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    rt = json.loads(RT.read_text(encoding="utf-8"))
    row3 = [r for r in rt["rows"] if r["Nc"] == 3][0]
    g = rt["gate_times_us"]
    qt_lo, qt_hi = rt["conventions"]["qt_bounds_us"]
    c = row3["qudit_counts"]                     # 34-op convention
    Tqb = row3["T_qubit_seq_us"]

    # parse T_ang per delta from the sweep table
    txt = SWEEP.read_text(encoding="utf-8", errors="replace")
    rows = []
    for ln in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                     r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$", ln)
        if m:
            rows.append(dict(delta=float(m.group(1)), small=float(m.group(2)),
                             T_ang=float(m.group(4)),
                             fit_vz=float(m.group(8)),
                             leak_g=float(m.group(9))))
    if len(rows) != 5:
        raise SystemExit(f"expected 5 sweep rows, parsed {len(rows)}")

    W("=" * 104)
    W("APPENDIX C, ITEM 2 -- Nc = 3 sequential ratio vs delta_disp, 34-op headline")
    W("=" * 104)
    W("  All numbers READ from files; nothing propagated here.")
    W(f"    T_ang(delta), fidelity, leakage : {SWEEP.relative_to(ROOT)}")
    W(f"    gate times, counts, T_qubit     : {RT.relative_to(ROOT)}")
    W()
    W("  FIXED ACROSS ALL ROWS (only T_QCR varies):")
    W(f"    T_qubit sequential = {Tqb:.4f} us   (unaffected by delta)")
    W(f"    QCD  {c['QCD']} x {g['QCD']:.4f} us")
    W(f"    Rot  {c['Rotation']} x {g['rotation']:.1f} us")
    W(f"    QT-SB {c['QT_SB']} x {g['QT_SB']:.6f} us")
    W(f"    QT   {c['QT_multi']} x [{qt_lo}, {qt_hi}] us   (34-op: QT NOT fused)")
    W(f"    QPS  {c['QPS']} x 0 us (virtual)")
    W(f"    QCR  {c['QCR']} x T_QCR(delta)")
    W()
    W(f"  T_QCR = 2*T_ang + t_pi,  t_pi = {T_PI} us ASSUMED")
    W("    echo structure, step_runtime_table.py:60 "
      "\"T_paper = 2*T + t_pi with t_pi = 5 us ASSUMED\"")
    W()

    base = (c["Rotation"] * g["rotation"] + c["QCD"] * g["QCD"]
            + c["QT_SB"] * g["QT_SB"])
    W(f"  non-(QCR,QT) part = {c['Rotation']}x{g['rotation']} + "
      f"{c['QCD']}x{g['QCD']:.4f} + {c['QT_SB']}x{g['QT_SB']:.6f} "
      f"= {base:.4f} us")
    W()

    W("=" * 104)
    W(f"{'delta':>7} {'Om/del':>7} {'T_ang':>9} {'T_QCR':>9} "
      f"{'T_qudit lo':>11} {'T_qudit hi':>11} {'ratio seq (Nc=3)':>20} "
      f"{'1-F fit+VZ':>12} {'leak P(g)':>11}")
    W(f"{'(kHz)':>7} {'':>7} {'(us)':>9} {'(us)':>9}")
    out = []
    for r in rows:
        T_qcr = 2 * r["T_ang"] + T_PI
        lo = base + c["QCR"] * T_qcr + c["QT_multi"] * qt_lo
        hi = base + c["QCR"] * T_qcr + c["QT_multi"] * qt_hi
        rlo, rhi = Tqb / hi, Tqb / lo
        out.append((r, T_qcr, lo, hi, rlo, rhi))
        W(f"{r['delta']:>7.0f} {r['small']:>7.3f} {r['T_ang']:>9.3f} "
          f"{T_qcr:>9.3f} {lo:>11.4f} {hi:>11.4f} "
          f"{'[%.2fx, %.2fx]' % (rlo, rhi):>20} "
          f"{r['fit_vz']:>12.4e} {r['leak_g']:>11.4e}")
    W()

    ref = [o for o in out if o[0]["delta"] == 40][0]
    W("  CROSS-CHECK against the adopted table:")
    W(f"    delta = 40 kHz gives T_QCR = {ref[1]:.3f} us; runtime_table.json "
      f"holds {g['QCR']} us   match: {abs(ref[1]-g['QCR']) < 5e-4}")
    W(f"    ratio there = [{ref[4]:.2f}x, {ref[5]:.2f}x]; runtime_table.json "
      f"holds [{row3['ratio_seq'][0]:.2f}x, {row3['ratio_seq'][1]:.2f}x]")
    W()
    W("  READING")
    W(f"    The bracket sweeps [{out[0][4]:.2f}x, {out[0][5]:.2f}x] at 20 kHz "
      f"down to [{out[-1][4]:.2f}x, {out[-1][5]:.2f}x] at 80 kHz -- a factor")
    W(f"    {out[0][5]/out[-1][4]:.1f} across the window, because T_QCR is "
      f"proportional to delta and QCR")
    W(f"    is {c['QCR']} of the qudit step's gates.")
    W("    Fidelity and leakage are printed alongside DELIBERATELY: smaller")
    W("    delta shortens the gate and flatters the ratio while the")
    W("    Schrieffer-Wolff parameter Omega/delta grows.  At 20 kHz the")
    W(f"    leakage out of |l> is {out[0][0]['leak_g']:.2e} -- that is not a")
    W("    working gate, whatever the ratio says.  A ratio-only column would")
    W("    invite exactly the wrong conclusion.")
    W()
    W("    40 kHz sits at Omega/delta = 0.100 (the usual 10% rule), mid-window,")
    W(f"    with leakage {ref[0]['leak_g']:.2e}.  Adopted; unchanged by this run.")
    W()
    W("  NO RERUNS.  Every value above is read from the two files named at top.")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
