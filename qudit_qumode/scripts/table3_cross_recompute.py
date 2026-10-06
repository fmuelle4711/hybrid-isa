"""Recompute the two cross-encoding entries of Table III, qubit vs qudit at
N = 4 and N = 8, from the stored 8-panel trajectories.

CONVENTION, identical to the other four columns of the table:
    per panel, per channel:  RMSE(x, y) = sqrt(mean((x - y)^2)), reported in
                             population percent, i.e. 100 * RMSE
    per panel:               PLAIN MEAN over the three channels A, B, C
                             (not the RMS over channels -- see
                             results/table3_provenance.txt section 3)
    table entry:             MEAN over the 8 dissipative panels
                             (fig6_closed excluded; it is the closed-system
                             panel and is not part of the dissipative table)

VERIFICATION GATE.  The same code path first recomputes the two N = 4
vs-reference numbers, which DO have a stored source
(scripts/N4_validation_report.txt, traced in results/table3_provenance.txt).
If those do not come back as 1.03 % and 1.09 % the cross numbers are not
reported, because a pipeline that cannot reproduce the sourced column cannot
be trusted on the unsourced one.

The Vu reference loader and the RMSE function are IMPORTED from
scripts/analyze_n4_results.py rather than re-implemented, so the reference
interpolation convention cannot silently drift from the one that produced the
stored report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import analyze_n4_results as A4                                  # noqa: E402

OUT = ROOT / "results" / "table3_cross_recomputed.txt"
L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


# panel -> which fig{N}_hqml_* file carries it
PANELS = [("fig8_top", 8), ("fig8_middle", 8), ("fig8_bottom", 8),
          ("fig9_top", 9), ("fig9_middle", 9), ("fig9_bottom", 9),
          ("fig10_combined", 10), ("fig10_damp_only", 10)]

METHOD = {"qubit": "b_qubit", "qudit": "a_qudit"}


def npz_path(fig, method, cutoff):
    return ROOT / "data" / f"fig{fig}_hqml_{method}_N{cutoff}_10000shots.npz"


def panel_traces(panel, fig, method, cutoff):
    """(t_ps, PA, PB, PC) for one panel, truncated to the timepoints actually
    computed when the file records a last_done_idx (the N = 8 schema does)."""
    p = npz_path(fig, method, cutoff)
    d = np.load(p, allow_pickle=True)
    t = np.asarray(d[f"{panel}_t"], float)
    P = [np.asarray(d[f"{panel}_P{c}"], float) for c in "ABC"]
    key = f"{panel}_last_done_idx"
    n = len(t)
    if key in d.files:
        ld = int(d[key])
        n = ld + 1 if ld >= 0 else 0
    return t[:n], [x[:n] for x in P], p


def column(cutoff, mode):
    """mode: 'qubit_vs_ref' | 'qudit_vs_ref' | 'cross'.
    Returns (per-panel rows, table entry in %)."""
    rows = []
    for panel, fig in PANELS:
        if mode == "cross":
            t1, P1, f1 = panel_traces(panel, fig, METHOD["qubit"], cutoff)
            t2, P2, f2 = panel_traces(panel, fig, METHOD["qudit"], cutoff)
            same = (len(t1) == len(t2)
                    and np.allclose(t1, t2, rtol=0, atol=1e-12))
            if not same:                      # align onto the shorter grid
                t = t1 if len(t1) <= len(t2) else t2
                P1 = [np.interp(t, t1, x) for x in P1]
                P2 = [np.interp(t, t2, x) for x in P2]
            per = [100 * A4.rmse(a, b) for a, b in zip(P1, P2)]
            src = f"{f1.name}  vs  {f2.name}"
            note = "grids identical" if same else "INTERPOLATED onto shorter grid"
        else:
            enc = "qubit" if mode.startswith("qubit") else "qudit"
            t, P, f1 = panel_traces(panel, fig, METHOD[enc], cutoff)
            vu = A4.load_vu_at_grid(panel, t)
            if vu is None:
                raise FileNotFoundError(f"Vu reference missing for {panel}")
            per = [100 * A4.rmse(a, b) for a, b in zip(P, vu)]
            src = f"{f1.name}  vs  {A4.VU_FOR_PANEL[panel]}"
            note = f"{len(t)} pts"
        rows.append((panel, per, float(np.mean(per)), src, note))
    return rows, float(np.mean([r[2] for r in rows]))


def show(title, rows, agg, paper, verdict_ok=None):
    W(f"  {title}")
    W(f"    {'panel':<17} {'A':>7} {'B':>7} {'C':>7} {'mean(A,B,C)':>13}   note")
    for panel, per, m, src, note in rows:
        W(f"    {panel:<17} {per[0]:>7.4f} {per[1]:>7.4f} {per[2]:>7.4f} "
          f"{m:>13.4f}   {note}")
    line = f"    mean over 8 panels = {agg:.4f} %  ->  {agg:.2f} %"
    if paper is not None:
        ok = abs(round(agg, 2) - paper) < 5e-3
        line += f"   paper {paper:.2f} %   {'MATCH' if ok else 'MISMATCH'}"
    W(line)
    W()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("=" * 112)
    W("TABLE III CROSS-ENCODING COLUMN, RECOMPUTED FROM THE STORED TRAJECTORIES")
    W("=" * 112)
    W("  convention, same as the other four columns:")
    W("    100 * sqrt(mean((x-y)^2)) per channel, PLAIN MEAN over A/B/C per")
    W("    panel, then mean over the 8 dissipative panels.  fig6_closed is not")
    W("    part of the dissipative table and is excluded.")
    W("  Vu loader and rmse() imported from scripts/analyze_n4_results.py so")
    W("  the reference-interpolation convention cannot drift.")
    W()

    W("  INPUT FILES")
    for cutoff in (4, 8):
        for enc, meth in METHOD.items():
            for fig in (8, 9, 10):
                p = npz_path(fig, meth, cutoff)
                d = np.load(p, allow_pickle=True)
                W(f"    N={cutoff}  {enc:<5} data/{p.name:<44} "
                  f"shots {int(d['shots'])}  fock {int(d['fock'])}  "
                  f"{str(d['backend'].item())}")
    W(f"    reference   {A4.VU_DIR.replace(str(ROOT), '.')}/<per-panel .out>, "
      f"mapped by analyze_n4_results.VU_FOR_PANEL")
    W()

    # ---------------- verification gate -------------------------------
    W("=" * 112)
    W("1  VERIFICATION GATE -- reproduce the two SOURCED N=4 numbers first")
    W("=" * 112)
    qb4, qb4a = column(4, "qubit_vs_ref")
    qd4, qd4a = column(4, "qudit_vs_ref")
    show("N=4  qubit vs reference", qb4, qb4a, 1.03)
    show("N=4  qudit vs reference", qd4, qd4a, 1.09)
    gate = (abs(round(qb4a, 2) - 1.03) < 5e-3
            and abs(round(qd4a, 2) - 1.09) < 5e-3)
    W(f"  GATE: {'PASSED' if gate else 'FAILED'} -- this code path "
      f"{'reproduces' if gate else 'does NOT reproduce'} the sourced column.")
    if not gate:
        W("  Cross-encoding numbers NOT reported.  A pipeline that cannot")
        W("  reproduce the sourced column must not be trusted on the unsourced")
        W("  one.")
        OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
        return
    W()

    # ---------------- the two requested numbers -----------------------
    W("=" * 112)
    W("2  THE TWO CROSS-ENCODING NUMBERS  (this is the requested computation)")
    W("=" * 112)
    x4, x4a = column(4, "cross")
    x8, x8a = column(8, "cross")
    show("N=4  qubit vs qudit", x4, x4a, 0.30)
    show("N=8  qubit vs qudit", x8, x8a, 0.45)

    # ---------------- N=8 vs-ref, checked against the commit msg ------
    W("=" * 112)
    W("3  N=8 vs-REFERENCE PAIR -- recomputed as a check on commit a2f3104")
    W("=" * 112)
    W("  These two have no stored artefact either; the only record of them is")
    W("  the commit message (1.0212 % and 1.0856 %).  Same code path, so this")
    W("  is a genuine independent check of that message.")
    W()
    qb8, qb8a = column(8, "qubit_vs_ref")
    qd8, qd8a = column(8, "qudit_vs_ref")
    show("N=8  qubit vs reference", qb8, qb8a, 1.02)
    show("N=8  qudit vs reference", qd8, qd8a, 1.09)

    # ---------------- all six -----------------------------------------
    W("=" * 112)
    W("4  ALL SIX TABLE III NUMBERS")
    W("=" * 112)
    six = [
        ("N=4", "qubit vs ref", qb4a, 1.03,
         "data/fig{8,9,10}_hqml_b_qubit_N4_10000shots.npz + Vu .out"),
        ("N=4", "qudit vs ref", qd4a, 1.09,
         "data/fig{8,9,10}_hqml_a_qudit_N4_10000shots.npz + Vu .out"),
        ("N=4", "qubit vs qudit", x4a, 0.30,
         "data/fig{8,9,10}_hqml_{b_qubit,a_qudit}_N4_10000shots.npz"),
        ("N=8", "qubit vs ref", qb8a, 1.02,
         "data/fig{8,9,10}_hqml_b_qubit_N8_10000shots.npz + Vu .out"),
        ("N=8", "qudit vs ref", qd8a, 1.09,
         "data/fig{8,9,10}_hqml_a_qudit_N8_10000shots.npz + Vu .out"),
        ("N=8", "qubit vs qudit", x8a, 0.45,
         "data/fig{8,9,10}_hqml_{b_qubit,a_qudit}_N8_10000shots.npz"),
    ]
    W(f"  {'N':<4} {'column':<15} {'recomputed':>11} {'paper':>7} {'':<9} files")
    bad = []
    for n, col, got, paper, files in six:
        ok = abs(round(got, 2) - paper) < 5e-3
        if not ok:
            bad.append((n, col, got, paper))
        W(f"  {n:<4} {col:<15} {got:>10.4f}% {paper:>6.2f}% "
          f"{'MATCH' if ok else 'MISMATCH':<9} {files}")
    W()

    W("=" * 112)
    W("VERDICT")
    W("=" * 112)
    if not bad:
        W("  All six reproduce the printed table.  The two cross-encoding")
        W("  entries and the two N=8 vs-reference entries now have a saved")
        W("  artefact for the first time; previously they existed only in")
        W("  commit message a2f3104.")
    else:
        W("  MISMATCHES FOUND -- reported, not fixed, and no paper number is")
        W("  altered here:")
        for n, col, got, paper in bad:
            W(f"    {n} {col}: recomputed {got:.4f} % vs printed {paper:.2f} %")
        W("  The recomputation above is fully specified: same files, same")
        W("  convention, same Vu loader as the column that does reproduce.")
    W()
    W("  Reminder from results/table3_provenance.txt: the N=4 row was produced")
    W("  on the classical_kraus_shots backend and the N=8 row on")
    W("  hqml_qnode_stinespring, so the two rows differ in solver as well as in")
    W("  Fock cutoff.  That is unchanged by this recomputation.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
