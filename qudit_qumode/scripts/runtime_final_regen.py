"""Regenerate tab:runtime and tab:deltasweep on the amplitude-calibrated base.

Everything here is derived from the SAME per-instance rows that Appendix D
prints (scripts/appendix_d_tables_tex.py), so the summary table, the Nc
projection, the detuning sweep and the multitone ratio cannot drift from
each other or from the two per-instance tables.

WHAT IS EXACT AND WHAT IS PROJECTED
  Nc = 3   every number is read from a compiled tape.  Exact.
  Nc > 3   there is no parametrized qudit builder in this repo
           (simulations/circuit_fig6_hqml.py is written for three modes),
           so the larger steps are PROJECTED: the per-class instance
           counts come from closed forms, checked against the Nc=3 tape,
           and each class is charged its Nc=3 MEAN instance time.  The
           angle spread within a class at Nc=3 is real -- chi_a, chi_b and
           chi_c differ -- so this assumes the extra instances at larger
           Nc look like the ones already measured.  Stated, not hidden.

  The MS-window column is a pure count, 5(Nc-1), and reproduces the
  10/25/35/55 already in the manuscript, which is the check that the
  count model is the same one the earlier table used.

DETUNING SWEEP
  Every dispersive gate runs for ONE DETUNING PERIOD, T = 2*pi/delta,
  and the compiled angle is delivered by the sideband AMPLITUDE.  The
  rows are READ from results/cpir_amplitude_sweep.txt rather than
  scaled here, and the read is echoed so it can be checked.
  Two consequences, both real:
    the slope REVERSES.  T = 2*pi/delta grows as delta falls, where the
    old T = theta/chi shrank, so the ratio now rises with delta.
    the leakage column is CONSTANT.  With T = 2*pi/delta the required
    amplitude is Omega_sb = delta sqrt(2 theta/pi), so both delta*T =
    2*pi and Omega_sb/delta = sqrt(2 theta/pi) are independent of
    delta.  In the simulated drive model the gate is then the same
    evolution at every detuning, rescaled in time.  Detuning buys
    duration and costs laser power rather than trading against
    fidelity.  Fixed hardware frequencies absent from the model can
    break that.
  The 1-F and leakage columns of results/qcr_delta_sweep.txt are
  SUPERSEDED -- measured at the uncalibrated amplitude and the formula
  duration.  Only its delta grid is still used.

Output: results/runtime_final_regen.txt
        results/runtime_final_regen.tex
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "scripts", "simulations", "circuits/trapped_ion_qubit"):
    sys.path.insert(0, str(ROOT / sub))

import appendix_d_tables_tex as A                              # noqa: E402

NC_REF = 3
NC_LIST = (3, 6, 8, 12)
SWEEP_SRC = ROOT / "results" / "qcr_delta_sweep.txt"
DELTA_REF = 40.0

L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


# class -> (count formula as a lambda of Nc, human-readable form, rule)
# Counts audited against Vu Eq. 18 to 21 for the chain with 2Nc-2 modes,
# second-order Trotter, two half steps
# (results/cluster_bound.md section 5d):
#   R1  one mode free evolution per mode per half step, modes
#       Nc high + (Nc-2) low = 2Nc-2                    -> 4Nc-4
#   R2  one dispersive sigma_z (x) n per chromophore, Eq. 20 term 1
#                                                       -> 2Nc
#   R3  one conditional displacement per site on its HIGH mode
#       (Eq. 20 term 2) plus one on its LOW mode (Eq. 21 term 1), the
#       two boundary low modes dropped with Eq. 52       -> 4Nc-4
#   R4  two mode-independent exchanges per bond          -> 2(Nc-1)
#   R5  link terms, one per end bond and two per interior bond
#                                                       -> 2(Nc-2)
QUDIT_N = {
    "Rotation":   (lambda n: 4 * n - 4,   "4Nc-4",   "R1"),
    "C_Pi R":     (lambda n: 2 * n,       "2Nc",     "R2"),
    "C_Pi_z D":   (lambda n: 4 * n - 4,   "4Nc-4",   "R3"),
    "X_E":        (lambda n: 2 * (n - 1), "2(Nc-1)", "R4"),
    "C_Pi_x D":   (lambda n: 2 * (n - 2), "2(Nc-2)", "R5"),
    "Z_d":        (lambda n: 2 * (n + 1), "2(Nc+1)", "R6"),
}
# The qubit CD splits by rule: the R3 instances are the real conditional
# displacements, the same terms and the same mean as the qudit's
# C_Pi_z D, while the R8 instances are the zero-time ones inside the J_l
# sandwiches.  Blending them, as the earlier 4Nc did, diluted the mean
# by exactly 12/8 and produced a spurious T0 difference.
QUBIT_N = {
    "Rotation":   (lambda n: 4 * n - 4,   "4Nc-4",   "R1"),
    "CR":         (lambda n: 2 * n,       "2Nc",     "R2"),
    "CD":         (lambda n: 4 * n - 4,   "4Nc-4",   "R3"),
    "CD_Jl":      (lambda n: 4 * (n - 2), "4(Nc-2)", "R8"),
    "r_z":        (lambda n: 6 * (n - 1), "6(Nc-1)", "R6"),
    "MS XX":      (lambda n: 2 * (n - 1), "2(Nc-1)", "R7"),
    "MS YY":      (lambda n: 2 * (n - 1), "2(Nc-1)", "R7"),
    "H":          (lambda n: 8 * (n - 1), "8(Nc-1)", "R8"),
    "CNOT":       (lambda n: 4 * (n - 1), "4(Nc-1)", "R8"),
}


def tapes():
    import pennylane as qml
    import circuit_fig6_hqml as QD
    import circuit_qubit_hybridlane as QB
    with qml.tape.QuantumTape() as t:
        QD.one_trotter_step(A.TAU)
    qd = A.build(list(t.operations), "qudit")
    with qml.tape.QuantumTape() as t:
        QB.one_hot_trotter_step()
    qb = split_cd(A.build(list(t.operations), "qubit"))
    return qd, qb


def split_cd(rows):
    """Relabel the zero-time J_l-internal CD as its own class.

    A.build tags them R8 and the real conditional displacements R3.
    per_class averages by class name, so without this the two are
    blended and the CD mean comes out low by 12/8 at Nc = 3.
    """
    return [(c + "_Jl" if (c == "CD" and r == "R8") else c, a, r, T)
            for c, a, r, T in rows]


def per_class(rows):
    """class -> (n_instances, total T, mean T)."""
    out = {}
    for c, _, _, T in rows:
        n, s = out.get(c, (0, 0.0))
        out[c] = (n + 1, s + T)
    return {c: (n, s, s / n) for c, (n, s) in out.items()}


def gate_counts(rows, model, name):
    """Verify every closed form against the Nc=3 tape.  Refuse on mismatch."""
    have = per_class(rows)
    bad = []
    for c, (f, txt, _) in model.items():
        got = have.get(c, (0, 0, 0))[0]
        if f(NC_REF) != got:
            bad.append(f"{name} {c}: formula {txt} gives {f(NC_REF)}, "
                       f"tape has {got}")
    for c in have:
        if c not in model:
            bad.append(f"{name} {c}: in the tape but has no count formula")
    return bad


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    qd, qb = tapes()
    cq, cb = per_class(qd), per_class(qb)

    W("=" * 104)
    W("RUNTIME REGENERATION ON THE PER-INSTANCE BASE")
    W("=" * 104)
    W(f"  commit {commit()}")
    W("  source of every row: the compiled tapes, via "
      "scripts/appendix_d_tables_tex.py")
    W()

    # ---- gate 1: the count model must reproduce the Nc=3 tape ------------
    bad = (gate_counts(qd, QUDIT_N, "qudit")
           + gate_counts(qb, QUBIT_N, "qubit"))
    W("0  VERIFICATION -- closed-form counts against the Nc=3 tape")
    if bad:
        for b in bad:
            W("   MISMATCH  " + b)
        W("   REFUSING TO PROJECT.")
        (ROOT / "results" / "runtime_final_regen.txt").write_text(
            "\n".join(L) + "\n", encoding="utf-8")
        return 1
    W(f"   all {len(QUDIT_N)} qudit and {len(QUBIT_N)} qubit closed forms "
      f"reproduce the tape at Nc=3.")
    tq = sum(v[0] for v in cq.values())
    tb = sum(v[0] for v in cb.values())
    W(f"   tape totals  qudit {tq} (16Nc-12 = {16*3-12})   "
      f"qubit {tb} (38Nc-44 = {38*3-44})")
    W(f"   logical      qudit {16*NC_REF-14}   qubit {16*NC_REF-16}   "
      f"(two qudit Z_d fuse, so 36 tape rows carry 34 ISA ops)")
    W()

    # ---- 1: per-class summary ------------------------------------------
    W("=" * 104)
    W("1  PER-CLASS SUMMARY AT Nc = 3  (this is the summary table)")
    W("=" * 104)
    T_QD3 = sum(v[1] for v in cq.values())
    T_QB3 = sum(v[1] for v in cb.values()) + A.N_JL * A.T_MS
    for nm, cc, model, extra in (("QUDIT", cq, QUDIT_N, None),
                                 ("QUBIT", cb, QUBIT_N,
                                  (f"J_l window", A.N_JL, A.T_MS))):
        W(f"  {nm}")
        W(f"    {'class':<11}{'rule':<7}{'count':<10}{'n':>4}"
          f"{'mean T (us)':>14}{'total (us)':>13}")
        tot = 0.0
        for c, (f, txt, rule) in model.items():
            n, s, m = cc[c]
            tot += s
            W(f"    {c:<11}{rule:<7}{txt:<10}{n:>4}{m:>14.4f}{s:>13.4f}")
        if extra:
            lbl, n, t = extra
            W(f"    {lbl:<11}{'R7':<7}{'(Nc-1)':<10}{n:>4}{t:>14.4f}"
              f"{n*t:>13.4f}")
            tot += n * t
        W(f"    {'':<11}{'':<7}{'':<10}{'':>4}{'TOTAL':>14}{tot:>13.4f}")
        W()

    mt, xeb, cxb, xe_seq, cx_seq = A.multitone(qd)
    W(f"  reference totals   qudit {T_QD3:.4f}   qubit {T_QB3:.4f}   "
      f"ratio {T_QB3/T_QD3:.4f}")
    W(f"  multitone qudit    {mt:.4f}   ratio {T_QB3/mt:.4f}")
    W(f"    X_E    {len(xeb)} blocks {sum(xeb):.4f} us, each "
      f"2*sqrt(sum theta^2)/(0.07 Omega_c) at the aggregate field "
      f"(was {xe_seq:.4f} over {cq['X_E'][0]} sequential tones)")
    W(f"    C_PixD {len(cxb)} block  {sum(cxb):.4f} us, the summed "
      f"exponential (was {cx_seq:.4f} over {cq['C_Pi_x D'][0]})")
    W(f"    Z_d fuses too but is virtual either way, T = 0.")
    W()

    # ---- 2: Nc projection ----------------------------------------------
    W("=" * 104)
    W("2  PROJECTION TO LARGER Nc   (PROJECTED, not compiled -- see header)")
    W("=" * 104)

    def totals(nc):
        qdt = sum(QUDIT_N[c][0](nc) * cq[c][2] for c in QUDIT_N)
        qbt = (sum(QUBIT_N[c][0](nc) * cb[c][2] for c in QUBIT_N)
               + 2 * (nc - 2) * A.T_MS)
        ms = 2 * (nc - 1) + 2 * (nc - 1) + 2 * (nc - 2)
        # multitone: X_E -> 2 blocks, C_Pi_x D -> 1 block
        mtt = (qdt
               - QUDIT_N["X_E"][0](nc) * cq["X_E"][2] + sum(xeb)
               - QUDIT_N["C_Pi_x D"][0](nc) * cq["C_Pi_x D"][2] + sum(cxb))
        return qdt, qbt, ms, mtt

    W(f"  {'Nc':>3}{'qudit ops':>11}{'qubit ops':>11}{'T_qubit':>12}"
      f"{'MS win':>8}{'T_qudit':>11}{'ratio':>8}"
      f"{'T_mt':>11}{'ratio_mt':>10}")
    rows_tex = []
    for nc in NC_LIST:
        qdt, qbt, ms, mtt = totals(nc)
        W(f"  {nc:>3}{16*nc-14:>11}{16*nc-16:>11}{qbt:>12.1f}{ms:>8}"
          f"{qdt:>11.1f}{qbt/qdt:>8.1f}{mtt:>11.1f}{qbt/mtt:>10.1f}")
        rows_tex.append((nc, 16*nc-14, 16*nc-16, qbt, ms, qdt, qbt/qdt))
    W()
    W("  AUDITED CHAIN COUNTS.  Rotation and the R3 conditional")
    W("  displacement are both 4Nc-4: one mode free evolution per mode")
    W("  per half step over the 2Nc-2 modes, and per site one R3 on its")
    W("  high mode (Vu Eq. 20 term 2) plus one on its low mode (Eq. 21")
    W("  term 1) with the two boundary low modes dropped.  The qubit CD")
    W("  is split by rule, 4Nc-4 real ones at R3 and 4(Nc-2) zero-time")
    W("  ones inside the J_l; blending them diluted the mean by 12/8.")
    W("  Closed forms: qudit ISA 16Nc-14, qubit logical 16Nc-16,")
    W("  multitone 10Nc-3, MS windows 6Nc-8.  T0 is now identical on")
    W("  both sides, so T0_qubit - T0_qudit = 0 at every Nc.")
    W("")
    W("  The link term C_Pi_x D, and the qubit J_l that")
    W("  mirrors it, are counted 2(Nc-2) per step, not Nc-1: Vu Eq. 21")
    W("  gives every bond two vibronic link terms, one per endpoint")
    W("  low-frequency mode, and Eq. 52 drops the two boundary modes, so")
    W("  the end bonds keep one each and interior bonds two.  The MS-window")
    W("  column is therefore 4(Nc-1) + 2(Nc-2) = 6Nc-8, giving 10/28/40/64.")
    W("  At Nc = 3 that is 10, unchanged, because both bonds are end bonds")
    W("  and 2(Nc-2) = Nc-1 = 2.  See results/cluster_bound.md section 5b.")
    W()

    # ---- 3: detuning sweep ---------------------------------------------
    W("=" * 104)
    W("3  DETUNING SWEEP ON THE AMPLITUDE-CALIBRATED BASE")
    W("=" * 104)
    W(f"  reading the delta grid from {SWEEP_SRC.relative_to(ROOT)}")
    W("  Its 1-F and leakage columns are SUPERSEDED: they were measured at")
    W("  the uncalibrated amplitude and the formula duration, neither of")
    W("  which this table now uses.")
    src = SWEEP_SRC.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"^\s{4,}(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                     r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$",
                     re.M)
    sweep = []
    for m in pat.finditer(src):
        sweep.append(dict(delta=float(m.group(1)), om_d=float(m.group(2)),
                          T=float(m.group(4)), fid=float(m.group(8)),
                          leak=float(m.group(9))))
    if not sweep:
        W("   COULD NOT PARSE THE SWEEP FILE -- refusing to invent columns.")
        (ROOT / "results" / "runtime_final_regen.txt").write_text(
            "\n".join(L) + "\n", encoding="utf-8")
        return 1
    for r in sweep:
        W(f"    read  delta {r['delta']:>4.0f} kHz   Om/del {r['om_d']:.3f}   "
          f"T {r['T']:>7.3f} us   1-F(fit+VZ) {r['fid']:.6e}   "
          f"leak P(g) {r['leak']:.6e}")
    W()
    pm_src = ROOT / "results" / "cpir_amplitude_sweep.txt"
    pm_txt = pm_src.read_text(encoding="utf-8", errors="replace")
    pm = {}
    for m in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                         r"([\d.]+)\s+([\d.e+-]+)\s+([\d.]+)\s+([\d.]+)"
                         r"\s+([\d.]+)\s+([\d.]+)\s*$", pm_txt, re.M):
        pm[float(m.group(1))] = dict(T=float(m.group(2)),
                                     amp=float(m.group(3)),
                                     om=float(m.group(4)),
                                     om_del=float(m.group(5)),
                                     leak=float(m.group(6)),
                                     six=float(m.group(7)))
    W(f"  reading the calibrated sweep from {pm_src.relative_to(ROOT)}")
    if set(pm) != {r["delta"] for r in sweep}:
        W(f"   DELTA GRIDS DO NOT MATCH: calibrated {sorted(pm)} against "
          f"sweep {sorted(r['delta'] for r in sweep)} -- refusing.")
        (ROOT / "results" / "runtime_final_regen.txt").write_text(
            "\n".join(L) + "\n", encoding="utf-8")
        return 1
    for d in sorted(pm):
        p = pm[d]
        W(f"    read  delta {d:>4.0f} kHz   T {p['T']:>7.3f} us   "
          f"Om_sb/2pi {p['om']:>6.3f} kHz   Om_sb/delta {p['om_del']:.4f}   "
          f"leak {p['leak']:.2e}   six-gate {p['six']:>8.3f} us")
    W()
    W("  NOTE: leak and Omega_sb/delta are IDENTICAL at every detuning.")
    W("  That is exact in this model, not a coincidence.  With")
    W("  T = 2*pi/delta the required amplitude is")
    W("  Omega_sb = delta sqrt(2 theta/pi), so Omega_sb/delta = sqrt(2 theta/pi)")
    W("  and delta*T = 2*pi are both independent of delta, so in the")
    W("  simulated drive model the whole gate is the same evolution up to a")
    W("  rescaling of time.  Under this rule the detuning buys duration and")
    W("  costs laser power rather than trading against fidelity, which is")
    W("  what the published sweep column showed.  Fixed hardware")
    W("  frequencies absent from the model can break that.")
    W()
    W(f"  {'delta':>6}{'T (us)':>10}{'Om_sb/2pi':>11}{'Om/del':>9}"
      f"{'leak':>10}{'T_qudit':>11}{'T_qubit':>11}{'ratio':>8}")
    sweep_tex = []
    for r in sweep:
        p = pm[r["delta"]]
        qdt = T_QD3 - cq["C_Pi R"][1] + p["six"]
        qbt = T_QB3 - cb["CR"][1] + p["six"]
        W(f"  {r['delta']:>6.0f}{p['T']:>10.3f}{p['om']:>11.3f}"
          f"{p['om_del']:>9.4f}{p['leak']:>10.2e}"
          f"{qdt:>11.1f}{qbt:>11.1f}{qbt/qdt:>8.1f}")
        sweep_tex.append((r["delta"], p["T"], p["om"], p["amp"],
                          p["leak"], qbt / qdt))
    W()
    W("  The slope REVERSES relative to the published table.  That priced")
    W("  T = theta/chi, which grows with delta, so a smaller detuning gave")
    W("  a shorter gate and a better ratio.  Here T = 2*pi/delta, so a")
    W("  smaller")
    W("  detuning gives a LONGER gate and a worse ratio, while asking for")
    W("  less laser power.")
    W()

    # ---- LaTeX ----------------------------------------------------------
    X = []
    X.append("% generated by scripts/runtime_final_regen.py at commit "
             + commit())
    X.append("% PROJECTED for Nc > 3; see the header of that script")
    X.append("")
    X.append(r"\begin{table*}[t]")
    X.append(r"  \centering")
    X.append(r"  \begin{tabular}{ccccccc}")
    X.append(r"\toprule")
    X.append(r"$N_c$ & Qudit ops & Qubit ops & $T_{\rm qubit}$ ($\mu$s) &")
    X.append(r"MS windows & $T_{\rm qudit}$ ($\mu$s) & Ratio \\")
    X.append(r"\midrule")
    for nc, qo, bo, qbt, ms, qdt, rat in rows_tex:
        X.append(f"{nc:<3}& {qo:<4}& {bo:<4}& {qbt:.0f} & {ms:<3}& "
                 f"{qdt:.0f} & {rat:.1f} " + r"\\")
    X.append(r"\bottomrule")
    X.append(r"\end{tabular}")
    # caption wording is the author's, synced from mas.tex.
    X.append(r"  \caption{"
             r"Projected time for one second-order Trotter step under the "
             r"per-instance rules above. Operation counts are logical, "
             r"$16N_c-14$ for the qudit and $16N_c-16$ for the qubit "
             r"reference, and $10N_c-3$ multitone control blocks. "
             r"At $N_c=3$ every entry is read from a compiled "
             r"circuit. The larger $N_c$ are projected, charging each gate "
             r"class its $N_c=3$ mean instance time at the instance count "
             r"its closed form gives, since this work compiles the qudit "
             r"circuit only at $N_c=3$. The ratio is $T_{\rm qubit}/T_{\rm "
             r"qudit}$. With the multitone $27$-control-block qudit "
             r"schedule at $N_c=3$ the qudit step is $209.5~\mu$s and the "
             r"ratio $10.5$."
             + r"}")
    X.append(r"  \label{tab:runtime}")
    X.append(r"\end{table*}")
    X.append("")
    X.append(r"\begin{table}[h]")
    X.append(r"  \centering")
    X.append(r"  \small")
    X.append(r"  \setlength{\tabcolsep}{4pt}")
    X.append(r"  \begin{tabular}{rrrrlc}")
    X.append(r"    \toprule")
    X.append(r"    $\delta/2\pi$ & $T$ & $\Omega_{\rm sb}/2\pi$ & "
             r"$\Omega_{\rm sb}/\Omega_{\rm sb}^{\rm meas}$ & "
             r"Leakage & Ratio \\")
    X.append(r"    (kHz) & ($\mu$s) & (kHz) & & & (seq.) \\")
    X.append(r"    \midrule")
    for d, T, om, amp, leak, rat in sweep_tex:
        e = f"{leak:.1e}".split("e")
        X.append(f"    {d:.0f} & {T:.2f} & {om:.2f} & {amp:.2f}$\\times$ & "
                 f"${e[0]}\\times10^{{{int(e[1])}}}$ & {rat:.1f} " + r"\\")
    X.append(r"    \bottomrule")
    X.append(r"  \end{tabular}")
    # caption wording is the author's, transcribed from mas.tex.
    X.append(r"  \caption{$C_{\Pi}R$ across the detuning window under "
             r"amplitude calibration. Every dispersive gate runs for one "
             r"detuning period, $T=2\pi/\delta$, and the compiled angle "
             r"is set by the sideband amplitude. To leading order "
             r"$\theta=\chi T$ with $\chi=\Omega_{\rm sb}^2/4\delta$ "
             r"gives $\Omega_{\rm sb}=\delta\sqrt{2\theta/\pi}$, "
             r"which fixes the scaling. The amplitudes tabulated here "
             r"are solved numerically under the full drive and run about "
             r"$2\%$ above it. In the simulated drive model the endpoint "
             r"leakage and the ratio "
             r"$\Omega_{\rm sb}/\delta\approx0.10$ come out the same "
             r"at every detuning, as expected when neither $\delta T$ "
             r"nor $\Omega_{\rm sb}/\delta$ depends on $\delta$. Fixed "
             r"hardware frequencies not in the model can break this. The "
             r"detuning trades duration against laser power. The last "
             r"column is the projected sequential runtime ratio at "
             r"$N_c=3$ from Appendix~\ref{app:runtime} with all other "
             r"gates unchanged. It rises with $\delta$. The ends of the "
             r"window ask for $2.06$ and $8.23$~kHz against a measured "
             r"$4.0$~kHz.}")
    X.append(r"  \label{tab:deltasweep}")
    X.append(r"\end{table}")

    # ---- 4: every other number the manuscript quotes --------------------
    W("=" * 104)
    W("4  DERIVED VALUES THE MANUSCRIPT QUOTES ELSEWHERE")
    W("=" * 104)
    r1q = T_QD3 - cq["Rotation"][1]
    r1b = T_QB3 - cb["Rotation"][1]
    W(f"  R1 removed from BOTH sides (the 5.0 us rotation assumption):")
    W(f"    qudit {T_QD3:.4f} - {cq['Rotation'][1]:.1f} = {r1q:.4f}")
    W(f"    qubit {T_QB3:.4f} - {cb['Rotation'][1]:.1f} = {r1b:.4f}")
    W(f"    ratio {r1b/r1q:.4f}  -> {r1b/r1q:.1f}   "
      f"(was quoted as '9.9 to 12.4' on the bracketed base)")
    W(f"  R1 share of the qudit step: {100*cq['Rotation'][1]/T_QD3:.1f}% "
      f"of {T_QD3:.1f}; of the qubit step "
      f"{100*cb['Rotation'][1]/T_QB3:.1f}% of {T_QB3:.1f}")
    lo = min(sweep_tex, key=lambda r: r[5])
    hi = max(sweep_tex, key=lambda r: r[5])
    W(f"  sweep ratio endpoints: {lo[5]:.1f} at {lo[0]:.0f} kHz, "
      f"{hi[5]:.1f} at {hi[0]:.0f} kHz")
    q3 = [r for r in rows_tex if r[0] == 3][0]
    q12 = [r for r in rows_tex if r[0] == 12][0]
    W(f"  ratio across Nc: {q3[6]:.1f} at Nc=3 -> {q12[6]:.1f} at Nc=12")
    W()

    (ROOT / "results" / "runtime_final_regen.tex").write_text(
        "\n".join(X) + "\n", encoding="utf-8")
    W("  wrote results/runtime_final_regen.tex")
    W()
    W("  READ ONLY.  No adoption.")
    (ROOT / "results" / "runtime_final_regen.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
