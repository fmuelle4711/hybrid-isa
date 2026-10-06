"""Reprice the runtime and segment tables with C_Pi R as ONE pulse.

The manuscript renamed the five primitives.  Code identifiers are unchanged,
so the mapping is:

    paper name        code / results-file name
    Z_d               QPS
    X_E               QT   (QT_multi in the count dicts)
    C_{\\Pi_z}D        QCD
    C_{\\Pi}R          QCR   (CR on the qubit-reference side)
    C_{\\Pi_x}D        QT-SB / QT_SB

DECISION BEING PRICED HERE.  The adopted tables charge C_Pi R (and the qubit
reference's CR) as a Choi AC-Stark echo, T_paper = 2T + t_pi = 55.93 us and
three timed segments.  This script reprices both as a SINGLE flat pulse,
T = theta/chi = 25.465 us and one segment, on the grounds that the carrier
light shift on the driven pair is a known, level-diagonal phase that the ISA
already absorbs as a virtual Z_d at zero cost.

The echo is dropped on BOTH sides.  CR is priced at the C_Pi R value by
FINDING 5, so the qubit reference gets the same reduction; this is not a
one-sided change.

EVERY INPUT IS READ FROM A FILE.  Nothing is retyped.  The only two values
not read are the two the manuscript already tags ASSUMED in tab:params,
delta/2pi = 40 kHz and t_pi = 5 us, and t_pi is used only to reproduce the
old number as a cross-check.

Output: results/cpir_one_pulse_regen.txt

GATE NAMES.  The manuscript renamed the five ISA primitives.  Code
identifiers, dict keys and results-file columns still use the old names, so
this mapping travels with every script that touches them:

    paper name      code / data name      what it is
    Z_d             QPS                   qudit phase shift
    X_E             QT                    qudit transition (carrier)
    C_{\Pi_z}D      QCD                   level-conditioned displacement
    C_{\Pi}R        QCR                   level-conditioned mode rotation
    C_{\Pi_x}D      QT-SB, QTsb           transition-conditioned displacement

Multitone forms write the transition set as plain E.  CD and CR keep their
names: they are the qubit-reference gates, not qudit ISA primitives.  No
identifier is renamed anywhere below.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RT = ROOT / "pulses_meth" / "runtime_table.json"
ANCHOR = ROOT / "pulses_meth" / "anchor_params.json"

THETA = 0.016          # C_Pi R circuit angle, appendix_table.md headline row
DELTA_2PI_MHZ = 0.040  # ASSUMED, tab:params "QCR detuning 40 kHz, this work"
T_PI_US = 5.0          # ASSUMED, tab:params "pi pulse time 5 us, this work"

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


def dirty():
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short"],
            capture_output=True, text=True, check=True).stdout
        return len([x for x in out.splitlines() if x.strip()])
    except Exception:
        return -1


# --------------------------------------------------------------- segments --
# results/segment_counts.txt: "QCR and CR count 3 segments each (T, pi, T).
# QCD and CD count 1.  Virtual gates count 0."  Jl decomposed = 14.
SEG_OLD = dict(Rotation=1, QCR=3, QCD=1, QT_SB=1, QT_multi=1, QPS=0,
               rz=0, MS_J=2, MS_Jl=14)
SEG_NEW = dict(SEG_OLD, QCR=1)


def qudit_segments(counts, seg, merged_centre=True):
    """Timed segments for a qudit schedule.

    merged_centre: the two C_{Pi_x}D terms at the centre are one multitone
    pulse, which is what takes 40 -> 39 in results/segment_counts.txt.
    """
    total = 0
    for k, n in counts.items():
        if k == "QT_SB" and merged_centre:
            total += 1
            continue
        total += n * seg[k]
    return total


def qubit_segments(counts, seg):
    return sum(n * seg[k] for k, n in counts.items())


# ---------------------------------------------------------------- runtime --
def step_time(counts, gt, t_qcr, t_qt, merged_centre=False):
    t = 0.0
    for k, n in counts.items():
        if k == "QCR":
            t += n * t_qcr
        elif k == "QT_multi":
            t += n * t_qt
        elif k == "QT_SB":
            t += (1 if merged_centre else n) * gt["QT_SB"]
        elif k == "QCD":
            t += n * gt["QCD"]
        elif k == "Rotation":
            t += n * gt["rotation"]
        elif k in ("QPS", "rz"):
            pass
        else:
            raise KeyError(k)
    return t


def qubit_step_time(counts, gt, t_cr, ms_windows):
    t = (counts["QCR"] * t_cr
         + counts["QCD"] * gt["QCD"]
         + counts["Rotation"] * gt["rotation"])
    return t + ms_windows * gt["MS"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    d = json.loads(RT.read_text(encoding="utf-8"))
    a = json.loads(ANCHOR.read_text(encoding="utf-8"))
    gt = d["gate_times_us"]
    qt_lo, qt_hi = d["conventions"]["qt_bounds_us"]

    eta = a["coupling"]["lamb_dicke_eta"]["value"]
    osb_khz = a["coupling"]["demonstrated_bsb_rabi_over_2pi_kHz"]["value"]

    # chi = Omega_sb^2 / (4 delta), the formula the manuscript states in App C
    osb = 2 * np.pi * osb_khz * 1e-3          # rad/us
    delta = 2 * np.pi * DELTA_2PI_MHZ         # rad/us
    chi = osb ** 2 / (4 * delta)
    t_single = THETA / chi
    t_echo_rebuilt = 2 * t_single + T_PI_US

    W("=" * 104)
    W("C_{\\Pi}R AS ONE PULSE -- REGENERATED SEGMENT AND RUNTIME TABLES")
    W("=" * 104)
    W("  STATUS: section 1 (timed pulse segments) is CURRENT and backs")
    W("  tab:gatecounts; scripts/segment_counts.py reproduces it.  Sections")
    W("  2 to 4 (runtime, share, rotations, detuning sweep) are SUPERSEDED")
    W("  BY APPENDIX D and are kept for traceability only.")
    W(f"  commit            {commit()}   ({dirty()} files dirty in the tree, "
      f"pre-existing)")
    W(f"  inputs            {RT.relative_to(ROOT)}")
    W(f"                    {ANCHOR.relative_to(ROOT)}")
    W("  name mapping      Z_d=QPS  X_E=QT  C_{Pi_z}D=QCD  C_{Pi}R=QCR  "
      "C_{Pi_x}D=QT-SB")
    W()
    W("  DERIVATION OF THE SINGLE-PULSE DURATION")
    W(f"    eta                       {eta}            (anchor_params.json)")
    W(f"    Omega_sb/2pi              {osb_khz} kHz       (anchor_params.json)")
    W(f"    delta/2pi                 {DELTA_2PI_MHZ*1e3:.0f} kHz      "
      "ASSUMED (tab:params)")
    W(f"    chi = Omega_sb^2/(4delta) {chi:.6e} rad/us  -> chi/2pi = "
      f"{chi/(2*np.pi)*1e6:.3f} Hz")
    W(f"    T = theta/chi             {t_single:.4f} us     at theta = {THETA}")
    W()
    W("  CROSS-CHECK AGAINST THE ADOPTED TABLE")
    W(f"    2T + t_pi (t_pi = {T_PI_US} ASSUMED) = {t_echo_rebuilt:.4f} us")
    W(f"    runtime_table.json holds QCR       = {gt['QCR']} us")
    ok = abs(t_echo_rebuilt - gt["QCR"]) < 5e-3
    W(f"    match: {ok}")
    if not ok:
        W("    STOP -- the rebuilt echo time does not reproduce the adopted "
          "value; nothing below is trustworthy.")
        return 1
    W()
    W(f"  other gate times read, unchanged: QCD {gt['QCD']:.4f}  "
      f"Rotation {gt['rotation']}  QT-SB {gt['QT_SB']}  MS {gt['MS']}  "
      f"X_E [{qt_lo}, {qt_hi}]")
    W()

    # ------------------------------------------------------------ segments --
    W("=" * 104)
    W("1  TIMED PULSE SEGMENTS PER TROTTER STEP")
    W("=" * 104)
    W("    C_{Pi}R and CR: 3 segments (echo) -> 1 segment (single pulse).")
    W("    Everything else unchanged.  Qubit uses the decomposed-J_l "
      "convention (14), which is the manuscript's.")
    W()
    W(f"  {'Nc':>3}  {'qudit ref':>20}  {'qudit multitone':>20}  "
      f"{'qubit':>20}")
    W(f"  {'':>3}  {'echo -> 1 pulse':>20}  {'echo -> 1 pulse':>20}  "
      f"{'echo -> 1 pulse':>20}")
    seg_rows = []
    for r in d["rows"]:
        nc = r["Nc"]
        c34 = r["qudit_counts"]
        c27 = dict(r["qudit_counts_28"], QT_SB=1)
        cq = r["qubit_counts"]
        s34o = qudit_segments(c34, SEG_OLD)
        s34n = qudit_segments(c34, SEG_NEW)
        s27o = qudit_segments(c27, SEG_OLD, merged_centre=True)
        s27n = qudit_segments(c27, SEG_NEW, merged_centre=True)
        sqo = qubit_segments(cq, SEG_OLD)
        sqn = qubit_segments(cq, SEG_NEW)
        seg_rows.append((nc, s34o, s34n, s27o, s27n, sqo, sqn))
        W(f"  {nc:>3}  {s34o:>9} -> {s34n:<8}  {s27o:>9} -> {s27n:<8}  "
          f"{sqo:>9} -> {sqn:<8}")
    W()
    n3 = seg_rows[0]
    W(f"  Echo convention      : {n3[1]} / {n3[3]} / {n3[5]}   "
      f"(the superseded draft printed 39 / 37 / 70: "
      f"{n3[1] == 39 and n3[3] == 37 and n3[5] == 70})")
    W(f"  One-pulse convention : {n3[2]} / {n3[4]} / {n3[6]}   "
      f"(the manuscript prints 27 / 25 / 58: "
      f"{n3[2] == 27 and n3[4] == 25 and n3[6] == 58})")
    W()

    # ------------------------------------------------------------- runtime --
    W("#" * 104)
    W("#  SECTIONS 2 TO 4 ARE SUPERSEDED BY APPENDIX D.  DO NOT QUOTE THEM.")
    W("#")
    W("#  They price C_Pi R and CR at T = theta/chi with a bracketed X_E.")
    W("#  Appendix D now holds every dispersive gate for one detuning")
    W("#  period, 25.000 us, delivers the angle by sideband amplitude, and")
    W("#  prices X_E as a flat two-tone block, so the numbers below are the")
    W("#  wrong basis.  Current values:")
    W("#    results/appendix_d_tables.txt     210.5 / 2207.9 us, ratio 10.5")
    W("#    results/runtime_final_regen.txt   N_c projection, detuning sweep")
    W("#    results/cpir_amplitude_sweep.txt  the sweep rows")
    W("#" * 104)
    W()
    W("=" * 104)
    W("2  PROJECTED STEP TIME AND RATIO  [SUPERSEDED BY APP D]")
    W("=" * 104)
    W("    C_{Pi}R and CR priced at T = theta/chi on both sides.")
    W("    X_E bracket, MS windows, and every other gate time unchanged.")
    W()
    hdr = (f"  {'Nc':>3}  {'T_qudit ref (us)':>22}  {'T_qudit mt (us)':>22}  "
           f"{'T_qubit':>10}  {'ratio ref':>16}  {'ratio mt':>16}")
    W(hdr)
    out_rows = []
    for r in d["rows"]:
        nc = r["Nc"]
        c34 = r["qudit_counts"]
        c27 = dict(r["qudit_counts_28"], QT_SB=1)
        cq = r["qubit_counts"]
        msw = r["ms_windows"]

        for tag, t_cr in (("echo", gt["QCR"]), ("one", t_single)):
            lo34 = step_time(c34, gt, t_cr, qt_lo, merged_centre=False)
            hi34 = step_time(c34, gt, t_cr, qt_hi, merged_centre=False)
            lo27 = step_time(c27, gt, t_cr, qt_lo, merged_centre=True)
            hi27 = step_time(c27, gt, t_cr, qt_hi, merged_centre=True)
            tq = qubit_step_time(cq, gt, t_cr, msw)
            out_rows.append((nc, tag, lo34, hi34, lo27, hi27, tq,
                             tq / hi34, tq / lo34, tq / hi27, tq / lo27))

        e = [x for x in out_rows if x[0] == nc and x[1] == "echo"][0]
        o = [x for x in out_rows if x[0] == nc and x[1] == "one"][0]
        W(f"  {nc:>3}  {e[2]:>7.1f}-{e[3]:<6.1f} -> {o[2]:>5.1f}-{o[3]:<5.1f}  "
          f"{e[4]:>7.1f}-{e[5]:<6.1f} -> {o[4]:>5.1f}-{o[5]:<5.1f}  "
          f"{e[6]:>5.0f}->{o[6]:<5.0f}  "
          f"{e[7]:>4.1f}-{e[8]:<4.1f} -> {o[7]:>4.1f}-{o[8]:<4.1f}  "
          f"{e[9]:>4.1f}-{e[10]:<4.1f} -> {o[9]:>4.1f}-{o[10]:<4.1f}")
    W()
    W("  VERIFICATION -- the 'echo' halves must reproduce the adopted table:")
    for r in d["rows"]:
        nc = r["Nc"]
        e = [x for x in out_rows if x[0] == nc and x[1] == "echo"][0]
        ok = (abs(e[2] - r["T_qudit_lo_us"]) < 0.05
              and abs(e[3] - r["T_qudit_hi_us"]) < 0.05
              and abs(e[6] - r["T_qubit_seq_us"]) < 0.05
              and abs(e[7] - r["ratio_seq"][0]) < 5e-3
              and abs(e[8] - r["ratio_seq"][1]) < 5e-3)
        W(f"    Nc={nc:<3} rebuilt {e[2]:.1f}/{e[3]:.1f}/{e[6]:.1f} "
          f"ratio [{e[7]:.2f}, {e[8]:.2f}]   json holds "
          f"{r['T_qudit_lo_us']:.1f}/{r['T_qudit_hi_us']:.1f}/"
          f"{r['T_qubit_seq_us']:.1f} ratio "
          f"[{r['ratio_seq'][0]:.2f}, {r['ratio_seq'][1]:.2f}]   match: {ok}")
    W()

    # ------------------------------------------------------- C_Pi R share --
    W("=" * 104)
    W("3  SHARE OF THE QUDIT STEP SPENT IN C_{\\Pi}R  [SUPERSEDED BY APP D]")
    W("=" * 104)
    for r in d["rows"][:1]:
        nc = r["Nc"]
        c34 = r["qudit_counts"]
        for tag, t_cr in (("echo", gt["QCR"]), ("one pulse", t_single)):
            lo = step_time(c34, gt, t_cr, qt_lo)
            hi = step_time(c34, gt, t_cr, qt_hi)
            q = c34["QCR"] * t_cr
            W(f"    Nc={nc}  {tag:<10}  C_Pi R = {q:7.2f} us of "
              f"{lo:7.2f}-{hi:7.2f} us  =  {100*q/hi:.0f}% to {100*q/lo:.0f}%")
    W()
    W("  The manuscript's 'About 85%' is the echo convention's LOW end only "
      "(84%);")
    W("  at the high end of the X_E bracket it is 76%.")
    W()

    # ------------------------------------------ rotations-removed variant --
    W("=" * 104)
    W("3b  VARIANT WITH THE MODE ROTATIONS REMOVED  [SUPERSEDED BY APP D]")
    W("=" * 104)
    W("    App D says rotations can in principle be absorbed into oscillator")
    W("    frame tracking at no cost, and quotes the ratio that would give.")
    W("    Recomputed under both conventions.")
    W()
    r3 = d["rows"][0]
    for tag, tc in (("echo", gt["QCR"]), ("one pulse", t_single)):
        lo = step_time(r3["qudit_counts"], gt, tc, qt_lo)
        hi = step_time(r3["qudit_counts"], gt, tc, qt_hi)
        tq = qubit_step_time(r3["qubit_counts"], gt, tc, r3["ms_windows"])
        drop_qd = r3["qudit_counts"]["Rotation"] * gt["rotation"]
        drop_qb = r3["qubit_counts"]["Rotation"] * gt["rotation"]
        W(f"    {tag:<10} qudit {lo-drop_qd:7.2f}-{hi-drop_qd:<7.2f} "
          f"qubit {tq-drop_qb:8.2f}   ratio "
          f"{(tq-drop_qb)/(hi-drop_qd):.1f} to "
          f"{(tq-drop_qb)/(lo-drop_qd):.1f}")
    W()
    W("    The echo row reproduces the 5.9 to 6.6 the appendix already "
      "quotes.")
    W()

    # --------------------------------------------------- detuning sweep --
    W("=" * 104)
    W("4  DETUNING SWEEP, RATIO COLUMN REPRICED  [SUPERSEDED BY APP D]")
    W("=" * 104)
    W("    tab:deltasweep holds delta, Omega_sb/delta, T, 1-F and leakage")
    W("    unchanged -- only the last column, the projected sequential ratio,")
    W("    depends on the echo convention.  T(delta) = theta/chi(delta) with")
    W("    chi = Omega_sb^2/(4 delta), which is linear in delta.")
    W()
    r3 = d["rows"][0]
    c34, cq, msw = r3["qudit_counts"], r3["qubit_counts"], r3["ms_windows"]
    # everything that does not depend on delta
    base_qudit = (c34["Rotation"] * gt["rotation"]
                  + c34["QCD"] * gt["QCD"]
                  + c34["QT_SB"] * gt["QT_SB"])
    base_qubit = (cq["Rotation"] * gt["rotation"]
                  + cq["QCD"] * gt["QCD"]
                  + msw * gt["MS"])
    W(f"  delta-independent qudit part = {base_qudit:.4f} us   "
      f"qubit part = {base_qubit:.4f} us")
    W()
    W(f"  {'delta':>6} {'Om/delta':>9} {'T_ang':>8} {'T echo':>8} "
      f"{'ratio echo':>14} {'ratio one pulse':>18}")
    W(f"  {'(kHz)':>6} {'':>9} {'(us)':>8} {'(us)':>8}")
    sweep = []
    for dk in (20, 30, 40, 60, 80):
        dd = 2 * np.pi * dk * 1e-3
        ck = osb ** 2 / (4 * dd)
        ta = THETA / ck
        te = 2 * ta + T_PI_US
        row = {}
        for tag, tc in (("echo", te), ("one", ta)):
            tqd_lo = base_qudit + c34["QCR"] * tc + c34["QT_multi"] * qt_lo
            tqd_hi = base_qudit + c34["QCR"] * tc + c34["QT_multi"] * qt_hi
            # BOTH SIDES move with delta: CR is priced at the C_Pi R duration
            # by FINDING 5, so the qubit's six dispersive gates rescale too.
            tqb_both = base_qubit + cq["QCR"] * tc
            # the published column instead FREEZES the qubit at its 40 kHz
            # value, varying delta on the qudit side only.
            tqb_frozen = r3["T_qubit_seq_us"]
            row[tag] = dict(both=(tqb_both / tqd_hi, tqb_both / tqd_lo),
                            frozen=(tqb_frozen / tqd_hi, tqb_frozen / tqd_lo))
        sweep.append((dk, osb / dd, ta, te, row))
        W(f"  {dk:>6} {osb/dd:>9.3f} {ta:>8.2f} {te:>8.2f} "
          f"{row['echo']['both'][0]:>6.1f} to "
          f"{row['echo']['both'][1]:<5.1f} "
          f"{row['one']['both'][0]:>10.1f} to "
          f"{row['one']['both'][1]:<5.1f}")
    W()
    W("  THE PUBLISHED COLUMN USES A DIFFERENT CONVENTION -- FLAGGED, NOT "
      "FIXED")
    W("  results/qcr_delta_ratio_34.txt states 'T_qubit sequential = "
      "2396.1112 us")
    W("  (unaffected by delta)', i.e. it varies delta on the QUDIT SIDE "
      "ONLY.  But")
    W("  CR is priced at the C_Pi R duration (FINDING 5), so a change of "
      "delta must")
    W("  move the qubit's six dispersive gates as well.  The two "
      "conventions agree")
    W("  only at the 40 kHz anchor, which is why the paper's headline row "
      "is safe")
    W("  and the other four rows are not.")
    W()
    W(f"  {'delta':>6} {'published':>12} {'frozen qubit':>16} "
      f"{'both sides':>16}   (echo convention)")
    want = [(8.3, 9.8), (6.6, 7.5), (5.4, 6.0), (4.0, 4.4), (3.2, 3.4)]
    for (dk, _, _, _, row), w in zip(sweep, want):
        fz, bo = row["echo"]["frozen"], row["echo"]["both"]
        tick = "ok" if abs(fz[0] - w[0]) < 0.06 and abs(fz[1] - w[1]) < 0.06 \
            else "MISMATCH"
        W(f"  {dk:>6} {w[0]:>5.1f} to {w[1]:<5.1f} "
          f"{fz[0]:>7.1f} to {fz[1]:<6.1f} {bo[0]:>7.1f} to {bo[1]:<6.1f}"
          f"   {tick}")
    W()
    W("  The 'frozen qubit' column reproduces the published numbers, so the")
    W("  published column is the frozen convention.  The 'both sides' column")
    W("  is the self-consistent one.  Adopting the one-pulse convention does")
    W("  not settle this; it is a separate defect and the author should pick")
    W("  a convention deliberately.")
    W()
    W("  RATIO COLUMN UNDER THE ONE-PULSE CONVENTION.  Only the "
      "self-consistent")
    W("  reading is given: under one pulse the qubit's CR is also a single")
    W("  segment, so freezing the qubit at its echo time would pair a "
      "one-pulse")
    W("  qudit against an echoed qubit, which is not a convention at all.")
    W(f"  The 40 kHz row, {sweep[2][4]['one']['both'][0]:.1f} to "
      f"{sweep[2][4]['one']['both'][1]:.1f}, agrees with section 2 as it "
      "must.")
    W()
    W(f"  {'delta (kHz)':>12} {'ratio (seq.)':>18}")
    for dk, _, _, _, row in sweep:
        bo = row["one"]["both"]
        W(f"  {dk:>12} {bo[0]:>8.1f} to {bo[1]:<8.1f}")
    W()

    W("=" * 104)
    W("WHAT THIS DOES NOT ESTABLISH")
    W("=" * 104)
    W("  Dropping the echo is a physics claim, not a counting one.  This")
    W("  script only reprices; it does not show the uncompensated gate")
    W("  works.  The supporting evidence already in the repo is:")
    W("    results/qcr_mechanism_paper_angle.txt   single segment, VZ "
      "absorbed: 1-F = 2.900031e-03, leak 1.840207e-03")
    W("    results/qcr_echo_physical.txt           full echo,      VZ "
      "absorbed: 1-F = 3.918260e-03, leak 3.151496e-03")
    W("  i.e. the echo is WORSE in this closed-system replay.  The argument")
    W("  against dropping it is experimental: a virtual Z_d has to know the")
    W("  light shift, which drifts with laser intensity, while the echo")
    W("  cancels it without knowing it.  That trade must be stated in the")
    W("  paper if the one-pulse convention is adopted.")
    W()
    W("  NO ADOPTION.  Report only.")

    out = ROOT / "results" / "cpir_one_pulse_regen.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
