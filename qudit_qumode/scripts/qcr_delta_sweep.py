"""QCR dispersive-detuning sweep: is delta_disp = 40 kHz a reasoned choice?

WHY THIS EXISTS
  QCR is the largest single cost in the qudit step (6 x 55.93 = 335.6 us of
  396.4 us, ~85%), and its duration comes from chi = Omega_sb^2 / (4 delta_disp)
  with delta_disp = 40 kHz -- a value this project CHOSE.  It is not a Meth
  device parameter; Meth drives resonant sidebands (delta = 0) and never uses a
  dispersive detuning.  Since T_QCR is proportional to delta, the headline
  runtime ratio moves with it, so the choice has to be defended rather than
  asserted.

THE TWO PHYSICAL LIMITS THAT BRACKET delta
  lower   delta >> Omega_sb (= 4 kHz), or the drive is not dispersive and the
          Schrieffer-Wolff expansion that defines chi does not hold.
          At 40 kHz, Omega_sb/delta = 0.1 -- the usual 10% rule.
  upper   delta << the gap to the nearest neighbouring motional mode, or the
          drive stops being spectrally selective and addresses two modes.
          NOTE: the ceiling is the MODE SPACING, not the trap frequency.  With
          a 3-ion chain there are 3 axial modes and the gaps between them are
          smaller than nu = 0.77 MHz.  This repo does not anchor the mode
          spectrum, so the upper limit is quoted against nu as a stand-in and
          flagged: a real chain will have a tighter ceiling.

WHAT IS SWEPT AND WHAT IS MEASURED
  For each delta:  chi = Omega_sb^2/(4 delta),  T scales as delta (chi ~ 1/delta).
  Then the FULL physical drive is propagated (same model as
  scripts/qcr_mechanism.py: H(t) = (Omega_sb/2)(|l><g| (x) a^dag e^{-i delta t}
  + h.c.), no RWA, no SW) and we measure, at the scaled duration:
      1-F against the effective generator exp(-i chi T Pi_l n)
      leakage P(g) out of |l>, and P(n>2)
      Omega_sb/delta, the SW small parameter
  Reporting fidelity NEXT TO the ratio is the point: T_QCR falls as delta
  falls, which makes the ratio look better, while the SW approximation the
  gate rests on gets worse.  A ratio-only sweep would make small delta look
  free.  It is not.

CORRECTION (v2).  Version 1 of this sweep was WRONG in two ways and its
fidelity column must not be quoted:
  (a) it propagated to T_paper = 55.93 us, which accumulates a gate angle of
      chi*T = 0.0351 rad -- 2.20x the paper angle theta = 0.016.  The correct
      duration for a given delta is T = theta/chi(delta), which reproduces
      25.465 us at 40 kHz.  The angle is now held FIXED at theta across the
      sweep, which is what makes the rows comparable to each other and to the
      adopted numbers.
  (b) it compared against the BARE Schrieffer-Wolff generator
      exp(-i chi_pred t Pi_l n) with no per-level virtual-Z freedom.  That
      reference omits the n-independent frame offset (a free virtual-Z) and the
      second-order correction to chi, so it reported the reference's error, not
      the gate's.  v2 compares against three references -- bare chi_pred*n,
      chi_pred*(n+1), and the FITTED generator (chi_fit*n + const) -- each with
      and without per-level virtual-Z absorption.

Writes results/qcr_delta_sweep.txt.  No adoption; runtime_table.md unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import qutip as qt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pulses_meth" / "hypulse_validation"))

import qcr_mechanism as QC                # noqa: E402
import step_runtime_table as RT           # noqa: E402

DELTAS_KHZ = (20.0, 30.0, 40.0, 60.0, 80.0)
DELTA_REF_KHZ = 40.0
THETA_QCR = 0.016          # paper angle, appendix_table.md:41
NC = 3
OUT = ROOT / "results" / "qcr_delta_sweep.txt"

LINES = []


def W(s=""):
    LINES.append(s)
    print(s, flush=True)


def fid_vz(U, V):
    """Fidelity after the optimal per-qudit-level virtual-Z frame update."""
    M = U @ np.conj(V.T)
    S = np.array([np.trace(M[q * QC.N_CUT:(q + 1) * QC.N_CUT,
                             q * QC.N_CUT:(q + 1) * QC.N_CUT])
                  for q in range(QC.D_QUDIT)])
    return float(np.sum(np.abs(S)) ** 2 / QC.DIM ** 2)


def fit_chi(tl, Us):
    """chi_fit and const from the per-n phase slopes on level |l>."""
    ph = np.zeros((QC.N_CUT, len(tl)))
    for n in range(QC.N_CUT):
        i = QC.row(QC.IDX["l"], n)
        ph[n] = np.unwrap(np.array([np.angle(U[i, i]) for U in Us]))
    sl = np.array([np.polyfit(tl, ph[n], 1)[0] for n in range(QC.N_CUT)])
    chi_fit, const = np.polyfit(np.arange(QC.N_CUT), sl, 1)
    return float(chi_fit), float(const)


def measure(delta_khz, T_us):
    """Propagate the real drive at this delta to T_us; measure SW quality."""
    QC.DELTA_DISP = 2 * math.pi * delta_khz * 1e-3        # rad/us
    QC.CHI_PRED = QC.OMEGA_SB ** 2 / (4 * QC.DELTA_DISP)
    tl = np.linspace(0.0, T_us, 21)
    Us = QC.propagate(tl, max_step=min(0.05, 1.0 / (delta_khz * 1e-3) / 20))
    U = Us[-1]
    chi_fit, const = fit_chi(tl, Us)

    refs = {
        "pred_n":   QC.U_eff_diag(T_us, QC.CHI_PRED, False, 0.0),
        "pred_n1":  QC.U_eff_diag(T_us, QC.CHI_PRED, True, 0.0),
        "fitted":   QC.U_eff_diag(T_us, chi_fit, False, const),
    }
    F = {k: QC.fid(U, V) for k, V in refs.items()}
    Fvz = {k: fid_vz(U, V) for k, V in refs.items()}

    # leakage out of |l> and above n=2
    pg = 0.0
    for n in range(QC.N_CUT):
        col = U[:, QC.row(QC.IDX["l"], n)]
        pg = max(pg, float(sum(abs(col[QC.row(QC.IDX["g"], m)]) ** 2
                               for m in range(QC.N_CUT))))
    phi = 0.0
    for q in range(QC.D_QUDIT):
        for n in range(3):
            col = U[:, QC.row(q, n)]
            phi = max(phi, float(sum(abs(col[QC.row(qq, m)]) ** 2
                                     for qq in range(QC.D_QUDIT)
                                     for m in range(3, QC.N_CUT))))
    best_k = max(Fvz, key=lambda k: Fvz[k])
    return dict(F=F, Fvz=Fvz, best_ref=best_k, best=Fvz[best_k],
                chi_fit=chi_fit, const=const,
                leak_g=pg, leak_n=phi,
                chi=QC.CHI_PRED, small=QC.OMEGA_SB / QC.DELTA_DISP)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    qcd = RT.read_qcd_alpha_scaled(0.008); qcr = RT.read_qcr_t_paper()
    qtsb = RT.read_qtsb_t_analytic(theta=0.0015); qt_b = RT.read_qt_bounds()
    rot = RT.read_rotation(); ms = RT.read_ms()
    T = dict(QCD=qcd["value_us"], QCR=qcr["value_us"], QT_SB=qtsb["value_us"],
             rotation=rot["value_us"], MS=ms["value_us"])
    T_QCR_REF = T["QCR"]

    W("=" * 116)
    W("QCR DISPERSIVE-DETUNING SWEEP -- defending delta_disp = 40 kHz")
    W("=" * 116)
    W(f"  Omega_sb/2pi = {QC.OMEGA_SB/(2*math.pi)*1e6/1e3:.3f} kHz "
      f"(MEASURED, Meth)")
    W(f"  nu/2pi       = 770 kHz (MEASURED, Meth) -- stand-in ceiling, see below")
    W(f"  T_QCR at delta = {DELTA_REF_KHZ:.0f} kHz = {T_QCR_REF} us "
      f"(read from appendix_table.md)")
    W()
    W("  delta_disp is OUR CHOICE, not a Meth parameter.  Meth drives resonant")
    W("  sidebands (delta = 0) and never uses a dispersive detuning.  The two")
    W("  limits that bracket a sensible value:")
    W(f"    lower: delta >> Omega_sb = {QC.OMEGA_SB/(2*math.pi)*1e6/1e3:.0f} kHz "
      f"(dispersive validity)")
    W("    upper: delta << gap to the nearest motional mode (spectral selectivity)")
    W("  CAVEAT: the upper limit is the MODE SPACING, not nu.  A 3-ion chain has")
    W("  3 axial modes with gaps smaller than nu = 770 kHz.  This repo does not")
    W("  anchor the mode spectrum, so nu is used as a stand-in and the real")
    W("  ceiling is tighter.  Flagged, not resolved.")
    W()
    W("  T_QCR scales LINEARLY with delta because chi = Omega_sb^2/(4 delta) and")
    W("  the gate angle is chi*T.  So T(delta) = T_ref * delta/delta_ref.")
    W()

    W("=" * 116)
    W("SWEEP")
    W("=" * 116)
    W(f"  angle held FIXED at theta = {THETA_QCR} for every row;")
    W(f"  duration T(delta) = theta / chi(delta), so T = 25.465 us at 40 kHz.")
    W(f"  (v1 wrongly used T_paper = {T_QCR_REF} us, i.e. angle "
      f"{T_QCR_REF/25.465*THETA_QCR:.4f} = "
      f"{T_QCR_REF/25.465:.2f}x theta.  Those numbers are superseded.)")
    W()
    W(f"{'delta':>7} {'Om/del':>7} {'chi/2pi':>9} {'T_ang':>9} "
      f"{'1-F pred_n':>12} {'1-F pred_n1':>12} {'1-F fitted':>12} "
      f"{'1-F fit+VZ':>12} {'leak P(g)':>11} {'leak n>2':>11}")
    W(f"{'(kHz)':>7} {'':>7} {'(Hz)':>9} {'(us)':>9}")
    rows = []
    for dk in DELTAS_KHZ:
        chi = QC.OMEGA_SB ** 2 / (4 * (2 * math.pi * dk * 1e-3))
        T_ang = THETA_QCR / chi                       # angle-correct duration
        m = measure(dk, T_ang)
        T_price = T_QCR_REF * dk / DELTA_REF_KHZ      # what the table prices
        rows.append((dk, T_ang, T_price, m))
        W(f"{dk:>7.0f} {m['small']:>7.3f} "
          f"{m['chi']/(2*math.pi)*1e6:>9.2f} {T_ang:>9.3f} "
          f"{1-m['F']['pred_n']:>12.4e} {1-m['F']['pred_n1']:>12.4e} "
          f"{1-m['F']['fitted']:>12.4e} {1-m['Fvz']['fitted']:>12.4e} "
          f"{m['leak_g']:>11.4e} {m['leak_n']:>11.4e}")
    W()
    W("  'fitted' = exp(-i[chi_fit n + const] t Pi_l), the effective generator")
    W("  actually produced by the drive; 'fit+VZ' additionally absorbs the free")
    W("  per-level virtual-Z.  The bare pred_n column is the reference's error,")
    W("  not the gate's -- it is kept only to show the difference.")
    W()

    # ---- runtime impact ----
    W("=" * 116)
    W("RUNTIME IMPACT AT Nc = 3")
    W("=" * 116)
    c = RT.qudit_counts(NC)
    base_no_qcr = (c["Rotation"] * T["rotation"] + c["QCD"] * T["QCD"]
                   + c["QT_SB"] * T["QT_SB"])
    Tqb = RT.qubit_step_us(NC, T, j_per_ms=2)
    Tqb_f = RT.qubit_step_us_figgatt(NC, T, n_windows=5)
    W(f"  qudit step minus QCR and QT = Rot {c['Rotation']}x{T['rotation']} + "
      f"QCD {c['QCD']}x{T['QCD']} + QT-SB {c['QT_SB']}x{T['QT_SB']:.4f} "
      f"= {base_no_qcr:.4f} us")
    W(f"  QCR appears {c['QCR']} times per step")
    W(f"  T_qubit sequential = {Tqb:.4f} us,  parallel-MS = {Tqb_f:.4f} us "
      f"(unaffected by delta)")
    W()
    W(f"{'delta':>8} {'T_QCR':>10} {'T_qudit lo':>12} {'T_qudit hi':>12} "
      f"{'ratio seq':>18} {'ratio par-MS':>18} {'1-F(SW)':>12}")
    for dk, T_ang, T_qcr, m in rows:
        lo = base_no_qcr + c["QCR"] * T_qcr + c["QT_multi"] * qt_b["low_us"]
        hi = base_no_qcr + c["QCR"] * T_qcr + c["QT_multi"] * qt_b["high_us"]
        rs = (Tqb / hi, Tqb / lo)
        rp = (Tqb_f / hi, Tqb_f / lo)
        W(f"{dk:>8.0f} {T_qcr:>10.3f} {lo:>12.4f} {hi:>12.4f} "
          f"{'[%.2fx, %.2fx]' % rs:>18} {'[%.2fx, %.2fx]' % rp:>18} "
          f"{1-m['Fvz']['fitted']:>12.6e}")
    W()

    ref = [r for r in rows if r[0] == DELTA_REF_KHZ][0]
    lo_r = base_no_qcr + c["QCR"] * ref[2] + c["QT_multi"] * qt_b["low_us"]
    hi_r = base_no_qcr + c["QCR"] * ref[2] + c["QT_multi"] * qt_b["high_us"]
    W("  READING")
    W(f"    The sequential ratio swings from "
      f"{Tqb/(base_no_qcr + c['QCR']*rows[0][2] + c['QT_multi']*qt_b['low_us']):.2f}x "
      f"at {DELTAS_KHZ[0]:.0f} kHz to "
      f"{Tqb/(base_no_qcr + c['QCR']*rows[-1][2] + c['QT_multi']*qt_b['high_us']):.2f}x "
      f"at {DELTAS_KHZ[-1]:.0f} kHz.")
    W("    Smaller delta makes the gate FASTER and the ratio BETTER, but the")
    W("    Schrieffer-Wolff parameter Omega_sb/delta grows and the SW infidelity")
    W("    and leakage grow with it -- read the 1-F and leakage columns before")
    W("    treating a small delta as free.  That is why both are tabulated here.")
    W(f"    At the chosen {DELTA_REF_KHZ:.0f} kHz: Omega/delta = {ref[3]['small']:.3f}, "
      f"1-F(fit+VZ) = {1-ref[3]['Fvz']['fitted']:.6e}, ratio seq = "
      f"[{Tqb/hi_r:.2f}x, {Tqb/lo_r:.2f}x].")
    W()
    W("  NO ADOPTION.  runtime_table.md keeps delta = 40 kHz and T_QCR = "
      f"{T_QCR_REF} us.")
    W("  This run turns that from an assumption into a bracketed, reasoned choice,")
    W("  and shows the cost of moving it in either direction.")

    np.savez(ROOT / "data" / "qcr_delta_sweep.npz",
             delta_khz=np.array([r[0] for r in rows]),
             T_angle=np.array([r[1] for r in rows]),
             T_priced=np.array([r[2] for r in rows]),
             one_minus_F_fit_vz=np.array([1 - r[3]["Fvz"]["fitted"] for r in rows]),
             one_minus_F_pred=np.array([1 - r[3]["F"]["pred_n"] for r in rows]),
             leak_g=np.array([r[3]["leak_g"] for r in rows]),
             leak_n=np.array([r[3]["leak_n"] for r in rows]),
             small=np.array([r[3]["small"] for r in rows]))
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
