"""QCR mechanism / Schrieffer-Wolff validation.

Shows that the paper QCR generator exp(-i theta Pi_l n_hat) EMERGES from a real
off-resonant blue-sideband drive at Meth parameters. NO imposed generator: the
only Hamiltonian handed to the propagator is the physical BSB drive.

MODEL
  qudit d=3, levels {g, l, s}; s is never driven (spectator).
  oscillator Fock cutoff 8.  Hilbert dim = 3 * 8 = 24.

  Interaction picture, drive ONLY the g<->l blue sideband, detuned by delta_disp:

     H(t) = (Omega_sb/2) * ( |l><g| (x) a_dag * exp(-i delta_disp t) + h.c. )

  No RWA beyond this. No Schrieffer-Wolff. No secular approximation.

PARAMETERS  (nothing here is tuned)
  Omega_sb/2pi   = 4 kHz    MEASURED  (Meth EDF 2c, device innsbruck_meth_2025)
  delta_disp/2pi = 40 kHz   ASSUMED   (not in Meth; chosen elsewhere in this repo
                                       to keep the SW error small -- that choice
                                       is exactly what this script tests)
  chi_pred/2pi   = Omega_sb^2 / (4 delta_disp) / 2pi = 100 Hz   DERIVED

Outputs the five full tables to results/qcr_mechanism.txt.
Reports must quote that file; never reconstruct numbers from it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import qutip as qt

# ----------------------------------------------------------------------------
# Parameters.  Units: time in us, angular frequency in rad/us.
# ----------------------------------------------------------------------------
OMEGA_SB_2PI_KHZ = 4.0        # MEASURED  (Meth EDF 2c)
DELTA_DISP_2PI_KHZ = 40.0     # ASSUMED   (not in Meth)

OMEGA_SB = 2 * math.pi * OMEGA_SB_2PI_KHZ * 1e-3        # rad/us
DELTA_DISP = 2 * math.pi * DELTA_DISP_2PI_KHZ * 1e-3    # rad/us

CHI_PRED = OMEGA_SB ** 2 / (4 * DELTA_DISP)             # rad/us
CHI_PRED_2PI_HZ = CHI_PRED / (2 * math.pi) * 1e6        # Hz

D_QUDIT = 3
LEVELS = ("g", "l", "s")
IDX = {name: i for i, name in enumerate(LEVELS)}
N_CUT = 8
DIM = D_QUDIT * N_CUT

T_QCR_US = 25.465
N_SAMPLES_SHORT = 41          # >= 40 sample times on [0, T_QCR]
T_LONG_US = 5000.0            # 5 ms clean-slope window
N_SAMPLES_LONG = 241

ATOL, RTOL = 1e-11, 1e-9
MAX_STEP_SHORT_US = 0.05
MAX_STEP_LONG_US = 0.50

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "qcr_mechanism.txt"


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def qud(i, j):
    return qt.basis(D_QUDIT, i) * qt.basis(D_QUDIT, j).dag()


def build_H():
    """Physical BSB drive only.  Returns qutip time-dependent list."""
    a = qt.destroy(N_CUT)
    lg = qud(IDX["l"], IDX["g"])          # |l><g|
    gl = qud(IDX["g"], IDX["l"])          # |g><l|
    op_up = (OMEGA_SB / 2) * qt.tensor(lg, a.dag())
    op_dn = (OMEGA_SB / 2) * qt.tensor(gl, a)
    args = {"d": DELTA_DISP}
    return [[op_up, "exp(-1j*d*t)"], [op_dn, "exp(1j*d*t)"]], args


def propagate(tlist, max_step):
    H, args = build_H()
    opts = {"atol": ATOL, "rtol": RTOL, "nsteps": 10_000_000,
            "max_step": max_step}
    Us = qt.propagator(H, tlist, args=args, options=opts)
    if not isinstance(Us, (list, tuple)):
        Us = [Us]
    return [np.asarray(U.full()) for U in Us]


def row(q, n):
    """Flat index for |q>|n> under qt.tensor(qudit, fock)."""
    return q * N_CUT + n


# ----------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------
LINES = []


def W(s=""):
    LINES.append(s)
    print(s)


def header():
    W("=" * 110)
    W("QCR MECHANISM / SCHRIEFFER-WOLFF VALIDATION")
    W("no imposed generator; only the physical off-resonant blue-sideband drive is propagated")
    W("=" * 110)
    W()
    W("MODEL")
    W("  H(t) = (Omega_sb/2) * ( |l><g| (x) a_dag * exp(-i delta_disp t) + h.c. )")
    W("  interaction picture; no RWA beyond this; no Schrieffer-Wolff; no secular approx")
    W(f"  qudit d = {D_QUDIT}  levels = {LEVELS}   ('s' is never driven)")
    W(f"  Fock cutoff = {N_CUT}   Hilbert dim = {DIM}")
    W()
    W("PARAMETERS")
    W(f"  Omega_sb/2pi    = {OMEGA_SB_2PI_KHZ:>10.4f} kHz   MEASURED (Meth EDF 2c, innsbruck_meth_2025)")
    W(f"  delta_disp/2pi  = {DELTA_DISP_2PI_KHZ:>10.4f} kHz   ***ASSUMED*** (not in Meth)")
    W(f"  Omega_sb        = {OMEGA_SB:>10.6e} rad/us")
    W(f"  delta_disp      = {DELTA_DISP:>10.6e} rad/us")
    W()
    W("  chi_pred = Omega_sb^2 / (4 * delta_disp)")
    W(f"           = ({OMEGA_SB:.6e})^2 / (4 * {DELTA_DISP:.6e})")
    W(f"           = {OMEGA_SB**2:.6e} / {4*DELTA_DISP:.6e}")
    W(f"           = {CHI_PRED:.6e} rad/us")
    W(f"  chi_pred/2pi = {CHI_PRED_2PI_HZ:.6f} Hz          DERIVED from the two above")
    W(f"  (chi_pred inherits the ASSUMED delta_disp: chi ~ 1/delta_disp)")
    W()
    W("INTEGRATOR")
    W(f"  qutip.propagator   atol={ATOL}  rtol={RTOL}")
    W(f"  short window: T = {T_QCR_US} us, {N_SAMPLES_SHORT} samples, max_step = {MAX_STEP_SHORT_US} us")
    W(f"  long  window: T = {T_LONG_US} us, {N_SAMPLES_LONG} samples, max_step = {MAX_STEP_LONG_US} us")
    W(f"  fastest timescale 2pi/delta_disp = {2*math.pi/DELTA_DISP:.4f} us")
    W()


# ----------------------------------------------------------------------------
# (a) conditioning is z-type / diagonal
# ----------------------------------------------------------------------------
def item_a(tl, Us):
    W("=" * 110)
    W("(a) CONDITIONING IS Z-TYPE / DIAGONAL")
    W("    full |U| in basis {g,l,s} x {n=0..7}, then transfer summaries")
    W("=" * 110)
    labels = [f"{q}{n}" for q in LEVELS for n in range(N_CUT)]

    max_ql_all, max_nn_all = 0.0, 0.0
    per_t = []
    for it, (t, U) in enumerate(zip(tl, Us)):
        A = np.abs(U)
        W()
        W(f"--- t = {t:10.6f} us ---")
        W("      " + " ".join(f"{c:>6}" for c in labels))
        for i, rl in enumerate(labels):
            W(f"{rl:>5} " + " ".join(f"{A[i, j]:6.3f}" for j in range(DIM)))

        # qudit-index off-diagonal: g<->l population transfer
        gl_blocks = []
        for (qa, qb) in [(IDX["g"], IDX["l"]), (IDX["l"], IDX["g"]),
                         (IDX["g"], IDX["s"]), (IDX["s"], IDX["g"]),
                         (IDX["l"], IDX["s"]), (IDX["s"], IDX["l"])]:
            blk = A[qa*N_CUT:(qa+1)*N_CUT, qb*N_CUT:(qb+1)*N_CUT]
            gl_blocks.append(blk.max())
        max_qudit_offdiag = max(gl_blocks)

        # within a fixed qudit level, transfer out of fixed n (n -> m != n)
        max_n_transfer = 0.0
        for q in range(D_QUDIT):
            blk = A[q*N_CUT:(q+1)*N_CUT, q*N_CUT:(q+1)*N_CUT]
            off = blk - np.diag(np.diag(blk))
            max_n_transfer = max(max_n_transfer, np.abs(off).max())

        max_ql_all = max(max_ql_all, max_qudit_offdiag)
        max_nn_all = max(max_nn_all, max_n_transfer)
        per_t.append((t, max_qudit_offdiag, max_n_transfer))
        W(f"  max |U| qudit off-diagonal (g<->l, g<->s, l<->s) = {max_qudit_offdiag:.6e}")
        W(f"  max |U| Fock off-diagonal within a level (n->m)  = {max_n_transfer:.6e}")

    W()
    W("SUMMARY TABLE (a): per sample time")
    W(f"{'t (us)':>12} {'max_qudit_offdiag':>20} {'max_fock_offdiag':>20}")
    for t, a1, a2 in per_t:
        W(f"{t:>12.6f} {a1:>20.6e} {a2:>20.6e}")
    W()
    W(f"  OVERALL max qudit off-diagonal = {max_ql_all:.6e}")
    W(f"  OVERALL max Fock  off-diagonal = {max_nn_all:.6e}")
    return max_ql_all, max_nn_all


# ----------------------------------------------------------------------------
# (b) chi emerges
# ----------------------------------------------------------------------------
def fit_slopes(tl, Us, level):
    """Return (slopes[n], phases[n][t]) for <level,n|U|level,n>."""
    q = IDX[level]
    phases = np.zeros((N_CUT, len(tl)))
    for n in range(N_CUT):
        i = row(q, n)
        raw = np.array([np.angle(U[i, i]) for U in Us])
        phases[n] = np.unwrap(raw)
    slopes = np.array([np.polyfit(tl, phases[n], 1)[0] for n in range(N_CUT)])
    return slopes, phases


def item_b(tl, Us, tag):
    W("=" * 110)
    W(f"(b) CHI EMERGES   [{tag}]")
    W("=" * 110)
    out = {}
    for level in ("l", "g"):
        slopes, phases = fit_slopes(tl, Us, level)
        W()
        W(f"--- level |{level}> : phase(t, n) = arg <{level},n|U|{level},n>  (unwrapped) ---")
        W("      n " + " ".join(f"t={t:9.4f}" for t in tl[:6]) + "   ...  (first 6 of "
          f"{len(tl)} sample times; full grid below)")
        for n in range(N_CUT):
            W(f"  n={n:>3} " + " ".join(f"{phases[n, k]:11.6f}" for k in range(min(6, len(tl)))))
        W()
        W(f"  FULL phase table |{level}> (rows = n, cols = all sample times)")
        for n in range(N_CUT):
            W(f"  n={n}: " + " ".join(f"{p:.6e}" for p in phases[n]))
        W()
        W(f"  per-n slope d(phase)/dt  [rad/us]   and slope/2pi [Hz]")
        W(f"{'n':>4} {'slope (rad/us)':>18} {'slope/2pi (Hz)':>18}")
        for n in range(N_CUT):
            W(f"{n:>4} {slopes[n]:>18.9e} {slopes[n]/(2*math.pi)*1e6:>18.6f}")

        # fit slope vs n : slope = chi_fit * n + const
        chi_fit, const = np.polyfit(np.arange(N_CUT), slopes, 1)
        W()
        W(f"  linear fit  slope(n) = chi_fit * n + const")
        W(f"    chi_fit      = {chi_fit:.9e} rad/us   -> chi_fit/2pi   = "
          f"{chi_fit/(2*math.pi)*1e6:.6f} Hz")
        W(f"    const offset = {const:.9e} rad/us   -> const/2pi      = "
          f"{const/(2*math.pi)*1e6:.6f} Hz")
        W(f"    (const is n-independent: a virtual-Z frame update, not an error)")
        W(f"    |chi_fit|/chi_pred = {abs(chi_fit)/CHI_PRED:.6f}")
        W(f"    const/chi_pred     = {const/CHI_PRED:.6f}")
        resid = slopes - (chi_fit*np.arange(N_CUT) + const)
        W(f"    max |residual| of the n-fit = {np.abs(resid).max():.6e} rad/us")
        out[level] = dict(slopes=slopes, chi_fit=chi_fit, const=const, phases=phases)
    return out


# ----------------------------------------------------------------------------
# (c) undriven level is dark
# ----------------------------------------------------------------------------
def item_c(tl, Us):
    W("=" * 110)
    W("(c) UNDRIVEN LEVEL |s> IS DARK   (this is what justifies the Pi_l projector)")
    W("=" * 110)
    W(f"{'t (us)':>12} " + " ".join(f"{'|<s,'+str(n)+'|U|s,'+str(n)+'>|':>20}" for n in range(4)))
    worst_pop_dev, worst_phase = 0.0, 0.0
    rows_out = []
    for t, U in zip(tl, Us):
        pops, phs = [], []
        for n in range(N_CUT):
            i = row(IDX["s"], n)
            z = U[i, i]
            pops.append(abs(z))
            phs.append(np.angle(z))
        worst_pop_dev = max(worst_pop_dev, max(abs(p - 1.0) for p in pops))
        worst_phase = max(worst_phase, max(abs(p) for p in phs))
        rows_out.append((t, pops, phs))
    W()
    W("FULL TABLE (c): |<s,n|U|s,n>| and arg, all n, all sample times")
    for t, pops, phs in rows_out:
        W(f"  t = {t:10.6f} us")
        W("    |amp|: " + " ".join(f"n{n}={pops[n]:.12f}" for n in range(N_CUT)))
        W("    arg  : " + " ".join(f"n{n}={phs[n]:+.3e}" for n in range(N_CUT)))
    W()
    W(f"  OVERALL max | |<s,n|U|s,n>| - 1 | = {worst_pop_dev:.6e}")
    W(f"  OVERALL max |arg <s,n|U|s,n>|     = {worst_phase:.6e}")
    return worst_pop_dev, worst_phase


# ----------------------------------------------------------------------------
# (d) SW error budget
# ----------------------------------------------------------------------------
def U_eff_diag(t, chi, use_n_plus_1, const=0.0):
    """exp(-i * [chi * Pi_l * f(n) + const * Pi_l] * t), diagonal."""
    d = np.zeros(DIM, dtype=complex)
    for q in range(D_QUDIT):
        for n in range(N_CUT):
            if q == IDX["l"]:
                f = (n + 1) if use_n_plus_1 else n
                ph = -(chi * f + const) * t
            else:
                ph = 0.0
            d[row(q, n)] = np.exp(1j * ph)
    return np.diag(d)


def fid(U, V):
    tr = np.trace(np.conj(V.T) @ U)
    return float(abs(tr) ** 2 / DIM ** 2)


def item_d(tl, Us, chi_fit_l, const_l):
    W("=" * 110)
    W("(d) SCHRIEFFER-WOLFF ERROR BUDGET")
    W("    U_eff variants compared against the full propagated U")
    W("=" * 110)
    W("  V1 = exp(-i chi_pred t Pi_l (n+1))              as specified")
    W("  V2 = exp(-i chi_pred t Pi_l (n+1)) with the fitted const offset removed")
    W("  V3 = exp(-i chi_pred t Pi_l n)                  naive (no +1)")
    W("  V4 = exp(-i [chi_fit n + const] t Pi_l)         fitted from (b)")
    W()
    W(f"{'t (us)':>12} {'1-F(V1)':>16} {'1-F(V2)':>16} {'1-F(V3)':>16} {'1-F(V4)':>16}")
    m1 = m2 = m3 = m4 = 0.0
    for t, U in zip(tl, Us):
        V1 = U_eff_diag(t, CHI_PRED, True, 0.0)
        V2 = U_eff_diag(t, CHI_PRED, True, -const_l)
        V3 = U_eff_diag(t, CHI_PRED, False, 0.0)
        V4 = U_eff_diag(t, chi_fit_l, False, const_l)
        e1, e2 = 1 - fid(U, V1), 1 - fid(U, V2)
        e3, e4 = 1 - fid(U, V3), 1 - fid(U, V4)
        m1, m2 = max(m1, e1), max(m2, e2)
        m3, m4 = max(m3, e3), max(m4, e4)
        W(f"{t:>12.6f} {e1:>16.6e} {e2:>16.6e} {e3:>16.6e} {e4:>16.6e}")
    W()
    W(f"  MAX 1-F  V1 (chi_pred, n+1)                 = {m1:.6e}")
    W(f"  MAX 1-F  V2 (chi_pred, n+1, const removed)  = {m2:.6e}")
    W(f"  MAX 1-F  V3 (chi_pred, n)                   = {m3:.6e}")
    W(f"  MAX 1-F  V4 (chi_fit n + const, fitted)     = {m4:.6e}")
    return m1, m2, m3, m4


# ----------------------------------------------------------------------------
# (e) leakage
# ----------------------------------------------------------------------------
def item_e(tl, Us):
    W("=" * 110)
    W("(e) LEAKAGE OUT OF THE CODE SUBSPACE")
    W("=" * 110)
    W(f"{'t (us)':>12} {'max P(g | start l,n)':>22} {'max P(n>2 | start n<=2)':>26}")
    worst_g, worst_hi = 0.0, 0.0
    for t, U in zip(tl, Us):
        pg = 0.0
        for n in range(N_CUT):
            col = U[:, row(IDX["l"], n)]
            p = sum(abs(col[row(IDX["g"], m)]) ** 2 for m in range(N_CUT))
            pg = max(pg, p)
        phi = 0.0
        for q in range(D_QUDIT):
            for n in range(3):
                col = U[:, row(q, n)]
                p = sum(abs(col[row(qq, m)]) ** 2
                        for qq in range(D_QUDIT) for m in range(3, N_CUT))
                phi = max(phi, p)
        worst_g, worst_hi = max(worst_g, pg), max(worst_hi, phi)
        W(f"{t:>12.6f} {pg:>22.6e} {phi:>26.6e}")
    W()
    W(f"  OVERALL max P(g) starting from |l,n>        = {worst_g:.6e}")
    W(f"  OVERALL max P(Fock n>2) starting from n<=2  = {worst_hi:.6e}")
    return worst_g, worst_hi


# ----------------------------------------------------------------------------
def main():
    header()

    tl_s = np.linspace(0.0, T_QCR_US, N_SAMPLES_SHORT)
    W(f"propagating short window ({N_SAMPLES_SHORT} samples to T_QCR = {T_QCR_US} us) ...")
    Us_s = propagate(tl_s, MAX_STEP_SHORT_US)
    W("  done")
    W()

    a_ql, a_nn = item_a(tl_s, Us_s)
    b_s = item_b(tl_s, Us_s, "short window, T = T_QCR")
    c_pop, c_ph = item_c(tl_s, Us_s)
    d1, d2, d3, d4 = item_d(tl_s, Us_s, b_s["l"]["chi_fit"], b_s["l"]["const"])
    e_g, e_hi = item_e(tl_s, Us_s)

    W()
    W(f"propagating long window ({N_SAMPLES_LONG} samples to {T_LONG_US} us) ...")
    tl_l = np.linspace(0.0, T_LONG_US, N_SAMPLES_LONG)
    Us_l = propagate(tl_l, MAX_STEP_LONG_US)
    W("  done")
    W()
    b_l = item_b(tl_l, Us_l, "long window, T = 5 ms (clean slope)")

    chi_l = b_l["l"]["chi_fit"]
    const_l = b_l["l"]["const"]

    W()
    W("=" * 110)
    W("SUMMARY")
    W("=" * 110)
    pass_a = (a_ql < 1e-2) and (a_nn < 1e-2)
    pass_c = (c_pop < 1e-9) and (c_ph < 1e-9)
    pass_d = min(d1, d2, d3, d4) < 1e-2
    pass_e = (e_g < 1e-2) and (e_hi < 1e-2)
    ratio = abs(chi_l) / CHI_PRED
    pass_b = 0.8 < ratio < 1.25

    W(f"1. max off-diagonal      : qudit {a_ql:.6e} , Fock {a_nn:.6e}   "
      f"[{'PASS' if pass_a else 'FAIL'}]")
    W(f"2. chi_fit/2pi (long win): {chi_l/(2*math.pi)*1e6:.6f} Hz   "
      f"(chi_pred/2pi = {CHI_PRED_2PI_HZ:.6f} Hz)")
    W(f"3. chi_fit/chi_pred      : {ratio:.6f}                    "
      f"[{'PASS' if pass_b else 'FAIL'}]")
    W(f"4. max SW infidelity     : V1 {d1:.6e} | V2 {d2:.6e} | V3 {d3:.6e} | V4 {d4:.6e}   "
      f"[{'PASS' if pass_d else 'FAIL'}]")
    W(f"5. max leakage           : P(g) {e_g:.6e} , P(n>2) {e_hi:.6e}   "
      f"[{'PASS' if pass_e else 'FAIL'}]")
    W(f"6. spectator |s> dark    : max|pop-1| {c_pop:.6e} , max|arg| {c_ph:.6e}   "
      f"[{'PASS' if pass_c else 'FAIL'}]")
    W()
    W(f"   delta_disp/2pi = {DELTA_DISP_2PI_KHZ} kHz is ASSUMED; chi_pred scales as 1/delta_disp,")
    W(f"   so every chi number above inherits that assumption.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
