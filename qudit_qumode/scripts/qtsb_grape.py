"""GRAPE rescue attempt for QT-SB on the FULL physical Hamiltonian.

qtsb_mechanism.py showed the paper QT-SB generator does not emerge from the real
drive (theta=0.5 endpoint 1-F = 8.664e-3), attributed to the LD-0 carrier.
qtsb_echo.py showed neither echo nor static detuning compensation rescues it.
This asks the last question: can GRAPE, given the two quadratures of the drive
envelope and the true time-dependent H, find a pulse that does?

MODEL -- identical to scripts/qtsb_mechanism.py, quoted:
  qtsb_mechanism.py:H_phys (bichromatic branch)
      C = OMEGA_C * math.cos(NU * t) * np.exp(-1j * ZETA)
      blk = C * np.kron(KG, M_ld(t, variant));  return blk + blk.conj().T
  qtsb_mechanism.py:M_ld  variant 'LDfull'
      alpha = 1j * ETA * ph
      return _expm(alpha * AD_NP - np.conj(alpha) * A_NP)      # = D(i eta e^{i nu t})
  i.e. BOTH the LD-0 carrier and the LD-1 sideband are retained, and they share
  one envelope because they come from the same laser.  That coupling is the
  whole difficulty: GRAPE cannot switch the carrier off independently.

CONTROLS
  The complex envelope c(t) = u_x(t) + i u_y(t) replaces the constant
  OMEGA_C * exp(-i ZETA).  With G(t) = cos(nu t) * |k><g| (x) M(t),
      H(t) = u_x(t) * (G + G^dag)  +  u_y(t) * i(G - G^dag)
  Both control operators are Hermitian and TIME-DEPENDENT.
  The flat physical pulse is u_x = 0, u_y = -OMEGA_C (zeta = +pi/2).

WHY NOT THE HyPulse / qutip-qtrl STACK
  optimize_pulse_unitary takes a FIXED list of control operators.  Here the
  control operators carry cos(nu t) and D(i eta e^{i nu t}) and change every
  slice.  Feeding qutip-qtrl a time-independent operator is exactly the
  imposed-effective-generator shortcut this run exists to avoid, so GRAPE is
  implemented here directly against the true H(t), with an analytic gradient.
  Control is piecewise constant over 200 slots; the propagation inside each
  slot uses SUBSTEPS so the carrier is resolved, not averaged away.

AMPLITUDE CAP, quoted from the device spec
  amp_bound_rad_per_us = OMEGA_C  (carrier Rabi) from
  hypulse calibrations 'innsbruck_meth_2025'.  Applied as a box bound
  |u_x|, |u_y| <= OMEGA_C / sqrt(2), so the aggregate |c| <= OMEGA_C -- the
  same Omega_max/sqrt(2) convention already used for QT in
  pulses_meth/hypulse_validation/step_qt_meth_aggcap.py.

REPORTED PER T
  F_grape, F_indep (independent repropagation at 4x finer substepping, the
  audit_v3 protocol), leakage, err/signal, and the flat-pulse baseline.

Writes results/qtsb_grape.txt.  Nothing adopted, no paper edits.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm as _expm
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import qtsb_mechanism as QM                       # noqa: E402

N_SLOTS = 200
SUBSTEPS = 10                 # per control slot
SUB_FINE = 40                 # independent repropagation
THETA = 0.5
T_LIST = (19.894368, 25.0, 30.0)
AMP = QM.OMEGA_C / math.sqrt(2.0)
MAXITER = 300

OUT = ROOT / "results" / "qtsb_grape.txt"
ECHO_TXT = ROOT / "results" / "qtsb_echo.txt"

LINES = []


def W(s=""):
    LINES.append(s)
    print(s)


KG_M = QM.KG          # |k><g| on the qudit


def G_of_t(t):
    """cos(nu t) * |k><g| (x) M_LDfull(t)."""
    return math.cos(QM.NU * t) * np.kron(KG_M, QM.M_ld(t, "LDfull"))


def ctrl_ops(t):
    G = G_of_t(t)
    Gd = G.conj().T
    return G + Gd, 1j * (G - Gd)


def build_grid(T, substeps):
    n = N_SLOTS * substeps
    dt = T / n
    ts = (np.arange(n) + 0.5) * dt
    return n, dt, ts


def _step(H, dt):
    """Exact propagator via eigendecomposition, plus the data needed for the
    EXACT derivative of exp(-i H dt) (H Hermitian).

    d/ds exp(-i(H+sA)dt) = W [Theta o (W^dag A W)] W^dag with
        Theta_mn = (e_m - e_n)/(lam_m - lam_n),  e = exp(-i dt lam)
        Theta_mm = -i dt e_m
    The first-order form -i dt A exp(-i H dt) is WRONG here: the two drive
    quadratures do not commute, so its error is O(dt [A,B]) -- first order,
    not second.  A finite-difference check of that form failed (ratios 0.7-3.9),
    which is why this exact form is used.
    """
    lam, Wv = np.linalg.eigh(H)
    e = np.exp(-1j * dt * lam)
    U = (Wv * e) @ Wv.conj().T
    dl = lam[:, None] - lam[None, :]
    de = e[:, None] - e[None, :]
    small = np.abs(dl) < 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        Theta = np.where(small, 0.0, de / np.where(small, 1.0, dl))
    Theta[small] = 0.0
    idx = np.arange(len(lam))
    Theta[small] = (-1j * dt * np.add.outer(e, np.zeros_like(e)))[small]
    Theta[idx, idx] = -1j * dt * e
    return U, Wv, Theta


def propagate(u, T, substeps, keep=False):
    """u shape (N_SLOTS, 2).  Returns U (and per-substep data if keep)."""
    n, dt, ts = build_grid(T, substeps)
    U = np.eye(QM.DIM, dtype=complex)
    Ps, As, Bs, Us, Ws, Th = [], [], [], [], [], []
    if keep:
        Ps.append(U.copy())
    for j in range(n):
        k = j // substeps
        A, B = ctrl_ops(ts[j])
        H = u[k, 0] * A + u[k, 1] * B
        Uj, Wv, Theta = _step(H, dt)
        U = Uj @ U
        if keep:
            As.append(A); Bs.append(B); Us.append(Uj)
            Ws.append(Wv); Th.append(Theta); Ps.append(U.copy())
    if keep:
        return U, dict(P=Ps, A=As, B=Bs, U=Us, Wv=Ws, Th=Th,
                       dt=dt, n=n, substeps=substeps)
    return U


def fid_and_grad(x, T, V):
    u = x.reshape(N_SLOTS, 2)
    U, d = propagate(u, T, SUBSTEPS, keep=True)
    D = QM.DIM
    tr = np.trace(V.conj().T @ U)
    F = abs(tr) ** 2 / D ** 2

    # backward chain Q_j = U_{n-1} ... U_{j+1}
    n = d["n"]
    Q = np.eye(D, dtype=complex)
    g = np.zeros((N_SLOTS, 2))
    for j in range(n - 1, -1, -1):
        Bmat = d["P"][j] @ V.conj().T @ Q          # P_j V^dag Q_j
        Wv, Theta = d["Wv"][j], d["Th"][j]
        Bt = Wv.conj().T @ Bmat @ Wv
        k = j // d["substeps"]
        for c, Op in ((0, d["A"][j]), (1, d["B"][j])):
            At = Wv.conj().T @ Op @ Wv
            # Tr(dU @ Bmat) = sum_mn Theta_mn * At_mn * Bt_nm
            val = np.einsum("mn,mn,nm->", Theta, At, Bt)
            g[k, c] += (2.0 / D ** 2) * np.real(np.conj(tr) * val)
        Q = Q @ d["U"][j]
    return -F, -g.reshape(-1)


def leakage(U, V):
    pr = pp = 0.0
    for q in QM.DRIVEN:
        for nn in range(3):
            cr, cp = U[:, QM.row(q, nn)], V[:, QM.row(q, nn)]
            sr = sum(abs(cr[QM.row(qq, m)]) ** 2
                     for qq in range(QM.D_QUDIT) for m in range(3, QM.N_CUT))
            sp = sum(abs(cp[QM.row(qq, m)]) ** 2
                     for qq in range(QM.D_QUDIT) for m in range(3, QM.N_CUT))
            pr, pp = max(pr, float(sr)), max(pp, float(sp))
    return abs(pr - pp)


def run_T(T, V, seeds):
    W("=" * 112)
    W(f"T = {T:.6f} us    theta = {THETA}    "
      f"({T/(2*math.pi/QM.NU):.3f} motional periods)")
    W("=" * 112)
    bounds = [(-AMP, AMP)] * (2 * N_SLOTS)
    best = None
    for name, x0 in seeds:
        t0 = time.time()
        r = minimize(fid_and_grad, x0, args=(T, V), jac=True, method="L-BFGS-B",
                     bounds=bounds, options=dict(maxiter=MAXITER, ftol=1e-15,
                                                 gtol=1e-12))
        u = r.x.reshape(N_SLOTS, 2)
        U = propagate(u, T, SUBSTEPS)
        F = QM.fid(U, V)
        Fvz = QM.fid_vz(U, V)[0]
        # independent repropagation, 4x finer substepping
        Ui = propagate(u, T, SUB_FINE)
        Fi = QM.fid(Ui, V)
        Fivz = QM.fid_vz(Ui, V)[0]
        sig = 1 - QM.fid(np.eye(QM.DIM, dtype=complex), V)
        lk = leakage(Ui, V)
        amax = float(np.abs(u).max())
        agg = float(np.sqrt((u ** 2).sum(axis=1)).max())
        row = dict(seed=name, F=F, Fvz=Fvz, Fi=Fi, Fivz=Fivz, sig=sig,
                   rs=(1 - Fvz) / sig, lk=lk, amax=amax, agg=agg,
                   it=r.nit, wall=time.time() - t0, u=u)
        W(f"  seed={name:<10} iters={r.nit:>4}  "
          f"1-F_grape={1-F:.6e}  1-F_vz={1-Fvz:.6e}")
        W(f"{'':>17} 1-F_indep={1-Fi:.6e}  1-F_indep_vz={1-Fivz:.6e}  "
          f"|match|={abs(F-Fi):.3e}")
        W(f"{'':>17} err/signal={row['rs']:.4f}  leak={lk:.6e}  "
          f"max|u|={amax:.4e} ({100*amax/AMP:.1f}% of bound)  "
          f"max|c|={agg:.4e} ({100*agg/QM.OMEGA_C:.1f}% of Omega_c)  "
          f"({row['wall']:.0f}s)")
        if best is None or Fvz > best["Fvz"]:
            best = row
    W()
    W(f"  BEST at T={T:.6f}: seed={best['seed']}  1-F_vz={1-best['Fvz']:.6e}  "
      f"1-F_indep_vz={1-best['Fivz']:.6e}  err/signal={best['rs']:.4f}")
    W()
    return best


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("=" * 112)
    W("QT-SB GRAPE RESCUE ATTEMPT -- full physical H, carrier included")
    W("=" * 112)
    W("MODEL quoted from scripts/qtsb_mechanism.py (H_phys bichromatic + M_ld LDfull):")
    W("    C = OMEGA_C * math.cos(NU * t) * np.exp(-1j * ZETA)")
    W("    blk = C * np.kron(KG, M_ld(t, variant));  return blk + blk.conj().T")
    W("    M_ld LDfull: _expm(alpha*AD - conj(alpha)*A),  alpha = 1j*ETA*e^{i nu t}")
    W("  -> LD-0 carrier and LD-1 sideband share ONE envelope (same laser).")
    W()
    W("CONTROLS: c(t) = u_x + i u_y replaces OMEGA_C*exp(-i ZETA)")
    W("    H(t) = u_x (G + G^dag) + u_y i(G - G^dag),  G = cos(nu t) |k><g| (x) M(t)")
    W(f"  N_SLOTS = {N_SLOTS}, substeps/slot = {SUBSTEPS} "
      f"(independent repropagation at {SUB_FINE})")
    W(f"  Fock cutoff = {QM.N_CUT}, Hilbert dim = {QM.DIM}")
    W()
    W("AMPLITUDE CAP, from hypulse device spec 'innsbruck_meth_2025':")
    W(f"    amp_bound_rad_per_us = OMEGA_C = {QM.OMEGA_C:.9e} rad/us")
    W(f"    box bound per quadrature = OMEGA_C/sqrt(2) = {AMP:.9e} rad/us")
    W("    (aggregate |c| <= OMEGA_C; same convention as step_qt_meth_aggcap.py)")
    W()
    W("NOT the HyPulse/qutip-qtrl stack: optimize_pulse_unitary takes FIXED")
    W("control operators, but these carry cos(nu t) and D(i eta e^{i nu t}).")
    W("Passing it a time-independent operator would re-impose the effective")
    W("generator this run exists to avoid.  GRAPE is implemented directly here.")
    W()
    W("BASELINE, quoted from results/qtsb_echo.txt:")
    if ECHO_TXT.exists():
        for ln in ECHO_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.strip()
            if s.startswith("0.5    C") or s.startswith("0.5    A") or \
               s.startswith("0.5    B"):
                W("    " + s)
    W("    flat-pulse endpoint 1-F at theta=0.5, T=19.894 us = 8.664350e-03")
    W()

    V = QM.U_paper(THETA)
    flat = np.zeros((N_SLOTS, 2))
    flat[:, 1] = -QM.OMEGA_C          # zeta = +pi/2 physical pulse
    flat = np.clip(flat, -AMP, AMP)

    rng = np.random.default_rng(42)
    results = {}
    for T in T_LIST:
        seeds = [("flat", flat.reshape(-1).copy()),
                 ("rand1", rng.uniform(-AMP, AMP, 2 * N_SLOTS)),
                 ("rand2", rng.uniform(-AMP, AMP, 2 * N_SLOTS))]
        results[T] = run_T(T, V, seeds)

    W("=" * 112)
    W("SUMMARY")
    W("=" * 112)
    W(f"{'T (us)':>10} {'seed':>8} {'1-F_grape':>14} {'1-F_vz':>14} "
      f"{'1-F_indep':>14} {'1-F_ind_vz':>14} {'err/signal':>11} {'leak':>12} "
      f"{'%bound':>8}")
    for T in T_LIST:
        r = results[T]
        W(f"{T:>10.4f} {r['seed']:>8} {1-r['F']:>14.6e} {1-r['Fvz']:>14.6e} "
          f"{1-r['Fi']:>14.6e} {1-r['Fivz']:>14.6e} {r['rs']:>11.4f} "
          f"{r['lk']:>12.6e} {100*r['amax']/AMP:>7.1f}%")
    W()
    W(f"{'baseline flat pulse, T=19.894':>44}  1-F = 8.664350e-03")
    best = min((1 - results[T]["Fivz"], T) for T in T_LIST)
    W(f"  best GRAPE (independent, vz) = {best[0]:.6e} at T = {best[1]} us")
    W(f"  improvement over flat pulse  = {8.664350e-03 / best[0]:.2f}x")
    W()
    W("  F_indep is an INDEPENDENT repropagation of the saved waveform at 4x")
    W("  finer substepping; |F_grape - F_indep| is printed per seed above and")
    W("  must be small for the optimiser's own number to be trusted.")
    W("  NOTHING ADOPTED.  No paper edits.")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
