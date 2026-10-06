"""
CITATION TAG "dev-tree:".  Paths marked that way point into the development
tree, tools/all_gates_pulse/, which the submission repository does not carry
-- nothing in the paper reads those files, only this script's prose cites
them.  Every passage cited below is quoted verbatim in place, so the
argument is complete without them.  They resolve in the full working
repository, https://github.com/mhakimih1988/qudit_qumode.
QT-SB mechanism validation.

Shows whether the paper QT-SB generator  exp(-i theta (|i><j|+|j><i|) (x) x_hat)
EMERGES from a real laser drive at Meth parameters.  NO imposed generator: the
only Hamiltonian handed to the propagator is the physical multi-tone drive with
its Lamb-Dicke structure.  Same discipline as scripts/qcr_mechanism.py.

WHY NOT A SINGLE RESONANT BSB
-----------------------------
The step spec proposed "resonant BSB on the relevant g-k transition".  The paper
QT-SB generator is  sigma_x^{gk} (x) x_hat , and a *single* resonant blue
sideband gives only the anti-Jaynes-Cummings piece

    H_BSB = i (Omega eta / 2) sigma_+^{gk} a^dag + h.c.

which is NOT sigma_x (x) x_hat.  This repo already states the correct tone
configuration, so per the step instruction ("if the paper QT-SB uses a different
tone configuration, quote it and use that instead, stating file+line") we use it:

  dev-tree:tools/all_gates_pulse/QT_SB/README.md:16-32
    "Simultaneous red + blue sideband tones on the E2 435 nm G <-> l_target
     transition, with SYMMETRIC bichromatic phase (both tones at the same
     zeta). ...
        H_RSB = i (Omega eta / 2) sigma_+^{Gl} a       + h.c.    (JC     )
        H_BSB = i (Omega eta / 2) sigma_+^{Gl} a^dag   + h.c.    (anti-JC)
        H_MS  = H_RSB + H_BSB
             = -(Omega eta / 2) sigma_y^{Gl} (x) (a + a^dag)    (zeta = 0)
             = +(Omega eta / 2) sigma_x^{Gl} (x) (a + a^dag)    (zeta = pi/2)
     So a bichromatic drive with zeta = +pi/2 delivers
     H = Omega eta sigma_x^{Gl} (x) x_hat and the analytic pulse time for
     angle theta is  T_analytic = theta / (Omega eta)."

  dev-tree:tools/all_gates_pulse/QT_SB/code/physical.py:5-8, 26-37, 88-125
    the same statement in code (H_bichrom_sb_ip), symmetric-phase convention.

The single-BSB variant is still propagated below, as section (f), precisely to
quantify how far it is from the paper generator -- i.e. to justify the swap
rather than assert it.

MODEL  (what is actually integrated; nothing here is a generator)
----------------------------------------------------------------
  qudit d = 3, levels {g, k, s}.  's' carries no drive term at all (spectator).
  oscillator Fock cutoff 8.  Hilbert dim = 3 * 8 = 24.

  Joint interaction picture w.r.t. H_0 = nu a^dag a + sum_l omega_l |l><l|,
  so H_drift = 0 (same frame as physical.py:10-13).  Two laser tones at
  omega_gk +/- nu, SYMMETRIC phase zeta on both, flat (constant) envelope:

     C(t) = (Omega_c/2) * ( e^{-i nu t - i zeta} + e^{+i nu t - i zeta} )
          = Omega_c cos(nu t) e^{-i zeta}

     H(t) = C(t) * |k><g| (x) M(t)  +  h.c.

  with the Lamb-Dicke factor M(t) treated in two variants:

     LD1     M(t) = I + i eta (a e^{-i nu t} + a^dag e^{+i nu t})
                    -- the LD-first-order truncation used by physical.py:120-124
     LDfull  M(t) = exp[ i eta (a e^{-i nu t} + a^dag e^{+i nu t}) ]
                  = D( i eta e^{+i nu t} )      (exact displacement operator)
                    -- no LD truncation at all

  NO per-tone RWA, NO secular approximation, NO carrier truncation: the LD-0
  carrier piece (amplitude Omega_c, detuning nu) and the 2-nu LD-1 cross terms
  are all retained.  That off-resonant carrier is the dominant physical error
  and it is exactly what an imposed generator hides.

  Slow (RWA) limit of the above, for reference only -- never propagated:
     cos(nu t) * (a e^{-i nu t} + a^dag e^{+i nu t}) = (a + a^dag)/2 + [2 nu]
     => H_eff = eta * Omega_c * sigma_x^{gk} (x) x_hat  at zeta = +pi/2
     => Omega_sb == eta * Omega_c  is the effective coupling, T = theta/Omega_sb.

PARAMETERS  (all read from the device spec; nothing tuned)
  eta                      MEASURED   Meth
  Omega_sideband_rad_per_s MEASURED   Meth  (this IS eta*Omega_c)
  amp_bound_rad_per_us     DERIVED    = Omega_c, the carrier Rabi
  omega_m_x_rad_per_s      MEASURED   Meth  = nu
  zeta = +pi/2             CONVENTION QT_SB/README.md:26-29

Outputs the full tables to results/qtsb_mechanism.txt.
Reports must quote that file; never reconstruct numbers from it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.linalg import expm as _expm

from hypulse.calibrations import get_device_spec

# ----------------------------------------------------------------------------
# Parameters.  Units: time in us, angular frequency in rad/us.
# ----------------------------------------------------------------------------
DEVICE_ID = "innsbruck_meth_2025"
_dev = get_device_spec(DEVICE_ID)

ETA = float(_dev.params["eta"])
OMEGA_SB = float(_dev.params["Omega_sideband_rad_per_s"]) / 1e6      # rad/us
OMEGA_C = float(_dev.params["amp_bound_rad_per_us"])                  # rad/us
NU = float(_dev.params["omega_m_x_rad_per_s"]) / 1e6                  # rad/us
ZETA = +math.pi / 2                                                   # sigma_x SDF

D_QUDIT = 3
LEVELS = ("g", "k", "s")
IDX = {name: i for i, name in enumerate(LEVELS)}
N_CUT = 8
DIM = D_QUDIT * N_CUT
DRIVEN = (IDX["g"], IDX["k"])
DIM_DRIVEN = 2 * N_CUT

# the two paper operating points, read below from the saved HyPulse JSON
THETAS = (0.0015, 0.5)

N_SAMPLES = 41                # >= 30 sample times per theta
N_SLICES = 40000              # exponential-midpoint slices per run
N_SLICES_CONV = 80000         # convergence re-run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "qtsb_mechanism.txt"
QTSB_JSON = ROOT / "pulses_meth" / "hypulse_validation" / "step_qtsb_meth_results.json"

LINES = []


def W(s=""):
    LINES.append(s)
    print(s)


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
A = qt.destroy(N_CUT)
AD = A.dag()
X_HAT = ((A + AD) / 2.0).full()
I_FOCK = np.eye(N_CUT, dtype=complex)
A_NP = A.full()
AD_NP = AD.full()


def qud(i, j):
    M = np.zeros((D_QUDIT, D_QUDIT), dtype=complex)
    M[i, j] = 1.0
    return M


KG = qud(IDX["k"], IDX["g"])          # |k><g|
SX_GK = qud(IDX["g"], IDX["k"]) + qud(IDX["k"], IDX["g"])


def row(q, n):
    """Flat index for |q>|n> under kron(qudit, fock)."""
    return q * N_CUT + n


def paper_generator():
    """G = sigma_x^{gk} (x) x_hat, identity (zero) on the spectator level."""
    return np.kron(SX_GK, X_HAT)


G_PAPER = paper_generator()


def U_paper(theta):
    """exp(-i theta sigma_x^{gk} (x) x_hat)."""
    return _expm(-1j * theta * G_PAPER)


# ----------------------------------------------------------------------------
# Physical Hamiltonian  (no generator anywhere in here)
# ----------------------------------------------------------------------------
def M_ld(t, variant):
    """Lamb-Dicke factor multiplying |k><g|."""
    ph = np.exp(1j * NU * t)
    if variant == "LD1":
        return I_FOCK + 1j * ETA * (A_NP * np.conj(ph) + AD_NP * ph)
    if variant == "LD1_nocarrier":
        # LD-1 sideband piece ONLY: the LD-0 carrier (the I_FOCK term) is deleted.
        # Diagnostic variant for error attribution -- NOT physical on its own.
        return 1j * ETA * (A_NP * np.conj(ph) + AD_NP * ph)
    if variant == "LDfull":
        # exp[i eta (a e^{-i nu t} + adg e^{+i nu t})] = D(i eta e^{+i nu t})
        alpha = 1j * ETA * ph
        return _expm(alpha * AD_NP - np.conj(alpha) * A_NP)
    raise ValueError(variant)


def H_phys(t, variant, tones="bichromatic"):
    """Physical drive Hamiltonian at time t.  Hermitian by construction."""
    if tones == "bichromatic":
        # (Omega_c/2) * (e^{-i nu t} + e^{+i nu t}) * e^{-i zeta}
        C = OMEGA_C * math.cos(NU * t) * np.exp(-1j * ZETA)
    elif tones == "bsb_only":
        # single resonant blue-sideband tone (the step spec's proposal)
        C = (OMEGA_C / 2.0) * np.exp(-1j * NU * t) * np.exp(-1j * ZETA)
    else:
        raise ValueError(tones)
    blk = C * np.kron(KG, M_ld(t, variant))
    return blk + blk.conj().T


def propagate(T_us, variant, tones="bichromatic", n_slices=N_SLICES,
              n_samples=N_SAMPLES):
    """Exponential-midpoint propagation.  Returns (tlist, [U(t)])."""
    dt = T_us / n_slices
    stride = n_slices // (n_samples - 1)
    U = np.eye(DIM, dtype=complex)
    tl = [0.0]
    Us = [U.copy()]
    for k in range(n_slices):
        t_c = (k + 0.5) * dt
        U = _expm(-1j * H_phys(t_c, variant, tones) * dt) @ U
        if (k + 1) % stride == 0 and len(Us) < n_samples:
            tl.append((k + 1) * dt)
            Us.append(U.copy())
    if len(Us) < n_samples:
        tl.append(T_us)
        Us.append(U.copy())
    return np.array(tl), Us


# ----------------------------------------------------------------------------
# Fidelity helpers
# ----------------------------------------------------------------------------
def fid(U, V, dim=DIM):
    tr = np.trace(np.conj(V.T) @ U)
    return float(abs(tr) ** 2 / dim ** 2)


def fid_vz(U, V):
    """Fidelity after the optimal per-level virtual-Z frame update.

    V_z = sum_q e^{-i z_q} P_q (x) I ; max over {z_q} of |Tr(V^dag V_z U)| is
    sum_q |S_q| with S_q = Tr[(P_q (x) I) U V^dag].  Same construction as
    dev-tree:tools/all_gates_pulse/QT_SB/code/physical.py:160-168 (qps_fidelity).
    """
    M = U @ np.conj(V.T)
    S = np.array([np.trace(M[q * N_CUT:(q + 1) * N_CUT,
                             q * N_CUT:(q + 1) * N_CUT])
                  for q in range(D_QUDIT)])
    return float(np.sum(np.abs(S)) ** 2 / DIM ** 2), np.angle(S)


def sub_driven(M):
    """Restrict a DIM x DIM matrix to the driven {g,k} block."""
    idx = [row(q, n) for q in DRIVEN for n in range(N_CUT)]
    return M[np.ix_(idx, idx)]


def fid_driven(U, V):
    return fid(sub_driven(U), sub_driven(V), dim=DIM_DRIVEN)


# ----------------------------------------------------------------------------
def header(claimed):
    W("=" * 118)
    W("QT-SB MECHANISM VALIDATION")
    W("no imposed generator; only the physical multi-tone laser drive is propagated")
    W("=" * 118)
    W()
    W("QUOTED FIRST -- paper QT-SB definition and claimed durations")
    W("-" * 118)
    W("  generator / target, quoted from")
    W("    pulses_meth/hypulse_validation/step_qtsb_meth.py:13-16 ->")
    W("      \"U_QTSB(theta) = exp( -i * theta * (|i><j| + |j><i|) (x) x_hat )\"")
    W("      \"x_hat = (a+a^dag)/2.  On {|0>,|1>} this is sigma_x, so the generator")
    W("       equals sigma_x (x) X (same structure as CD).\"")
    W("    pulses_meth/appendix_table.md:63 ->")
    W("      \"generator `sigma_x (x) X` (single-control imposed).\"   <-- the caveat this run addresses")
    W("    dev-tree:tools/all_gates_pulse/QT_SB/README.md:7-8 ->")
    W("      \"QT-SB(theta, i, j) = exp[ -i theta ( |i><j| + |j><i| ) (x) x_hat ]\"")
    W()
    W("  Hamiltonian the paper assumes (the imposed one), quoted from")
    W("    pulses_meth/hypulse_validation/step_qtsb_meth.py:50-54 ->")
    W("      def build_generator(cutoff=CUTOFF):")
    W("          \\\"\\\"\\\"H_c = sigma_x (x) X ,  X = (a+adg)/2\\\"\\\"\\\"")
    W("      -> H_drift = 0, H_control = sigma_x (x) X  (step_qtsb_meth.py:65)")
    W("      i.e. control operator == gate generator.  That is the caveat.")
    W()
    W("  claimed T, quoted from pulses_meth/hypulse_validation/step_qtsb_meth_results.json")
    W(f"{'theta':>10} {'T_analytic_us':>16} {'T_grape_us':>14} {'F_grape':>18} {'amp_bound':>14}")
    for c in claimed:
        W(f"{c['theta']:>10.4f} {c['T_analytic_us']:>16.6f} {c['T_grape_us']:>14.6f} "
          f"{c['F_grape']:>18.12f} {c['amp_bound_rad_per_us']:>14.6e}")
    W("    appendix_table.md:166,168  ->  QT-SB theta=0.0015 : 0.0597 us ; theta=0.5 : 19.894 us")
    W()
    W("TONE CONFIGURATION -- the step spec's single resonant BSB is NOT what the paper uses")
    W("-" * 118)
    W("  dev-tree:tools/all_gates_pulse/QT_SB/README.md:16-32:")
    W("    \"Simultaneous red + blue sideband tones on the E2 435 nm G <-> l_target")
    W("     transition, with SYMMETRIC bichromatic phase (both tones at the same zeta).\"")
    W("    \"H_MS = H_RSB + H_BSB = +(Omega eta / 2) sigma_x^{Gl} (x) (a + a^dag)  (zeta = pi/2)\"")
    W("    \"T_analytic = theta / (Omega eta).\"")
    W("  dev-tree:tools/all_gates_pulse/QT_SB/code/physical.py:5-8:")
    W("    \"the DRIVE is a red+blue sideband bichromatic on the |i><j| E2 transition\"")
    W("  dev-tree:tools/all_gates_pulse/QT_SB/code/physical.py:32-35:")
    W("    \"The phase convention is SYMMETRIC (both tones carry the same zeta) ...")
    W("     Symmetric-phase red+blue yields the sigma_x SDF that QT-SB needs\"")
    W("  => bichromatic red+blue at zeta=+pi/2 is used below.  A single resonant BSB")
    W("     gives only the anti-JC term i(Omega eta/2) sigma_+ a^dag + h.c., which is")
    W("     not sigma_x (x) x_hat; that variant is propagated in section (f) to show it.")
    W()
    W("MODEL ACTUALLY INTEGRATED")
    W("-" * 118)
    W("  H(t) = C(t) |k><g| (x) M(t) + h.c.,   C(t) = Omega_c cos(nu t) e^{-i zeta}")
    W("  M(t) = I + i eta (a e^{-i nu t} + adg e^{+i nu t})        [variant LD1]")
    W("  M(t) = exp[i eta (a e^{-i nu t} + adg e^{+i nu t})]       [variant LDfull, exact]")
    W("       = D(i eta e^{+i nu t})")
    W("  frame: joint interaction picture, H_drift = 0 (physical.py:10-13)")
    W("  retained: LD-0 off-resonant carrier (amplitude Omega_c, detuning nu),")
    W("            2-nu LD-1 cross terms, both tones.  No per-tone RWA, no secular approx.")
    W(f"  qudit d = {D_QUDIT}  levels = {LEVELS}   ('s' carries no drive term)")
    W(f"  Fock cutoff = {N_CUT}   Hilbert dim = {DIM}   driven-block dim = {DIM_DRIVEN}")
    W()
    W("PARAMETERS (read from hypulse device spec, not retyped)")
    W(f"  device                    = '{DEVICE_ID}'")
    W(f"  eta                       = {ETA:.6f}                MEASURED (Meth)")
    W(f"  Omega_sideband_rad_per_s  = {_dev.params['Omega_sideband_rad_per_s']:.6f}   MEASURED (Meth)")
    W(f"    -> Omega_sb             = {OMEGA_SB:.9e} rad/us  = Omega_sb/2pi = "
      f"{OMEGA_SB/(2*math.pi)*1e6:.4f} Hz")
    W(f"  amp_bound_rad_per_us      = {OMEGA_C:.9e} rad/us  = Omega_c (carrier Rabi)")
    W(f"    -> Omega_c/2pi          = {OMEGA_C/(2*math.pi)*1e6/1e3:.4f} kHz")
    W(f"  omega_m_x_rad_per_s       = {_dev.params['omega_m_x_rad_per_s']:.6f}   MEASURED (Meth)")
    W(f"    -> nu                   = {NU:.9e} rad/us  = nu/2pi = {NU/(2*math.pi):.6f} MHz")
    W(f"  zeta                      = {ZETA:.9f} rad (= +pi/2)   CONVENTION (README.md:26-29)")
    W()
    W("  consistency of the effective coupling:")
    W(f"    eta * Omega_c = {ETA:.6f} * {OMEGA_C:.9e} = {ETA*OMEGA_C:.9e} rad/us")
    W(f"    Omega_sb      = {OMEGA_SB:.9e} rad/us")
    W(f"    ratio eta*Omega_c / Omega_sb = {ETA*OMEGA_C/OMEGA_SB:.12f}")
    W("    -> the spec's Omega_sideband IS eta*Omega_c; T_analytic = theta/Omega_sb.")
    W()
    W("  ADIABATICITY / RWA VALIDITY (this is the whole question)")
    W(f"    motional period      2pi/nu   = {2*math.pi/NU:.6f} us")
    W(f"    carrier/detuning     Omega_c/nu = {OMEGA_C/NU:.6f}   (RWA small parameter)")
    for th in THETAS:
        T = th / OMEGA_SB
        W(f"    theta = {th:<8g} T_paper = {T:>11.6f} us  =  {T/(2*math.pi/NU):>10.4f} motional periods")
    W()
    W("INTEGRATOR")
    W(f"  exponential midpoint, {N_SLICES} slices per run, {N_SAMPLES} recorded sample times")
    W(f"  convergence re-run at {N_SLICES_CONV} slices (reported per theta)")
    W()


# ----------------------------------------------------------------------------
# (a) U_real vs U_paper
# ----------------------------------------------------------------------------
def item_a(theta, tl, Us, variant):
    W("=" * 118)
    W(f"(a) U_real vs U_paper-generator   [theta = {theta}, variant = {variant}]")
    W("    U_paper(t) = exp(-i Omega_sb t sigma_x^{gk} (x) x_hat)   (Omega_sb t = the accrued angle)")
    W("    'vz' columns = after the optimal per-level virtual-Z frame update (physical.py:160-168)")
    W("=" * 118)
    W("    'signal' = 1-F(U_paper(t), I): how much the INTENDED gate itself differs from")
    W("    doing nothing.  err/signal > 1 means the drive's error exceeds the gate it performs,")
    W("    so a small absolute 1-F at tiny theta is a dilution artifact, not agreement.")
    W(f"{'t (us)':>12} {'angle=Om_sb*t':>15} {'1-F (full)':>15} {'1-F vz (full)':>15} "
      f"{'1-F (driven)':>15} {'1-F vz(driven)':>15} {'signal':>15} {'err/signal':>13}")
    m_f = m_fvz = m_d = m_dvz = m_rs = 0.0
    end = None
    for t, U in zip(tl, Us):
        V = U_paper(OMEGA_SB * t)
        e_f = 1 - fid(U, V)
        e_fvz, zs = fid_vz(U, V)
        e_fvz = 1 - e_fvz
        e_d = 1 - fid_driven(U, V)
        # driven-block vz: reuse the same per-level phases, restricted
        Vz = np.diag(np.repeat(np.exp(-1j * zs), N_CUT))
        e_dvz = 1 - fid_driven(Vz @ U, V)
        sig = 1 - fid(np.eye(DIM, dtype=complex), V)
        rs = e_fvz / sig if sig > 0 else float("nan")
        m_f, m_fvz = max(m_f, e_f), max(m_fvz, e_fvz)
        m_d, m_dvz = max(m_d, e_d), max(m_dvz, e_dvz)
        if np.isfinite(rs):
            m_rs = max(m_rs, rs)
        end = (e_f, e_fvz, e_d, e_dvz, sig, rs)
        W(f"{t:>12.8f} {OMEGA_SB*t:>15.9f} {e_f:>15.6e} {e_fvz:>15.6e} "
          f"{e_d:>15.6e} {e_dvz:>15.6e} {sig:>15.6e} {rs:>13.4f}")
    W()
    W(f"  ENDPOINT 1-F  full={end[0]:.6e}  full+vz={end[1]:.6e}  "
      f"driven={end[2]:.6e}  driven+vz={end[3]:.6e}")
    W(f"  ENDPOINT signal = {end[4]:.6e}   err/signal = {end[5]:.6f}")
    W(f"  MAX      1-F  full={m_f:.6e}  full+vz={m_fvz:.6e}  "
      f"driven={m_d:.6e}  driven+vz={m_dvz:.6e}")
    W(f"  MAX      err/signal = {m_rs:.6f}")
    return dict(end=end, mx=(m_f, m_fvz, m_d, m_dvz, m_rs))


# ----------------------------------------------------------------------------
# (b) conditioning structure
# ----------------------------------------------------------------------------
def item_b(theta, tl, Us, variant):
    W("=" * 118)
    W(f"(b) CONDITIONING STRUCTURE   [theta = {theta}, variant = {variant}]")
    W("    intended transfer block = {g,k} x {g,k} (any Fock).  'outside' = any block touching |s>.")
    W("    Fock metric is vs U_paper, since the intended gate IS a displacement and")
    W("    therefore legitimately fills the Fock block: we report max | |U_real| - |U_paper| |.")
    W("=" * 118)
    W(f"{'t (us)':>12} {'max|U| qudit outside':>22} {'max||U_r|-|U_p|| Fock':>24} "
      f"{'max|U_r[n,m]|, |n-m|>3':>24}")
    mq = mf = mfar = 0.0
    end = None
    for t, U in zip(tl, Us):
        Aabs = np.abs(U)
        outside = 0.0
        for (qa, qb) in [(IDX["g"], IDX["s"]), (IDX["s"], IDX["g"]),
                         (IDX["k"], IDX["s"]), (IDX["s"], IDX["k"])]:
            blk = Aabs[qa * N_CUT:(qa + 1) * N_CUT, qb * N_CUT:(qb + 1) * N_CUT]
            outside = max(outside, blk.max())
        V = U_paper(OMEGA_SB * t)
        dr, dp = np.abs(sub_driven(U)), np.abs(sub_driven(V))
        fock_dev = float(np.abs(dr - dp).max())
        far = 0.0
        for qa in range(2):
            for qb in range(2):
                blk = dr[qa * N_CUT:(qa + 1) * N_CUT, qb * N_CUT:(qb + 1) * N_CUT]
                for n in range(N_CUT):
                    for m in range(N_CUT):
                        if abs(n - m) > 3:
                            far = max(far, blk[n, m])
        mq, mf, mfar = max(mq, outside), max(mf, fock_dev), max(mfar, far)
        end = (outside, fock_dev, far)
        W(f"{t:>12.8f} {outside:>22.6e} {fock_dev:>24.6e} {far:>24.6e}")
    W()
    W(f"  ENDPOINT  qudit_outside={end[0]:.6e}  fock_dev={end[1]:.6e}  far_fock={end[2]:.6e}")
    W(f"  MAX       qudit_outside={mq:.6e}  fock_dev={mf:.6e}  far_fock={mfar:.6e}")
    W("  NOTE: 'qudit outside' is BY CONSTRUCTION zero -- no drive term in H(t) touches |s>.")
    W("        Its value here only confirms the integrator preserves that block structure.")
    return dict(end=end, mx=(mq, mf, mfar))


# ----------------------------------------------------------------------------
# (c) spectator
# ----------------------------------------------------------------------------
def item_c(theta, tl, Us, variant):
    W("=" * 118)
    W(f"(c) SPECTATOR LEVEL |s|   [theta = {theta}, variant = {variant}]")
    W("    BY CONSTRUCTION: H(t) contains no |s>-coupling term at all (only |k><g| + h.c.).")
    W("    This table therefore tests the integrator, not the physics of level selectivity.")
    W("    A genuine selectivity test needs an anchored cross-coupling ratio for the Meth")
    W("    level scheme; none exists in this repo (physical.py's r = 0.385 is RQC 171Yb+,")
    W("    dev-tree:tools/all_gates_pulse/QT_SB/code/physical.py:7 -- NOT transferable to 40Ca+).")
    W("=" * 118)
    worst_pop, worst_ph, worst_leak = 0.0, 0.0, 0.0
    for t, U in zip(tl, Us):
        pops, phs = [], []
        for n in range(N_CUT):
            i = row(IDX["s"], n)
            z = U[i, i]
            pops.append(abs(z))
            phs.append(np.angle(z))
            col = U[:, i]
            leak = float(np.sum(np.abs(col) ** 2)
                         - np.sum(np.abs(col[IDX["s"] * N_CUT:(IDX["s"] + 1) * N_CUT]) ** 2))
            worst_leak = max(worst_leak, leak)
        worst_pop = max(worst_pop, max(abs(p - 1.0) for p in pops))
        worst_ph = max(worst_ph, max(abs(p) for p in phs))
        W(f"  t = {t:12.8f} us")
        W("    |amp|: " + " ".join(f"n{n}={pops[n]:.12f}" for n in range(N_CUT)))
        W("    arg  : " + " ".join(f"n{n}={phs[n]:+.3e}" for n in range(N_CUT)))
    W()
    W(f"  OVERALL max | |<s,n|U|s,n>| - 1 | = {worst_pop:.6e}   [BY CONSTRUCTION ~0]")
    W(f"  OVERALL max |arg <s,n|U|s,n>|     = {worst_ph:.6e}   [BY CONSTRUCTION ~0]")
    W(f"  OVERALL max P(leave |s>)          = {worst_leak:.6e}   [BY CONSTRUCTION ~0]")
    return dict(pop=worst_pop, ph=worst_ph, leak=worst_leak)


# ----------------------------------------------------------------------------
# (d) leakage
# ----------------------------------------------------------------------------
def item_d(theta, tl, Us, variant):
    W("=" * 118)
    W(f"(d) LEAKAGE   [theta = {theta}, variant = {variant}]")
    W("    code subspace = {g,k} (x) all Fock.  P(n>2) is reported for U_real AND U_paper,")
    W("    because the intended gate is a displacement and populates n>2 BY DESIGN;")
    W("    only the DIFFERENCE is an error.")
    W("=" * 118)
    W(f"{'t (us)':>12} {'P out of {g,k}':>18} {'P(n>2) real':>16} {'P(n>2) paper':>16} "
      f"{'|difference|':>16}")
    mo = mr = mp = md = 0.0
    end = None
    for t, U in zip(tl, Us):
        V = U_paper(OMEGA_SB * t)
        p_out = 0.0
        for q in DRIVEN:
            for n in range(N_CUT):
                col = U[:, row(q, n)]
                p = float(np.sum(np.abs(col[IDX["s"] * N_CUT:(IDX["s"] + 1) * N_CUT]) ** 2))
                p_out = max(p_out, p)
        pr = pp = 0.0
        for q in DRIVEN:
            for n in range(3):
                cr, cp = U[:, row(q, n)], V[:, row(q, n)]
                sr = sum(abs(cr[row(qq, m)]) ** 2
                         for qq in range(D_QUDIT) for m in range(3, N_CUT))
                sp = sum(abs(cp[row(qq, m)]) ** 2
                         for qq in range(D_QUDIT) for m in range(3, N_CUT))
                pr, pp = max(pr, float(sr)), max(pp, float(sp))
        diff = abs(pr - pp)
        mo, mr, mp, md = max(mo, p_out), max(mr, pr), max(mp, pp), max(md, diff)
        end = (p_out, pr, pp, diff)
        W(f"{t:>12.8f} {p_out:>18.6e} {pr:>16.6e} {pp:>16.6e} {diff:>16.6e}")
    W()
    W(f"  ENDPOINT  P_out={end[0]:.6e}  P(n>2)real={end[1]:.6e}  "
      f"P(n>2)paper={end[2]:.6e}  diff={end[3]:.6e}")
    W(f"  MAX       P_out={mo:.6e}  P(n>2)real={mr:.6e}  "
      f"P(n>2)paper={mp:.6e}  diff={md:.6e}")
    W("  NOTE: 'P out of {g,k}' is BY CONSTRUCTION zero (no |s> drive term).")
    return dict(end=end, mx=(mo, mr, mp, md))


# ----------------------------------------------------------------------------
# (e) effective coupling
# ----------------------------------------------------------------------------
def item_e(theta, tl, Us, variant):
    W("=" * 118)
    W(f"(e) EFFECTIVE COUPLING FROM THE TRANSFER OSCILLATION   [theta = {theta}, variant = {variant}]")
    W("=" * 118)
    W("  For the paper generator, starting from |g,0>:")
    W("    exp(-i th sigma_x (x) x_hat)|g,0> ; in the sigma_x eigenbasis |+->,")
    W("    |phi_+-> = D(-+ i th/2)|0> and  <phi_+|phi_-> = exp(-th^2/2)  (real)")
    W("    => P_k(th) = (1/2)(1 - exp(-th^2/2))     [exact, no small-angle assumption]")
    W("    => invert:  th_est(t) = sqrt( -2 ln(1 - 2 P_k(t)) )")
    W("    => a linear fit of th_est vs t has slope = the effective coupling.")
    W()
    W(f"{'t (us)':>12} {'P_k real':>16} {'P_k paper':>16} {'th_est real':>16} "
      f"{'th_est paper':>16} {'Om_sb*t':>16}")
    ts, th_r, th_p = [], [], []
    for t, U in zip(tl, Us):
        V = U_paper(OMEGA_SB * t)
        col_r, col_p = U[:, row(IDX["g"], 0)], V[:, row(IDX["g"], 0)]
        pk_r = float(np.sum(np.abs(col_r[IDX["k"] * N_CUT:(IDX["k"] + 1) * N_CUT]) ** 2))
        pk_p = float(np.sum(np.abs(col_p[IDX["k"] * N_CUT:(IDX["k"] + 1) * N_CUT]) ** 2))

        def inv(p):
            arg = 1.0 - 2.0 * p
            return math.sqrt(-2.0 * math.log(arg)) if arg > 0 else float("nan")

        er, ep = inv(pk_r), inv(pk_p)
        ts.append(t); th_r.append(er); th_p.append(ep)
        W(f"{t:>12.8f} {pk_r:>16.9e} {pk_p:>16.9e} {er:>16.9e} {ep:>16.9e} "
          f"{OMEGA_SB*t:>16.9e}")
    ts = np.array(ts); th_r = np.array(th_r); th_p = np.array(th_p)
    ok = np.isfinite(th_r) & np.isfinite(th_p) & (ts > 0)
    slope_r = float(np.polyfit(ts[ok], th_r[ok], 1)[0])
    slope_p = float(np.polyfit(ts[ok], th_p[ok], 1)[0])
    W()
    W("  linear fit  th_est(t) = slope * t + c   (all t > 0 with a real inverse)")
    W(f"    slope_real  = {slope_r:.9e} rad/us   -> /2pi = {slope_r/(2*math.pi)*1e6:.6f} Hz")
    W(f"    slope_paper = {slope_p:.9e} rad/us   -> /2pi = {slope_p/(2*math.pi)*1e6:.6f} Hz")
    W(f"    prediction eta*Omega_c = {ETA*OMEGA_C:.9e} rad/us "
      f"(= Omega_sb = {OMEGA_SB:.9e})")
    W(f"    RATIO  slope_real / (eta*Omega_c)  = {slope_r/(ETA*OMEGA_C):.9f}")
    W(f"    RATIO  slope_paper/ (eta*Omega_c)  = {slope_p/(ETA*OMEGA_C):.9f}   "
      f"[sanity: paper curve must give 1]")
    W(f"    RATIO  slope_real / slope_paper    = {slope_r/slope_p:.9f}")
    return dict(slope_r=slope_r, slope_p=slope_p,
                ratio=slope_r / (ETA * OMEGA_C),
                ratio_p=slope_p / (ETA * OMEGA_C))


# ----------------------------------------------------------------------------
# (f) single resonant BSB -- the variant the step spec proposed
# ----------------------------------------------------------------------------
def item_f(theta):
    W("=" * 118)
    W(f"(f) SINGLE RESONANT BSB (the step spec's proposed drive)   [theta = {theta}]")
    W("    Propagated for comparison only, to justify using the bichromatic drive instead.")
    W("    Same Omega_c, same eta, same nu; ONLY the red tone is removed.")
    W("=" * 118)
    T = theta / OMEGA_SB
    tl, Us = propagate(T, "LDfull", tones="bsb_only", n_samples=11)
    W(f"{'t (us)':>12} {'1-F vs U_paper':>18} {'1-F vz':>18} {'1-F driven':>18}")
    m1 = m2 = m3 = 0.0
    end = None
    for t, U in zip(tl, Us):
        V = U_paper(OMEGA_SB * t)
        e1 = 1 - fid(U, V)
        e2 = 1 - fid_vz(U, V)[0]
        e3 = 1 - fid_driven(U, V)
        m1, m2, m3 = max(m1, e1), max(m2, e2), max(m3, e3)
        end = (e1, e2, e3)
        W(f"{t:>12.8f} {e1:>18.6e} {e2:>18.6e} {e3:>18.6e}")
    W()
    W(f"  ENDPOINT  1-F={end[0]:.6e}  vz={end[1]:.6e}  driven={end[2]:.6e}")
    W(f"  MAX       1-F={m1:.6e}  vz={m2:.6e}  driven={m3:.6e}")
    return dict(end=end, mx=(m1, m2, m3))


# ----------------------------------------------------------------------------
# (g) error attribution -- delete the LD-0 carrier and re-measure
# ----------------------------------------------------------------------------
def item_g(theta):
    W("=" * 118)
    W(f"(g) ERROR ATTRIBUTION: DELETE THE LD-0 CARRIER   [theta = {theta}]")
    W("    Diagnostic, not a physical drive.  variant 'LD1_nocarrier' keeps only the")
    W("    LD-first-order sideband piece and deletes the LD-0 carrier term.  If the")
    W("    residual collapses, the off-resonant carrier is the error source.")
    W("=" * 118)
    W("  scale comparison, computed from the read parameters:")
    stark = OMEGA_C ** 2 / (4 * NU)
    W(f"    carrier Rabi          Omega_c        = {OMEGA_C:.9e} rad/us")
    W(f"    mode frequency        nu             = {NU:.9e} rad/us")
    W(f"    RWA small parameter   Omega_c/nu     = {OMEGA_C/NU:.9f}")
    W(f"    carrier AC Stark      Omega_c^2/4nu  = {OMEGA_C:.6e}^2 / (4*{NU:.6e})")
    W(f"                                         = {stark:.9e} rad/us")
    W(f"    sideband coupling     Omega_sb       = {OMEGA_SB:.9e} rad/us")
    W(f"    RATIO  (Omega_c^2/4nu) / Omega_sb    = {stark/OMEGA_SB:.9f}")
    W("    -> if this ratio is O(1) the carrier light shift is comparable to the gate")
    W("       coupling itself, and no amount of GRAPE on a single control fixes it.")
    W()
    T = theta / OMEGA_SB
    out = {}
    for variant in ("LDfull", "LD1_nocarrier"):
        tl, Us = propagate(T, variant, n_samples=11)
        V = U_paper(OMEGA_SB * T)
        e = 1 - fid(Us[-1], V)
        evz = 1 - fid_vz(Us[-1], V)[0]
        # coupling slope, same estimator as (e)
        ts, th = [], []
        for t, U in zip(tl, Us):
            col = U[:, row(IDX["g"], 0)]
            pk = float(np.sum(np.abs(col[IDX["k"] * N_CUT:(IDX["k"] + 1) * N_CUT]) ** 2))
            arg = 1.0 - 2.0 * pk
            if t > 0 and arg > 0:
                ts.append(t); th.append(math.sqrt(-2.0 * math.log(arg)))
        slope = float(np.polyfit(ts, th, 1)[0]) if len(ts) > 1 else float("nan")
        out[variant] = (e, evz, slope, slope / (ETA * OMEGA_C))
        W(f"  {variant:>14}:  1-F endpoint = {e:.6e}   vz = {evz:.6e}   "
          f"slope = {slope:.9e}   slope/(eta*Om_c) = {slope/(ETA*OMEGA_C):.6f}")
    W()
    f_full, f_noc = out["LDfull"][1], out["LD1_nocarrier"][1]
    W(f"  carrier contribution to 1-F(vz): {f_full:.6e} (with) -> {f_noc:.6e} (without), "
      f"factor {f_full/f_noc if f_noc > 0 else float('inf'):.1f}")
    return out


# ----------------------------------------------------------------------------
def convergence(theta, variant):
    T = theta / OMEGA_SB
    _, U1 = propagate(T, variant, n_slices=N_SLICES, n_samples=2)
    _, U2 = propagate(T, variant, n_slices=N_SLICES_CONV, n_samples=2)
    return float(np.abs(U1[-1] - U2[-1]).max())


def main():
    import json
    claimed = json.load(open(QTSB_JSON))["summary"]
    header(claimed)

    results = {}
    for theta in THETAS:
        T = theta / OMEGA_SB
        W("#" * 118)
        W(f"# THETA = {theta}    T_paper = {T:.8f} us    "
          f"({T/(2*math.pi/NU):.4f} motional periods)")
        W("#" * 118)
        W()
        for variant in ("LD1", "LDfull"):
            W(f"propagating theta={theta} variant={variant} "
              f"({N_SLICES} slices, {N_SAMPLES} samples) ...")
            tl, Us = propagate(T, variant)
            W("  done")
            W()
            ra = item_a(theta, tl, Us, variant)
            rb = item_b(theta, tl, Us, variant)
            rc = item_c(theta, tl, Us, variant)
            rd = item_d(theta, tl, Us, variant)
            re = item_e(theta, tl, Us, variant)
            results[(theta, variant)] = dict(a=ra, b=rb, c=rc, d=rd, e=re)
        conv = convergence(theta, "LDfull")
        W(f"  integrator convergence, theta={theta}: max|U({N_SLICES}) - "
          f"U({N_SLICES_CONV})| = {conv:.6e}")
        W()
        results[(theta, "conv")] = conv
        results[(theta, "bsb")] = item_f(theta)
        W()
        results[(theta, "attr")] = item_g(theta)
        W()

    # ------------------------------------------------------------------
    W("=" * 118)
    W("VERDICT")
    W("=" * 118)
    W()
    W("Table 1 -- (a) U_real vs paper generator.  ENDPOINT and MAX, both variants.")
    W(f"{'theta':>8} {'variant':>8} | {'1-F end':>13} {'1-F end vz':>13} "
      f"{'1-F end drv':>13} | {'1-F max':>13} {'1-F max vz':>13} {'1-F max drv':>13}")
    for theta in THETAS:
        for variant in ("LD1", "LDfull"):
            r = results[(theta, variant)]["a"]
            W(f"{theta:>8g} {variant:>8} | {r['end'][0]:>13.6e} {r['end'][1]:>13.6e} "
              f"{r['end'][2]:>13.6e} | {r['mx'][0]:>13.6e} {r['mx'][1]:>13.6e} "
              f"{r['mx'][2]:>13.6e}")
    W()
    W("Table 1b -- (a) the same error measured AGAINST THE GATE'S OWN SIGNAL.")
    W("  signal = 1-F(U_paper, I).  err/signal > 1 => the drive's error is larger than")
    W("  the gate it is supposed to perform, so the absolute 1-F above is diluted.")
    W(f"{'theta':>8} {'variant':>8} | {'1-F end vz':>14} {'signal end':>14} "
      f"{'err/signal end':>15} {'err/signal MAX':>15}")
    for theta in THETAS:
        for variant in ("LD1", "LDfull"):
            r = results[(theta, variant)]["a"]
            W(f"{theta:>8g} {variant:>8} | {r['end'][1]:>14.6e} {r['end'][4]:>14.6e} "
              f"{r['end'][5]:>15.4f} {r['mx'][4]:>15.4f}")
    W()
    W("Table 2 -- (b) conditioning, (c) spectator, (d) leakage.  ENDPOINT / MAX.")
    W(f"{'theta':>8} {'variant':>8} | {'qudit_out MAX':>14} {'fock_dev MAX':>14} "
      f"{'spec_pop MAX':>14} {'P_out MAX':>12} {'P(n>2)diff MAX':>15}")
    for theta in THETAS:
        for variant in ("LD1", "LDfull"):
            r = results[(theta, variant)]
            W(f"{theta:>8g} {variant:>8} | {r['b']['mx'][0]:>14.6e} {r['b']['mx'][1]:>14.6e} "
              f"{r['c']['pop']:>14.6e} {r['d']['mx'][0]:>12.6e} {r['d']['mx'][3]:>15.6e}")
    W("  columns 'qudit_out', 'spec_pop', 'P_out' are BY CONSTRUCTION zero (see (c)).")
    W()
    W("Table 3 -- (e) effective coupling vs eta*Omega prediction.")
    W(f"{'theta':>8} {'variant':>8} | {'slope_real':>16} {'eta*Omega_c':>16} "
      f"{'RATIO real':>13} {'RATIO paper':>13}")
    for theta in THETAS:
        for variant in ("LD1", "LDfull"):
            r = results[(theta, variant)]["e"]
            W(f"{theta:>8g} {variant:>8} | {r['slope_r']:>16.9e} {ETA*OMEGA_C:>16.9e} "
              f"{r['ratio']:>13.6f} {r['ratio_p']:>13.6f}")
    W()
    W("Table 4 -- (f) single resonant BSB (step spec's proposal) vs the paper generator.")
    W(f"{'theta':>8} | {'1-F end':>14} {'1-F end vz':>14} {'1-F max':>14} {'1-F max vz':>14}")
    for theta in THETAS:
        r = results[(theta, "bsb")]
        W(f"{theta:>8g} | {r['end'][0]:>14.6e} {r['end'][1]:>14.6e} "
          f"{r['mx'][0]:>14.6e} {r['mx'][1]:>14.6e}")
    W()
    W("Table 5 -- integrator convergence (LDfull, endpoint).")
    for theta in THETAS:
        W(f"  theta = {theta:<8g}  max|U({N_SLICES}) - U({N_SLICES_CONV})| = "
          f"{results[(theta,'conv')]:.6e}")
    W()
    W("Table 6 -- (g) error attribution, LD-0 carrier deleted.  Endpoint, LDfull vs no-carrier.")
    W(f"{'theta':>8} {'variant':>15} | {'1-F end':>14} {'1-F end vz':>14} "
      f"{'slope':>16} {'slope/(eta*Om_c)':>18}")
    for theta in THETAS:
        for variant in ("LDfull", "LD1_nocarrier"):
            e, evz, sl, rt = results[(theta, "attr")][variant]
            W(f"{theta:>8g} {variant:>15} | {e:>14.6e} {evz:>14.6e} {sl:>16.9e} {rt:>18.6f}")
    W()

    # ---- pass/fail ----
    W("PASS / FAIL")
    W("-" * 118)
    W("  Criterion 1 is reported BOTH ways.  The absolute form (1-F < 1e-3) is the one")
    W("  originally written; it is kept visible because at theta = 0.0015 it passes for a")
    W("  reason that is not agreement -- both unitaries are near identity.  The")
    W("  signal-relative form is the meaningful test and both are printed, nothing dropped.")
    W()
    for theta in THETAS:
        T = theta / OMEGA_SB
        r = results[(theta, "LDfull")]
        e_vz = r["a"]["end"][1]
        sig = r["a"]["end"][4]
        rs = r["a"]["end"][5]
        ratio = r["e"]["ratio"]
        periods = T / (2 * math.pi / NU)
        p1a = e_vz < 1e-3
        p1b = rs < 0.1
        p2 = 0.9 < ratio < 1.1
        W(f"  theta = {theta:<8g}  T = {T:>10.6f} us  ({periods:.3f} motional periods)")
        W(f"    1a. absolute   1-F endpoint (vz) < 1e-3            : "
          f"{e_vz:.6e}   [{'PASS' if p1a else 'FAIL'}]")
        W(f"    1b. relative   err/signal < 0.1  (signal={sig:.3e})  : "
          f"{rs:.6f}   [{'PASS' if p1b else 'FAIL'}]")
        W(f"    2.  effective coupling = eta*Omega_c within 10%     : "
          f"{ratio:.6f}   [{'PASS' if p2 else 'FAIL'}]")
    W()
    W("SCOPE / WHAT IS NOT TESTED")
    W("-" * 118)
    W("  * Level selectivity is BY CONSTRUCTION, not measured: H(t) has no |s> coupling.")
    W("    Testing it needs an anchored Meth cross-coupling ratio, which this repo lacks.")
    W("  * Single motional mode only; no spectator modes, no heating, no dephasing.")
    W("  * Flat envelope with instantaneous on/off; no ramps (appendix_table.md:270).")
    W("  * 'Meth Eq. 2' is not transcribed anywhere in this repo; the drive form used")
    W("    here is the standard Lamb-Dicke expansion as implemented by")
    W("    dev-tree:tools/all_gates_pulse/QT_SB/code/physical.py:88-125, with the LD truncation")
    W("    lifted (LDfull).  The FORM is therefore repo-anchored, not Meth-verbatim.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
