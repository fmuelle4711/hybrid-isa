"""QCD validation against the Bazavan full time-dependent Hamiltonian (after
OPTICAL RWA only). No Bazavan-side RWA applied — all mode/spin time
dependences kept explicit (cos(delta*t), exp(±i omega_z t)).

Amendments to STEP 1 setup:
  A. Name: "Bazavan full time-dependent Hamiltonian after optical RWA".
  B. At delta = omega_z/2 the emergent conditioning basis is sigma_z (verified
     TEST B in qcd_bazavan_phys.py). U_QCD and U_eff are DEFINED in this basis
     with |0>, |1> as sigma_z eigenstates.
  C. Fit an unconditional mode rotation R(phi) = I ⊗ exp(-i phi n_hat) before
     the target fidelity. Report F with and without the fit; report fitted phi.
  D. sesolve options: max_step = 30 ns EXPLICITLY (tlist alone does not bound
     internal steps).
  E. u_p waveform is JITTERED (verified: 170/200 slots exactly at bound, 30
     distinct values, std = 1.14e-5 rad/us). Use per-slot piecewise-constant
     Omega_c(t) inverted from u_p:
        Omega_c(t) = u_p_slot(t) / [ eta * (J1+J3)(2*Omega_c/delta) ]
     (the Bessel factor is treated as constant at its Meth value 0.248,
      since Omega_c varies only within +-2% of nominal).
  F. u_x = 0 in the drive (artifact; zeroing improves plugin F by 1e-5).
  G. Compare fidelities in TWO subspace conventions:
        - Full-dim: F = |Tr(U_targ^dag · U)|^2 / dim^2, dim = 2*N_cut
        - Code-subspace: restrict to |q>|n> with n < n_code = N_cut - 2
     Run at N_cut in {8, 10, 12} for cutoff convergence.
  H. Constant Ω_c (u_p-mapped) drive; NO ramps (later step).
"""
from __future__ import annotations
import math, json, time
from pathlib import Path
import numpy as np
import qutip as qt


HERE = Path(__file__).resolve().parent
FINAL_V3 = HERE / "final_v3.json"


# =============================================================================
# Constants (SI, rad/s and dimensionless)
# =============================================================================
ETA         = 0.041
OMEGA_Z     = 2 * math.pi * 0.77e6       # 4.8381e6 rad/s
DELTA       = OMEGA_Z / 2                # 2.4190e6 rad/s
OMEGA_C_2PI_KHZ_NOMINAL = 97.56           # Meth Ω_c derived from Ω_sb/eta
OMEGA_C_NOMINAL = 2 * math.pi * OMEGA_C_2PI_KHZ_NOMINAL * 1e3   # 6.130e5 rad/s

# Bessel factor at Meth (2 Omega_c / delta) = 2*97.56/385 = 0.507
from scipy.special import jv
X_BESSEL = 2 * OMEGA_C_NOMINAL / DELTA
J1_PLUS_J3 = float(jv(1, X_BESSEL) + jv(3, X_BESSEL))
print(f"Bessel factor (J1+J3)(x={X_BESSEL:.4f}) = {J1_PLUS_J3:.6f}   (should be ~0.248)")

ALPHA = 0.5
T_US  = 160.4                              # QCD paper headline
T_SEC = T_US * 1e-6

# dt / integrator
DT_OUT_NS = 30.0
DT_OUT_SEC = DT_OUT_NS * 1e-9
MAX_STEP_NS = 30.0                         # per amendment D
MAX_STEP_SEC = MAX_STEP_NS * 1e-9
ATOL, RTOL = 1e-10, 1e-8
NSTEPS = 500_000

N_tlist = int(math.ceil(T_SEC / DT_OUT_SEC)) + 1
tlist = np.linspace(0.0, T_SEC, N_tlist)
print(f"N_tlist = {N_tlist}   dt_out = {DT_OUT_NS} ns   max_step = {MAX_STEP_NS} ns")


# =============================================================================
# Load GRAPE waveform and build piecewise-constant Omega_c(t) callable
# =============================================================================
final = json.loads(FINAL_V3.read_text())
u_p = np.asarray(final["headline"]["Omega_p"], dtype=float)          # rad/us
u_x = np.asarray(final["headline"]["Omega_x"], dtype=float)
N_slots = len(u_p)                                                    # 200
dt_slot_us = float(final["headline"]["T_us"]) / N_slots               # 0.802 us
print(f"loaded QCD final_v3 waveform: N_slots={N_slots}, dt_slot={dt_slot_us} us")
print(f"  u_p range: [{u_p.min():.4e}, {u_p.max():.4e}]   std={u_p.std():.4e}")
print(f"  amendment F: u_x set to 0 (originally |mean|={abs(u_x.mean()):.2e})")

# Invert: Omega_c_slot = u_p_slot / (eta * (J1+J3))
# u_p in rad/us -> rad/s by *1e6
Omega_c_slot_rad_per_s = (u_p * 1e6) / (ETA * J1_PLUS_J3)             # rad/s per slot
print(f"  Omega_c(t) range: [{Omega_c_slot_rad_per_s.min()/(2*math.pi)/1e3:.3f}, "
      f"{Omega_c_slot_rad_per_s.max()/(2*math.pi)/1e3:.3f}] kHz "
      f"(nominal Meth: {OMEGA_C_2PI_KHZ_NOMINAL} kHz)")


def Omega_c_of_t(t):
    """Return Omega_c(t) [rad/s] at time t [sec] as piecewise-constant per slot."""
    idx = int((t * 1e6) / dt_slot_us)
    if idx < 0: idx = 0
    if idx >= N_slots: idx = N_slots - 1
    return Omega_c_slot_rad_per_s[idx]


# =============================================================================
# Build Bazavan H at a chosen cutoff and return time-dep list for sesolve
# =============================================================================
def build_H_list(n_cut):
    a = qt.destroy(n_cut); adg = a.dag()
    I_m = qt.qeye(n_cut)
    sx = qt.sigmax(); sy = qt.sigmay()

    # base operators (Omega_c-independent shape)
    op_A  = qt.tensor(-sy, I_m)                       # multiplied by Omega_c(t) * cos(delta t)
    op_B1 = qt.tensor( sx, a  )                       # multiplied by eta*Omega_c(t) * cos(delta t) * exp(-i omega_z t)
    op_B2 = qt.tensor( sx, adg)                       # multiplied by eta*Omega_c(t) * cos(delta t) * exp(+i omega_z t)

    def coeff_A(t, args):
        return Omega_c_of_t(t) * math.cos(DELTA * t)

    def coeff_B1(t, args):
        return ETA * Omega_c_of_t(t) * math.cos(DELTA * t) * complex(math.cos(OMEGA_Z * t), -math.sin(OMEGA_Z * t))

    def coeff_B2(t, args):
        return ETA * Omega_c_of_t(t) * math.cos(DELTA * t) * complex(math.cos(OMEGA_Z * t), +math.sin(OMEGA_Z * t))

    H_list = [[op_A, coeff_A], [op_B1, coeff_B1], [op_B2, coeff_B2]]
    return H_list


# =============================================================================
# Targets in sigma_z eigenbasis
# =============================================================================
def build_U_QCD(n_cut, alpha=ALPHA):
    """DRIVE-PHASE convention: Bazavan Eq. 1 at phi=0 produces effective
    A < 0 (i.e., |0>-branch displaces to NEGATIVE alpha per TEST A rate_X_up<0).
    So the U_QCD target we compare against in this frame is D(-alpha) on |0>,
    D(+alpha) on |1>."""
    Dp = qt.displace(n_cut, +alpha); Dm = qt.displace(n_cut, -alpha)
    P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
    P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
    # sign-flipped to match Bazavan phi=0 drive convention:
    return qt.tensor(P0, Dm) + qt.tensor(P1, Dp)


def build_U_eff(n_cut, T=T_SEC):
    """RWA effective propagator: exp(-i A T sigma_z ⊗ P), with
    A = -eta * Omega_c_nominal * (J1+J3)  (SIGN-FLIPPED to match Bazavan phi=0)."""
    a = qt.destroy(n_cut); adg = a.dag()
    P = (a - adg) / (2.0j)
    A = -ETA * OMEGA_C_NOMINAL * J1_PLUS_J3                                  # note sign
    H_eff = A * qt.tensor(qt.sigmaz(), P)
    return (-1j * H_eff * T).expm()


def build_mode_rotation(n_cut, phi):
    """R(phi) = I ⊗ exp(-i phi n_hat)  (unconditional mode phase)."""
    a = qt.destroy(n_cut); adg = a.dag()
    n_op = adg * a
    return qt.tensor(qt.qeye(2), (-1j * phi * n_op).expm())


# =============================================================================
# Fidelities
# =============================================================================
def F_full(U, U_targ, n_cut):
    dim = 2 * n_cut
    tr = complex((U_targ.dag() * U).tr())
    return float(abs(tr) ** 2 / dim ** 2)


def F_code(U, U_targ, n_cut, n_code=None):
    """Trace only over |q>|n> with n < n_code (default N_cut-2, avoid boundary)."""
    if n_code is None:
        n_code = n_cut - 2
    dim_code = 2 * n_code
    U_arr    = U.full()
    Ut_arr   = U_targ.full()
    # tensor ordering: qubit outer, mode inner. index = q * n_cut + n
    idx = [q * n_cut + n for q in (0, 1) for n in range(n_code)]
    U_sub  = U_arr[np.ix_(idx, idx)]
    Ut_sub = Ut_arr[np.ix_(idx, idx)]
    tr = np.trace(np.conj(Ut_sub.T) @ U_sub)
    return float(abs(tr) ** 2 / dim_code ** 2)


def fit_mode_phi(U, U_targ, n_cut, coarse=101, fine_width=0.05):
    """Fit unconditional phi that maximizes F_full( R(phi) U, U_targ ).
    Two-pass: coarse then fine bracket."""
    def F(phi):
        R = build_mode_rotation(n_cut, phi)
        return F_full(R * U, U_targ, n_cut)
    phis_coarse = np.linspace(-math.pi, math.pi, coarse)
    Fs = [F(p) for p in phis_coarse]
    i_best = int(np.argmax(Fs))
    phi_best = phis_coarse[i_best]
    lo = phi_best - fine_width; hi = phi_best + fine_width
    phis_fine = np.linspace(lo, hi, 201)
    Fs_fine = [F(p) for p in phis_fine]
    i_best_fine = int(np.argmax(Fs_fine))
    phi_best_fine = phis_fine[i_best_fine]
    return phi_best_fine, Fs_fine[i_best_fine]


def fit_su2_joint(U, U_targ, n_cut, n_trials=25):
    """Joint fit: mode phi + unconditional SU(2) spin rotation
    R_spin(n̂ , angle) = exp(-i angle/2 (n_x sx + n_y sy + n_z sz)).
    Returns dict with fitted params and F_max."""
    from scipy.optimize import minimize
    sx = qt.sigmax(); sy = qt.sigmay(); sz = qt.sigmaz()
    a = qt.destroy(n_cut); adg = a.dag(); n_op = adg * a

    def R_spin(nx, ny, nz, angle):
        n = np.array([nx, ny, nz])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            return qt.qeye(2)
        n = n / norm
        return (-1j * angle / 2 * (n[0]*sx + n[1]*sy + n[2]*sz)).expm()

    def R_mode(phi):
        return qt.tensor(qt.qeye(2), (-1j * phi * n_op).expm())

    def objective(params):
        phi_m, nx, ny, nz, angle = params
        Rs = qt.tensor(R_spin(nx, ny, nz, angle), qt.qeye(n_cut))
        Rm = R_mode(phi_m)
        return -F_full(Rs * Rm * U, U_targ, n_cut)

    best = None
    np.random.seed(0)
    for trial in range(n_trials):
        x0 = np.random.uniform(-1, 1, 5)
        x0[4] = np.random.uniform(-math.pi, math.pi)
        x0[0] = np.random.uniform(-math.pi, math.pi)
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
        if best is None or -res.fun > -best.fun:
            best = res
    return dict(phi_mode=float(best.x[0]),
                spin_axis=(float(best.x[1]), float(best.x[2]), float(best.x[3])),
                spin_angle=float(best.x[4]),
                F_max=float(-best.fun))


# =============================================================================
# Propagate + observe
# =============================================================================
def propagate_U_and_quadratures(n_cut):
    """Build U_lab(T) via qt.propagator; also return <X>(t), <P>(t) for |q>|0>."""
    H_list = build_H_list(n_cut)

    # observables per branch
    a = qt.destroy(n_cut); adg = a.dag()
    X = (a + adg) / 2.0; P = (a - adg) / (2.0j)
    X_op = qt.tensor(qt.qeye(2), X)
    P_op = qt.tensor(qt.qeye(2), P)

    options = qt.Options(atol=ATOL, rtol=RTOL, nsteps=NSTEPS, method="adams",
                          store_states=True, max_step=MAX_STEP_SEC)

    # 1. Propagator for U_lab
    print(f"  building U_lab at N_cut={n_cut}, dim={2*n_cut}... ", end="", flush=True)
    t0 = time.time()
    U_prop = qt.propagator(H_list, T_SEC, args={}, options=options)
    if isinstance(U_prop, list):                                       # qutip 5 sometimes returns [U]
        U_lab = U_prop[-1] if len(U_prop) > 1 else U_prop[0]
    else:
        U_lab = U_prop
    print(f"done ({time.time()-t0:.1f} s)")

    # 2. Per-branch <X>(t), <P>(t)  (only for N_cut = 8 to keep output small)
    xy = None
    if n_cut == 8:
        print(f"  sesolve <X>, <P> per branch... ", end="", flush=True)
        t1 = time.time()
        vac = qt.basis(n_cut, 0)
        xy = {}
        for branch, lbl in [(0, "up"), (1, "dn")]:
            psi0 = qt.tensor(qt.basis(2, branch), vac)
            res = qt.sesolve(H_list, psi0, tlist, e_ops=[X_op, P_op],
                              args={}, options=options)
            xy[lbl] = dict(X=res.expect[0].tolist(), P=res.expect[1].tolist())
        print(f"done ({time.time()-t1:.1f} s)")

    return U_lab, xy


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 76)
    print("QCD validation vs Bazavan full time-dependent H (optical RWA only)")
    print("=" * 76)
    print(f"T = {T_US} us   alpha = {ALPHA}   Omega_c/2pi = {OMEGA_C_2PI_KHZ_NOMINAL} kHz nominal")
    print(f"eta = {ETA}   omega_z/2pi = 0.77 MHz   delta = omega_z/2 = 2pi * 385 kHz")
    print()

    results = {}
    for n_cut in [8, 10, 12]:
        print(f"--- N_cut = {n_cut} ---")
        U_lab, xy = propagate_U_and_quadratures(n_cut)
        U_QCD = build_U_QCD(n_cut)
        U_eff = build_U_eff(n_cut)

        # phi fit for U_lab vs U_QCD
        phi_fit, F_fit_full = fit_mode_phi(U_lab, U_QCD, n_cut)
        R = build_mode_rotation(n_cut, phi_fit)
        U_lab_fit = R * U_lab

        F_lab_vs_QCD_raw_full = F_full(U_lab, U_QCD, n_cut)
        F_lab_vs_QCD_raw_code = F_code(U_lab, U_QCD, n_cut)
        F_lab_vs_QCD_fit_full = F_full(U_lab_fit, U_QCD, n_cut)
        F_lab_vs_QCD_fit_code = F_code(U_lab_fit, U_QCD, n_cut)

        # joint SU(2) spin + mode rotation fit
        su2 = fit_su2_joint(U_lab, U_QCD, n_cut)
        F_lab_vs_QCD_su2_full = su2["F_max"]

        F_lab_vs_eff_full = F_full(U_lab, U_eff, n_cut)
        F_lab_vs_eff_code = F_code(U_lab, U_eff, n_cut)

        F_eff_vs_QCD_full = F_full(U_eff, U_QCD, n_cut)
        F_eff_vs_QCD_code = F_code(U_eff, U_QCD, n_cut)

        results[n_cut] = dict(
            phi_fit=phi_fit,
            F_lab_vs_QCD_raw_full=F_lab_vs_QCD_raw_full,
            F_lab_vs_QCD_raw_code=F_lab_vs_QCD_raw_code,
            F_lab_vs_QCD_fit_full=F_lab_vs_QCD_fit_full,
            F_lab_vs_QCD_fit_code=F_lab_vs_QCD_fit_code,
            F_lab_vs_QCD_su2_full=F_lab_vs_QCD_su2_full,
            su2_phi_mode=su2["phi_mode"],
            su2_spin_axis=su2["spin_axis"],
            su2_spin_angle=su2["spin_angle"],
            F_lab_vs_eff_full=F_lab_vs_eff_full,
            F_lab_vs_eff_code=F_lab_vs_eff_code,
            F_eff_vs_QCD_full=F_eff_vs_QCD_full,
            F_eff_vs_QCD_code=F_eff_vs_QCD_code,
            xy=xy,
        )
        print(f"  phi_fit                 = {phi_fit:+.5f} rad")
        print(f"  F_lab_vs_QCD  raw full  = {F_lab_vs_QCD_raw_full:.6f}")
        print(f"  F_lab_vs_QCD  raw code  = {F_lab_vs_QCD_raw_code:.6f}   (n_code = {n_cut-2})")
        print(f"  F_lab_vs_QCD  R(phi)    = {F_lab_vs_QCD_fit_full:.6f}   (mode-phase only)")
        print(f"  F_lab_vs_QCD  SU2+R     = {F_lab_vs_QCD_su2_full:.6f}   <-- joint mode phi + SU(2) spin")
        print(f"     su2_spin_axis        = {su2['spin_axis']}")
        print(f"     su2_spin_angle       = {su2['spin_angle']:+.4f} rad")
        print(f"     su2_phi_mode         = {su2['phi_mode']:+.4f} rad")
        print(f"  F_lab_vs_eff  raw full  = {F_lab_vs_eff_full:.6f}")
        print(f"  F_eff_vs_QCD  raw full  = {F_eff_vs_QCD_full:.6f}   (analytic prediction)")
        print()

    # ---- save + summary table ----
    outp = HERE / "step_qcd_bazavan_full_lab_results.json"
    payload = {
        "run_params": dict(T_us=T_US, alpha=ALPHA, eta=ETA,
                            Omega_c_2pi_kHz_nominal=OMEGA_C_2PI_KHZ_NOMINAL,
                            omega_z_2pi_MHz=0.77, delta_2pi_kHz=385.0,
                            dt_out_ns=DT_OUT_NS, max_step_ns=MAX_STEP_NS,
                            atol=ATOL, rtol=RTOL, N_tlist=N_tlist,
                            J1_plus_J3=J1_PLUS_J3, u_x_zeroed=True,
                            n_slots=N_slots, dt_slot_us=dt_slot_us),
        "results_by_ncut": {str(k): {kk: vv for kk, vv in v.items() if kk != "xy"}
                              for k, v in results.items()},
        "quadrature_trajectories_ncut8": results[8]["xy"],
        "tlist_sec": tlist.tolist(),
    }
    outp.write_text(json.dumps(payload, indent=2))
    print(f"saved {outp}")

    print()
    print("=" * 76)
    print(f"SUMMARY TABLE  (T = {T_US} us, alpha = {ALPHA}, u_x = 0)")
    print("=" * 76)
    print(f"{'N_cut':>5}  {'F_raw_full':>10}  {'raw_code':>10}  "
          f"{'R(phi)':>10}  {'SU2+R':>10}  {'phi_mode':>9}  {'spin_ang':>9}  "
          f"{'F_lab_eff':>10}  {'F_eff_QCD':>10}")
    for n_cut, r in results.items():
        print(f"{n_cut:>5}  "
              f"{r['F_lab_vs_QCD_raw_full']:>10.6f}  "
              f"{r['F_lab_vs_QCD_raw_code']:>10.6f}  "
              f"{r['F_lab_vs_QCD_fit_full']:>10.6f}  "
              f"{r['F_lab_vs_QCD_su2_full']:>10.6f}  "
              f"{r['su2_phi_mode']:>+9.4f}  "
              f"{r['su2_spin_angle']:>+9.4f}  "
              f"{r['F_lab_vs_eff_full']:>10.6f}  "
              f"{r['F_eff_vs_QCD_full']:>10.6f}")


if __name__ == "__main__":
    main()
