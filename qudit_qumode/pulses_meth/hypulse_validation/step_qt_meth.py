"""QT carrier gate, qutip-qtrl GRAPE, innsbruck_meth_2025 anchor.

Purpose: replace the archived non-HyPulse L-BFGS-B QT run with a run on the
same optimizer stack (qutip-qtrl GRAPE) that every other gate in this repo uses.

APPROVED SETUP (see conversation):
  (A) imposed treatment -- direct qutip-qtrl call, NOT via the plugin registry
      (HyPulse has no QT plugin; registry = CR, CD, QCD, CS only).
  (i) 5-dim model: code {G,A,B,C} (idx 0-3) + S1/2 auxiliary |g> (idx 4).
      Target acts on the {G,A} pair; B,C are spectators; no mode.
      Transitions go THROUGH the auxiliary (no direct D5/2<->D5/2 quadrupole
      coupling), exactly as archive/v1_pergate/QT/grape_meth_qt.py.
  frozen worst-case Stark in the optimiser + exact |Omega(t)|^2 audit after.
  per-quadrature bound = Omega_max / sqrt(2)  (total tone field <= Omega_max).

T picker: F_grape >= 0.99999  AND  aux leakage < 1e-3  AND  |spectator_dF| < 1e-3.

STRUCTURAL NOTE (surfaced, not silently handled)
------------------------------------------------
The frozen Stark drift is diagonal on {B,C}; the drive has matrix elements only
among {G,A,g}. The two act on DISJOINT subspaces, so they commute EXACTLY:
    [H_stark, H_drive] = 0
Therefore U(T) = exp(-i H_stark T) . U_drive(T) factorises, and the spectator
Stark contributes nothing but a diagonal phase exp(-i s_l T) on |B>,|C>.
That phase is a virtual-Z. Meth compensates exactly this in software
(anchor_params ac_stark_spectator_shifts_kHz._note: "Compensated in software by
frame updates"), and the ISA prices it at T_QPS = 0.

So the GRAPE target is defined as
    U_target = exp(-i H_stark_frozen T) . U_QT_bare
i.e. "do QT on {G,A}; let B,C accrue their known, freely-correctable phase".
We report BOTH F vs this target and F vs the bare U_QT after an explicit
virtual-Z fit, plus the fitted VZ angles (the QPS frame update, cost 0 us).

STARK REFERENCE -- TWO VARIANTS RUN
-----------------------------------
The archive sets OMEGA_REF_STARK = 2*pi*4 kHz, treating Meth's quoted 4 kHz as
the CARRIER Rabi. Meth's 4 kHz is the SIDEBAND Rabi (= eta * Omega_carrier), so
the carrier Rabi at the point where the shifts were measured was
4 kHz / 0.041 = 97.56 kHz. This is the FINDING-3 class of error (missing eta).
  variant "archive"    : Omega_ref = 2*pi*4 kHz      -> scale factor 1250
  variant "eta_corrected": Omega_ref = 2*pi*97.56 kHz -> scale factor ~2.1
Both are run; the archive variant is primary (for comparability with the
archived 36 us), the eta-corrected variant is reported as a flagged sensitivity.
"""
from __future__ import annotations
import json, math, time, logging
from pathlib import Path
import numpy as np
import qutip as qt
from qutip.control.pulseoptim import optimize_pulse_unitary


HERE = Path(__file__).resolve().parent
ANCHOR = json.loads((HERE.parent / "anchor_params.json").read_text(encoding="utf-8"))

# ---------------- anchors (READ, not typed) ----------------
OMEGA_MAX_2PI_KHZ = float(
    ANCHOR["hardware_bounds_NOT_in_meth"]["carrier_omega_max_over_2pi_kHz"]["value"])
_stark = ANCHOR["ac_stark_spectator_shifts_kHz"]
SHIFT_B_KHZ = float(_stark["D_minus_5_2"]["value"])     # 1.9
SHIFT_C_KHZ = float(_stark["D_plus_1_2"]["value"])      # 1.4
SHIFT_G_KHZ = float(_stark["D_plus_3_2"]["value"])      # 5.3  (assigned to G in
                                                        # the archive; G is DRIVEN
                                                        # here so it is skipped)
ETA = float(ANCHOR["sideband_rabi_interpretation"]["value_sideband_rabi_2pi_kHz"]) \
      if "sideband_rabi_interpretation" in ANCHOR else 4.0
OMEGA_SB_2PI_KHZ = 4.0
ETA_LD = 0.041
OMEGA_C_REF_2PI_KHZ = OMEGA_SB_2PI_KHZ / ETA_LD          # 97.56 kHz

# units: rad/us throughout
OMEGA_MAX = 2 * math.pi * OMEGA_MAX_2PI_KHZ * 1e-3       # rad/us
BOUND_QUAD = OMEGA_MAX / math.sqrt(2.0)                  # per-quadrature bound

STARK_REFS = {
    "archive":       2 * math.pi * OMEGA_SB_2PI_KHZ    * 1e-3,   # rad/us (4 kHz)
    "eta_corrected": 2 * math.pi * OMEGA_C_REF_2PI_KHZ * 1e-3,   # rad/us (97.56 kHz)
}

DIM     = 5
IDX     = {"G": 0, "A": 1, "B": 2, "C": 3, "g": 4}
D_CODE  = 4
DRIVEN  = ["G", "A"]
THETA   = math.pi / 4
NUM_TSLOTS = 200


# ---------------- operators ----------------
def _proj(i, j):
    m = np.zeros((DIM, DIM), dtype=complex); m[i, j] = 1.0
    return m


def control_ops():
    """2 tones x 2 quadratures. Convention matches the archive:
         H[k,g] = 0.5*Omega*exp(-i phi),  H[g,k] = 0.5*Omega*exp(+i phi)
       with Omega*exp(-i phi) = u_x + i u_y, giving
         Hx[k,g]=Hx[g,k]=0.5 ;  Hy[k,g]=+0.5i, Hy[g,k]=-0.5i.
    """
    ops, labels = [], []
    g = IDX["g"]
    for name in DRIVEN:
        k = IDX[name]
        Hx = 0.5 * (_proj(k, g) + _proj(g, k))
        Hy = 0.5j * _proj(k, g) - 0.5j * _proj(g, k)
        ops += [qt.Qobj(Hx), qt.Qobj(Hy)]
        labels += [f"u_{name}x", f"u_{name}y"]
    return ops, labels


def stark_diag(scale):
    """Frozen/instantaneous Stark drift, diagonal on undriven spectators.
    Archive guard: actively driven levels are skipped."""
    d = np.zeros((DIM, DIM), dtype=complex)
    assign = {"B": SHIFT_B_KHZ, "C": SHIFT_C_KHZ, "G": SHIFT_G_KHZ}
    for name, s_khz in assign.items():
        if name in DRIVEN:
            continue                                   # archive guard
        d[IDX[name], IDX[name]] = 2 * math.pi * s_khz * 1e-3 * scale   # rad/us
    return qt.Qobj(d)


def stark_scale(sum_om2, omega_ref):
    return sum_om2 / (omega_ref ** 2)


def U_QT_bare(theta=THETA):
    M = _proj(IDX["G"], IDX["A"]) + _proj(IDX["A"], IDX["G"])
    return (-1j * theta * qt.Qobj(M)).expm()


# ---------------- GRAPE ----------------
def run_grape(T_us, H_d, bound=BOUND_QUAD, seed=42):
    ops, labels = control_ops()
    U_0 = qt.qeye(DIM)
    U_targ = (-1j * H_d * T_us).expm() * U_QT_bare()     # commuting => exact
    np.random.seed(seed)
    t0 = time.time()
    res = optimize_pulse_unitary(
        H_d, ops, U_0, U_targ,
        num_tslots=NUM_TSLOTS, evo_time=float(T_us),
        amp_lbound=-float(bound), amp_ubound=+float(bound),
        fid_err_targ=1e-5, min_grad=1e-20, max_iter=20000,
        max_wall_time=300.0, init_pulse_type="RND",
        log_level=logging.WARN, gen_stats=False, alg="GRAPE",
    )
    wall = time.time() - t0
    u = np.asarray(res.final_amps, dtype=float)          # (num_tslots, n_ctrl)
    if u.shape[0] != NUM_TSLOTS:
        u = u.T
    return u, float(1.0 - res.fid_err), getattr(res, "termination_reason", None), wall, U_targ


# ---------------- propagation / audits ----------------
def propagate(u, T_us, omega_ref, true_stark: bool, H_d_frozen):
    """Slot-by-slot expm. true_stark=True uses the instantaneous
    sum|Omega_k(t)|^2; False uses the frozen worst-case drift."""
    ops, _ = control_ops()
    N = u.shape[0]; dt = T_us / N
    U = qt.qeye(DIM)
    for kslot in range(N):
        H = qt.Qobj(np.zeros((DIM, DIM), dtype=complex))
        for j, op in enumerate(ops):
            H = H + u[kslot, j] * op
        if true_stark:
            sum_om2 = sum(u[kslot, 2*t]**2 + u[kslot, 2*t+1]**2
                          for t in range(len(DRIVEN)))
            H = H + stark_diag(stark_scale(sum_om2, omega_ref))
        else:
            H = H + H_d_frozen
        U = (-1j * H * dt).expm() * U
    return U


def fit_virtual_z(U, U_bare):
    """Absorb free diagonal phases (virtual-Z / QPS, cost 0 us) on ALL levels,
    then report F vs the bare QT target. Closed form: the optimal diagonal
    phase per level maximises |Tr(D^dag U_bare^dag U)|, i.e. D = phase of the
    diagonal of (U_bare^dag U)."""
    M = (U_bare.dag() * U).full()
    phases = np.angle(np.diag(M))
    D = qt.Qobj(np.diag(np.exp(1j * phases)))
    U_fit = D.dag() * U
    tr = complex((U_bare.dag() * U_fit).tr())
    F = float(abs(tr) ** 2 / DIM ** 2)
    return F, phases


def aux_leakage(U):
    """max over code basis states of P(|g>) at t=T."""
    g = IDX["g"]
    return float(max(abs(U.full()[g, c])**2 for c in range(D_CODE)))


def block_leakage(U):
    """max leakage out of the {G,A} block, from |G> and |A> inputs."""
    out = []
    for c in (IDX["G"], IDX["A"]):
        col = U.full()[:, c]
        p_out = sum(abs(col[IDX[n]])**2 for n in ("B", "C", "g"))
        out.append(p_out)
    return float(max(out))


def F_code_block(U, U_bare):
    """Fidelity restricted to the 4-dim code block (project out |g>)."""
    idx = list(range(D_CODE))
    Us  = U.full()[np.ix_(idx, idx)]
    Ut  = U_bare.full()[np.ix_(idx, idx)]
    # absorb free diagonal phases on the code block too
    M = np.conj(Ut.T) @ Us
    D = np.diag(np.exp(1j * np.angle(np.diag(M))))
    tr = np.trace(np.conj(Ut.T) @ (np.conj(D.T) @ Us))
    return float(abs(tr) ** 2 / D_CODE ** 2)


# ---------------- sweep ----------------
def sweep(variant, omega_ref, T_grid):
    print(f"\n{'='*78}")
    print(f"VARIANT '{variant}'   Omega_ref/2pi = {omega_ref/(2*math.pi)*1e3:.2f} kHz")
    sum_om2_worst = len(DRIVEN) * OMEGA_MAX ** 2
    scale = stark_scale(sum_om2_worst, omega_ref)
    H_d = stark_diag(scale)
    print(f"  frozen Stark scale factor = {scale:.4f}")
    print(f"  frozen drift diag (rad/us) = "
          f"B {H_d.full()[IDX['B'],IDX['B']].real:.6f}, "
          f"C {H_d.full()[IDX['C'],IDX['C']].real:.6f}   "
          f"(bound/quad = {BOUND_QUAD:.6f})")
    print(f"{'='*78}")

    U_bare = U_QT_bare()
    rows = []
    for T_us in T_grid:
        u, F_grape, term, wall, U_targ = run_grape(T_us, H_d)
        U_frozen = propagate(u, T_us, omega_ref, False, H_d)
        U_true   = propagate(u, T_us, omega_ref, True,  H_d)

        # F vs the GRAPE target (Stark-included), independent reprop
        tr = complex((U_targ.dag() * U_frozen).tr())
        F_indep = float(abs(tr) ** 2 / DIM ** 2)

        F_vz_frozen, phases = fit_virtual_z(U_frozen, U_bare)
        F_vz_true,   _      = fit_virtual_z(U_true,   U_bare)
        spectator_dF = F_vz_true - F_vz_frozen

        leak_aux  = aux_leakage(U_frozen)
        leak_blk  = block_leakage(U_frozen)
        F_code    = F_code_block(U_frozen, U_bare)

        om = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        om2 = np.sqrt(u[:, 2]**2 + u[:, 3]**2)
        max_field = float(max(om.max(), om2.max()))

        ok = (F_grape >= 0.99999) and (leak_aux < 1e-3) and (abs(spectator_dF) < 1e-3)
        rows.append(dict(T_us=T_us, F_grape=F_grape, F_indep=F_indep,
                         F_vz_frozen=F_vz_frozen, F_vz_true=F_vz_true,
                         F_code=F_code, spectator_dF=spectator_dF,
                         leak_aux=leak_aux, leak_block=leak_blk,
                         max_field_rad_per_us=max_field,
                         max_field_pct_of_Omega_max=100*max_field/OMEGA_MAX,
                         termination=term, walltime_s=wall,
                         picker_pass=bool(ok),
                         vz_phases_rad=[float(p) for p in phases],
                         u=u.tolist()))
        print(f"  T={T_us:>6.2f}us  F_grape={F_grape:.6f}  F_indep={F_indep:.6f}  "
              f"leak_aux={leak_aux:.2e}  dF_spec={spectator_dF:+.2e}  "
              f"|Om|max={100*max_field/OMEGA_MAX:5.1f}%  {'PASS' if ok else '--'}  "
              f"({wall:.1f}s)", flush=True)
    return rows, H_d


def main():
    print("=" * 78)
    print("QT carrier gate -- qutip-qtrl GRAPE, innsbruck_meth_2025 anchor")
    print("=" * 78)
    print(f"[read] Omega_max/2pi = {OMEGA_MAX_2PI_KHZ} kHz  (source 'assumed', tag ASSUMED)")
    print(f"       Omega_max     = {OMEGA_MAX:.6f} rad/us")
    print(f"       bound/quad    = Omega_max/sqrt(2) = {BOUND_QUAD:.6f} rad/us")
    print(f"[read] Stark shifts (kHz): B(D_-5/2)={SHIFT_B_KHZ}  C(D_+1/2)={SHIFT_C_KHZ}  "
          f"G(D_+3/2)={SHIFT_G_KHZ} [G is DRIVEN -> skipped by archive guard]")
    print(f"       dim = {DIM} (code {D_CODE} + aux), driven tones = {DRIVEN}, "
          f"theta = pi/4, num_tslots = {NUM_TSLOTS}")

    T_grid = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 36.0, 45.0, 60.0]

    out = {}
    for variant, omega_ref in STARK_REFS.items():
        rows, H_d = sweep(variant, omega_ref, T_grid)
        passing = [r for r in rows if r["picker_pass"]]
        best = min(passing, key=lambda r: r["T_us"]) if passing else None
        out[variant] = dict(
            omega_ref_2pi_kHz=omega_ref/(2*math.pi)*1e3,
            frozen_scale=stark_scale(len(DRIVEN)*OMEGA_MAX**2, omega_ref),
            frozen_drift_B_rad_per_us=float(H_d.full()[IDX["B"], IDX["B"]].real),
            frozen_drift_C_rad_per_us=float(H_d.full()[IDX["C"], IDX["C"]].real),
            rows=[{k: v for k, v in r.items() if k != "u"} for r in rows],
            picked=None if best is None else {k: v for k, v in best.items() if k != "u"},
        )
        if best:
            print(f"  --> PICKED T = {best['T_us']} us   "
                  f"F_grape = {best['F_grape']:.6f}   "
                  f"leak_aux = {best['leak_aux']:.2e}   "
                  f"dF_spec = {best['spectator_dF']:+.2e}")
        else:
            print("  --> NO T satisfied the picker")

    outp = HERE / "step_qt_meth_results.json"
    outp.write_text(json.dumps(dict(
        setup=dict(
            treatment="(A) imposed: direct qutip-qtrl GRAPE, NOT via plugin registry "
                      "(HyPulse has no QT plugin)",
            model="(i) 5-dim: code {G,A,B,C} + S1/2 auxiliary |g>; tones via auxiliary",
            Omega_max_2pi_kHz=OMEGA_MAX_2PI_KHZ, Omega_max_tag="ASSUMED",
            bound_per_quadrature_rad_per_us=BOUND_QUAD,
            bound_convention="Omega_max/sqrt(2) per quadrature => total tone field <= Omega_max",
            num_tslots=NUM_TSLOTS, theta="pi/4",
            stark_shifts_kHz=dict(B=SHIFT_B_KHZ, C=SHIFT_C_KHZ, G_skipped=SHIFT_G_KHZ),
            picker="F_grape >= 0.99999 AND leak_aux < 1e-3 AND |spectator_dF| < 1e-3",
            commutation_note="[H_stark, H_drive] = 0 exactly (disjoint subspaces); "
                             "spectator Stark is a pure virtual-Z, QPS-correctable at 0 us",
        ),
        variants=out,
    ), indent=2))
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
