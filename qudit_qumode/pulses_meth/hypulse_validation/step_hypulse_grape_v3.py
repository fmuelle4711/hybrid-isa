"""HyPulse GRAPE — ETA_AND_BESSEL_CORRECTED amp bound (audit CHECK B fix).

Reads amp_bound_rad_per_us_eta_bessel_corrected from the innsbruck_meth_2025
calibration. That bound equals Bazavan's Omega_eff = 0.248 * Omega_sb =
0.00623 rad/us. NO post-hoc BESSEL_FACTOR is applied here — the calibration
already includes it. NO ramp masking.

Layout:
  PART 0 — analytic constant-pulse sanity check. Propagates constant
           u_p = amp_bound with the plugin's static H_ctrl. Reads |beta|
           per branch at multiple T points. Fits |beta|(T) = slope * T
           and prints slope. Compares slope to amp_bound and amp_bound/2.
           This settles the 80-vs-161-us prediction analytically.

  PART 1 — GRAPE T-sweep at alpha = 0.5. Coarse grid then refinement.

  PART 2 — write results, report shortest T with F >= 0.99.
"""
from __future__ import annotations
import math, json, time, tempfile
from pathlib import Path
import numpy as np
import qutip as qt

from hypulse.calibrations import get_device_spec
from hypulse.core import GateSpec, ModelSpec, compile_pulse


CUTOFF        = 8
NUM_TSLOTS    = 200
DELTA_RAD_PER_S = 0.0
GATE_NAME     = "QCD"
ALPHA         = 0.5


def build_static_H(gate_name):
    a = qt.destroy(CUTOFF); adg = a.dag()
    X = (a + adg) / 2.0; P = (a - adg) / (2.0j)
    sig = qt.sigmax() if gate_name == "CD" else qt.sigmaz()
    return qt.tensor(sig, X), qt.tensor(sig, P)


def target_U(gate_name, alpha):
    Dp = qt.displace(CUTOFF, +alpha); Dm = qt.displace(CUTOFF, -alpha)
    if gate_name == "CD":
        plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        mnus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
        Pp = plus * plus.dag(); Pm = mnus * mnus.dag()
        return qt.tensor(Pp, Dp) + qt.tensor(Pm, Dm)
    else:
        P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        return qt.tensor(P0, Dp) + qt.tensor(P1, Dm)


def eval_F(gate_name, alpha, T_us, u_x, u_p):
    Hcx, Hcp = build_static_H(gate_name)
    Utarg = target_U(gate_name, alpha)
    N = len(u_x); dt_us = T_us / N
    U = qt.tensor(qt.qeye(2), qt.qeye(CUTOFF))
    for k in range(N):
        H = (u_x[k] * 1e6) * Hcx + (u_p[k] * 1e6) * Hcp
        Uk = (-1j * H * (dt_us * 1e-6)).expm()
        U = Uk * U
    dim = 2 * CUTOFF
    tr = (Utarg.dag() * U).tr()
    F = float(abs(tr) ** 2 / dim ** 2)
    return F


def propagate_constant(gate_name, T_us, u_x_const, u_p_const, N=400):
    """Propagate constant u_x, u_p (rad/us) for T_us. Return U."""
    Hcx, Hcp = build_static_H(gate_name)
    dt_us = T_us / N
    U = qt.tensor(qt.qeye(2), qt.qeye(CUTOFF))
    H = (u_x_const * 1e6) * Hcx + (u_p_const * 1e6) * Hcp
    Uk = (-1j * H * (dt_us * 1e-6)).expm()
    for _ in range(N):
        U = Uk * U
    return U


def displacement_of_branch(U, branch=0):
    """Extract |beta| for the |branch> qubit sector.
    Apply U to |branch>|0> and read <n>, then |beta| = sqrt(<n>)."""
    q = qt.basis(2, branch)
    vac = qt.basis(CUTOFF, 0)
    psi = qt.tensor(q, vac)
    out = U * psi
    a = qt.destroy(CUTOFF); n_op = qt.tensor(qt.qeye(2), a.dag() * a)
    nbar = float(qt.expect(n_op, out))
    return math.sqrt(max(0.0, nbar))


def part0_analytic_sanity(amp_bound):
    print("=" * 72)
    print("PART 0 -- analytic constant-pulse sanity check")
    print("=" * 72)
    print(f"amp_bound in use = {amp_bound:.6f} rad/us")
    print(f"applying constant u_p = amp_bound, u_x = 0")
    print()
    T_points = [10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 200.0]
    print(f"{'T (us)':>10}  {'|beta| |0>':>14}  {'|beta| |1>':>14}  "
          f"{'slope = beta/T':>16}")
    slopes = []
    for T in T_points:
        U = propagate_constant(GATE_NAME, T, u_x_const=0.0, u_p_const=amp_bound)
        b0 = displacement_of_branch(U, branch=0)
        b1 = displacement_of_branch(U, branch=1)
        slope = b0 / T if T > 0 else float("nan")
        slopes.append(slope)
        print(f"  {T:>7.1f}  {b0:>14.6f}  {b1:>14.6f}  {slope:>16.6f}")
    print()
    # slope_predicted candidates:
    slope_predicted_u = amp_bound              # if beta_dot = u
    slope_predicted_u_over_2 = amp_bound / 2   # if beta_dot = u/2  (plugin X=(a+adg)/2 -> this)
    slope_measured = float(np.median(slopes[:3]))    # use early points (before saturation)
    print(f"predicted beta/T if beta_dot = u        : {slope_predicted_u:.6f} rad/us  -> T for alpha=0.5 : {0.5/slope_predicted_u:.2f} us")
    print(f"predicted beta/T if beta_dot = u/2      : {slope_predicted_u_over_2:.6f} rad/us  -> T for alpha=0.5 : {0.5/slope_predicted_u_over_2:.2f} us")
    print(f"MEASURED beta/T (median of first 3 pts) : {slope_measured:.6f} rad/us  -> T for alpha=0.5 : {0.5/slope_measured:.2f} us")
    print()
    # analytic minimum T for GRAPE:
    T_analytic_min = 0.5 / slope_measured
    print(f"ANALYTIC MINIMUM T for alpha=0.5 at this amp bound = {T_analytic_min:.2f} us")
    print()
    return T_analytic_min


def compile_one(gate_name, alpha, T_us, device, amp_bound_effective):
    gate = GateSpec(
        name=gate_name,
        params={
            "alpha_re": float(alpha), "alpha_im": 0.0,
            "Delta_rad_per_s": float(DELTA_RAD_PER_S),
            "duration_us": float(T_us),
            "num_tslots": int(NUM_TSLOTS),
            "amp_bound": float(amp_bound_effective),
        },
        targets=(0, 0), meta={},
    )
    model = ModelSpec(cutoff=CUTOFF, options={})
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        pulse = compile_pulse(gate, device, model,
                                strategy="synthesize", library_root=tmp)
        walltime = time.time() - t0
    ch = pulse.channels or {}
    opt_meta = pulse.meta.get("optimizer", {}).get("result", {}) if isinstance(pulse.meta, dict) else {}
    return dict(
        Omega_x=list(ch.get("Omega_x", [])),
        Omega_p=list(ch.get("Omega_p", [])),
        F_grape=opt_meta.get("fidelity"),
        fid_err_grape=opt_meta.get("fid_err"),
        termination_grape=opt_meta.get("termination_reason"),
        walltime_s=walltime,
    )


def part1_grape_sweep(device, amp_bound_effective, T_analytic_min):
    print("=" * 72)
    print("PART 1 -- GRAPE T-sweep at alpha = 0.5")
    print("=" * 72)

    # Grid spans a wide range around T_analytic_min so we can see F drop and rise.
    # We want the shortest T with F >= 0.99.
    T_grid = sorted(set([
        20.0, 40.0, 60.0,
        round(0.5 * T_analytic_min, 1), round(0.75 * T_analytic_min, 1),
        round(0.9 * T_analytic_min, 1), round(T_analytic_min, 1),
        round(1.1 * T_analytic_min, 1), round(1.25 * T_analytic_min, 1),
        round(1.5 * T_analytic_min, 1), round(2.0 * T_analytic_min, 1),
        320.0,
    ]))
    T_grid = [t for t in T_grid if 5.0 <= t <= 400.0]
    print(f"T grid (us): {T_grid}\n")

    rows = []
    for T_us in T_grid:
        try:
            r = compile_one(GATE_NAME, ALPHA, T_us, device, amp_bound_effective)
            u_x = np.asarray(r["Omega_x"]); u_p = np.asarray(r["Omega_p"])
            max_ux = float(np.abs(u_x).max()) if u_x.size else 0.0
            max_up = float(np.abs(u_p).max()) if u_p.size else 0.0
            F = r["F_grape"] if r["F_grape"] is not None else eval_F(GATE_NAME, ALPHA, T_us, u_x, u_p)
            row = dict(
                gate=GATE_NAME, alpha=ALPHA, T_us=T_us,
                F_grape=F,
                fid_err_grape=r["fid_err_grape"],
                termination=r["termination_grape"],
                walltime_s=r["walltime_s"],
                max_ux=max_ux, max_up=max_up,
                Omega_x=list(u_x), Omega_p=list(u_p),
            )
            rows.append(row)
            print(f"  T={T_us:>7.2f}us  F_grape={F:.6f}  "
                  f"max|u_x|={max_ux:.4f}  max|u_p|={max_up:.4f}  "
                  f"term='{r['termination_grape']}'  "
                  f"({r['walltime_s']:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"  T={T_us:>7.2f}us  FAILED: {e}", flush=True)
            rows.append(dict(gate=GATE_NAME, alpha=ALPHA, T_us=T_us, error=str(e)))
    return rows


def main():
    out_dir = Path(__file__).resolve().parent

    device = get_device_spec("innsbruck_meth_2025")
    amp_bound_eff = float(device.params["amp_bound_rad_per_us_eta_bessel_corrected"])
    calib_carrier_bound = float(device.params["amp_bound_rad_per_us"])

    print("=" * 72)
    print("HyPulse GRAPE v3 -- ETA_AND_BESSEL_CORRECTED bound")
    print("=" * 72)
    print(f"device                = 'innsbruck_meth_2025'")
    print(f"cutoff                = {CUTOFF}")
    print(f"num_tslots            = {NUM_TSLOTS}")
    print(f"carrier amp bound     = {calib_carrier_bound:.6f} rad/us   (uncorrected reference)")
    print(f"ETA_AND_BESSEL amp bound = {amp_bound_eff:.6f} rad/us   (this run)")
    print(f"ratio (uncorrected / corrected) = {calib_carrier_bound/amp_bound_eff:.2f}x")
    print()

    T_analytic_min = part0_analytic_sanity(amp_bound_eff)
    rows = part1_grape_sweep(device, amp_bound_eff, T_analytic_min)

    out = {
        "device_id": "innsbruck_meth_2025",
        "amp_bound_rad_per_us": amp_bound_eff,
        "amp_bound_tag": "ETA_AND_BESSEL_CORRECTED",
        "cutoff": CUTOFF,
        "num_tslots": NUM_TSLOTS,
        "T_analytic_min_us": T_analytic_min,
        "rows": rows,
    }
    outp = out_dir / "step_hypulse_grape_v3_results.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {outp}")

    print()
    print("=" * 72)
    print("SHORTEST T with F_grape >= 0.99")
    print("=" * 72)
    passing = [r for r in rows if r.get("F_grape") is not None and r["F_grape"] >= 0.99]
    if passing:
        best = min(passing, key=lambda r: r["T_us"])
        print(f"  shortest T = {best['T_us']:.2f} us    F_grape = {best['F_grape']:.6f}")
        print(f"  max|u_x|   = {best['max_ux']:.4f} rad/us  ({100*best['max_ux']/amp_bound_eff:.1f}% of bound)")
        print(f"  max|u_p|   = {best['max_up']:.4f} rad/us  ({100*best['max_up']/amp_bound_eff:.1f}% of bound)")
    else:
        best_F = max((r.get("F_grape") or 0 for r in rows), default=0)
        print(f"  NO T reached F_grape >= 0.99.  best F = {best_F:.4f}")

    # SANITY GATE
    print()
    print("=" * 72)
    print("SANITY GATE")
    print("=" * 72)
    if passing:
        best = min(passing, key=lambda r: r["T_us"])
        T_paper = best["T_us"]
        ratio_to_analytic = T_paper / T_analytic_min
        print(f"T_paper / T_analytic_min = {ratio_to_analytic:.3f}")
        if 0.9 <= ratio_to_analytic <= 1.3:
            print("  PASS: within optimizer overhead of analytic minimum.")
        elif ratio_to_analytic < 0.9:
            print("  WARN: T_paper < 0.9 * analytic minimum. GRAPE beat the analytic bound;")
            print("        suggests amp_bound is not really binding or the convention is off.")
        else:
            print("  WARN: T_paper > 1.3 * analytic minimum. Optimizer overhead is large;")
            print("        double-check the convention is not being double-applied.")


if __name__ == "__main__":
    main()
