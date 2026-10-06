"""v2: same imposed CHI_CORRECTED treatment, two fixes from v1:

  (i)  Use init_pulse_type="LIN" (constant-linear ramp) to break the sign
       ambiguity of the sigma_z generator.  Random init let GRAPE settle
       into the wrong-sign minimum for CR at Meth theta=0.5.

  (ii) Pick shortest T with (F >= 0.99999 AND |int(u dt) - theta| < 5%).
       At small theta, the identity solution has F ~ 1 too, so the F bar
       alone is not distinguishing.  Requiring the integrated pulse area
       to match theta rules out identity-like local minima.
"""
from __future__ import annotations
import math, json, time
from pathlib import Path
import numpy as np
import qutip as qt

from step_qcr_cr_two_points import (
    CUTOFF, NUM_TSLOTS, ETA, T_PI_US, OPERATING_POINTS,
    compute_chi_rad_per_us, build_generator, build_target,
)


def compile_grape(gate_name, theta, chi_rad_per_us, T_us,
                    init_pulse_type="LIN"):
    from qutip.control.pulseoptim import optimize_pulse_unitary
    import logging

    H_c = build_generator(gate_name)
    U_targ = build_target(gate_name, theta)
    H_d = 0 * H_c
    U_0 = qt.tensor(qt.qeye(2), qt.qeye(CUTOFF))

    t0 = time.time()
    result = optimize_pulse_unitary(
        H_d, [H_c], U_0, U_targ,
        num_tslots=NUM_TSLOTS,
        evo_time=float(T_us),
        amp_lbound=-float(chi_rad_per_us),
        amp_ubound=+float(chi_rad_per_us),
        fid_err_targ=1e-5,
        min_grad=1e-20,
        max_iter=20000,
        max_wall_time=300.0,
        init_pulse_type=str(init_pulse_type),
        log_level=logging.WARN,
        gen_stats=False,
        alg="GRAPE",
    )
    walltime = time.time() - t0

    u = np.asarray(result.final_amps, dtype=float)
    if u.ndim == 2 and u.shape[1] == 1:  u = u[:, 0]
    elif u.ndim == 2 and u.shape[0] == 1: u = u[0]
    u = u.reshape(-1)

    fid_err = float(getattr(result, "fid_err", np.nan))
    return dict(
        u=u.tolist(),
        F_grape=float(1.0 - fid_err),
        fid_err_grape=fid_err,
        termination=getattr(result, "termination_reason", None),
        walltime_s=walltime,
    )


def sweep_and_pick(gate_name, theta, chi_rad_per_us):
    """Sweep T, pick shortest T with F>=0.99999 AND |int(u)-theta|/theta < 5%."""
    T_min = theta / chi_rad_per_us
    T_grid = sorted(set([
        round(0.9 * T_min, 3),
        round(0.95 * T_min, 3),
        round(T_min, 3),
        round(1.02 * T_min, 3),
        round(1.05 * T_min, 3),
        round(1.10 * T_min, 3),
        round(1.20 * T_min, 3),
        round(1.35 * T_min, 3),
        round(1.55 * T_min, 3),
    ]))
    rows = []
    for T in T_grid:
        r = compile_grape(gate_name, theta, chi_rad_per_us, T, "LIN")
        u_arr = np.asarray(r["u"])
        int_u = float(u_arr.sum() * T / len(u_arr))
        row = dict(
            gate=gate_name, theta=theta, T_us=T,
            F_grape=r["F_grape"],
            max_u=float(np.abs(u_arr).max()),
            int_u=int_u,
            int_u_err=abs(int_u - theta) / theta,
            termination=r["termination"],
            walltime_s=r["walltime_s"],
            u=r["u"],
        )
        rows.append(row)
        print(f"    T={T:>9.3f}us  F={r['F_grape']:.6f}  max|u|={row['max_u']:.4e}  "
              f"int_u={int_u:+.5f} (target {theta:+.4f}, err {100*row['int_u_err']:.1f}%)  "
              f"term='{r['termination']}'  ({r['walltime_s']:.1f}s)",
              flush=True)
    return rows


def pick_paper_T(rows, theta, F_bar=0.99999, int_tol=0.05):
    """Shortest T with F>=F_bar AND ∫u matches theta within int_tol."""
    passing = [r for r in rows
                if r["F_grape"] >= F_bar and r["int_u_err"] < int_tol]
    if not passing:
        return None
    return min(passing, key=lambda r: r["T_us"])


def audit_check_A(row, chi_rad_per_us):
    max_u = row["max_u"]
    return max_u > 0.9 * chi_rad_per_us, max_u, 100 * max_u / chi_rad_per_us


def audit_check_C(row, gate_name, theta):
    T_us = row["T_us"]
    u = np.asarray(row["u"])
    N = len(u); dt = T_us / N

    H_c = build_generator(gate_name)
    U_targ = build_target(gate_name, theta)
    U = qt.tensor(qt.qeye(2), qt.qeye(CUTOFF))
    for k in range(N):
        H = u[k] * H_c
        U = (-1j * H * dt).expm() * U

    dim = 2 * CUTOFF
    tr = (U_targ.dag() * U).tr()
    F_indep = float(abs(tr) ** 2 / dim ** 2)
    int_u = float(u.sum() * dt)
    return dict(F_indep=F_indep, F_grape=row["F_grape"],
                F_match=(abs(F_indep - row["F_grape"]) < 1e-3),
                int_u=int_u, int_u_target=theta,
                int_u_match=(abs(int_u - theta) / theta < 0.05))


def main():
    out_dir = Path(__file__).resolve().parent
    print("=" * 78)
    print("QCR + CR   v2   (LIN init;  paper T requires F AND int_u match)")
    print("=" * 78)

    summary_rows = []
    all_rows = []

    for pname, point in OPERATING_POINTS.items():
        chi_rad_per_s, chi_rad_per_us = compute_chi_rad_per_us(point)
        print()
        print("-" * 78)
        print(f"OPERATING POINT {pname}  ({point['tag']})")
        print("-" * 78)
        print(f"  Omega_c/2pi     = {point['Omega_c_2pi_kHz']:.2f} kHz")
        print(f"  Omega_sb/2pi    = {point['Omega_sb_2pi_kHz']:.2f} kHz")
        print(f"  delta_disp/2pi  = {point['delta_disp_2pi_kHz']:.1f} kHz")
        print(f"  chi/2pi         = {chi_rad_per_s / (2 * math.pi):.2f} Hz")
        print(f"  chi (rad/us)    = {chi_rad_per_us:.5e}")

        for gate_name in ["CR", "QCR"]:
            for theta in [0.016, 0.5]:
                T_min = theta / chi_rad_per_us
                print()
                print(f"  === {gate_name}   theta = {theta}   T_analytic = {T_min:.3f} us ===")
                rows = sweep_and_pick(gate_name, theta, chi_rad_per_us)
                for r in rows:
                    r.update(dict(point=pname, chi_rad_per_us=chi_rad_per_us,
                                    T_analytic_us=T_min))
                all_rows.extend(rows)

                best = pick_paper_T(rows, theta)
                if best is None:
                    print(f"    NO T passed (F>=0.99999 AND int_u within 5% of theta)")
                    # relax int_u to 15%
                    relaxed = [r for r in rows
                                if r["F_grape"] >= 0.99999 and r["int_u_err"] < 0.15]
                    if relaxed:
                        best = min(relaxed, key=lambda r: r["T_us"])
                        print(f"    relaxed to 15% int_u tol -> T = {best['T_us']:.2f}")

                if best is None:
                    summary_rows.append(dict(
                        point=pname, tag=point["tag"], gate=gate_name, theta=theta,
                        chi_2pi_Hz=chi_rad_per_s / (2 * math.pi),
                        T_analytic_us=T_min, T_grape_us=None, F_grape=None,
                        T_paper_us=None,
                        check_A_binds=None, check_A_max_u_pct=None,
                        check_C_F_indep=None, check_C_F_match=None,
                        check_C_int_u=None, check_C_int_u_match=None,
                        note="GRAPE did not converge with F+int_u both matched",
                    ))
                    continue

                bindA, maxu, pct = audit_check_A(best, chi_rad_per_us)
                checkC = audit_check_C(best, gate_name, theta)
                T_paper = 2 * best["T_us"] + T_PI_US

                summary_rows.append(dict(
                    point=pname, tag=point["tag"], gate=gate_name, theta=theta,
                    chi_2pi_Hz=chi_rad_per_s / (2 * math.pi),
                    chi_rad_per_us=chi_rad_per_us,
                    T_analytic_us=T_min,
                    T_grape_us=best["T_us"],
                    F_grape=best["F_grape"],
                    T_paper_us=T_paper,
                    check_A_binds=bindA, check_A_max_u_pct=pct,
                    check_C_F_indep=checkC["F_indep"],
                    check_C_F_match=checkC["F_match"],
                    check_C_int_u=checkC["int_u"],
                    check_C_int_u_target=checkC["int_u_target"],
                    check_C_int_u_match=checkC["int_u_match"],
                ))
                print(f"    -> paper T_grape = {best['T_us']:.2f} us   T_paper (2T+t_pi) = {T_paper:.2f} us")
                print(f"    CHECK A: max|u| = {maxu:.4e}  ({pct:.1f}% of bound)   "
                      f"{'PASS' if bindA else 'FAIL'}")
                print(f"    CHECK C: F_indep = {checkC['F_indep']:.6f}  vs F_grape = {best['F_grape']:.6f}   "
                      f"match {'PASS' if checkC['F_match'] else 'FAIL'}")
                print(f"    CHECK C: int u dt = {checkC['int_u']:+.5f} rad  (target {theta})   "
                      f"{'PASS' if checkC['int_u_match'] else 'FAIL'}")

    print()
    print("=" * 78)
    print("SUMMARY  (T_paper = 2 * T_grape + t_pi;  t_pi = 5.0 us  ASSUMED)")
    print("=" * 78)
    hdr = f"{'point':<10}  {'gate':<4}  {'theta':>6}  {'chi/2pi Hz':>11}  " \
          f"{'T_analytic':>11}  {'T_grape':>9}  {'T_paper':>9}  {'F':>9}  {'A':>4}  {'C':>4}"
    print(hdr)
    for s in summary_rows:
        A_tag = ("PASS" if s["check_A_binds"] else "----") if s["check_A_binds"] is not None else "----"
        C_tag = ("PASS" if (s.get("check_C_F_match") and s.get("check_C_int_u_match")) else "FAIL") \
                if s.get("check_C_F_match") is not None else "----"
        T_g = f"{s['T_grape_us']:>9.2f}" if s['T_grape_us'] is not None else "     ----"
        T_p = f"{s['T_paper_us']:>9.2f}" if s['T_paper_us'] is not None else "     ----"
        F   = f"{s['F_grape']:>9.6f}" if s['F_grape'] is not None else "     ----"
        print(f"{s['point']:<10}  {s['gate']:<4}  {s['theta']:>6.3f}  "
              f"{s['chi_2pi_Hz']:>11.2f}  {s['T_analytic_us']:>11.2f}  "
              f"{T_g}  {T_p}  {F}  {A_tag:>4}  {C_tag:>4}")

    outp = out_dir / "step_qcr_cr_two_points_v2_results.json"
    with open(outp, "w") as f:
        json.dump({"summary": summary_rows, "all_rows": all_rows}, f, indent=2, default=str)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
