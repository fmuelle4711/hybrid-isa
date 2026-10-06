"""QCR + CR via HyPulse GRAPE (imposed sigma_z (x) n_hat plugin, CHI_CORRECTED bound).

Mirrors the QCD discipline (FINDING 3 style):
  - The user-facing CR plugin uses raw BSB (sigma_x (x) X + sigma_y (x) Q).
    If GRAPE is given carrier-Rabi amp bounds it will find non-dispersive
    fast solutions that beat the analytic dispersive floor T = theta/chi.
    This is the CR analog of QCD's amp-bound-too-optimistic bug.

  - Here we IMPOSE the dispersive target operator directly:
        CR :  H_c = sigma_z (x) n_hat            u_max = chi
        QCR:  H_c = Pi_l    (x) n_hat            u_max = chi
    where Pi_l = |l><l| = (I - sigma_z)/2 for encoding |l> = |1>.
    Amp bound = chi (Kerr rate) forces the physical dispersive floor.

  - QCR ~= CR in duration:  Pi_l (x) n = (1/2) I (x) n + (1/2) sigma_z (x) n.
    The identity half is a mode rotation, absorbed elsewhere in the ISA
    (mode Rotation is a native, unconditional gate).  The sigma_z half runs
    at rate chi/2 within the sigma_z-only component, but the Pi_l control
    itself runs at rate chi so T_QCR = phi/chi = T_CR at matched angle.

Two operating points:
  A. Meth   Omega_c/2pi = 97.56 kHz (from Omega_sb = 4 kHz),  delta_disp/2pi = 40 kHz
     chi = (eta * Omega_c)^2 / (4 * delta_disp)
     chi/2pi = 100.0 Hz          chi = 6.283e-4 rad/us
     tag: METH_OPERATING_POINT
  B. Choi   chi/2pi = 471 Hz (MEASURED, arXiv:2601.04591; overrides the
                              formula's 84 Hz due to off-resonant multi-level
                              Rabi contributions Choi calibrated)
     Omega_c/2pi = 100 kHz, delta_disp/2pi = 50 kHz (Choi's setup)
     chi = 2.960e-3 rad/us
     tag: CHOI_MEASURED

Two theta values per gate:
  theta = 0.016 (Meth's actual QCR angle; anchors the paper number)
  theta = 0.5   (stress test)

AC-Stark cancellation:
  Physical implementation uses 2 pulse segments + one microwave pi pulse
  in between (Choi scheme). T_paper = 2 * T_grape + t_pi.
  t_pi = 5 us  ASSUMED (typical 171Yb+ hyperfine pi; not quoted in Choi or Meth).

Audit checks per waveform:
  A. amp bound BINDS (max|u| = chi within 90%)
  C. independent qutip re-propagation matches HyPulse F to 1e-3
     integrated pulse area = theta (within a few %).
"""
from __future__ import annotations
import math, json, time
from pathlib import Path
import numpy as np
import qutip as qt


CUTOFF        = 8
NUM_TSLOTS    = 200
ETA           = 0.041
T_PI_US       = 5.0                              # ASSUMED microwave pi


OPERATING_POINTS = {
    "A_meth": {
        "tag": "METH_OPERATING_POINT",
        "Omega_c_2pi_kHz": 97.56,
        "Omega_sb_2pi_kHz": 4.0,
        "delta_disp_2pi_kHz": 40.0,
        "chi_2pi_Hz": None,        # derived from formula
    },
    "B_choi": {
        "tag": "CHOI_MEASURED",
        "Omega_c_2pi_kHz": 100.0,
        "Omega_sb_2pi_kHz": 4.1,   # eta*Omega_c
        "delta_disp_2pi_kHz": 50.0,
        "chi_2pi_Hz": 471.0,       # MEASURED by Choi
    },
}


def compute_chi_rad_per_us(point):
    """chi = (eta * Omega_c)^2 / (4 * delta_disp) in rad/s, converted to rad/us."""
    if point.get("chi_2pi_Hz") is not None:
        chi_rad_per_s = 2 * math.pi * point["chi_2pi_Hz"]
    else:
        Omega_sb_rad_per_s = 2 * math.pi * point["Omega_sb_2pi_kHz"] * 1e3
        delta_disp_rad_per_s = 2 * math.pi * point["delta_disp_2pi_kHz"] * 1e3
        chi_rad_per_s = Omega_sb_rad_per_s ** 2 / (4 * delta_disp_rad_per_s)
    return chi_rad_per_s, chi_rad_per_s / 1e6         # rad/s, rad/us


def build_generator(gate_name, cutoff=CUTOFF):
    """Return the IMPOSED single control operator: sigma_z (x) n_hat  or  Pi_l (x) n_hat."""
    a = qt.destroy(cutoff); n_op = a.dag() * a
    I2 = qt.qeye(2)
    if gate_name == "CR":
        # H = u * sigma_z (x) n
        return qt.tensor(qt.sigmaz(), n_op)
    elif gate_name == "QCR":
        # Pi_l = (I - sigma_z)/2   with |l> = |1>
        P_l = (I2 - qt.sigmaz()) / 2
        return qt.tensor(P_l, n_op)
    else:
        raise ValueError(gate_name)


def build_target(gate_name, theta, cutoff=CUTOFF):
    """CR target: exp(-i theta sigma_z (x) n).  QCR target: P_g (x) I + P_l (x) exp(-i theta n)."""
    a = qt.destroy(cutoff); n_op = a.dag() * a
    I2 = qt.qeye(2); Im = qt.qeye(cutoff)
    if gate_name == "CR":
        return (-1j * theta * qt.tensor(qt.sigmaz(), n_op)).expm()
    elif gate_name == "QCR":
        P_g = (I2 + qt.sigmaz()) / 2                 # |g> = |0>
        P_l = (I2 - qt.sigmaz()) / 2                 # |l> = |1>
        U = qt.tensor(P_g, Im) + qt.tensor(P_l, (-1j * theta * n_op).expm())
        return U
    else:
        raise ValueError(gate_name)


def compile_grape(gate_name, theta, chi_rad_per_us, T_us):
    """Run qutip-qtrl GRAPE (analogous to how HyPulse calls it internally)."""
    from qutip.control.pulseoptim import optimize_pulse_unitary
    import logging

    H_c = build_generator(gate_name)
    U_targ = build_target(gate_name, theta)
    H_d = 0 * H_c                                    # no drift for imposed model
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
        init_pulse_type="RND",
        log_level=logging.WARN,
        gen_stats=False,
        alg="GRAPE",
    )
    walltime = time.time() - t0

    u = np.asarray(result.final_amps, dtype=float)
    if u.ndim == 2 and u.shape[1] == 1:
        u = u[:, 0]
    elif u.ndim == 2 and u.shape[0] == 1:
        u = u[0]
    u = u.reshape(-1)

    fid_err = float(getattr(result, "fid_err", np.nan))
    return dict(
        u=list(u),
        F_grape=float(1.0 - fid_err),
        fid_err_grape=fid_err,
        termination=getattr(result, "termination_reason", None),
        walltime_s=walltime,
    )


def sweep_T_for_convergence(gate_name, theta, chi_rad_per_us):
    """Sweep T around the analytic minimum theta/chi. Report shortest with F >= 0.99999."""
    T_min = theta / chi_rad_per_us
    T_grid = sorted(set([
        round(0.6 * T_min, 2),
        round(0.85 * T_min, 2),
        round(0.95 * T_min, 2),
        round(T_min, 2),
        round(1.05 * T_min, 2),
        round(1.2 * T_min, 2),
        round(1.5 * T_min, 2),
    ]))
    rows = []
    for T in T_grid:
        r = compile_grape(gate_name, theta, chi_rad_per_us, T)
        u_arr = np.asarray(r["u"])
        row = dict(
            gate=gate_name, theta=theta, T_us=T,
            F_grape=r["F_grape"],
            max_u=float(np.abs(u_arr).max()),
            mean_u=float(u_arr.mean()),
            int_u=float(u_arr.sum() * T / len(u_arr)),
            termination=r["termination"],
            walltime_s=r["walltime_s"],
            u=r["u"],
        )
        rows.append(row)
        print(f"    T={T:>7.2f}us  F={r['F_grape']:.6f}  max|u|={row['max_u']:.4e}  "
              f"int_u={row['int_u']:+.5f}  term='{r['termination']}'  ({r['walltime_s']:.1f}s)",
              flush=True)
    return rows


def shortest_T(rows, F_bar=0.99999):
    ok = [r for r in rows if r["F_grape"] >= F_bar]
    if not ok:
        return None
    return min(ok, key=lambda r: r["T_us"])


def audit_check_A(row, chi_rad_per_us):
    """max |u| within 90% of the amp bound = chi ?"""
    max_u = row["max_u"]
    return max_u > 0.9 * chi_rad_per_us, max_u, 100 * max_u / chi_rad_per_us


def audit_check_C(row, gate_name, theta, chi_rad_per_us):
    """Independent qutip re-propagation. Report F and integrated pulse area = theta."""
    T_us = row["T_us"]
    u = np.asarray(row["u"])
    N = len(u); dt = T_us / N

    H_c = build_generator(gate_name)
    U_targ = build_target(gate_name, theta)
    U = qt.tensor(qt.qeye(2), qt.qeye(CUTOFF))
    for k in range(N):
        H = u[k] * H_c
        Uk = (-1j * H * dt).expm()
        U = Uk * U

    dim = 2 * CUTOFF
    tr = (U_targ.dag() * U).tr()
    F_indep = float(abs(tr) ** 2 / dim ** 2)
    int_u = float(u.sum() * dt)

    return dict(
        F_indep=F_indep, F_grape=row["F_grape"],
        F_match=(abs(F_indep - row["F_grape"]) < 1e-3),
        int_u=int_u, int_u_target=theta,
        int_u_match=(abs(int_u - theta) / theta < 0.05),
    )


def main():
    out_dir = Path(__file__).resolve().parent
    print("=" * 78)
    print("QCR + CR   HyPulse GRAPE (imposed, CHI_CORRECTED)   two operating points")
    print("=" * 78)

    all_rows = []
    summary_rows = []

    for pname, point in OPERATING_POINTS.items():
        chi_rad_per_s, chi_rad_per_us = compute_chi_rad_per_us(point)
        print()
        print("-" * 78)
        print(f"OPERATING POINT {pname}  ({point['tag']})")
        print("-" * 78)
        print(f"  Omega_c/2pi     = {point['Omega_c_2pi_kHz']:.2f} kHz")
        print(f"  Omega_sb/2pi    = {point['Omega_sb_2pi_kHz']:.2f} kHz")
        print(f"  delta_disp/2pi  = {point['delta_disp_2pi_kHz']:.1f} kHz")
        print(f"  chi/2pi         = {chi_rad_per_s / (2 * math.pi):.2f} Hz "
              f"({'MEASURED' if point.get('chi_2pi_Hz') else 'FORMULA'})")
        print(f"  chi (rad/us)    = {chi_rad_per_us:.5e}")
        print()

        for gate_name in ["CR", "QCR"]:
            for theta in [0.016, 0.5]:
                T_min_analytic = theta / chi_rad_per_us
                print(f"  === {gate_name}   theta = {theta}   T_analytic = {T_min_analytic:.2f} us ===")
                rows = sweep_T_for_convergence(gate_name, theta, chi_rad_per_us)
                for r in rows:
                    r["point"] = pname; r["gate"] = gate_name; r["theta"] = theta
                    r["chi_rad_per_us"] = chi_rad_per_us
                    r["T_analytic_us"] = T_min_analytic
                all_rows.extend(rows)

                best = shortest_T(rows, F_bar=0.99999) or shortest_T(rows, F_bar=0.999)
                if best is None:
                    print(f"    NO T reached F >= 0.999")
                    continue

                # audit
                bindA, maxu, pct = audit_check_A(best, chi_rad_per_us)
                checkC = audit_check_C(best, gate_name, theta, chi_rad_per_us)

                T_paper = 2 * best["T_us"] + T_PI_US
                summary_rows.append(dict(
                    point=pname, tag=point["tag"], gate=gate_name, theta=theta,
                    chi_2pi_Hz=chi_rad_per_s / (2 * math.pi),
                    chi_rad_per_us=chi_rad_per_us,
                    T_analytic_us=T_min_analytic,
                    T_grape_us=best["T_us"],
                    F_grape=best["F_grape"],
                    T_paper_us=T_paper,      # 2*T + t_pi (AC-Stark cancellation)
                    check_A_binds=bindA,
                    check_A_max_u_pct=pct,
                    check_C_F_indep=checkC["F_indep"],
                    check_C_F_match=checkC["F_match"],
                    check_C_int_u=checkC["int_u"],
                    check_C_int_u_target=checkC["int_u_target"],
                    check_C_int_u_match=checkC["int_u_match"],
                ))
                print(f"    -> T_grape = {best['T_us']:.2f} us    "
                      f"T_paper (2T + t_pi) = {T_paper:.2f} us")
                print(f"    CHECK A: max|u| = {maxu:.4e} = {pct:.1f}% of bound   "
                      f"{'PASS' if bindA else 'WARN (not binding)'}")
                print(f"    CHECK C: F_indep = {checkC['F_indep']:.6f}  vs F_grape = {best['F_grape']:.6f}   "
                      f"match {'PASS' if checkC['F_match'] else 'FAIL'}")
                print(f"    CHECK C: int u dt = {checkC['int_u']:+.5f} rad  (target = theta = {theta})   "
                      f"{'PASS' if checkC['int_u_match'] else 'FAIL'}")
                print()

    # ----- final table -----
    print("=" * 78)
    print("SUMMARY  (T_paper = 2 * T_grape + t_pi;  t_pi = 5.0 us  ASSUMED)")
    print("=" * 78)
    print(f"{'point':<10}  {'gate':<4}  {'theta':>6}  {'chi/2pi (Hz)':>13}  "
          f"{'T_analytic':>11}  {'T_grape':>9}  {'T_paper':>9}  {'F':>9}  {'A':>3}  {'C':>3}")
    for s in summary_rows:
        A_tag = "PASS" if s["check_A_binds"] else "FAIL"
        C_tag = "PASS" if (s["check_C_F_match"] and s["check_C_int_u_match"]) else "FAIL"
        print(f"{s['point']:<10}  {s['gate']:<4}  {s['theta']:>6.3f}  "
              f"{s['chi_2pi_Hz']:>13.2f}  {s['T_analytic_us']:>11.2f}  "
              f"{s['T_grape_us']:>9.2f}  {s['T_paper_us']:>9.2f}  "
              f"{s['F_grape']:>9.6f}  {A_tag:>3}  {C_tag:>3}")

    # ----- save -----
    outp = out_dir / "step_qcr_cr_two_points_results.json"
    with open(outp, "w") as f:
        json.dump({"summary": summary_rows, "all_rows": all_rows}, f, indent=2, default=str)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
