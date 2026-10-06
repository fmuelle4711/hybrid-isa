"""QT with the AGGREGATE field cap: sqrt(sum_k |Omega_k|^2) <= Omega_max.

Rationale: anchor_params 'Rejects any pulse whose Omega_peak/2pi > 100 kHz' is
read here as a bound on the TOTAL laser field, not per tone. With 4 real
quadrature controls (2 tones x 2 quadratures) each bounded by b, the aggregate
is sqrt(sum u_j^2) <= 2b, so the guaranteeing per-quadrature bound is
    b = Omega_max / 2
(vs Omega_max/sqrt(2) for the per-tone convention used in step_qt_meth.py).

Everything else identical to step_qt_meth.py: 5-dim model, tones via the S1/2
auxiliary, frozen Stark drift + exact |Omega(t)|^2 audit, virtual-Z absorption,
picker F_grape >= 0.99999 AND leak_aux < 1e-3 AND |spectator_dF| < 1e-3.

Bright-state floor: the two resonant tones drive only the bright combination at
Omega_tot = aggregate, so a 2pi bright pulse costs 2*pi/Omega_max = 10.000 us
under this cap (vs 7.071 us under the per-tone cap).
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import qutip as qt

from step_qt_meth import (
    HERE, OMEGA_MAX, OMEGA_MAX_2PI_KHZ, STARK_REFS, DRIVEN, IDX, DIM, D_CODE,
    NUM_TSLOTS, THETA, SHIFT_B_KHZ, SHIFT_C_KHZ, SHIFT_G_KHZ,
    stark_diag, stark_scale, run_grape, propagate, fit_virtual_z,
    aux_leakage, block_leakage, F_code_block, U_QT_bare,
)

BOUND_AGG = OMEGA_MAX / 2.0        # guarantees aggregate <= Omega_max


def sweep_aggcap(variant, omega_ref, T_grid):
    sum_om2_worst = len(DRIVEN) * OMEGA_MAX ** 2      # frozen Stark unchanged (worst case)
    H_d = stark_diag(stark_scale(sum_om2_worst, omega_ref))
    U_bare = U_QT_bare()

    print(f"\n{'='*78}")
    print(f"AGGREGATE-CAPPED QT   variant '{variant}'")
    print(f"  Omega_max        = {OMEGA_MAX:.6f} rad/us  (Omega_max/2pi = {OMEGA_MAX_2PI_KHZ} kHz, ASSUMED)")
    print(f"  bound/quadrature = Omega_max/2 = {BOUND_AGG:.6f} rad/us")
    print(f"  => aggregate sqrt(sum u^2) <= 2b = {2*BOUND_AGG:.6f} = Omega_max  (guaranteed)")
    print(f"  bright-state 2pi floor at Omega_tot = Omega_max: {2*math.pi/OMEGA_MAX:.3f} us")
    print(f"{'='*78}")

    rows = []
    for T_us in T_grid:
        u, F_grape, term, wall, U_targ = run_grape(T_us, H_d, bound=BOUND_AGG)
        U_fro = propagate(u, T_us, omega_ref, False, H_d)
        U_tru = propagate(u, T_us, omega_ref, True,  H_d)

        tr = complex((U_targ.dag() * U_fro).tr())
        F_indep = float(abs(tr) ** 2 / DIM ** 2)
        F_vz_f, phases = fit_virtual_z(U_fro, U_bare)
        F_vz_t, _      = fit_virtual_z(U_tru, U_bare)
        dF = F_vz_t - F_vz_f
        la = aux_leakage(U_fro)
        lb = block_leakage(U_fro)
        Fc = F_code_block(U_fro, U_bare)

        omG = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        omA = np.sqrt(u[:, 2]**2 + u[:, 3]**2)
        agg = np.sqrt(omG**2 + omA**2)
        agg_max = float(agg.max())
        tone_max = float(max(omG.max(), omA.max()))

        ok = (F_grape >= 0.99999) and (la < 1e-3) and (abs(dF) < 1e-3) \
             and (agg_max <= OMEGA_MAX * (1 + 1e-9))

        rows.append(dict(
            T_us=T_us, F_grape=F_grape, F_indep=F_indep,
            F_vz_frozen=F_vz_f, F_vz_true=F_vz_t, F_code=Fc,
            spectator_dF=dF, leak_aux=la, leak_block=lb,
            agg_max_rad_per_us=agg_max,
            agg_pct_of_Omega_max=100*agg_max/OMEGA_MAX,
            tone_max_pct_of_Omega_max=100*tone_max/OMEGA_MAX,
            termination=term, walltime_s=wall, picker_pass=bool(ok),
            vz_phases_rad=[float(p) for p in phases],
        ))
        print(f"  T={T_us:>6.2f}us  F_grape={F_grape:.6f}  F_indep={F_indep:.6f}  "
              f"leak={la:.2e}  dF={dF:+.1e}  agg={100*agg_max/OMEGA_MAX:5.1f}%  "
              f"tone={100*tone_max/OMEGA_MAX:5.1f}%  {'PASS' if ok else '--'}  ({wall:.1f}s)",
              flush=True)
    return rows


def main():
    print("=" * 78)
    print("QT -- AGGREGATE FIELD CAP  sqrt(sum_k |Omega_k|^2) <= Omega_max")
    print("=" * 78)

    T_grid = [9.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 14.0, 16.0, 20.0]

    out = {}
    for variant, omega_ref in STARK_REFS.items():
        rows = sweep_aggcap(variant, omega_ref, T_grid)
        passing = [r for r in rows if r["picker_pass"]]
        best = min(passing, key=lambda r: r["T_us"]) if passing else None
        out[variant] = dict(rows=rows, picked=best)
        if best:
            print(f"  --> PICKED T = {best['T_us']} us   F_grape = {best['F_grape']:.6f}   "
                  f"leak = {best['leak_aux']:.2e}   agg = {best['agg_pct_of_Omega_max']:.1f}%")
        else:
            print("  --> NO T satisfied the picker")

    outp = HERE / "step_qt_meth_aggcap_results.json"
    outp.write_text(json.dumps(dict(
        setup=dict(
            cap="aggregate: sqrt(sum_k |Omega_k|^2) <= Omega_max",
            bound_per_quadrature_rad_per_us=BOUND_AGG,
            bound_rule="b = Omega_max/2 so that 2b = Omega_max (4 quadrature controls)",
            Omega_max_2pi_kHz=OMEGA_MAX_2PI_KHZ, Omega_max_tag="ASSUMED",
            bright_state_2pi_floor_us=2*math.pi/OMEGA_MAX,
            num_tslots=NUM_TSLOTS, theta="pi/4",
            picker=("F_grape >= 0.99999 AND leak_aux < 1e-3 AND |spectator_dF| < 1e-3 "
                    "AND aggregate <= Omega_max"),
        ),
        variants=out,
    ), indent=2))
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
