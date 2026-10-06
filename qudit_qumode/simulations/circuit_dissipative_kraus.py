"""circuit_dissipative_kraus.py -- Vu-equivalent dissipative simulation via
density-matrix Trotter + per-step Kraus channels.

Mirrors Vu's 3_chromophore_vibronic_simulation.py channel topology (one
Trotter step of the unitary part + amplitude-damping & dephasing channels per
chromophore) but applies the channel directly to the density matrix instead
of via measure-and-reset ancillas. Mathematically equivalent up to the
Stinespring dilation; no MCM/reset needed.

Method A: H_lab    in qudit encoding (Vu eq. 1; second-order Suzuki-Trotter)
Method B: H_cqed   in qudit encoding (Vu eqs 14-17; second-order Suzuki-Trotter)

Per Trotter step (at dt = 5 fs):
    1. U_step = e^{-i H_QTSB tau} sandwiched by half-step expms of
       (free, QCR, QCD, QPS, QT) on each side  -- second-order symmetric.
    2. Amp damping per chromophore R:
         p_R = gamma_amp_R * dt
         K0_R = sqrt(p_R) * |G><R| (x I_modes)
         K1_R = (I_qudit - (1 - sqrt(1-p_R)) |R><R|) (x I_modes)
         rho -> K0 rho K0^dag + K1 rho K1^dag
    3. Dephasing per chromophore R:
         q_R = gamma_dep_R * dt
         L_R = (I_qudit - 2 |R><R|) (x I_modes)
         rho -> (1 - q_R) rho + q_R L_R rho L_R^dag

CLI:
    python -m simulations.circuit_dissipative_kraus --validate          # Task 2: fig8_top vs mesolve
    python -m simulations.circuit_dissipative_kraus --all               # Task 3: all 8 panels
    python -m simulations.circuit_dissipative_kraus --panel fig8_top
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import qutip as qt
from scipy.linalg import expm as dense_expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P
from simulations.reproduce_vu_dissipative_figs import (
    PANELS, FIG_LAYOUT, COLORS, vu_panel,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
FIG_DIR    = os.path.join(ROOT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
# Trotter timing                                                      #
# ------------------------------------------------------------------ #
N_STEPS = 401             # matches Vu's sim_steps
DT = 5e-15                # 5 fs Trotter timestep, matches Vu
T_GRID_PS = np.arange(N_STEPS) * (DT * 1e12)  # 0..2 ps in 5 fs increments

_RATE_KEYS = ('gamma_amp_a', 'gamma_amp_b', 'gamma_amp_c',
              'gamma_dep_a', 'gamma_dep_b', 'gamma_dep_c')

# ------------------------------------------------------------------ #
# Sub-Hamiltonian builders (mirror trotter_method_{a,b}.py splits)    #
# ------------------------------------------------------------------ #
def _qudit_helpers(N):
    I4 = qt.qeye(4); IN = qt.qeye(N)
    _G, _A, _B, _C = 0, 1, 2, 3
    def proj(i):
        v = qt.basis(4, i); return qt.tensor(v*v.dag(), IN, IN, IN, IN)
    def trans(i, j):
        return qt.tensor(qt.basis(4, i)*qt.basis(4, j).dag(), IN, IN, IN, IN)
    def md_op(op, idx):
        ops = [I4] + [IN]*4; ops[1+idx] = op; return qt.tensor(ops)
    return I4, IN, proj, trans, md_op, (_G, _A, _B, _C)


def build_sub_hamiltonians(method, N):
    """Return ordered list [(name, H_sub), ...] for the second-order Trotter
    split, plus (PiA, PiB, PiC, psi0) for population readout / initial state.
    """
    I4, IN, proj_d, trans_d, md_op, (_G, _A, _B, _C) = _qudit_helpers(N)
    PiA = proj_d(_A); PiB = proj_d(_B); PiC = proj_d(_C)
    I_d = qt.tensor(I4, IN, IN, IN, IN)
    ad = md_op(qt.destroy(N), 0); bd = md_op(qt.destroy(N), 1)
    cd = md_op(qt.destroy(N), 2); ld = md_op(qt.destroy(N), 3)
    xad = ad+ad.dag(); xbd = bd+bd.dag(); xcd = cd+cd.dag(); xld = ld+ld.dag()
    nad = ad.dag()*ad; nbd = bd.dag()*bd; ncd = cd.dag()*cd; nld = ld.dag()*ld

    if method == 'A':
        H_free = (P.omega_ga*I_d*nad + P.omega_gb*I_d*nbd
                  + P.omega_gc*I_d*ncd + P.omega_l*I_d*nld)
        H_QCR  = (P.chi_a*PiA*nad + P.chi_b*PiB*nbd + P.chi_c*PiC*ncd)
        H_QCD  = (-P.omega_ea*np.sqrt(P.S_a)*PiA*xad
                  -P.omega_l *np.sqrt(P.S_l)*PiA*xld
                  -P.omega_eb*np.sqrt(P.S_b)*PiB*xbd
                  -P.omega_ec*np.sqrt(P.S_c)*PiC*xcd)
        H_QPS  = (P.omega_ea*P.S_a*PiA + P.omega_l*P.S_l*PiA
                  + P.omega_eb*P.S_b*PiB + P.omega_ec*P.S_c*PiC)
        H_QT   = (P.J_AB0*(trans_d(_A,_B)+trans_d(_B,_A))
                  + P.J_AC0*(trans_d(_A,_C)+trans_d(_C,_A)))
        H_QTSB = (P.J_AB0*P.eta_AB*(trans_d(_A,_B)+trans_d(_B,_A))*xld
                  + P.J_AC0*P.eta_AC*(trans_d(_A,_C)+trans_d(_C,_A))*xld)
    elif method == 'B':
        H_free = (P.omega_a*I_d*nad + P.omega_b*I_d*nbd
                  + P.omega_c*I_d*ncd + P.omega_l*I_d*nld)
        H_QCR  = (-P.chi_a/2*I_d*nad + P.chi_a*PiA*nad
                  -P.chi_b/2*I_d*nbd + P.chi_b*PiB*nbd
                  -P.chi_c/2*I_d*ncd + P.chi_c*PiC*ncd)
        H_QCD  = (P.g_cd_a/2*I_d*xad - P.g_cd_a*PiA*xad
                  + P.g_cd_b/2*I_d*xbd - P.g_cd_b*PiB*xbd
                  + P.g_cd_c/2*I_d*xcd - P.g_cd_c*PiC*xcd
                  + P.g_cdl /2*I_d*xld - P.g_cdl *PiA*xld)
        H_QPS  = (P.delta_ab*PiB + P.delta_ac*PiC)
        H_QT   = (P.g_ab*(trans_d(_A,_B)+trans_d(_B,_A))
                  + P.g_ac*(trans_d(_A,_C)+trans_d(_C,_A)))
        H_QTSB = (P.g_abl*(trans_d(_A,_B)+trans_d(_B,_A))*xld
                  + P.g_acl*(trans_d(_A,_C)+trans_d(_C,_A))*xld)
    else:
        raise ValueError(f"method must be 'A' or 'B', got {method}")

    H_subs = [('free', H_free), ('QCR', H_QCR), ('QCD', H_QCD),
              ('QPS',  H_QPS),  ('QT',  H_QT),  ('QTSB', H_QTSB)]

    psi0 = qt.tensor(qt.basis(4, _A), qt.basis(N, 0), qt.basis(N, 0),
                     qt.basis(N, 0), qt.basis(N, 0))
    return H_subs, PiA, PiB, PiC, psi0


# ------------------------------------------------------------------ #
# Kraus operator builders                                             #
# ------------------------------------------------------------------ #
def build_kraus_qudit(N, rates, dt):
    """Build per-chromophore Kraus operators for one Trotter step.

    rates: dict with 'gamma_amp_{a,b,c}', 'gamma_dep_{a,b,c}'.

    Returns: (amp_pairs, dep_pairs) where
      amp_pairs is a list of (K0, K1) Qobj pairs (one per active R)
      dep_pairs is a list of (q_R, L_R) pairs (one per active R)
    """
    I4, IN, proj_d, _trans, _md, (_G, _A, _B, _C) = _qudit_helpers(N)
    I_d = qt.tensor(I4, IN, IN, IN, IN)
    G_levels = {1: _A, 2: _B, 3: _C}                  # index -> qudit level
    amp_keys = ('gamma_amp_a', 'gamma_amp_b', 'gamma_amp_c')
    dep_keys = ('gamma_dep_a', 'gamma_dep_b', 'gamma_dep_c')

    amp_pairs = []
    for k, lvl in zip(amp_keys, (_A, _B, _C)):
        gamma = rates.get(k, getattr(P, k))
        if gamma <= 0.0: continue
        p = gamma * dt
        if p > 1.0:
            raise ValueError(f"gamma*dt = {p:.3f} > 1; pick smaller dt")
        sqrt_p = np.sqrt(p); sqrt_1mp = np.sqrt(1.0 - p)
        # K0 = sqrt(p) |G><R|       (the jump operator)
        # K1 = I - (1 - sqrt(1-p)) |R><R|   (the no-jump operator)
        K_jump = qt.tensor(sqrt_p * qt.basis(4, _G) * qt.basis(4, lvl).dag(),
                           IN, IN, IN, IN)
        Pi_R   = qt.tensor(qt.basis(4, lvl)*qt.basis(4, lvl).dag(),
                           IN, IN, IN, IN)
        K_noj  = I_d - (1.0 - sqrt_1mp) * Pi_R
        amp_pairs.append((K_jump, K_noj))

    dep_pairs = []
    for k, lvl in zip(dep_keys, (_A, _B, _C)):
        gamma = rates.get(k, getattr(P, k))
        if gamma <= 0.0: continue
        q = gamma * dt
        if q > 1.0:
            raise ValueError(f"gamma_dep*dt = {q:.3f} > 1; pick smaller dt")
        Pi_R = qt.tensor(qt.basis(4, lvl)*qt.basis(4, lvl).dag(),
                         IN, IN, IN, IN)
        L_R = I_d - 2.0 * Pi_R   # eigenvalues +1 on G,B,C and -1 on R
        dep_pairs.append((q, L_R))

    return amp_pairs, dep_pairs


# ------------------------------------------------------------------ #
# Build U_step from second-order Trotter splitting                    #
# ------------------------------------------------------------------ #
def build_u_step(H_subs, dt):
    """Second-order symmetric Trotter step:
        free(h) QCR(h) QCD(h) QPS(h) QT(h) | QTSB(t) | QT(h) QPS(h) QCD(h) QCR(h) free(h)
    where h = dt/2.  Returns the dim^2 unitary as a Qobj.
    """
    h = dt / 2
    name_to_h = {n: H for n, H in H_subs}
    def U(name, t):
        H = name_to_h[name]
        return qt.Qobj(dense_expm(-1j * H.full() * t), dims=H.dims)
    seq = ['free', 'QCR', 'QCD', 'QPS', 'QT']
    Us = [U(n, h) for n in seq]
    Ufwd = Us[0]
    for u in Us[1:]:
        Ufwd = Ufwd * u
    Umid = U('QTSB', dt)
    Ubwd = Us[-1]
    for u in reversed(Us[:-1]):
        Ubwd = Ubwd * u
    return Ufwd * Umid * Ubwd


# ------------------------------------------------------------------ #
# evolve_kraus                                                        #
# ------------------------------------------------------------------ #
def evolve_kraus(rho0=None, encoding='qudit', method='A', config=None,
                 N=4, n_steps=N_STEPS, dt=DT, verbose=True):
    """Density-matrix Trotter+Kraus evolution.

    Args:
        rho0:     initial Qobj density matrix; if None, |A,vac><A,vac| is used.
        encoding: only 'qudit' supported (qubit encoding has its own H).
        method:   'A' (H_lab) or 'B' (H_cqed_qudit).
        config:   dict of rate overrides (panel rates dict).
        N:        Fock cutoff.
        n_steps:  number of Trotter steps (records n_steps populations).
        dt:       Trotter timestep (s).

    Returns:
        dict with t_ps, PA, PB, PC, wall.
    """
    assert encoding == 'qudit'
    config = config or {}
    t_total0 = time.time()

    H_subs, PiA, PiB, PiC, psi0 = build_sub_hamiltonians(method, N)
    if rho0 is None:
        rho0 = qt.ket2dm(psi0)
    if verbose:
        print(f"  [Kraus M{method} N={N}] dim={rho0.shape[0]}, n_steps={n_steps}")

    t0 = time.time()
    U_step = build_u_step(H_subs, dt)
    Ud = U_step.dag()
    if verbose:
        print(f"  [U_step build] {time.time()-t0:.1f}s")

    amp_pairs, dep_pairs = build_kraus_qudit(N, config, dt)
    if verbose:
        print(f"  [kraus] {len(amp_pairs)} amp + {len(dep_pairs)} dep "
              f"per step (config={config})")

    PA = np.empty(n_steps); PB = np.empty(n_steps); PC = np.empty(n_steps)
    rho = rho0
    PA[0] = qt.expect(PiA, rho).real
    PB[0] = qt.expect(PiB, rho).real
    PC[0] = qt.expect(PiC, rho).real

    t_loop0 = time.time()
    for k in range(1, n_steps):
        # Unitary Trotter step
        rho = U_step * rho * Ud
        # Amplitude damping channels
        for K_jump, K_noj in amp_pairs:
            rho = K_jump * rho * K_jump.dag() + K_noj * rho * K_noj.dag()
        # Dephasing channels
        for q_R, L_R in dep_pairs:
            rho = (1.0 - q_R) * rho + q_R * (L_R * rho * L_R)
        PA[k] = qt.expect(PiA, rho).real
        PB[k] = qt.expect(PiB, rho).real
        PC[k] = qt.expect(PiC, rho).real
    if verbose:
        print(f"  [evolution] {time.time()-t_loop0:.1f}s for {n_steps-1} steps")

    return dict(t_ps=T_GRID_PS[:n_steps], PA=PA, PB=PB, PC=PC,
                wall=time.time() - t_total0)


# ------------------------------------------------------------------ #
# Per-panel runner                                                    #
# ------------------------------------------------------------------ #
def run_kraus_panel(panel, N=4, save=True):
    if panel not in PANELS:
        raise KeyError(panel)
    spec = PANELS[panel]
    print(f"\n[panel {panel}] N={N}  -- {spec['title']}")
    res_b = evolve_kraus(method='B', config=spec['rates'], N=N)
    res_a = evolve_kraus(method='A', config=spec['rates'], N=N)
    PA_vu, PB_vu, PC_vu = vu_panel(panel)
    # Vu reference is at 201 pts (10 fs); align by interp onto 5 fs grid
    t_vu = np.linspace(0, 2.0, 201)
    PA_vu_g = np.interp(T_GRID_PS, t_vu, PA_vu)
    PB_vu_g = np.interp(T_GRID_PS, t_vu, PB_vu)
    PC_vu_g = np.interp(T_GRID_PS, t_vu, PC_vu)

    out = dict(panel=panel, fig=spec['fig'], tag=spec['tag'],
               title=spec['title'], cutoff=N, t_ps=T_GRID_PS,
               PA_a=res_a['PA'], PB_a=res_a['PB'], PC_a=res_a['PC'],
               PA_b=res_b['PA'], PB_b=res_b['PB'], PC_b=res_b['PC'],
               PA_vu=PA_vu_g, PB_vu=PB_vu_g, PC_vu=PC_vu_g,
               wall_method_a=res_a['wall'], wall_method_b=res_b['wall'])
    if save:
        path = os.path.join(DATA_DIR, f'panel_kraus_{panel}_N{N}.npz')
        np.savez(path, **{k: v for k, v in out.items() if not isinstance(v, str)})
        print(f"  saved: {path}")
    return out


# ------------------------------------------------------------------ #
# Comparison utilities                                                #
# ------------------------------------------------------------------ #
def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _load_mesolve_panel(panel, N):
    path = os.path.join(DATA_DIR, f'panel_{panel}_N{N}.npz')
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def panel_rmses(kraus_out, mesolve_out):
    """RMSE table for a single panel: Kraus vs mesolve, Kraus vs Vu, mesolve vs Vu."""
    t_kraus = kraus_out['t_ps']
    t_meso  = mesolve_out['t_ps'] if mesolve_out is not None else None
    rows = {}
    for R in ('A', 'B', 'C'):
        kA = kraus_out[f'P{R}_a']; kB = kraus_out[f'P{R}_b']
        kV = kraus_out[f'P{R}_vu']
        if mesolve_out is not None:
            mA = np.interp(t_kraus, t_meso, mesolve_out[f'P{R}_a'])
            mB = np.interp(t_kraus, t_meso, mesolve_out[f'P{R}_b'])
            mV = np.interp(t_kraus, t_meso, mesolve_out[f'P{R}_vu'])
            rows[f'KA-mA_{R}'] = rmse(kA, mA)
            rows[f'KB-mB_{R}'] = rmse(kB, mB)
            rows[f'mA-Vu_{R}'] = rmse(mA, mV)
        rows[f'KA-KB_{R}'] = rmse(kA, kB)
        rows[f'KA-Vu_{R}'] = rmse(kA, kV)
    return rows


# ------------------------------------------------------------------ #
# Plotting                                                            #
# ------------------------------------------------------------------ #
def plot_panel(ax, p, title=None):
    t = p['t_ps']
    ax.plot(t, p['PA_vu'], '--', color=COLORS['A'], lw=1.5, label='Vu A')
    ax.plot(t, p['PB_vu'], '--', color=COLORS['B'], lw=1.5, label='Vu B')
    ax.plot(t, p['PC_vu'], '--', color=COLORS['C'], lw=1.5, label='Vu C')
    ax.plot(t, p['PA_b'],  ':',  color=COLORS['A'], lw=1.5, label='K-MB A')
    ax.plot(t, p['PB_b'],  ':',  color=COLORS['B'], lw=1.5, label='K-MB B')
    ax.plot(t, p['PC_b'],  ':',  color=COLORS['C'], lw=1.5, label='K-MB C')
    ax.plot(t, p['PA_a'],  '-',  color=COLORS['A'], lw=1.5, label='K-MA A')
    ax.plot(t, p['PB_a'],  '-',  color=COLORS['B'], lw=1.5, label='K-MA B')
    ax.plot(t, p['PC_a'],  '-',  color=COLORS['C'], lw=1.5, label='K-MA C')
    ax.set_ylim(-0.02, 1.05); ax.set_ylabel('Population')
    ax.set_title(title or f"{p.get('fig')} {p.get('tag')}: {p.get('title','')}",
                 fontsize=10)


def assemble_figure_kraus(fig_id, N):
    panels = []
    for name in FIG_LAYOUT[fig_id]:
        path = os.path.join(DATA_DIR, f'panel_kraus_{name}_N{N}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = dict(np.load(path, allow_pickle=True))
        d['fig'] = fig_id; d['tag'] = name.split('_', 1)[1]
        d['title'] = PANELS[name]['title']
        panels.append((name, d))

    n = len(panels)
    fig, axs = plt.subplots(n, 1, figsize=(8, 3*n + 1), sharex=True)
    if n == 1: axs = [axs]
    for ax, (_, p) in zip(axs, panels):
        plot_panel(ax, p)
    axs[-1].set_xlabel('Time (ps)')
    axs[0].legend(loc='upper right', fontsize=7, ncol=3)
    fig.suptitle(f'Vu Fig {fig_id} -- Kraus Trotter (cutoff={N})')
    fig.tight_layout()
    suffix = {'8': 'fig8_amplitude_kraus', '9': 'fig9_dephasing_kraus',
              '10': 'fig10_combined_kraus'}[fig_id]
    png = os.path.join(FIG_DIR, f'{suffix}.png')
    pdf = os.path.join(FIG_DIR, f'{suffix}.pdf')
    fig.savefig(png, dpi=150); fig.savefig(pdf); plt.close(fig)
    return panels, png, pdf


# ------------------------------------------------------------------ #
# Gate count printer (Task 4)                                         #
# ------------------------------------------------------------------ #
def print_gate_counts():
    # From trotter_method_a.py / trotter_method_b.py
    method_a_unitary = 36   # 2 * (4 free + 3 QCR + 4 QCD + 4 QPS + 2 QT) + 2 QTSB
    method_b_unitary = 32   # 2 * (4 free + 3 QCR + 4 QCD + 2 QPS + 2 QT) + 2 QTSB
    # Per Vu's circuit: 4 native ops per chromophore-channel (cry/cx/measure/reset
    # for amp; ry/cz/measure/reset for dep). On a TI qudit the unitary part is
    # ~5 native ops per channel after qudit-controlled decomposition.
    native_per_channel = 5
    n_channels = 6                                  # 3 amp + 3 dep
    ancilla_unitary = n_channels * native_per_channel
    n_mcm_reset = n_channels                        # one M+R per channel use
    print("\n" + "="*72)
    print("PHYSICAL GATE COUNT PER TROTTER STEP (with ancilla channels)")
    print("="*72)
    print(f"{'Method':<10s} {'Unitary':>10s} {'Anc-unitary':>14s} "
          f"{'MCM+reset':>12s} {'Total':>10s}")
    print('-'*72)
    print(f"{'A (qudit)':<10s} {method_a_unitary:>10d} {ancilla_unitary:>14d} "
          f"{n_mcm_reset:>12d} {method_a_unitary + ancilla_unitary:>10d}"
          f"  + {n_mcm_reset} M+R")
    print(f"{'B (qubit)':<10s} {method_b_unitary:>10d} {ancilla_unitary:>14d} "
          f"{n_mcm_reset:>12d} {method_b_unitary + ancilla_unitary:>10d}"
          f"  + {n_mcm_reset} M+R")
    print('-'*72)
    print(f"Channel breakdown (per chromophore, per Trotter step):")
    print(f"  amp damping  : ~{native_per_channel} native gates + 1 MCM + 1 reset")
    print(f"  pure dephase : ~{native_per_channel} native gates + 1 MCM + 1 reset")
    print(f"  6 channels   : {ancilla_unitary} native gates + {n_mcm_reset} MCM + "
          f"{n_mcm_reset} reset")
    print("="*72)


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #
def cmd_validate(N):
    """Task 2: run fig8_top via Kraus, compare to mesolve baseline."""
    print(f"\n[VALIDATE] running fig8_top via Kraus at N={N}")
    out = run_kraus_panel('fig8_top', N=N, save=True)
    meso = _load_mesolve_panel('fig8_top', N)
    if meso is None:
        print("  WARNING: no mesolve panel_fig8_top_N4.npz found; skipping comparison")
        return
    rs = panel_rmses(out, meso)
    print("\nValidation RMSE (population units, %):")
    print(f"  K-MA vs mesolve-MA  A/B/C : "
          f"{100*rs['KA-mA_A']:.3f}/{100*rs['KA-mA_B']:.3f}/{100*rs['KA-mA_C']:.3f}")
    print(f"  K-MB vs mesolve-MB  A/B/C : "
          f"{100*rs['KB-mB_A']:.3f}/{100*rs['KB-mB_B']:.3f}/{100*rs['KB-mB_C']:.3f}")
    print(f"  K-MA vs Vu          A/B/C : "
          f"{100*rs['KA-Vu_A']:.3f}/{100*rs['KA-Vu_B']:.3f}/{100*rs['KA-Vu_C']:.3f}")
    print(f"  mesolve-MA vs Vu    A/B/C : "
          f"{100*rs['mA-Vu_A']:.3f}/{100*rs['mA-Vu_B']:.3f}/{100*rs['mA-Vu_C']:.3f}")


def cmd_all(N):
    """Task 3: run all 8 panels, assemble figures, print full RMSE table."""
    paths = []
    summary = []
    for name in PANELS:
        out = run_kraus_panel(name, N=N, save=True)
        meso = _load_mesolve_panel(name, N)
        rs = panel_rmses(out, meso)
        summary.append((name, rs))

    for fig_id in ('8', '9', '10'):
        _panels, png, pdf = assemble_figure_kraus(fig_id, N)
        paths.extend([png, pdf])

    print("\nFigure files:")
    for p in paths: print(f"  {p}")

    print("\n" + "="*108)
    print("RMSE TABLE (population units, %)")
    print("="*108)
    print(f"{'Panel':<18s} "
          f"{'K-MA vs m-MA  A/B/C':<22s} "
          f"{'K-MB vs m-MB  A/B/C':<22s} "
          f"{'K-MA vs Vu   A/B/C':<22s} "
          f"{'m-MA vs Vu   A/B/C':<22s}")
    print("-"*108)
    for name, rs in summary:
        f = lambda key: (f"{100*rs[key+'_A']:.2f}/"
                         f"{100*rs[key+'_B']:.2f}/"
                         f"{100*rs[key+'_C']:.2f}"
                         if key+'_A' in rs else 'n/a')
        print(f"{name:<18s} "
              f"{f('KA-mA'):<22s} {f('KB-mB'):<22s} "
              f"{f('KA-Vu'):<22s} {f('mA-Vu'):<22s}")
    print("="*108)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', '-c', type=int, default=4)
    ap.add_argument('--validate', action='store_true', help='Task 2')
    ap.add_argument('--all',      action='store_true', help='Task 3 (all panels)')
    ap.add_argument('--panel',    type=str, default=None)
    ap.add_argument('--gate-counts', action='store_true', help='Task 4 only')
    args = ap.parse_args()

    if args.gate_counts:
        print_gate_counts(); return
    if args.validate:
        cmd_validate(args.cutoff); print_gate_counts(); return
    if args.panel:
        run_kraus_panel(args.panel, N=args.cutoff); return
    if args.all:
        cmd_all(args.cutoff); print_gate_counts(); return

    cmd_validate(args.cutoff)


if __name__ == '__main__':
    main()
