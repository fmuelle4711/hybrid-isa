"""reproduce_vu_dissipative_figs.py -- Reproduce Vu Figs 8, 9, 10.

For each Vu dissipative figure, generate three curves per chromophore:

    Vu reference  : Vu's published .out trajectory (cqed-vibronic-simulation)
    Method B (qd) : H_cqed in qudit encoding + qudit Lindblad ops (mesolve)
    Method A (qd) : H_lab (eq 1) in qudit encoding + qudit Lindblad ops (mesolve)

Outputs:
    figures/fig{8,9,10}_*.{png,pdf}
    data/panel_<name>_N<cutoff>.npz   (one per panel)
    data/fig{8,9,10}_*_data.npz       (assembled per figure)

Usage:
    python -m simulations.reproduce_vu_dissipative_figs                         # all panels, N=8
    python -m simulations.reproduce_vu_dissipative_figs --cutoff 4              # all panels, N=4
    python -m simulations.reproduce_vu_dissipative_figs --panel fig8_top -c 4   # one panel
    python -m simulations.reproduce_vu_dissipative_figs --assemble              # plot from saved .npz
"""

import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import qutip as qt
import sys, os, re, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hamiltonians.hlab import build_hlab
from hamiltonians.hcqed_qudit import build_hcqed_qudit
from hamiltonians.lindblad_qudit import build_lindblad_ops_qudit
import params as P

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
FIG_DIR    = os.path.join(ROOT_DIR, 'figures')
VU_DATA    = os.path.join(ROOT_DIR, 'legacy', 'vu_reference',
                          'cqed-vibronic-simulation-main', 'data', 'general')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

T_MAX = 2.0
N_PTS = 201                          # 0.01-ps grid matches Vu .out files
T_PS  = np.linspace(0, T_MAX, N_PTS)
TLIST_S = T_PS * 1e-12

MESOLVE_OPTS = {
    'nsteps': 20000,
    'store_states': False,
    'store_final_state': False,
    'atol': 1e-8,
    'rtol': 1e-6,
    'progress_bar': False,
    # Adams (default) — BDF tested 60% slower; normalize_output=False also slower.
}

# ------------------------------------------------------------------ #
# Panel registry: rate overrides + Vu reference file                  #
# ------------------------------------------------------------------ #
G_AMP = P.gamma_amp_all
G_DEP = P.gamma_dep_all

def _amp(a, b, c):
    return {'gamma_amp_a': a, 'gamma_amp_b': b, 'gamma_amp_c': c,
            'gamma_dep_a': 0.0, 'gamma_dep_b': 0.0, 'gamma_dep_c': 0.0}

def _dep(a, b, c):
    return {'gamma_amp_a': 0.0, 'gamma_amp_b': 0.0, 'gamma_amp_c': 0.0,
            'gamma_dep_a': a, 'gamma_dep_b': b, 'gamma_dep_c': c}

def _both(amps, deps):
    a, b, c = amps; d, e, f = deps
    return {'gamma_amp_a': a, 'gamma_amp_b': b, 'gamma_amp_c': c,
            'gamma_dep_a': d, 'gamma_dep_b': e, 'gamma_dep_c': f}

PANELS = {
    'fig8_top':       dict(fig='8',  tag='top',
        title=f'gamma_amp = {G_AMP:.2e} (all)',
        rates=_amp(G_AMP, G_AMP, G_AMP),
        vu_file='amp_3.15-3.15-3.15_Fock8_10k.out'),
    'fig8_middle':    dict(fig='8',  tag='middle',
        title=f'gamma_amp,B = 3x ({3.0*G_AMP:.2e})',
        rates=_amp(G_AMP, 3.0*G_AMP, G_AMP),
        vu_file='amp_3.15-9.45-3.15_Fock8_10k.out'),
    'fig8_bottom':    dict(fig='8',  tag='bottom',
        title=f'gamma_amp,B = 1/3x ({G_AMP/3.0:.2e})',
        rates=_amp(G_AMP, G_AMP/3.0, G_AMP),
        vu_file='amp_3.15-1.05-3.15_Fock8_10k.out'),

    'fig9_top':       dict(fig='9',  tag='top',
        title=f'gamma_dep = {G_DEP:.2e} (all)',
        rates=_dep(G_DEP, G_DEP, G_DEP),
        vu_file='dep_0.9-0.9-0.9_Fock8_10k.out'),
    'fig9_middle':    dict(fig='9',  tag='middle',
        title=f'gamma_dep,B = 3x ({3.0*G_DEP:.2e})',
        rates=_dep(G_DEP, 3.0*G_DEP, G_DEP),
        vu_file='dep_0.9-2.7-0.9_Fock8_10k.out'),
    'fig9_bottom':    dict(fig='9',  tag='bottom',
        title=f'gamma_dep,B = 1/3x ({G_DEP/3.0:.2e})',
        rates=_dep(G_DEP, G_DEP/3.0, G_DEP),
        vu_file='dep_0.9-0.3-0.9_Fock8_10k.out'),

    'fig10_combined': dict(fig='10', tag='combined',
        title=f'damped + dephased (amp={G_AMP:.2e}, dep={G_DEP:.2e})',
        rates=_both((G_AMP, G_AMP, G_AMP), (G_DEP, G_DEP, G_DEP)),
        vu_file='amp_all3.15_dep_all0.9_Fock8_10k.out'),
    'fig10_damp_only':dict(fig='10', tag='damp_only',
        title=f'damped only (amp={G_AMP:.2e})',
        rates=_amp(G_AMP, G_AMP, G_AMP),
        vu_file='amp_3.15-3.15-3.15_Fock8_10k.out'),
}

FIG_LAYOUT = {
    '8':  ['fig8_top', 'fig8_middle', 'fig8_bottom'],
    '9':  ['fig9_top', 'fig9_middle', 'fig9_bottom'],
    '10': ['fig10_combined', 'fig10_damp_only'],
}


# ------------------------------------------------------------------ #
# Vu .out parser                                                      #
# ------------------------------------------------------------------ #
_TUPLE_RE = re.compile(
    r'\(\s*([+-]?\d+\.\d+e[+-]\d+)\s*([+-]\d+\.\d+e[+-]\d+)j\s*\)')

def load_vu_out(path):
    """Return (t_ps, P_A, P_B, P_C) from a Vu .out file (rows 0-3, real parts)."""
    with open(path, 'r') as f:
        lines = [ln for ln in f if ln.strip()]
    parsed = []
    for ln in lines[:4]:
        toks = _TUPLE_RE.findall(ln)
        parsed.append(np.array([float(re_) for re_, _ in toks]))
    if len(parsed) < 4:
        raise ValueError(f"Could not parse 4 rows from {path}")
    return parsed[0], parsed[1], parsed[2], parsed[3]


def vu_panel(panel_name):
    """Load and time-align Vu reference for a panel."""
    spec = PANELS[panel_name]
    t_vu, A, B, C = load_vu_out(os.path.join(VU_DATA, spec['vu_file']))
    if not (len(t_vu) == len(T_PS) and np.allclose(t_vu, T_PS, atol=1e-9)):
        A = np.interp(T_PS, t_vu, A)
        B = np.interp(T_PS, t_vu, B)
        C = np.interp(T_PS, t_vu, C)
    return A, B, C


# ------------------------------------------------------------------ #
# Caches: H per (encoding, cutoff); c_ops per (rates tuple, cutoff)   #
# ------------------------------------------------------------------ #
_H_CACHE = {}        # (encoding, N) -> (H, PiA, PiB, PiC, psi0)
_COPS_CACHE = {}     # (rates_tuple, N) -> list of c_ops

_RATE_KEYS = ('gamma_amp_a', 'gamma_amp_b', 'gamma_amp_c',
              'gamma_dep_a', 'gamma_dep_b', 'gamma_dep_c',
              'gamma_exc_a', 'gamma_exc_b', 'gamma_exc_c')

def _rates_tuple(override):
    return tuple(override.get(k, getattr(P, k)) for k in _RATE_KEYS)

def get_hamiltonian(encoding, N):
    key = (encoding, N)
    if key not in _H_CACHE:
        t0 = time.time()
        if encoding == 'lab':
            _H_CACHE[key] = build_hlab(N)
        elif encoding == 'cqed':
            _H_CACHE[key] = build_hcqed_qudit(N)
        else:
            raise ValueError(encoding)
        print(f"  [H_{encoding} N={N}] built in {time.time()-t0:.1f}s, "
              f"dim={_H_CACHE[key][0].shape[0]}")
    return _H_CACHE[key]

def get_cops(override, N):
    key = (_rates_tuple(override), N)
    if key not in _COPS_CACHE:
        _COPS_CACHE[key] = build_lindblad_ops_qudit(N, params_override=override)
    return _COPS_CACHE[key]


# ------------------------------------------------------------------ #
# Single mesolve                                                      #
# ------------------------------------------------------------------ #
def _mesolve_pops(H, PiA, PiB, PiC, psi0, c_ops, label):
    t0 = time.time()
    res = qt.mesolve(H, psi0, TLIST_S, c_ops,
                     e_ops=[PiA, PiB, PiC], options=MESOLVE_OPTS)
    dt = time.time() - t0
    print(f"  [{label}] mesolve done in {dt:.1f}s")
    return (np.array(res.expect[0]), np.array(res.expect[1]),
            np.array(res.expect[2]), dt)


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #
def run_panel(panel='fig8_top', cutoff=4, save=True):
    """Run Method A and Method B mesolve for a single panel.

    Returns dict with t_ps, PA_a..PC_b, PA_vu..PC_vu, timings, plus the panel
    spec metadata. Optionally writes data/panel_<name>_N<cutoff>.npz.
    """
    if panel not in PANELS:
        raise KeyError(f"Unknown panel '{panel}'. Choices: {sorted(PANELS)}")
    spec = PANELS[panel]
    print(f"\n[panel {panel}] cutoff={cutoff}  -- {spec['title']}")

    H_b, PiA, PiB, PiC, psi0 = get_hamiltonian('cqed', cutoff)
    H_a, _,   _,   _,   _    = get_hamiltonian('lab',  cutoff)
    c_ops = get_cops(spec['rates'], cutoff)
    print(f"  c_ops: {len(c_ops)} (from rates {spec['rates']})")

    PA_b, PB_b, PC_b, t_b = _mesolve_pops(H_b, PiA, PiB, PiC, psi0, c_ops, 'Method B')
    PA_a, PB_a, PC_a, t_a = _mesolve_pops(H_a, PiA, PiB, PiC, psi0, c_ops, 'Method A')
    PA_vu, PB_vu, PC_vu = vu_panel(panel)

    out = dict(panel=panel, fig=spec['fig'], tag=spec['tag'], title=spec['title'],
               cutoff=cutoff, t_ps=T_PS,
               PA_a=PA_a, PB_a=PB_a, PC_a=PC_a,
               PA_b=PA_b, PB_b=PB_b, PC_b=PC_b,
               PA_vu=PA_vu, PB_vu=PB_vu, PC_vu=PC_vu,
               wall_method_a=t_a, wall_method_b=t_b)
    if save:
        path = os.path.join(DATA_DIR, f'panel_{panel}_N{cutoff}.npz')
        np.savez(path, **{k: v for k, v in out.items() if not isinstance(v, str)})
        print(f"  saved: {path}")
    return out


# ------------------------------------------------------------------ #
# Plotting                                                            #
# ------------------------------------------------------------------ #
COLORS = dict(A='#2f0033', B='#0500b4', C='#ff0000')

def plot_panel(ax, p):
    t = p['t_ps']
    ax.plot(t, p['PA_vu'], '--', color=COLORS['A'], lw=1.5, label='Vu A')
    ax.plot(t, p['PB_vu'], '--', color=COLORS['B'], lw=1.5, label='Vu B')
    ax.plot(t, p['PC_vu'], '--', color=COLORS['C'], lw=1.5, label='Vu C')
    ax.plot(t, p['PA_b'],  ':',  color=COLORS['A'], lw=1.5, label='MB A')
    ax.plot(t, p['PB_b'],  ':',  color=COLORS['B'], lw=1.5, label='MB B')
    ax.plot(t, p['PC_b'],  ':',  color=COLORS['C'], lw=1.5, label='MB C')
    ax.plot(t, p['PA_a'],  '-',  color=COLORS['A'], lw=1.5, label='MA A')
    ax.plot(t, p['PB_a'],  '-',  color=COLORS['B'], lw=1.5, label='MA B')
    ax.plot(t, p['PC_a'],  '-',  color=COLORS['C'], lw=1.5, label='MA C')
    ax.set_ylim(-0.02, 1.05); ax.set_ylabel('Population')
    title = f"{p.get('fig')} {p.get('tag')}: {p.get('title','')}"
    ax.set_title(title.strip(), fontsize=10)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))

def panel_rmse(p):
    out = {}
    for R in ['A', 'B', 'C']:
        out[f'MA-MB_{R}'] = rmse(p[f'P{R}_a'], p[f'P{R}_b'])
        out[f'MA-Vu_{R}'] = rmse(p[f'P{R}_a'], p[f'P{R}_vu'])
    return out


def assemble_figure(fig_id, cutoff):
    """Build a multi-panel figure from cached panel .npz files."""
    panels = []
    for name in FIG_LAYOUT[fig_id]:
        path = os.path.join(DATA_DIR, f'panel_{name}_N{cutoff}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run with --panel {name} --cutoff {cutoff}.")
        d = dict(np.load(path, allow_pickle=True))
        # restore string fields stripped at save time
        d['fig'] = fig_id; d['tag'] = name.split('_', 1)[1]
        d['title'] = PANELS[name]['title']
        panels.append((name, d))

    n = len(panels)
    fig, axs = plt.subplots(n, 1, figsize=(8, 3*n + 1), sharex=True)
    if n == 1: axs = [axs]
    summaries = []
    for ax, (name, p) in zip(axs, panels):
        plot_panel(ax, p)
        summaries.append((p['tag'].item() if hasattr(p['tag'],'item') else p['tag'],
                          panel_rmse(p)))
    axs[-1].set_xlabel('Time (ps)')
    axs[0].legend(loc='upper right', fontsize=7, ncol=3)
    fig.suptitle(f'Vu Fig {fig_id} reproduction (cutoff={cutoff})')
    fig.tight_layout()

    suffix = {'8': 'fig8_amplitude', '9': 'fig9_dephasing', '10': 'fig10_combined'}[fig_id]
    png = os.path.join(FIG_DIR, f'{suffix}.png')
    pdf = os.path.join(FIG_DIR, f'{suffix}.pdf')
    fig.savefig(png, dpi=150); fig.savefig(pdf); plt.close(fig)

    npz = os.path.join(DATA_DIR, f'{suffix}_data.npz')
    merged = {'t_ps': T_PS}
    for name, p in panels:
        for k in ('PA_a','PB_a','PC_a','PA_b','PB_b','PC_b','PA_vu','PB_vu','PC_vu'):
            merged[f'{name}_{k}'] = p[k]
    np.savez(npz, **merged)
    return summaries, png, pdf


def print_summary(all_summaries):
    print("\n" + "="*78)
    print("RMSE SUMMARY (population units, %)")
    print("="*78)
    print(f"{'Figure':<7s} {'Sub':<11s} "
          f"{'RMSE(MA-MB) %  A/B/C':<26s} {'RMSE(MA-Vu) %  A/B/C':<26s}")
    print("-"*78)
    for fig_id, panels in all_summaries:
        for tag, r in panels:
            mab = f"{100*r['MA-MB_A']:.2f}/{100*r['MA-MB_B']:.2f}/{100*r['MA-MB_C']:.2f}"
            mav = f"{100*r['MA-Vu_A']:.2f}/{100*r['MA-Vu_B']:.2f}/{100*r['MA-Vu_C']:.2f}"
            print(f"{fig_id:<7s} {tag:<11s} {mab:<26s} {mav:<26s}")
    print("="*78)


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cutoff_pos', nargs='?', type=int, default=None,
                    help='positional cutoff (alternative to --cutoff)')
    ap.add_argument('-c', '--cutoff', type=int, default=8)
    ap.add_argument('--panel', type=str, default=None, help='single panel name')
    ap.add_argument('--assemble', action='store_true',
                    help='assemble figures from saved panel .npz files only')
    args = ap.parse_args()
    if args.cutoff_pos is not None:
        args.cutoff = args.cutoff_pos

    if args.panel:
        run_panel(panel=args.panel, cutoff=args.cutoff)
        return

    if not args.assemble:
        for name in PANELS:
            run_panel(panel=name, cutoff=args.cutoff)

    all_summaries = []
    paths = []
    for fig_id in ('8', '9', '10'):
        s, png, pdf = assemble_figure(fig_id, args.cutoff)
        all_summaries.append((fig_id, s))
        paths.extend([png, pdf])

    print("\nFigure files:")
    for p in paths: print(f"  {p}")
    print_summary(all_summaries)


if __name__ == '__main__':
    main()
