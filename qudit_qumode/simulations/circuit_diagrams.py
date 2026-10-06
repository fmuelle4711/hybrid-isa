"""circuit_diagrams.py -- One-step Trotter circuit diagrams for the four
implementations (qubit, qudit, qudit-d, Vu cQED).

Outputs:
    figures/circuit_<impl>_one_step.{png,pdf,svg}
    data/circuit_<impl>_one_step.json

Each diagram = exactly ONE second-order Suzuki-Trotter step. The JSON gate
list captures (gate, wires, params, source_term) tuples for downstream
analysis or alternative renderings.

Implementation map:
    qudit   = Method A native qudit (Hybridlane: R, QCR, QCD, QPS, QT, QT-SB)
              -- 36 gates per second-order step
    qudit_d = Method A with Pi_A QPS gates fused -- 34 gates per step
    qubit   = Method B-qubit (literal Vu encoding, our TI gate count) -- 32 gates
    vu      = Vu cQED literal pipeline (full c2qa decomposition with SWAPs,
              H/CX sandwiches, etc.) -- ~96 gates per Vu paper count
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import params as P

FIG_DIR  = os.path.join(ROOT, 'figures')
DATA_DIR = os.path.join(ROOT, 'data')

TAU = 0.005e-12          # 5 fs in seconds (Hamiltonian in Hz)
TAU_H = TAU / 2

# Vu's THz*ps unit convention used inside the c2qa circuits
TS_PS = 0.005           # 5 fs in ps
omega_THz_arr = np.array([P.omega_a, P.omega_b, P.omega_c, P.omega_l]) / 1e12
delta_qa     = np.array([P.delta_ab, P.delta_ac]) / 1e12
chi          = np.array([P.chi_a, P.chi_b, P.chi_c]) / 1e12
g_cd         = np.array([P.g_cd_a, P.g_cd_b, P.g_cd_c]) / 1e12
g_cdl        = (P.g_cdl + 0.0j) / 1e12
g_a_arr      = np.array([P.J_AB0, P.J_AC0]) / 1e12
g_al_arr     = np.array([P.J_AB0 * P.eta_AB, P.J_AC0 * P.eta_AC]) / 1e12

NUM_MODES = 4


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------
def save_outputs(name, fig, gate_list, gate_count, source_files=None):
    fig_path = os.path.join(FIG_DIR, f'circuit_{name}_one_step')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{fig_path}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    json_path = os.path.join(DATA_DIR, f'circuit_{name}_one_step.json')
    payload = {
        'implementation': name,
        'gate_count_total': gate_count,
        'gate_count_actual': len(gate_list),
        'source_files': source_files or [],
        'gates': gate_list,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    return fig_path, json_path


# ---------------------------------------------------------------------------
# QUDIT (Method A, 36 gates/step) -- Hybridlane native qudit
# ---------------------------------------------------------------------------
def make_qudit():
    import pennylane as qml
    import hybridlane as hqml

    _G, _A, _B, _C = 0, 1, 2, 3

    dev = qml.device('bosonicqiskit.hybrid', wires=['q','a','b','c','l'],
                     max_fock_level=4)

    @qml.qnode(dev)
    def one_step():
        qml.PauliX(wires='q')
        # === forward half-step: free, QCR, QCD, QPS, QT
        hqml.Rotation(P.omega_ga * TAU_H, wires='a')
        hqml.Rotation(P.omega_gb * TAU_H, wires='b')
        hqml.Rotation(P.omega_gc * TAU_H, wires='c')
        hqml.Rotation(P.omega_l  * TAU_H, wires='l')
        hqml.QCR(-2*P.chi_a * TAU_H, level=_A, wires=['q','a'])
        hqml.QCR(-2*P.chi_b * TAU_H, level=_B, wires=['q','b'])
        hqml.QCR(-2*P.chi_c * TAU_H, level=_C, wires=['q','c'])
        hqml.QCD(P.omega_ea*np.sqrt(P.S_a) * TAU_H, np.pi/2, level=_A, wires=['q','a'])
        hqml.QCD(P.omega_eb*np.sqrt(P.S_b) * TAU_H, np.pi/2, level=_B, wires=['q','b'])
        hqml.QCD(P.omega_ec*np.sqrt(P.S_c) * TAU_H, np.pi/2, level=_C, wires=['q','c'])
        hqml.QCD(P.omega_l*np.sqrt(P.S_l)  * TAU_H, np.pi/2, level=_A, wires=['q','l'])
        hqml.QPS(P.omega_ea*P.S_a * TAU_H, level=_A, wires='q')
        hqml.QPS(P.omega_l *P.S_l * TAU_H, level=_A, wires='q')
        hqml.QPS(P.omega_eb*P.S_b * TAU_H, level=_B, wires='q')
        hqml.QPS(P.omega_ec*P.S_c * TAU_H, level=_C, wires='q')
        hqml.QT(P.J_AB0 * TAU_H, level_i=_A, level_j=_B, wires='q')
        hqml.QT(P.J_AC0 * TAU_H, level_i=_A, level_j=_C, wires='q')

        # === center: H_QTSB full tau
        hqml.QT(P.J_AB0*P.eta_AB * TAU, level_i=_A, level_j=_B,
                wires=['q','l'], sideband=True)
        hqml.QT(P.J_AC0*P.eta_AC * TAU, level_i=_A, level_j=_C,
                wires=['q','l'], sideband=True)

        # === backward half-step (reverse order)
        hqml.QT(P.J_AC0 * TAU_H, level_i=_A, level_j=_C, wires='q')
        hqml.QT(P.J_AB0 * TAU_H, level_i=_A, level_j=_B, wires='q')
        hqml.QPS(P.omega_ec*P.S_c * TAU_H, level=_C, wires='q')
        hqml.QPS(P.omega_eb*P.S_b * TAU_H, level=_B, wires='q')
        hqml.QPS(P.omega_l *P.S_l * TAU_H, level=_A, wires='q')
        hqml.QPS(P.omega_ea*P.S_a * TAU_H, level=_A, wires='q')
        hqml.QCD(P.omega_l*np.sqrt(P.S_l)  * TAU_H, np.pi/2, level=_A, wires=['q','l'])
        hqml.QCD(P.omega_ec*np.sqrt(P.S_c) * TAU_H, np.pi/2, level=_C, wires=['q','c'])
        hqml.QCD(P.omega_eb*np.sqrt(P.S_b) * TAU_H, np.pi/2, level=_B, wires=['q','b'])
        hqml.QCD(P.omega_ea*np.sqrt(P.S_a) * TAU_H, np.pi/2, level=_A, wires=['q','a'])
        hqml.QCR(-2*P.chi_c * TAU_H, level=_C, wires=['q','c'])
        hqml.QCR(-2*P.chi_b * TAU_H, level=_B, wires=['q','b'])
        hqml.QCR(-2*P.chi_a * TAU_H, level=_A, wires=['q','a'])
        hqml.Rotation(P.omega_l  * TAU_H, wires='l')
        hqml.Rotation(P.omega_gc * TAU_H, wires='c')
        hqml.Rotation(P.omega_gb * TAU_H, wires='b')
        hqml.Rotation(P.omega_ga * TAU_H, wires='a')

        return hqml.state(wires=['q','a','b','c','l'])

    fig, _ = hqml.draw_mpl(one_step, style='sketch')()
    fig.suptitle(
        'qudit (Method A, H_lab): one second-order Trotter step  |  '
        '4 R + 3 QCR + 3 QCD-local + 1 QCD-l + 4 QPS + 2 QT (x2 half-step)'
        '  +  2 QT-sideband  =  36 total',
        fontsize=8)

    # Build the JSON gate list (forward half-step + center + backward) — 36 entries
    gates = []
    sources = {
        'free': 'H_free: omega_gR * I * n_R',
        'QCR_local': 'H_QCR: chi_R * Pi_R * n_R',
        'QCD_local': 'H_QCD: -omega_eR * sqrt(S_R) * Pi_R * x_R',
        'QCD_l_A':   'H_QCD: -omega_l * sqrt(S_l) * Pi_A * x_l (mode-l only on Pi_A)',
        'QPS_S':     'H_QPS: omega_eR * S_R * Pi_R',
        'QPS_l_A':   'H_QPS: omega_l * S_l * Pi_A',
        'QT':        'H_QT: J_AB or J_AC carrier transition |R><R\'| + h.c.',
        'QT_sb':     'H_QTSB: J * eta * (|R><R\'| + h.c.) * x_l',
    }
    fwd = [
        # 4 free rotations
        ('Rotation', ['a'], [P.omega_ga * TAU_H], sources['free']),
        ('Rotation', ['b'], [P.omega_gb * TAU_H], sources['free']),
        ('Rotation', ['c'], [P.omega_gc * TAU_H], sources['free']),
        ('Rotation', ['l'], [P.omega_l  * TAU_H], sources['free']),
        # 3 QCR (one per chromophore-local mode)
        ('QCR', ['q','a'], [-2*P.chi_a*TAU_H, 'level=A'], sources['QCR_local']),
        ('QCR', ['q','b'], [-2*P.chi_b*TAU_H, 'level=B'], sources['QCR_local']),
        ('QCR', ['q','c'], [-2*P.chi_c*TAU_H, 'level=C'], sources['QCR_local']),
        # 4 QCD (3 local + 1 mode-l on Pi_A)
        ('QCD', ['q','a'], [P.omega_ea*np.sqrt(P.S_a)*TAU_H, 'phi=pi/2', 'level=A'], sources['QCD_local']),
        ('QCD', ['q','b'], [P.omega_eb*np.sqrt(P.S_b)*TAU_H, 'phi=pi/2', 'level=B'], sources['QCD_local']),
        ('QCD', ['q','c'], [P.omega_ec*np.sqrt(P.S_c)*TAU_H, 'phi=pi/2', 'level=C'], sources['QCD_local']),
        ('QCD', ['q','l'], [P.omega_l *np.sqrt(P.S_l)*TAU_H, 'phi=pi/2', 'level=A'], sources['QCD_l_A']),
        # 4 QPS (Pi_A local-mode + Pi_A mode-l + Pi_B + Pi_C)
        ('QPS', ['q'],     [P.omega_ea*P.S_a*TAU_H, 'level=A'], sources['QPS_S']),
        ('QPS', ['q'],     [P.omega_l *P.S_l*TAU_H, 'level=A'], sources['QPS_l_A']),
        ('QPS', ['q'],     [P.omega_eb*P.S_b*TAU_H, 'level=B'], sources['QPS_S']),
        ('QPS', ['q'],     [P.omega_ec*P.S_c*TAU_H, 'level=C'], sources['QPS_S']),
        # 2 QT carrier
        ('QT',  ['q'],     [P.J_AB0*TAU_H, 'level_i=A,level_j=B'], sources['QT']),
        ('QT',  ['q'],     [P.J_AC0*TAU_H, 'level_i=A,level_j=C'], sources['QT']),
    ]
    center = [
        ('QT-SB', ['q','l'], [P.J_AB0*P.eta_AB*TAU, 'level_i=A,level_j=B,sideband'], sources['QT_sb']),
        ('QT-SB', ['q','l'], [P.J_AC0*P.eta_AC*TAU, 'level_i=A,level_j=C,sideband'], sources['QT_sb']),
    ]
    bwd = list(reversed(fwd))   # symmetric Trotter
    seq = fwd + center + bwd
    for g, w, p, s in seq:
        gates.append({'gate': g, 'wires': w, 'params': p, 'source_term': s})

    return fig, gates, 36, ['simulations/trotter_method_a.py', 'hamiltonians/hlab.py']


# ---------------------------------------------------------------------------
# QUDIT-D (Method A reduced, 34 gates/step) -- Pi_A QPS gates fused
# ---------------------------------------------------------------------------
def make_qudit_d():
    import pennylane as qml
    import hybridlane as hqml

    _G, _A, _B, _C = 0, 1, 2, 3

    dev = qml.device('bosonicqiskit.hybrid', wires=['q','a','b','c','l'],
                     max_fock_level=4)

    @qml.qnode(dev)
    def one_step():
        qml.PauliX(wires='q')
        # forward half-step
        hqml.Rotation(P.omega_ga * TAU_H, wires='a')
        hqml.Rotation(P.omega_gb * TAU_H, wires='b')
        hqml.Rotation(P.omega_gc * TAU_H, wires='c')
        hqml.Rotation(P.omega_l  * TAU_H, wires='l')
        hqml.QCR(-2*P.chi_a * TAU_H, level=_A, wires=['q','a'])
        hqml.QCR(-2*P.chi_b * TAU_H, level=_B, wires=['q','b'])
        hqml.QCR(-2*P.chi_c * TAU_H, level=_C, wires=['q','c'])
        hqml.QCD(P.omega_ea*np.sqrt(P.S_a) * TAU_H, np.pi/2, level=_A, wires=['q','a'])
        hqml.QCD(P.omega_eb*np.sqrt(P.S_b) * TAU_H, np.pi/2, level=_B, wires=['q','b'])
        hqml.QCD(P.omega_ec*np.sqrt(P.S_c) * TAU_H, np.pi/2, level=_C, wires=['q','c'])
        hqml.QCD(P.omega_l *np.sqrt(P.S_l) * TAU_H, np.pi/2, level=_A, wires=['q','l'])
        # FUSED Pi_A QPS:
        hqml.QPS((P.omega_ea*P.S_a + P.omega_l*P.S_l) * TAU_H, level=_A, wires='q')
        hqml.QPS(P.omega_eb*P.S_b * TAU_H, level=_B, wires='q')
        hqml.QPS(P.omega_ec*P.S_c * TAU_H, level=_C, wires='q')
        hqml.QT(P.J_AB0 * TAU_H, level_i=_A, level_j=_B, wires='q')
        hqml.QT(P.J_AC0 * TAU_H, level_i=_A, level_j=_C, wires='q')
        # center
        hqml.QT(P.J_AB0*P.eta_AB * TAU, level_i=_A, level_j=_B,
                wires=['q','l'], sideband=True)
        hqml.QT(P.J_AC0*P.eta_AC * TAU, level_i=_A, level_j=_C,
                wires=['q','l'], sideband=True)
        # backward half-step
        hqml.QT(P.J_AC0 * TAU_H, level_i=_A, level_j=_C, wires='q')
        hqml.QT(P.J_AB0 * TAU_H, level_i=_A, level_j=_B, wires='q')
        hqml.QPS(P.omega_ec*P.S_c * TAU_H, level=_C, wires='q')
        hqml.QPS(P.omega_eb*P.S_b * TAU_H, level=_B, wires='q')
        hqml.QPS((P.omega_ea*P.S_a + P.omega_l*P.S_l) * TAU_H, level=_A, wires='q')
        hqml.QCD(P.omega_l *np.sqrt(P.S_l) * TAU_H, np.pi/2, level=_A, wires=['q','l'])
        hqml.QCD(P.omega_ec*np.sqrt(P.S_c) * TAU_H, np.pi/2, level=_C, wires=['q','c'])
        hqml.QCD(P.omega_eb*np.sqrt(P.S_b) * TAU_H, np.pi/2, level=_B, wires=['q','b'])
        hqml.QCD(P.omega_ea*np.sqrt(P.S_a) * TAU_H, np.pi/2, level=_A, wires=['q','a'])
        hqml.QCR(-2*P.chi_c * TAU_H, level=_C, wires=['q','c'])
        hqml.QCR(-2*P.chi_b * TAU_H, level=_B, wires=['q','b'])
        hqml.QCR(-2*P.chi_a * TAU_H, level=_A, wires=['q','a'])
        hqml.Rotation(P.omega_l  * TAU_H, wires='l')
        hqml.Rotation(P.omega_gc * TAU_H, wires='c')
        hqml.Rotation(P.omega_gb * TAU_H, wires='b')
        hqml.Rotation(P.omega_ga * TAU_H, wires='a')
        return hqml.state(wires=['q','a','b','c','l'])

    # FM 2026-08-25: remove title (per manuscript request).  Font size left
    # at the 'sketch' default -- 14 pt overflowed some of the tighter gate
    # boxes (QPS_{1,2,3}, QT_{1,2/3}) and pushed labels outside their boxes.
    fig, _ = hqml.draw_mpl(one_step, style='sketch')()

    gates = []
    src = {
        'free': 'H_free: omega_gR * I * n_R',
        'QCR_local': 'H_QCR: chi_R * Pi_R * n_R',
        'QCD_local': 'H_QCD: -omega_eR*sqrt(S_R) * Pi_R * x_R',
        'QCD_l_A':   'H_QCD: -omega_l*sqrt(S_l) * Pi_A * x_l',
        'QPS_A_fused': 'H_QPS (fused): (omega_ea*S_a + omega_l*S_l) * Pi_A',
        'QPS_S':      'H_QPS: omega_eR * S_R * Pi_R',
        'QT':        'H_QT: J carrier |R><R\'| + h.c.',
        'QT_sb':     'H_QTSB: J*eta*(|R><R\'| + h.c.)*x_l',
    }
    fwd = [
        ('Rotation', ['a'], [P.omega_ga*TAU_H], src['free']),
        ('Rotation', ['b'], [P.omega_gb*TAU_H], src['free']),
        ('Rotation', ['c'], [P.omega_gc*TAU_H], src['free']),
        ('Rotation', ['l'], [P.omega_l *TAU_H], src['free']),
        ('QCR', ['q','a'], [-2*P.chi_a*TAU_H, 'level=A'], src['QCR_local']),
        ('QCR', ['q','b'], [-2*P.chi_b*TAU_H, 'level=B'], src['QCR_local']),
        ('QCR', ['q','c'], [-2*P.chi_c*TAU_H, 'level=C'], src['QCR_local']),
        ('QCD', ['q','a'], [P.omega_ea*np.sqrt(P.S_a)*TAU_H, 'phi=pi/2', 'level=A'], src['QCD_local']),
        ('QCD', ['q','b'], [P.omega_eb*np.sqrt(P.S_b)*TAU_H, 'phi=pi/2', 'level=B'], src['QCD_local']),
        ('QCD', ['q','c'], [P.omega_ec*np.sqrt(P.S_c)*TAU_H, 'phi=pi/2', 'level=C'], src['QCD_local']),
        ('QCD', ['q','l'], [P.omega_l *np.sqrt(P.S_l)*TAU_H, 'phi=pi/2', 'level=A'], src['QCD_l_A']),
        ('QPS', ['q'], [(P.omega_ea*P.S_a + P.omega_l*P.S_l)*TAU_H, 'level=A'], src['QPS_A_fused']),
        ('QPS', ['q'], [P.omega_eb*P.S_b*TAU_H, 'level=B'], src['QPS_S']),
        ('QPS', ['q'], [P.omega_ec*P.S_c*TAU_H, 'level=C'], src['QPS_S']),
        ('QT',  ['q'], [P.J_AB0*TAU_H, 'level_i=A,level_j=B'], src['QT']),
        ('QT',  ['q'], [P.J_AC0*TAU_H, 'level_i=A,level_j=C'], src['QT']),
    ]
    center = [
        ('QT-SB', ['q','l'], [P.J_AB0*P.eta_AB*TAU, 'level_i=A,level_j=B,sideband'], src['QT_sb']),
        ('QT-SB', ['q','l'], [P.J_AC0*P.eta_AC*TAU, 'level_i=A,level_j=C,sideband'], src['QT_sb']),
    ]
    seq = fwd + center + list(reversed(fwd))
    for g, w, p, s in seq:
        gates.append({'gate': g, 'wires': w, 'params': p, 'source_term': s})

    return fig, gates, 34, ['simulations/trotter_method_a_fused_qps.py']


# ---------------------------------------------------------------------------
# QUBIT (Method B-qubit, 32 gates/step) -- via qiskit/c2qa
# ---------------------------------------------------------------------------
def make_qubit():
    from qiskit import QuantumRegister, QuantumCircuit
    import bosonic_qiskit as c2qa

    qmr = c2qa.QumodeRegister(num_qumodes=NUM_MODES, num_qubits_per_qumode=2)
    qbr = QuantumRegister(NUM_MODES - 1, name='q')   # 3 chromophore qubits
    ckt = c2qa.CVCircuit(qmr, qbr)

    omega_t = -TS_PS * omega_THz_arr
    delta_t = -TS_PS * delta_qa
    chi_t   = TS_PS * chi / 2.0
    g_cdt   = TS_PS * g_cd / 2.0 * 1j * (-1)
    g_cdl_t = -1j * g_cdl * TS_PS / 2.0
    g_at    = 2 * TS_PS * g_a_arr
    g_alt   = -1j * TS_PS * g_al_arr

    def free_rot(reverse=False):
        if not reverse:
            for i in range(NUM_MODES): ckt.cv_r(omega_t[i] / 2, qmr[i])
            for i in range(1, NUM_MODES - 1): ckt.rz(delta_t[i-1] / 2, qbr[i])
        else:
            for i in range(1, NUM_MODES - 1): ckt.rz(delta_t[i-1] / 2, qbr[i])
            for i in range(NUM_MODES): ckt.cv_r(omega_t[i] / 2, qmr[i])

    def chi_disp(reverse=False):
        order = range(NUM_MODES - 1) if not reverse else range(NUM_MODES - 2, -1, -1)
        for i in order:
            ckt.cv_c_r(chi_t[i] / 2, qmr[i], qbr[i])

    def cd_local(reverse=False):
        order = range(NUM_MODES - 1) if not reverse else range(NUM_MODES - 2, -1, -1)
        for i in order:
            ckt.cv_c_d(g_cdt[i] / 2, qmr[i], qbr[i])

    def cd_l_A(reverse=False):
        # mode-l conditional displacement on chromophore A: from H2XX+H2YY g_cdl/2
        ckt.cv_c_d(g_cdl_t / 2, qmr[3], qbr[0])

    def carrier(reverse=False):
        # 2 J flip-flops (XX+YY equivalent at TI native gate set)
        if not reverse:
            for i in range(2):
                ckt.rxx(g_at[i] / 4, qbr[0], qbr[1+i])
                ckt.ryy(g_at[i] / 4, qbr[0], qbr[1+i])
        else:
            for i in range(1, -1, -1):
                ckt.ryy(g_at[i] / 4, qbr[0], qbr[1+i])
                ckt.rxx(g_at[i] / 4, qbr[0], qbr[1+i])

    def sideband():
        # H_Jl: 2 sideband couplings, full tau (centered)
        for i in range(2):
            ckt.rxx(g_alt[i].real / 2, qbr[0], qbr[1+i])
            ckt.ryy(g_alt[i].real / 2, qbr[0], qbr[1+i])

    free_rot(); chi_disp(); cd_local(); cd_l_A(); carrier()
    sideband()
    carrier(reverse=True); cd_l_A(reverse=True); cd_local(reverse=True)
    chi_disp(reverse=True); free_rot(reverse=True)

    # Render
    fig = ckt.draw('mpl', fold=80, justify='left',
                   style={'name': 'iqx-dark', 'fontsize': 8})
    fig.suptitle(
        'qubit (Method B, H_cqed): one second-order Trotter step  |  '
        '4 cv_r + 2 r_z + 3 cv_c_r + 4 cv_c_d + 2 J carrier (x2)  +  '
        '2 J sideband  =  32 abstract gates per step',
        fontsize=8)

    gates = []
    src = {
        'free_cv_r': 'H_det: omega_R * n_R (4 cv_r)',
        'free_rz':  'H_det: -delta_R * sigma_z^R / 2 (single-qubit rz)',
        'chi':      'H_chi: -chi_R * n_R * sigma_z^R / 2',
        'g_cd':     'H_cd: g_cd_R * x_R * sigma_z^R / 2',
        'g_cdl':    'H_cd: g_cdl * x_l * sigma_z^A / 2 (from H_2XX + H_2YY)',
        'J':        'H_J: g_RR\' * (sxsx + sysy) / 2 carrier flip-flop',
        'Jl':       'H_Jl: g_RR\'l * (sxsx + sysy) * x_l / 2 sideband',
    }
    fwd = [
        ('cv_r', ['m_a'], [omega_t[0]/2], src['free_cv_r']),
        ('cv_r', ['m_b'], [omega_t[1]/2], src['free_cv_r']),
        ('cv_r', ['m_c'], [omega_t[2]/2], src['free_cv_r']),
        ('cv_r', ['m_l'], [omega_t[3]/2], src['free_cv_r']),
        ('rz',   ['q_B'], [delta_t[0]/2], src['free_rz']),
        ('rz',   ['q_C'], [delta_t[1]/2], src['free_rz']),
        ('cv_c_r', ['q_A','m_a'], [chi_t[0]/2], src['chi']),
        ('cv_c_r', ['q_B','m_b'], [chi_t[1]/2], src['chi']),
        ('cv_c_r', ['q_C','m_c'], [chi_t[2]/2], src['chi']),
        ('cv_c_d', ['q_A','m_a'], [g_cdt[0]/2], src['g_cd']),
        ('cv_c_d', ['q_B','m_b'], [g_cdt[1]/2], src['g_cd']),
        ('cv_c_d', ['q_C','m_c'], [g_cdt[2]/2], src['g_cd']),
        ('cv_c_d', ['q_A','m_l'], [g_cdl_t/2], src['g_cdl']),
        ('J(XX+YY)', ['q_A','q_B'], [g_at[0]/2], src['J']),
        ('J(XX+YY)', ['q_A','q_C'], [g_at[1]/2], src['J']),
    ]
    center = [
        ('Jl(XX+YY)x_l', ['q_A','q_B','m_l'], [g_alt[0]], src['Jl']),
        ('Jl(XX+YY)x_l', ['q_A','q_C','m_l'], [g_alt[1]], src['Jl']),
    ]
    seq = fwd + center + list(reversed(fwd))
    for g, w, p, s in seq:
        gates.append({'gate': g, 'wires': w, 'params': p, 'source_term': s})

    return fig, gates, 32, ['simulations/trotter_method_b_qubit.py']


# ---------------------------------------------------------------------------
# VU cQED (literal pipeline, ~96 gates/step) -- via c2qa/qiskit
# Mirrors 3_chromophore_vibronic_simulation.py H0/H1/H2XX/H2YY for ONE step
# ---------------------------------------------------------------------------
def make_vu():
    from qiskit import QuantumRegister, ClassicalRegister
    import bosonic_qiskit as c2qa

    qmr = c2qa.QumodeRegister(num_qumodes=NUM_MODES, num_qubits_per_qumode=2)
    qbr = QuantumRegister(NUM_MODES - 1, name='q')
    anc = QuantumRegister(1, name='a')
    cb  = ClassicalRegister(NUM_MODES, name='c')
    ckt = c2qa.CVCircuit(qmr, qbr, anc, cb)

    omega_t = -TS_PS * omega_THz_arr
    delta_t = -TS_PS * delta_qa
    chi_t   = TS_PS * chi / 2.0
    g_cdt   = TS_PS * g_cd / 2.0 * 1j * (-1)
    g_at    = 2 * TS_PS * g_a_arr
    g_alt   = -1j * TS_PS * g_al_arr

    # Gate counter alongside the JSON list
    json_gates = []
    src_map = {
        'H0':   'H_0 (eq 14): omega_R*n_R, -delta_RR\'*sigma_z/2',
        'H1':   'H_1 (eq 15): -chi_R*n_R*sigma_z/2 + g_cd_R*x_R*sigma_z/2',
        'H2XX': 'H_2XX (eq 16): g_cdl/4 sigma_a^x x_l + g_RR\' sigma_x sigma_x/2 + g_RR\'l sigma_x sigma_x x_l/2',
        'H2YY': 'H_2YY (eq 17): g_cdl/4 sigma_a^x x_l + g_RR\' sigma_y sigma_y/2 + g_RR\'l sigma_y sigma_y x_l/2',
    }

    def add(gate, wires, params, src):
        json_gates.append({'gate': gate, 'wires': wires,
                           'params': params, 'source_term': src})

    def H0(reverse=False):
        if not reverse:
            for i in range(NUM_MODES):
                ckt.cv_r(omega_t[i] / 2, qmr[i]); add('cv_r', [f'm_{i}'], [omega_t[i]/2], src_map['H0'])
            for i in range(1, NUM_MODES - 1):
                ckt.rz(delta_t[i-1] / 2, qbr[i]); add('rz', [f'q_{i}'], [delta_t[i-1]/2], src_map['H0'])
        else:
            for i in range(1, NUM_MODES - 1):
                ckt.rz(delta_t[i-1] / 2, qbr[i]); add('rz', [f'q_{i}'], [delta_t[i-1]/2], src_map['H0'])
            for i in range(NUM_MODES):
                ckt.cv_r(omega_t[i] / 2, qmr[i]); add('cv_r', [f'm_{i}'], [omega_t[i]/2], src_map['H0'])

    def H1(reverse=False):
        if not reverse:
            for i in range(NUM_MODES - 1):
                ckt.cv_c_r(chi_t[i] / 2, qmr[i], qbr[i]); add('cv_c_r', [f'q_{i}', f'm_{i}'], [chi_t[i]/2], src_map['H1'])
                ckt.cv_c_d(g_cdt[i] / 2, qmr[i], qbr[i]); add('cv_c_d', [f'q_{i}', f'm_{i}'], [g_cdt[i]/2], src_map['H1'])
        else:
            for i in range(NUM_MODES - 2, -1, -1):
                ckt.cv_c_d(g_cdt[i] / 2, qmr[i], qbr[i]); add('cv_c_d', [f'q_{i}', f'm_{i}'], [g_cdt[i]/2], src_map['H1'])
                ckt.cv_c_r(chi_t[i] / 2, qmr[i], qbr[i]); add('cv_c_r', [f'q_{i}', f'm_{i}'], [chi_t[i]/2], src_map['H1'])

    def H2XX(reverse=False):
        if not reverse:
            for i in range(2):
                ckt.rxx(g_at[i] / 4, qbr[0], qbr[1+i]); add('rxx', ['q_0', f'q_{1+i}'], [g_at[i]/4], src_map['H2XX'])
            ckt.swap(qbr[0], anc[0]);                add('swap', ['q_0','anc'], [], src_map['H2XX'])
            ckt.cv_c_d(-1j*g_cdl*TS_PS/4, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [-1j*g_cdl*TS_PS/4], src_map['H2XX'])
            ckt.swap(qbr[0], anc[0]);                add('swap', ['q_0','anc'], [], src_map['H2XX'])
            for i in range(1, 3):
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2XX'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2XX'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2XX'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2XX'])
                ckt.cv_c_d(g_alt[i-1] / 2, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [g_alt[i-1]/2], src_map['H2XX'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2XX'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2XX'])
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2XX'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2XX'])
        else:
            for i in range(1, 3)[::-1]:
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2XX'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2XX'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2XX'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2XX'])
                ckt.cv_c_d(g_alt[i-1] / 2, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [g_alt[i-1]/2], src_map['H2XX'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2XX'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2XX'])
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2XX'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2XX'])
            ckt.swap(qbr[0], anc[0]);    add('swap', ['q_0','anc'], [], src_map['H2XX'])
            ckt.cv_c_d(-1j*g_cdl*TS_PS/4, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [-1j*g_cdl*TS_PS/4], src_map['H2XX'])
            ckt.swap(qbr[0], anc[0]);    add('swap', ['q_0','anc'], [], src_map['H2XX'])
            for i in range(2)[::-1]:
                ckt.rxx(g_at[i] / 4, qbr[0], qbr[1+i]); add('rxx', ['q_0', f'q_{1+i}'], [g_at[i]/4], src_map['H2XX'])

    def H2YY(reverse=False):
        if not reverse:
            for i in range(2):
                ckt.ryy(g_at[i] / 4, qbr[0], qbr[1+i]); add('ryy', ['q_0', f'q_{1+i}'], [g_at[i]/4], src_map['H2YY'])
            ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
            ckt.cv_c_d(-1j*g_cdl*TS_PS/4, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [-1j*g_cdl*TS_PS/4], src_map['H2YY'])
            ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
            for i in range(1, 3):
                ckt.rz(-np.pi/2, qbr[0]); add('rz', ['q_0'], [-np.pi/2], src_map['H2YY'])
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2YY'])
                ckt.rz(-np.pi/2, qbr[i]); add('rz', [f'q_{i}'], [-np.pi/2], src_map['H2YY'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2YY'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2YY'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
                ckt.cv_c_d(g_alt[i-1] / 2, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [g_alt[i-1]/2], src_map['H2YY'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2YY'])
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2YY'])
                ckt.rz(np.pi/2, qbr[0]); add('rz', ['q_0'], [np.pi/2], src_map['H2YY'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2YY'])
                ckt.rz(np.pi/2, qbr[i]); add('rz', [f'q_{i}'], [np.pi/2], src_map['H2YY'])
        else:
            for i in range(1, 3)[::-1]:
                ckt.rz(np.pi/2, qbr[0]); add('rz', ['q_0'], [np.pi/2], src_map['H2YY'])
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2YY'])
                ckt.rz(np.pi/2, qbr[i]); add('rz', [f'q_{i}'], [np.pi/2], src_map['H2YY'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2YY'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2YY'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
                ckt.cv_c_d(g_alt[i-1] / 2, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [g_alt[i-1]/2], src_map['H2YY'])
                ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
                ckt.cx(qbr[i], qbr[0]); add('cx', [f'q_{i}','q_0'], [], src_map['H2YY'])
                ckt.h(qbr[0]); add('h', ['q_0'], [], src_map['H2YY'])
                ckt.rz(-np.pi/2, qbr[0]); add('rz', ['q_0'], [-np.pi/2], src_map['H2YY'])
                ckt.h(qbr[i]); add('h', [f'q_{i}'], [], src_map['H2YY'])
                ckt.rz(-np.pi/2, qbr[i]); add('rz', [f'q_{i}'], [-np.pi/2], src_map['H2YY'])
            ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
            ckt.cv_c_d(-1j*g_cdl*TS_PS/4, qmr[3], anc[0]); add('cv_c_d', ['anc','m_l'], [-1j*g_cdl*TS_PS/4], src_map['H2YY'])
            ckt.swap(qbr[0], anc[0]); add('swap', ['q_0','anc'], [], src_map['H2YY'])
            for i in range(2)[::-1]:
                ckt.ryy(g_at[i] / 4, qbr[0], qbr[1+i]); add('ryy', ['q_0', f'q_{1+i}'], [g_at[i]/4], src_map['H2YY'])

    H0(); H1(); H2XX(); H2YY()
    H2YY(reverse=True); H2XX(reverse=True); H1(reverse=True); H0(reverse=True)

    fig = ckt.draw('mpl', fold=120, justify='left',
                   style={'name': 'iqx-dark', 'fontsize': 6})
    fig.suptitle(
        f'Vu cQED literal pipeline: one second-order Trotter step  '
        f'(actual c2qa primitive count = {len(json_gates)};  '
        f'Vu paper "96 gates per step" reflects the abstract operator decomposition)',
        fontsize=8)

    return fig, json_gates, 96, ['legacy/vu_reference/cqed-vibronic-simulation-main/3_chromophore_vibronic_simulation.py']


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None,
                    help='Subset of impls to render (qudit qudit_d qubit vu)')
    args = ap.parse_args()

    targets = {
        'qudit':   make_qudit,
        'qudit_d': make_qudit_d,
        'qubit':   make_qubit,
        'vu':      make_vu,
    }
    selected = args.only if args.only else list(targets.keys())

    for name in selected:
        print(f"\n=== building circuit for: {name} ===")
        fig, gates, count, sources = targets[name]()
        fig_path, json_path = save_outputs(name, fig, gates, count, sources)
        print(f"  saved: {fig_path}.{{png,pdf,svg}}")
        print(f"  saved: {json_path}")
        print(f"  total gate count (abstract): {count}")
        print(f"  total gate count (concrete entries in JSON): {len(gates)}")


if __name__ == '__main__':
    main()
