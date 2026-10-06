"""circuit_fig89_10_hqml_qnode_b_qubit.py -- Method B (3-qubit / qumode
encoding of Vu's H_cqed) dissipative trajectories for Fig 8 / 9 / 10
via Hybridlane qnodes.

Gate-for-gate port of ``simulations/vu_stack/method_b_qubit_vu_stack.py``
(which builds the same circuit in c2qa.CVCircuit directly) into a
Hybridlane qnode running through the ``bosonicqiskit.hybrid`` PennyLane
device. The point of the port is to validate that

* the new ``hqml.QubitReset`` primitive correctly implements
  measure-and-reset on a shared Stinespring ancilla, and
* the existing PennyLane <-> bosonic-qiskit lowering produces identical
  Trotter dynamics to Vu's direct c2qa script.

Wires
-----
* ``qb_a, qb_b, qb_c``  chromophore qubits (|0> = ground, |1> = excited).
* ``mode_a, mode_b, mode_c, mode_l``  four qumodes.
* ``anc``  one shared Stinespring ancilla qubit (reused via
  :class:`hqml.QubitReset` across all 6 channel calls per Trotter step,
  matching Vu's pattern).

Initial state: |A> chromophore (``qb_a`` flipped to |1>), all qumodes in
vacuum, ancilla in |0>.

Trotter step (symmetric 2nd order): forward [H0, H1, H2XX, H2YY] -> reverse,
matching c2qa H0/H1/H2XX/H2YY definitions term-for-term (signs verified
against the bosonic_qiskit -> Hybridlane gate map in
``hybridlane/src/hybridlane/devices/bosonic_qiskit/simulate.py``).

Per-step dissipation (Vu pattern, shared ancilla):
* amp_R:  CRY(theta_amp_R, qb_R -> anc) ; CNOT(anc -> qb_R) ; QubitReset(anc)
* dep_R:  RY(theta_dep_R, anc)          ; CZ(qb_R, anc)      ; QubitReset(anc)
with ``theta = 2 arcsin(sqrt(gamma * dt))`` for both channels (matches
Vu's K_jump = sqrt(p)|0><1| and L = I - 2P_1 in the qubit subspace).

CLI
---
    python -m simulations.circuit_fig89_10_hqml_qnode_b_qubit \\
        --panel fig8_top --fock 4 --shots 1000 --n-timepoints 5

Output
------
    data/fig{8,9,10}_hqml_b_qubit_qnode_{panel}_N{fock}_{shots}shots.npz
schema: {t_ps, PA, PB, PC, panel, fock, shots, wall}
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pennylane as qml
import hybridlane as hqml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

import params as P                                            # noqa: E402
from simulations.reproduce_vu_dissipative_figs import PANELS  # noqa: E402

DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Wire labels
# ----------------------------------------------------------------------

QB = ("qb_a", "qb_b", "qb_c")                 # chromophore qubits
MODES = ("mode_a", "mode_b", "mode_c", "mode_l")
ANC = "anc"

WIRES = list(QB) + list(MODES) + [ANC]

# ----------------------------------------------------------------------
# Trotter sub-Hamiltonians (gate-for-gate port of vu_stack/method_b_qubit_vu_stack.py)
# ----------------------------------------------------------------------

def _H0(tau: float, reverse: bool = False) -> None:
    """H_free: cv_r on each mode (sign +omega*tau in hqml because
    hqml.Rotation -> qc.cv_r(-theta)), plus RZ on qb_b/qb_c for the
    delta detunings."""
    omegas = (P.omega_a, P.omega_b, P.omega_c, P.omega_l)
    deltas = (P.delta_ab, P.delta_ac)
    body_modes = lambda: [hqml.Rotation(omegas[i] * tau, wires=MODES[i])
                          for i in range(4)]
    body_qb    = lambda: [qml.RZ(-deltas[i] * tau, wires=QB[i + 1])
                          for i in range(2)]
    if not reverse:
        body_modes(); body_qb()
    else:
        body_qb();    body_modes()


def _H1(tau: float, reverse: bool = False) -> None:
    """H_QCR + H_QCD per chromophore: cv_c_r(chi*tau/2) + cv_c_d(-1j*g_cd*tau/2)."""
    chis  = (P.chi_a, P.chi_b, P.chi_c)
    g_cds = (P.g_cd_a, P.g_cd_b, P.g_cd_c)
    indices = range(3) if not reverse else reversed(range(3))
    for i in indices:
        # cv_c_r(theta_qis = chi*tau/2)  <==>  hqml.CR(-2*theta_qis) = hqml.CR(-chi*tau)
        if reverse:
            hqml.ConditionalDisplacement(g_cds[i] * tau / 2.0, -np.pi / 2,
                                         wires=[QB[i], MODES[i]])
            hqml.ConditionalRotation(-chis[i] * tau,
                                     wires=[QB[i], MODES[i]])
        else:
            hqml.ConditionalRotation(-chis[i] * tau,
                                     wires=[QB[i], MODES[i]])
            # cv_c_d(-1j * g_cd * tau / 2): r=g_cd*tau/2, phi=-pi/2
            hqml.ConditionalDisplacement(g_cds[i] * tau / 2.0, -np.pi / 2,
                                         wires=[QB[i], MODES[i]])


def _H2XX(tau: float, reverse: bool = False) -> None:
    """RXX(qb_a, qb_{B,C}) coupling + ancilla-mediated mode-l displacement
    pieces, matching c2qa H2XX. Uses SWAP(qb_a, anc) twice to satisfy
    Vu's hardware-mapping convention -- equivalent under PennyLane, but
    preserved for faithfulness."""
    g_at = (P.J_AB0, P.J_AC0)           # because g_at[i]/2 = (2*tau*g_a[i])/2 = tau*g_a[i]
    g_alt = (-1j * P.J_AB0 * P.eta_AB, -1j * P.J_AC0 * P.eta_AC)
    g_cdl = P.g_cdl                     # real
    if not reverse:
        # rxx couplings
        for i in range(2):
            qml.IsingXX(g_at[i] * tau, wires=[QB[0], QB[1 + i]])
        # swap-mediated mode-l displacement on sigma_x_a
        qml.SWAP(wires=[QB[0], ANC])
        _val = -1j * g_cdl * tau / 4.0          # phi=-pi/2, r=g_cdl*tau/4
        hqml.ConditionalDisplacement(g_cdl * tau / 4.0, -np.pi / 2,
                                     wires=[ANC, MODES[3]])
        qml.SWAP(wires=[QB[0], ANC])
        # H * CNOT * cv_c_d * CNOT * H sandwiches for sigma_x_a sigma_x_{B,C}
        for i in (1, 2):
            qml.Hadamard(wires=QB[0]); qml.Hadamard(wires=QB[i])
            qml.CNOT(wires=[QB[i], QB[0]])
            qml.SWAP(wires=[QB[0], ANC])
            val = g_alt[i - 1] * tau / 2.0
            r, phi = float(np.abs(val)), float(np.angle(val))
            hqml.ConditionalDisplacement(r, phi, wires=[ANC, MODES[3]])
            qml.SWAP(wires=[QB[0], ANC])
            qml.CNOT(wires=[QB[i], QB[0]])
            qml.Hadamard(wires=QB[0]); qml.Hadamard(wires=QB[i])
    else:
        for i in (1, 2):
            r_idx = 2 - i + 1                   # mirror of forward: i=1->idx=2, i=2->idx=1
            # using c2qa convention: in reverse, i in (1,2) maps to r = NUM_MODES-1-i,
            # and g_alt[-i] -- here -1->index 1, -2->index 0
            qml.Hadamard(wires=QB[0]); qml.Hadamard(wires=QB[3 - i])
            qml.CNOT(wires=[QB[3 - i], QB[0]])
            qml.SWAP(wires=[QB[0], ANC])
            val = g_alt[-i] * tau / 2.0          # -1 -> index 1, -2 -> index 0
            r, phi = float(np.abs(val)), float(np.angle(val))
            hqml.ConditionalDisplacement(r, phi, wires=[ANC, MODES[3]])
            qml.SWAP(wires=[QB[0], ANC])
            qml.CNOT(wires=[QB[3 - i], QB[0]])
            qml.Hadamard(wires=QB[0]); qml.Hadamard(wires=QB[3 - i])
        qml.SWAP(wires=[QB[0], ANC])
        hqml.ConditionalDisplacement(g_cdl * tau / 4.0, -np.pi / 2,
                                     wires=[ANC, MODES[3]])
        qml.SWAP(wires=[QB[0], ANC])
        for i in range(2):
            r = 1 - i
            qml.IsingXX(g_at[1 - i] * tau, wires=[QB[0], QB[1 + r]])


def _H2YY(tau: float, reverse: bool = False) -> None:
    """Same as H2XX but the off-diagonal mode-l displacement gates are
    sandwiched by RZ +/- pi/2 + H to rotate the XX coupling into a YY
    coupling. (Vu's c2qa does this manually with rz/h/cx around cv_c_d.)"""
    g_at = (P.J_AB0, P.J_AC0)
    g_alt = (-1j * P.J_AB0 * P.eta_AB, -1j * P.J_AC0 * P.eta_AC)
    g_cdl = P.g_cdl
    if not reverse:
        for i in range(2):
            qml.IsingYY(g_at[i] * tau, wires=[QB[0], QB[1 + i]])
        qml.SWAP(wires=[QB[0], ANC])
        hqml.ConditionalDisplacement(g_cdl * tau / 4.0, -np.pi / 2,
                                     wires=[ANC, MODES[3]])
        qml.SWAP(wires=[QB[0], ANC])
        for i in (1, 2):
            qml.RZ(-np.pi / 2, wires=QB[0]); qml.Hadamard(wires=QB[0])
            qml.RZ(-np.pi / 2, wires=QB[i]); qml.Hadamard(wires=QB[i])
            qml.CNOT(wires=[QB[i], QB[0]])
            qml.SWAP(wires=[QB[0], ANC])
            val = g_alt[i - 1] * tau / 2.0
            r, phi = float(np.abs(val)), float(np.angle(val))
            hqml.ConditionalDisplacement(r, phi, wires=[ANC, MODES[3]])
            qml.SWAP(wires=[QB[0], ANC])
            qml.CNOT(wires=[QB[i], QB[0]])
            qml.Hadamard(wires=QB[0]); qml.RZ(np.pi / 2, wires=QB[0])
            qml.Hadamard(wires=QB[i]); qml.RZ(np.pi / 2, wires=QB[i])
    else:
        for i in (1, 2):
            qml.RZ(np.pi / 2, wires=QB[0]); qml.Hadamard(wires=QB[0])
            qml.RZ(np.pi / 2, wires=QB[3 - i]); qml.Hadamard(wires=QB[3 - i])
            qml.CNOT(wires=[QB[3 - i], QB[0]])
            qml.SWAP(wires=[QB[0], ANC])
            val = g_alt[-i] * tau / 2.0
            r, phi = float(np.abs(val)), float(np.angle(val))
            hqml.ConditionalDisplacement(r, phi, wires=[ANC, MODES[3]])
            qml.SWAP(wires=[QB[0], ANC])
            qml.CNOT(wires=[QB[3 - i], QB[0]])
            qml.Hadamard(wires=QB[0]); qml.RZ(-np.pi / 2, wires=QB[0])
            qml.Hadamard(wires=QB[3 - i]); qml.RZ(-np.pi / 2, wires=QB[3 - i])
        qml.SWAP(wires=[QB[0], ANC])
        hqml.ConditionalDisplacement(g_cdl * tau / 4.0, -np.pi / 2,
                                     wires=[ANC, MODES[3]])
        qml.SWAP(wires=[QB[0], ANC])
        for i in range(2):
            r = 1 - i
            qml.IsingYY(g_at[1 - i] * tau, wires=[QB[0], QB[1 + r]])


def one_trotter_step(tau_full: float) -> None:
    """Second-order symmetric Trotter step of H_cqed (qubit encoding)."""
    h = tau_full / 2.0
    _H0(h)
    _H1(h)
    _H2XX(h)
    _H2YY(h)
    _H2YY(h, reverse=True)
    _H2XX(h, reverse=True)
    _H1(h,  reverse=True)
    _H0(h,  reverse=True)


# ----------------------------------------------------------------------
# Stinespring channels (per-chromophore qubit, shared ancilla)
# ----------------------------------------------------------------------

def _theta(gamma: float, dt: float) -> float:
    """Vu's universal angle: 2 * arcsin(sqrt(gamma * dt)) for both amp and
    dep (the qubit Stinespring is identical for amp's CRY+CNOT and dep's
    RY+CZ patterns up to constant rescaling, modulo the operator type)."""
    p = float(np.clip(gamma * dt, 0.0, 1.0))
    return 2.0 * float(np.arcsin(np.sqrt(p)))


def apply_dissipation_step(rates: dict, dt: float) -> None:
    """Queue per-chromophore amp + dep Stinespring channels.

    Amp (R = chromophore A,B,C):
        CRY(theta, qb_R -> anc) ; CNOT(anc -> qb_R) ; QubitReset(anc)
    Dep:
        RY(theta, anc) ; CZ(qb_R, anc) ; QubitReset(anc)
    """
    amp = (rates.get("gamma_amp_a", 0.0),
           rates.get("gamma_amp_b", 0.0),
           rates.get("gamma_amp_c", 0.0))
    dep = (rates.get("gamma_dep_a", 0.0),
           rates.get("gamma_dep_b", 0.0),
           rates.get("gamma_dep_c", 0.0))
    if any(g > 0 for g in amp):
        for q, g in enumerate(amp):
            if g <= 0.0:
                continue
            qml.CRY(_theta(g, dt), wires=[QB[q], ANC])
            qml.CNOT(wires=[ANC, QB[q]])
            hqml.QubitReset(wires=ANC)
    if any(g > 0 for g in dep):
        for q, g in enumerate(dep):
            if g <= 0.0:
                continue
            qml.RY(_theta(g, dt), wires=ANC)
            qml.CZ(wires=[QB[q], ANC])
            hqml.QubitReset(wires=ANC)


# ----------------------------------------------------------------------
# Top-level qnode builder
# ----------------------------------------------------------------------

def make_qnode(fock_cutoff: int, n_steps: int, rates: dict, dt: float, shots: int):
    """Build a fresh Hybridlane qnode that returns the three chromophore
    excited-state populations P_A, P_B, P_C after ``n_steps`` Trotter
    steps with per-step Stinespring dissipation."""
    dev = qml.device(
        "bosonicqiskit.hybrid",
        wires=WIRES,
        max_fock_level=fock_cutoff,
        shots=shots,
    )

    @qml.qnode(dev)
    def run():
        # |A> on chromophore A; |0> on B and C
        qml.PauliX(wires=QB[0])
        # Type-inference hints (zero-angle ops so the static analyzer can
        # type wires we haven't yet touched at k=0)
        qml.RZ(0.0, wires=QB[1])
        qml.RZ(0.0, wires=QB[2])
        qml.RZ(0.0, wires=ANC)
        for m in MODES:
            hqml.Rotation(0.0, wires=m)

        for _ in range(int(n_steps)):
            one_trotter_step(dt)
            apply_dissipation_step(rates, dt)

        # Per-qubit Pauli-Z; population P_R(|1>) = (1 - <Z_R>) / 2.
        return (
            qml.expval(qml.PauliZ(wires=QB[0])),
            qml.expval(qml.PauliZ(wires=QB[1])),
            qml.expval(qml.PauliZ(wires=QB[2])),
        )

    return run


def populations_from_z(z_triplet) -> tuple:
    """Convert (<Z_A>, <Z_B>, <Z_C>) to (P_A, P_B, P_C)."""
    pA = 0.5 * (1.0 - float(z_triplet[0]))
    pB = 0.5 * (1.0 - float(z_triplet[1]))
    pC = 0.5 * (1.0 - float(z_triplet[2]))
    return pA, pB, pC


# ----------------------------------------------------------------------
# Per-panel runner (mirrors run_fig_dissipative_hqml_qnode in
# generate_figs_hqml_comparison.py but specialised to one panel)
# ----------------------------------------------------------------------

TAU_S = 5e-15
N_STEPS = 401


def _try_load_partial(path):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}
    except Exception as e:
        print(f"  [resume] cannot read {path}: {e}; deleting and starting fresh")
        try: os.remove(path)
        except OSError: pass
        return None


def _atomic_savez(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def run_panel(panel: str, fock_cutoff: int, shots: int, n_timepoints: int,
              out_path: str):
    if panel not in PANELS:
        raise KeyError(f"unknown panel {panel!r}; choose from {list(PANELS)}")
    rates = PANELS[panel]["rates"]

    k_grid = np.unique(np.round(
        np.linspace(0, N_STEPS - 1, n_timepoints)
    ).astype(int))
    t_ps = k_grid * TAU_S * 1e12
    n = len(k_grid)

    # ---- Resume state ----
    PA = np.zeros(n); PB = np.zeros(n); PC = np.zeros(n)
    start_idx = 0
    d = _try_load_partial(out_path)
    if d is not None:
        if ("t_ps" in d and d["t_ps"].shape == t_ps.shape
                and np.allclose(d["t_ps"], t_ps)):
            last = int(d.get("_last_done_idx", -1))
            if last >= 0:
                last = min(last, n - 1)
                PA = d["PA"].copy(); PB = d["PB"].copy(); PC = d["PC"].copy()
                start_idx = last + 1
                print(f"  [resume] picking up at idx={start_idx}/{n} "
                      f"(file: {os.path.basename(out_path)})")
        else:
            print(f"  [resume] existing {out_path} has different t-grid; deleting")
            try: os.remove(out_path)
            except OSError: pass

    print(f"[fig {PANELS[panel]['fig']} / b_qubit qnode + Stinespring / "
          f"panel={panel}]")
    print(f"  rates={rates}")
    print(f"  fock={fock_cutoff} shots={shots} k_grid={list(k_grid)}  "
          f"start_idx={start_idx}")

    if start_idx >= n:
        print(f"  all {n} timepoints already complete; nothing to do.")
        return t_ps, PA, PB, PC, 0.0

    save_meta = dict(
        panel=panel, fock=fock_cutoff, shots=shots,
        n_steps=N_STEPS, n_timepoints=n,
        backend="hqml_qnode_stinespring_b_qubit",
    )

    t_start = time.time()
    for idx in range(start_idx, n):
        k = int(k_grid[idx])
        t0 = time.time()
        qn = make_qnode(fock_cutoff, k, rates, TAU_S, shots)
        zs = qn()
        PA[idx], PB[idx], PC[idx] = populations_from_z(zs)
        _atomic_savez(
            out_path,
            t_ps=t_ps, PA=PA, PB=PB, PC=PC,
            _last_done_idx=np.int64(idx),
            **save_meta,
        )
        print(f"    t={t_ps[idx]:.4f}ps  k={k:3d}  PA={PA[idx]:.4f}  "
              f"PB={PB[idx]:.4f}  PC={PC[idx]:.4f}  ({time.time()-t0:.1f}s)")
    wall = time.time() - t_start
    print(f"  total wall: {wall:.1f}s")
    return t_ps, PA, PB, PC, wall


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=tuple(PANELS),
                    help="Dissipative panel to reproduce (fig8_*, fig9_*, fig10_*).")
    ap.add_argument("--fock",  type=int, default=4, choices=(2, 4, 8, 16),
                    help="Fock cutoff per qumode (power of 2).")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--n-timepoints", type=int, default=41,
                    help="Timepoints to sample over [0, 2 ps] (default 41).")
    args = ap.parse_args()

    fig_id = PANELS[args.panel]["fig"]
    out = os.path.join(
        DATA_DIR,
        f"fig{fig_id}_hqml_b_qubit_qnode_{args.panel}_N{args.fock}_{args.shots}shots.npz",
    )
    # No skip-if-exists: run_panel() resumes from an existing partial .npz
    # at `out` and exits immediately if all timepoints are done.

    t_ps, PA, PB, PC, wall = run_panel(
        args.panel, args.fock, args.shots, args.n_timepoints, out)

    # run_panel() saves after every timepoint; append the final wall time
    # without overwriting the per-timepoint metadata.
    with np.load(out, allow_pickle=True) as f:
        existing = {k: f[k] for k in f.files}
    existing["wall"] = np.asarray(wall)
    _atomic_savez(out, **existing)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
