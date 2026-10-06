"""circuit_fig6_hqml.py -- closed-system Fig 6 trajectory via Hybridlane.

Hybridlane qnode for Method A (H_lab) using NATIVE 4-LEVEL QUDIT gates --
the analog of `simulations/circuit_fig6_bq.py --method a`, but expressed
in hqml.QCR / QCD / QPS / QT / Rotation primitives instead of the c2qa
2-qubit-encoded form.

Runs through the bosonicqiskit.hybrid PennyLane device, which is backed
by bosonic_qiskit + qiskit-aer. Populations P_A(t), P_B(t), P_C(t) are
read out as expvals of hqml.QuditLevelProjector on the qudit wire,
sampled with `shots` (default 10 000) per timepoint.

Cost model: we build one fresh qnode per requested timepoint, each
applying `k` second-order Trotter steps to |A, vac, vac, vac, vac> and
returning 3 expvals. With 10 000 shots and Fock N=8, each qnode call
runs the full k-step circuit on qiskit-aer. To keep wall-time tractable
on ARC's 48 h walltime, the default time grid is 41 points (every
50 fs), not 401 (every 5 fs) -- the curves remain visually identical
on a Fig 6 panel.

Usage:
    python -m simulations.circuit_fig6_hqml --fock 4 --n-timepoints 21 --shots 1000
    python -m simulations.circuit_fig6_hqml --fock 8 --n-timepoints 41 --shots 10000

Output:
    data/fig6_circuit_hqml_method_a_N<fock>.npz
    Schema = {t, PA, PB, PC, shots, n_timepoints, fock,
              method='a', backend='hqml'}
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
import params as P

DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TAU_S       = 5e-15          # 5 fs in seconds (H in Hz)
T_MAX_S     = 2.0e-12        # 2 ps total
N_STEPS_MAX = 400            # full-resolution Trotter step count

# Qudit-level convention (matches hamiltonians/hlab.py and trotter_method_a.py)
_L_G, _L_A, _L_B, _L_C = 0, 1, 2, 3


def one_trotter_step(tau_full):
    """Apply one symmetric second-order Trotter step of H_lab using native
    hqml gates on wires ['q','a','b','c','l'].

    Matches the classical builder
        simulations.circuit_dissipative_kraus.build_sub_hamiltonians('A', N)
    term-for-term (verified by direct unitary comparison). The previous
    version of this function inherited two unrelated bugs from
    simulations/trotter_method_a.py's qnode (which was only ever used by
    hqml.draw_mpl, never as a simulator, so the bugs went undetected):

      (a) Wrong QCR / QPS scaling. Docstrings advertise
          QCR/QPS(theta) = exp(-i theta/2 G); the actual Hybridlane
          implementation (_qudit_gate_unitary_compute in
          devices/bosonic_qiskit/simulate.py) uses exp(-i theta G) with
          no 1/2 factor. The old code passed -2*coeff*tau_half which
          gave exp(+i 2 coeff tau_half G) -- wrong magnitude AND sign.
      (b) Six extra mode-l terms that have no counterpart in H_lab
          (mode-l QCR on Pi_B,Pi_C; mode-l QCD on Pi_B,Pi_C; mode-l QPS
          on Pi_B,Pi_C). H_lab couples mode l ONLY to Pi_A.

    Correct schedule mirroring classical H_lab (Method A):
        H_free  : 4 R-gates (one per mode, unconditional)
        H_QCR   : 3 QCR (chi_R Pi_R n_R; no mode-l terms)
        H_QCD   : 4 QCD (3 chromophore-local + 1 mode-l on Pi_A)
        H_QPS   : 4 QPS (3 chromophore-local + 1 mode-l on Pi_A)
        H_QT    : 2 carrier transitions
        H_QTSB  : 2 sideband transitions (middle of symmetric step, full tau)
      = 19 gates per half-step + 2 sideband = 40 gates per second-order
      step (consistent with the 36-gate count in writeup once you collapse
      forward/backward Identity ordering -- which we do NOT do here for
      clarity).
    """
    h = tau_full / 2.0   # half-step

    # --- H_free (4 R) ---
    hqml.Rotation(P.omega_ga * h, wires='a')
    hqml.Rotation(P.omega_gb * h, wires='b')
    hqml.Rotation(P.omega_gc * h, wires='c')
    hqml.Rotation(P.omega_l  * h, wires='l')

    # --- H_QCR (3 QCR; native exp(-i theta Pi_R n)) ---
    hqml.QCR(P.chi_a * h, level=_L_A, wires=['q', 'a'])
    hqml.QCR(P.chi_b * h, level=_L_B, wires=['q', 'b'])
    hqml.QCR(P.chi_c * h, level=_L_C, wires=['q', 'c'])

    # --- H_QCD (4 QCD; native exp(+i a Pi_R x_b) with phi=pi/2;
    #     H_QCD has  -omega_eR*sqrt(S_R)*Pi_R*x_R  so the Trotter sub-step
    #     wants exp(+i omega_eR sqrt(S_R) h Pi_R x_R), i.e. a = +coeff*h) ---
    hqml.QCD(P.omega_ea * np.sqrt(P.S_a) * h, np.pi / 2, level=_L_A, wires=['q', 'a'])
    hqml.QCD(P.omega_eb * np.sqrt(P.S_b) * h, np.pi / 2, level=_L_B, wires=['q', 'b'])
    hqml.QCD(P.omega_ec * np.sqrt(P.S_c) * h, np.pi / 2, level=_L_C, wires=['q', 'c'])
    hqml.QCD(P.omega_l  * np.sqrt(P.S_l) * h, np.pi / 2, level=_L_A, wires=['q', 'l'])

    # --- H_QPS (4 QPS; native exp(-i theta Pi_R)) ---
    hqml.QPS(P.omega_ea * P.S_a * h, level=_L_A, wires='q')
    hqml.QPS(P.omega_eb * P.S_b * h, level=_L_B, wires='q')
    hqml.QPS(P.omega_ec * P.S_c * h, level=_L_C, wires='q')
    hqml.QPS(P.omega_l  * P.S_l * h, level=_L_A, wires='q')   # mode-l contrib on Pi_A only

    # --- H_QT carrier (2 QT; native exp(-i theta (|i><j|+|j><i|))) ---
    hqml.QT(P.J_AB0 * h, level_i=_L_A, level_j=_L_B, wires='q')
    hqml.QT(P.J_AC0 * h, level_i=_L_A, level_j=_L_C, wires='q')

    # --- H_QTSB centre, full tau (2 sideband) ---
    hqml.QT(P.J_AB0 * P.eta_AB * tau_full, level_i=_L_A, level_j=_L_B,
            wires=['q', 'l'], sideband=True)
    hqml.QT(P.J_AC0 * P.eta_AC * tau_full, level_i=_L_A, level_j=_L_C,
            wires=['q', 'l'], sideband=True)

    # --- Backward half-step (reverse order) ---
    hqml.QT(P.J_AC0 * h, level_i=_L_A, level_j=_L_C, wires='q')
    hqml.QT(P.J_AB0 * h, level_i=_L_A, level_j=_L_B, wires='q')

    hqml.QPS(P.omega_l  * P.S_l * h, level=_L_A, wires='q')
    hqml.QPS(P.omega_ec * P.S_c * h, level=_L_C, wires='q')
    hqml.QPS(P.omega_eb * P.S_b * h, level=_L_B, wires='q')
    hqml.QPS(P.omega_ea * P.S_a * h, level=_L_A, wires='q')

    hqml.QCD(P.omega_l  * np.sqrt(P.S_l) * h, np.pi / 2, level=_L_A, wires=['q', 'l'])
    hqml.QCD(P.omega_ec * np.sqrt(P.S_c) * h, np.pi / 2, level=_L_C, wires=['q', 'c'])
    hqml.QCD(P.omega_eb * np.sqrt(P.S_b) * h, np.pi / 2, level=_L_B, wires=['q', 'b'])
    hqml.QCD(P.omega_ea * np.sqrt(P.S_a) * h, np.pi / 2, level=_L_A, wires=['q', 'a'])

    hqml.QCR(P.chi_c * h, level=_L_C, wires=['q', 'c'])
    hqml.QCR(P.chi_b * h, level=_L_B, wires=['q', 'b'])
    hqml.QCR(P.chi_a * h, level=_L_A, wires=['q', 'a'])

    hqml.Rotation(P.omega_l  * h, wires='l')
    hqml.Rotation(P.omega_gc * h, wires='c')
    hqml.Rotation(P.omega_gb * h, wires='b')
    hqml.Rotation(P.omega_ga * h, wires='a')


def make_qnode(fock_cutoff: int, n_steps: int):
    """Build a fresh qnode that applies `n_steps` Trotter steps to |A, vac>
    and returns the reduced 4x4 qudit density matrix.

    Statevector mode (shots=None) is used because the Hybridlane
    bosonicqiskit.hybrid device rejects `qml.expval(QuditLevelProjector)`
    and `qml.probs(wires='q')` with finite shots (device-level limitation,
    not a circuit problem). Shot noise is applied classically downstream
    via multinomial sampling -- exact populations from the statevector
    density matrix, then resampled with `shots` draws.
    """
    dev = qml.device(
        'bosonicqiskit.hybrid',
        wires=['q', 'a', 'b', 'c', 'l'],
        max_fock_level=fock_cutoff,
        shots=None,                           # statevector readout
    )

    @qml.qnode(dev)
    def fig6_at_step():
        # Initialize chromophore qudit to |A>. Native primitive: QT at
        # theta=pi/2 swaps |0> <-> |1> up to a global phase that doesn't
        # affect populations.
        hqml.QT(np.pi / 2, level_i=_L_G, level_j=_L_A, wires='q')

        # Apply zero-angle Rotations to each mode wire so Hybridlane's
        # static analyzer can infer their Qumode wire type even at
        # n_steps=0 (otherwise it errors with "Unable to infer wire
        # types for Wires([...])" when n_steps=0 means no QCR/QCD/etc.
        # fires).
        for mode in ('a', 'b', 'c', 'l'):
            hqml.Rotation(0.0, wires=mode)

        for _ in range(n_steps):
            one_trotter_step(TAU_S)

        # Hybridlane's bosonicqiskit.hybrid device requires hqml.state(...)
        # (the Hybridlane-aware state measurement; bare qml.state() / qml.
        # density_matrix() / qml.probs() are rejected by its measurement-
        # validator). We trace out the modes classically below.
        return hqml.state(wires=['q', 'a', 'b', 'c', 'l'])

    return fig6_at_step


def populations_from_state(state_vec, fock_cutoff: int, shots: int, rng):
    """Trace out modes from the full statevector, return diag(rho_qudit)
    populations and a shot-sampled version.

    Wire ordering on bosonicqiskit.hybrid: ['q','a','b','c','l'] -> qudit
    is the leftmost subsystem (dim 4), modes follow. The exact mode
    encoding may include padding qubits, so we don't hard-code the rest
    dimension -- we just reshape to (4, -1) and partial-trace.
    """
    psi = np.asarray(state_vec).ravel()
    if psi.size % 4 != 0:
        raise ValueError(
            f"unexpected state size {psi.size}; not a multiple of qudit dim 4")
    psi_mat = psi.reshape(4, -1)                         # (qudit, rest)
    rho_q = psi_mat @ psi_mat.conj().T                   # (4, 4)
    exact = np.clip(np.real(np.diag(rho_q)), 0.0, 1.0)
    exact = exact / exact.sum()
    samp = rng.multinomial(shots, exact) / shots
    return exact, samp


def _try_load_partial(path):
    """Open path as npz and return dict, or None if missing/corrupt."""
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


def run(fock_cutoff: int, n_timepoints: int, shots: int, out_path: str):
    """Loop over `n_timepoints` evenly spaced over [0, 2 ps]; per timepoint
    build a fresh qnode that applies k Trotter steps and return expvals.

    Saves cumulative results plus ``_last_done_idx`` to ``out_path`` after
    every timepoint. Re-running the script picks up at the first undone
    index instead of restarting from k=0.
    """
    k_grid = np.unique(np.round(
        np.linspace(0, N_STEPS_MAX, n_timepoints)
    ).astype(int))
    n = len(k_grid)
    t_ps = k_grid * TAU_S * 1e12

    # ---- Resume state ----
    PA = np.zeros(n); PB = np.zeros(n); PC = np.zeros(n)
    PA_exact = np.zeros(n); PB_exact = np.zeros(n); PC_exact = np.zeros(n)
    start_idx = 0
    d = _try_load_partial(out_path)
    if d is not None:
        # Validate t-grid matches (shape must agree before allclose)
        if ("t" in d and d["t"].shape == t_ps.shape
                and np.allclose(d["t"], t_ps)):
            last = int(d.get("_last_done_idx", -1))
            if last >= 0:
                last = min(last, n - 1)
                PA = d["PA"].copy(); PB = d["PB"].copy(); PC = d["PC"].copy()
                if "PA_exact" in d: PA_exact = d["PA_exact"].copy()
                if "PB_exact" in d: PB_exact = d["PB_exact"].copy()
                if "PC_exact" in d: PC_exact = d["PC_exact"].copy()
                start_idx = last + 1
                print(f"  [resume] picking up at idx={start_idx}/{n} "
                      f"(file: {os.path.basename(out_path)})")
        else:
            print(f"  [resume] existing {out_path} has different t-grid; deleting")
            try: os.remove(out_path)
            except OSError: pass

    rng = np.random.default_rng(0)

    print("=" * 70)
    print(f"circuit_fig6_hqml  --  method=a (native qudit)  "
          f"fock_cutoff={fock_cutoff}  n_timepoints={n}  shots={shots}  "
          f"start_idx={start_idx}")
    print("=" * 70)

    if start_idx >= n:
        print(f"  all {n} timepoints already complete; nothing to do.")
        return t_ps, PA, PB, PC, PA_exact, PB_exact, PC_exact, 0.0

    t0_total = time.time()
    for idx in range(start_idx, n):
        k = int(k_grid[idx])
        t0 = time.time()
        qnode = make_qnode(fock_cutoff, k)
        state_vec = qnode()
        exact, samp = populations_from_state(state_vec, fock_cutoff, shots, rng)
        PA_exact[idx], PB_exact[idx], PC_exact[idx] = exact[_L_A], exact[_L_B], exact[_L_C]
        PA[idx],       PB[idx],       PC[idx]       = samp[_L_A],  samp[_L_B],  samp[_L_C]
        _atomic_savez(
            out_path,
            t=t_ps, PA=PA, PB=PB, PC=PC,
            PA_exact=PA_exact, PB_exact=PB_exact, PC_exact=PC_exact,
            _last_done_idx=np.int64(idx),
            shots=shots, n_timepoints=n, fock=fock_cutoff,
            method='a', backend='hqml',
        )
        print(f"  t={t_ps[idx]:.4f}ps  (k={k:3d}, {time.time()-t0:.1f}s)  "
              f"P_A={PA[idx]:.4f}/{PA_exact[idx]:.4f}  "
              f"P_B={PB[idx]:.4f}/{PB_exact[idx]:.4f}  "
              f"P_C={PC[idx]:.4f}/{PC_exact[idx]:.4f}  (shots/exact)")

    wall = time.time() - t0_total
    print(f"\nTotal wall: {wall:.1f}s")
    return t_ps, PA, PB, PC, PA_exact, PB_exact, PC_exact, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fock", type=int, default=8,
                    help="Fock cutoff N per qumode (default 8)")
    ap.add_argument("--n-timepoints", type=int, default=41,
                    help="Number of Trotter timesteps to sample over "
                         "[0, 2 ps] (default 41 = every 50 fs)")
    ap.add_argument("--shots", type=int, default=10000)
    args = ap.parse_args()

    out = os.path.join(DATA_DIR, f"fig6_circuit_hqml_method_a_N{args.fock}.npz")
    # No skip-if-exists: run() now resumes from a partial .npz at out_path
    # and exits immediately if all timepoints are already done.

    t_ps, PA, PB, PC, PA_exact, PB_exact, PC_exact, wall = run(
        args.fock, args.n_timepoints, args.shots, out)

    # run() saves after every timepoint; append the final `wall` for
    # provenance without overwriting the per-timepoint metadata.
    with np.load(out, allow_pickle=True) as f:
        existing = {k: f[k] for k in f.files}
    existing["wall"] = np.asarray(wall)
    _atomic_savez(out, **existing)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
