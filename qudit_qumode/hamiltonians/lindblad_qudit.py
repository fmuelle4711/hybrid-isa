"""lindblad_qudit.py -- Qudit-encoded Lindblad operators (Vu Table 2).

4-level qudit (G=0, A=1, B=2, C=3) + 4 bosonic modes (a, b, c, l).
Hilbert dim = 4 * N^4.

Translates Vu's per-chromophore qubit Lindblad operators to the qudit basis:

  Amplitude damping:  L_amp_R = sqrt(gamma_amp_R) * |G><R|
  Excitation:         L_exc_R = sqrt(gamma_exc_R) * |R><G|
  Dephasing:          L_dep_R = sqrt(gamma_dep_R) * (Pi_R - Pi_G)

The dephasing form (Pi_R - Pi_G) is the qudit equivalent of Vu's sigma_z_R
acting on the {|G>, |R>} subspace -- it phases |R> relative to |G> while
leaving the spectator chromophore states untouched.

All rates read from params module AT CALL TIME.
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P


def build_lindblad_ops_qudit(N: int, params_override: dict | None = None) -> list:
    """Build qudit-encoded Lindblad collapse operators.

    Args:
        N: Fock cutoff per mode.
        params_override: dict overriding any of the rate names
            ('gamma_amp_a', 'gamma_amp_b', 'gamma_amp_c',
             'gamma_exc_a', 'gamma_exc_b', 'gamma_exc_c',
             'gamma_dep_a', 'gamma_dep_b', 'gamma_dep_c').

    Returns:
        List of collapse operators (Qobj) on the full 4*N^4-dim space,
        suitable as c_ops for qutip.mesolve. Operators with zero rate
        are omitted.
    """
    def p(name):
        if params_override and name in params_override:
            return params_override[name]
        return getattr(P, name)

    rates_amp = {1: p('gamma_amp_a'), 2: p('gamma_amp_b'), 3: p('gamma_amp_c')}
    rates_exc = {1: p('gamma_exc_a'), 2: p('gamma_exc_b'), 3: p('gamma_exc_c')}
    rates_dep = {1: p('gamma_dep_a'), 2: p('gamma_dep_b'), 3: p('gamma_dep_c')}

    IN = qt.qeye(N)
    _G = 0  # ground qudit level

    def lift(op_q):
        """Tensor a 4x4 qudit op with I on each of the 4 modes."""
        return qt.tensor(op_q, IN, IN, IN, IN)

    def trans(i, j):
        return qt.basis(4, i) * qt.basis(4, j).dag()

    def proj(i):
        return qt.basis(4, i) * qt.basis(4, i).dag()

    Pi_G = proj(_G)

    c_ops = []

    # Amplitude damping: |G><R|
    for R, gamma in rates_amp.items():
        if gamma > 0.0:
            c_ops.append(np.sqrt(gamma) * lift(trans(_G, R)))

    # Excitation: |R><G|
    for R, gamma in rates_exc.items():
        if gamma > 0.0:
            c_ops.append(np.sqrt(gamma) * lift(trans(R, _G)))

    # Vu's qubit sigma_z^R in the single-excitation manifold acts as I - 2*Pi_R
    # (eigenvalues +1 on |G>,|B>,|C> and -1 on |R>). This dephases all
    # coherences involving |R>, including |R> <-> |R'> with R' != R.
    I4 = qt.qeye(4)
    for R, gamma in rates_dep.items():
        if gamma > 0.0:
            c_ops.append(np.sqrt(gamma) * lift(I4 - 2.0 * proj(R)))

    return c_ops
