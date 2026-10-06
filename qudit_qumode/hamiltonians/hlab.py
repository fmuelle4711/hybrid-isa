"""hlab.py -- H_lab (Eq. 1, raw molecular vibronic Hamiltonian, qudit encoding).

4-level qudit (G=0, A=1, B=2, C=3) + 4 bosonic modes (a, b, c, l).
Hilbert dim = 4 * N^4.

Uses completeness: PiG + PiA + PiB + PiC = I
  => omega_ga*I*na from spectator terms
  => omega_l*I*nl unconditional

All parameters read from params module AT CALL TIME (not import time).
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P


def build_hlab(N: int, override: dict | None = None):
    """Build H_lab in qudit encoding.

    Args:
        N: Fock cutoff per mode
        override: dict of parameter overrides, e.g. {"J_AB0": 0.0}
                  Keys must match params.py attribute names.

    Returns (H_lab, PiA, PiB, PiC, psi0) where:
      H_lab: full Hamiltonian (Qobj, dim = 4*N^4)
      PiA, PiB, PiC: projectors for populations
      psi0: initial state |A, 0, 0, 0, 0>
    """
    # Read parameters at call time
    def p(name):
        if override and name in override:
            return override[name]
        return getattr(P, name)

    omega_ga = p('omega_ga'); omega_ea = p('omega_ea')
    omega_gb = p('omega_gb'); omega_eb = p('omega_eb')
    omega_gc = p('omega_gc'); omega_ec = p('omega_ec')
    omega_l  = p('omega_l')
    S_a = p('S_a'); S_b = p('S_b'); S_c = p('S_c'); S_l = p('S_l')
    chi_a = p('chi_a'); chi_b = p('chi_b'); chi_c = p('chi_c')
    J_AB0 = p('J_AB0'); J_AC0 = p('J_AC0')
    eta_AB = p('eta_AB'); eta_AC = p('eta_AC')

    I4 = qt.qeye(4); IN = qt.qeye(N)
    _G, _A, _B, _C = 0, 1, 2, 3

    def proj_d(i):
        v = qt.basis(4, i); return qt.tensor(v*v.dag(), IN, IN, IN, IN)

    def trans_d(i, j):
        return qt.tensor(qt.basis(4,i)*qt.basis(4,j).dag(), IN, IN, IN, IN)

    def md_op(op, idx):
        ops = [I4]+[IN]*4; ops[1+idx] = op; return qt.tensor(ops)

    PiA = proj_d(_A); PiB = proj_d(_B); PiC = proj_d(_C)
    I_d = qt.tensor(I4, IN, IN, IN, IN)
    ad = md_op(qt.destroy(N),0); bd = md_op(qt.destroy(N),1)
    cd = md_op(qt.destroy(N),2); ld = md_op(qt.destroy(N),3)
    xad = ad+ad.dag(); xbd = bd+bd.dag(); xcd = cd+cd.dag(); xld = ld+ld.dag()
    nad = ad.dag()*ad; nbd = bd.dag()*bd; ncd = cd.dag()*cd; nld = ld.dag()*ld

    # Unconditional free rotations (from completeness)
    H_free = omega_ga*I_d*nad + omega_gb*I_d*nbd + omega_gc*I_d*ncd + omega_l*I_d*nld

    # Mode l couples ONLY to chromophore A (Vu eqs 2-3):
    #   h_A^e has omega_l*(l†l + S_l - sqrt(S_l)*(l+l†))
    #   h_B^g, h_C^g have omega_l*l†l only (= unconditional, already in H_free)
    # So mode l conditional terms appear ONLY on PiA.

    H_lab = (
        H_free
      + chi_a*PiA*nad + omega_ea*S_a*PiA - omega_ea*np.sqrt(S_a)*PiA*xad  # PiA local mode
      + omega_l*S_l*PiA - omega_l*np.sqrt(S_l)*PiA*xld                     # PiA mode l (no extra nld -- already in H_free)
      + chi_b*PiB*nbd + omega_eb*S_b*PiB - omega_eb*np.sqrt(S_b)*PiB*xbd  # PiB local mode
      + chi_c*PiC*ncd + omega_ec*S_c*PiC - omega_ec*np.sqrt(S_c)*PiC*xcd  # PiC local mode
      + J_AB0*(trans_d(_A,_B)+trans_d(_B,_A))
      + J_AC0*(trans_d(_A,_C)+trans_d(_C,_A))
      + J_AB0*eta_AB*(trans_d(_A,_B)+trans_d(_B,_A))*xld
      + J_AC0*eta_AC*(trans_d(_A,_C)+trans_d(_C,_A))*xld
    )

    psi0 = qt.tensor(qt.basis(4, _A), qt.basis(N,0), qt.basis(N,0),
                     qt.basis(N,0), qt.basis(N,0))

    return H_lab, PiA, PiB, PiC, psi0
