"""hcqed_qudit.py -- H_cqed in qudit encoding (Method B reference Hamiltonian).

Qudit encoding of Vu eqs 14-17 obtained by substituting
    sigma_z_R = I_qudit - 2 * Pi_R
in the {|G>=0, |A>=1, |B>=2, |C>=3} basis. Within the single-excitation
manifold, sigma_x_R sigma_x_R' + sigma_y_R sigma_y_R' acts as
2 * (|R><R'| + |R'><R|).

Built directly from the Trotter Method B splitting in trotter_method_b.py,
but consumes parameters from params.py at call time.

Hilbert dim = 4 * N^4.
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P


def build_hcqed_qudit(N: int, override: dict | None = None):
    """Build H_cqed in qudit encoding (Method B exact reference).

    Returns (H, PiA, PiB, PiC, psi0) on the 4*N^4 space, with psi0 = |A,vac>.
    """
    def p(name):
        if override and name in override:
            return override[name]
        return getattr(P, name)

    omega_a = p('omega_a'); omega_b = p('omega_b')
    omega_c = p('omega_c'); omega_l = p('omega_l')
    chi_a = p('chi_a'); chi_b = p('chi_b'); chi_c = p('chi_c')
    g_cd_a = p('g_cd_a'); g_cd_b = p('g_cd_b'); g_cd_c = p('g_cd_c')
    g_cdl = p('g_cdl')
    delta_ab = p('delta_ab'); delta_ac = p('delta_ac')
    g_ab = p('g_ab'); g_ac = p('g_ac')
    g_abl = p('g_abl'); g_acl = p('g_acl')

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

    H = (
        omega_a*I_d*nad + omega_b*I_d*nbd + omega_c*I_d*ncd + omega_l*I_d*nld
      + delta_ab*PiB + delta_ac*PiC
      - chi_a/2*I_d*nad + chi_a*PiA*nad
      - chi_b/2*I_d*nbd + chi_b*PiB*nbd
      - chi_c/2*I_d*ncd + chi_c*PiC*ncd
      + g_cd_a/2*I_d*xad - g_cd_a*PiA*xad
      + g_cd_b/2*I_d*xbd - g_cd_b*PiB*xbd
      + g_cd_c/2*I_d*xcd - g_cd_c*PiC*xcd
      + g_cdl/2*I_d*xld  - g_cdl*PiA*xld
      + g_ab*(trans_d(_A,_B)+trans_d(_B,_A))
      + g_ac*(trans_d(_A,_C)+trans_d(_C,_A))
      + g_abl*(trans_d(_A,_B)+trans_d(_B,_A))*xld
      + g_acl*(trans_d(_A,_C)+trans_d(_C,_A))*xld
    )

    psi0 = qt.tensor(qt.basis(4, _A), qt.basis(N,0), qt.basis(N,0),
                     qt.basis(N,0), qt.basis(N,0))

    return H, PiA, PiB, PiC, psi0
