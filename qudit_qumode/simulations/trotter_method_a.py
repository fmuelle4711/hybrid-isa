"""
trotter_method_a.py  --  Second-order Suzuki-Trotter circuit for Method A
=========================================================================
Implements H_qq_A (exact qudit encoding of H_lab) using Hybridlane gates.

Hamiltonian splitting:
  H_qq_A = H_QCR + H_QCD + H_QPS + H_QT + H_QTSB

Trotter step (second-order symmetric):
  U(tau) = QCR(tau/2) * QCD(tau/2) * QPS(tau/2)
         * QT(tau/2) * QTSB(tau)
         * QT(tau/2) * QPS(tau/2)
         * QCD(tau/2) * QCR(tau/2)

Gate count per step: 10 QCR + 6 QCD + 6 QPS + 2 QT + 2 QTSB = 26 gates

Usage:
    python -m simulations.trotter_method_a         # cutoff=8
    python -m simulations.trotter_method_a 4       # cutoff=4
    python -m simulations.trotter_method_a 6       # cutoff=6
"""

import numpy as np
import re, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import qutip as qt
import os, time
from scipy.interpolate import interp1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
def dp(f): return os.path.join(DATA_DIR, f)

# ================================================================== #
# RAW PARAMETERS (Table 1)                                             #
# ================================================================== #
omega_ga = 4.95e13;  omega_ea = 4.63e13
omega_gb = 4.98e13;  omega_eb = 4.62e13
omega_gc = 4.92e13;  omega_ec = 4.65e13
omega_l  = 6.00e12
J_AB0 = 3.00e12;  J_AC0 = 2.70e12
eta_AB = -0.1;     eta_AC = 0.15
S_a = 0.005;  S_b = 0.004;  S_c = 0.006;  S_l = 0.05

chi_a = omega_ea - omega_ga
chi_b = omega_eb - omega_gb
chi_c = omega_ec - omega_gc

# ================================================================== #
# SIMULATION SETTINGS                                                  #
# ================================================================== #
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TAU = 0.005e-12   # 5 fs in seconds (Hamiltonian in Hz)
N_STEPS = 400
T_MAX = 2.0       # ps

print("="*70)
print(f"Trotter Method A  --  cutoff={N}, tau={TAU*1e15:.0f}fs, {N_STEPS} steps")
print("="*70)

# ================================================================== #
# QUDIT OPERATORS                                                      #
# ================================================================== #
I4 = qt.qeye(4); IN = qt.qeye(N)
_G, _A, _B, _C = 0, 1, 2, 3

def proj_d(i):
    v = qt.basis(4, i); return qt.tensor(v*v.dag(), IN, IN, IN, IN)

def trans_d(i, j):
    return qt.tensor(qt.basis(4,i)*qt.basis(4,j).dag(), IN, IN, IN, IN)

def md_op(op, idx):
    ops = [I4]+[IN]*4; ops[1+idx] = op; return qt.tensor(ops)

PiG = proj_d(_G); PiA = proj_d(_A); PiB = proj_d(_B); PiC = proj_d(_C)
I_d = qt.tensor(I4, IN, IN, IN, IN)
ad = md_op(qt.destroy(N),0); bd = md_op(qt.destroy(N),1)
cd = md_op(qt.destroy(N),2); ld = md_op(qt.destroy(N),3)
xad = ad+ad.dag(); xbd = bd+bd.dag(); xcd = cd+cd.dag(); xld = ld+ld.dag()
nad = ad.dag()*ad; nbd = bd.dag()*bd; ncd = cd.dag()*cd; nld = ld.dag()*ld

psi0 = qt.tensor(qt.basis(4, _A), qt.basis(N,0), qt.basis(N,0),
                 qt.basis(N,0), qt.basis(N,0))

# ================================================================== #
# BUILD H_qq_A (for exact reference)                                   #
#                                                                      #
# Completeness relation: PiG + PiA + PiB + PiC = I                    #
# omega_ga*I*nad comes from spectator terms:                           #
#   |G>: omega_ga*nad, |A>: omega_ga is part of omega_ea=omega_ga+chi #
#   |B>: omega_ga*nad (spectator), |C>: omega_ga*nad (spectator)      #
#   Total: omega_ga*(PiG+PiB+PiC)*nad + omega_ea*PiA*nad              #
#        = omega_ga*I*nad + chi_a*PiA*nad                              #
# omega_l*I*nld comes from:                                            #
#   (PiG+PiA+PiB+PiC)*omega_l*nld = I*omega_l*nld -- unconditional   #
# ================================================================== #

# Unconditional free rotations (from completeness: PiG+PiA+PiB+PiC = I)
H_free_a = omega_ga*I_d*nad + omega_gb*I_d*nbd + omega_gc*I_d*ncd + omega_l*I_d*nld

# Mode l couples ONLY to chromophore A (Vu eqs 2-3).
# h_B^g and h_C^g have omega_l*l†l only = unconditional, already in H_free_a.
# No extra PiB*nld or PiC*nld terms.

H_qq_A = (
    H_free_a
  + chi_a*PiA*nad + omega_ea*S_a*PiA - omega_ea*np.sqrt(S_a)*PiA*xad   # PiA local mode
  + omega_l*S_l*PiA - omega_l*np.sqrt(S_l)*PiA*xld                      # PiA mode l (no extra nld)
  + chi_b*PiB*nbd + omega_eb*S_b*PiB - omega_eb*np.sqrt(S_b)*PiB*xbd   # PiB local mode
  + chi_c*PiC*ncd + omega_ec*S_c*PiC - omega_ec*np.sqrt(S_c)*PiC*xcd   # PiC local mode
  + J_AB0*(trans_d(_A,_B)+trans_d(_B,_A)) + J_AC0*(trans_d(_A,_C)+trans_d(_C,_A))
  + J_AB0*eta_AB*(trans_d(_A,_B)+trans_d(_B,_A))*xld
  + J_AC0*eta_AC*(trans_d(_A,_C)+trans_d(_C,_A))*xld
)

# Cross-check: verify inline H_qq_A matches hamiltonians/hlab.py
from hamiltonians.hlab import build_hlab as _build_hlab
_H_ref, _, _, _, _ = _build_hlab(N)
_ref_err = np.max(np.abs((H_qq_A - _H_ref).diag()))
print(f"  Cross-check vs hlab.py: max|diag diff| = {_ref_err:.2e}  {'OK' if _ref_err < 1e-6 else 'MISMATCH!'}")

# ================================================================== #
# BUILD TROTTER SUB-HAMILTONIANS                                       #
# Using completeness: PiG + PiA + PiB + PiC = I                        #
#                                                                      #
# H_free: unconditional free rotations (from completeness)             #
#   omega_ga*I*nad, omega_gb*I*nbd, omega_gc*I*ncd, omega_l*I*nld      #
#   Mode l is fully unconditional (same freq on all electronic states)  #
# H_QCR: conditional dispersive shifts (chi_R on local mode only)       #
#   No mode-l QCR — mode l has no excited-state frequency shift         #
# ================================================================== #

# H_free: 4 unconditional Rotation gates (happen naturally on hardware)
H_free = (omega_ga*I_d*nad + omega_gb*I_d*nbd + omega_gc*I_d*ncd + omega_l*I_d*nld)

# H_QCR: 3 conditional QCR gates (local dispersive shifts only)
H_QCR = (chi_a*PiA*nad + chi_b*PiB*nbd + chi_c*PiC*ncd)

# H_QCD: 4 QCD gates (vibronic displacement: 3 local + mode l for A only)
H_QCD = (
    - omega_ea*np.sqrt(S_a)*PiA*xad
    - omega_l*np.sqrt(S_l)*PiA*xld    # mode l only for A
    - omega_eb*np.sqrt(S_b)*PiB*xbd
    - omega_ec*np.sqrt(S_c)*PiC*xcd
)

# H_QPS: 4 QPS gates (constant energy offsets: 3 local + mode l for A only)
H_QPS = (
    omega_ea*S_a*PiA + omega_l*S_l*PiA  # mode l only for A
  + omega_eb*S_b*PiB
  + omega_ec*S_c*PiC
)

# H_QT: 2 carrier gates
H_QT = (
    J_AB0*(trans_d(_A,_B)+trans_d(_B,_A))
  + J_AC0*(trans_d(_A,_C)+trans_d(_C,_A))
)

# H_QTSB: 2 sideband gates
H_QTSB = (
    J_AB0*eta_AB*(trans_d(_A,_B)+trans_d(_B,_A))*xld
  + J_AC0*eta_AC*(trans_d(_A,_C)+trans_d(_C,_A))*xld
)

# Verify: H_qq_A = H_free + H_QCR + H_QCD + H_QPS + H_QT + H_QTSB
H_sum = H_free + H_QCR + H_QCD + H_QPS + H_QT + H_QTSB
diff_diag = (H_qq_A - H_sum).diag()
print(f"\nSplitting check: max|diag(H_qq_A - H_sum)| = {np.max(np.abs(diff_diag)):.2e}")

# ================================================================== #
# GATE COUNTS                                                          #
# ================================================================== #
n_free = 4   # H_free terms (omega_gr*I*nr, one per mode)
n_QCR  = 3   # H_QCR terms (chi_R*PiR*nR, local modes only)
n_QCD  = 4   # H_QCD terms (3 local + mode l for A only)
n_QPS  = 4   # H_QPS terms (3 local + mode l for A only)
n_QT   = 2   # H_QT carrier terms
n_QTSB = 2   # H_QTSB sideband terms

n_first_order  = n_free + n_QCR + n_QCD + n_QPS + n_QT + n_QTSB
n_half_step    = n_free + n_QCR + n_QCD + n_QPS + n_QT
n_second_order = 2 * n_half_step + n_QTSB

print(f"Gate count -- first order:  {n_first_order}")
print(f"Gate count -- second order: {n_second_order}")

# ================================================================== #
# PRE-COMPUTE PROPAGATORS                                              #
# Second-order Trotter:                                                 #
# U = free(h)*QCR(h)*QCD(h)*QPS(h)*QT(h) * QTSB(f) *                 #
#     QT(h)*QPS(h)*QCD(h)*QCR(h)*free(h)                              #
# ================================================================== #
dim = 4 * N**4
USE_DENSE = (dim <= 1024)  # dense expm for cutoff<=4, sesolve for larger

if USE_DENSE:
    print(f"\nPre-computing dense propagators (dim={dim})...")
    t0 = time.time()
    from scipy.linalg import expm as dense_expm
    def make_U(H, dt):
        return qt.Qobj(dense_expm(-1j * H.full() * dt), dims=H.dims)
    U_free_h = make_U(H_free, TAU/2)
    U_QCR_h  = make_U(H_QCR,  TAU/2)
    U_QCD_h  = make_U(H_QCD,  TAU/2)
    U_QPS_h  = make_U(H_QPS,  TAU/2)
    U_QT_h   = make_U(H_QT,   TAU/2)
    U_QTSB_f = make_U(H_QTSB, TAU)
    U_fwd_h = U_free_h * U_QCR_h * U_QCD_h * U_QPS_h * U_QT_h
    U_bwd_h = U_QT_h * U_QPS_h * U_QCD_h * U_QCR_h * U_free_h
    U_step = U_fwd_h * U_QTSB_f * U_bwd_h
    print(f"  done in {time.time()-t0:.1f}s")
else:
    print(f"\nUsing sesolve sub-steps (dim={dim}, too large for dense expm)")
    sesolve_opts = {'nsteps': 5000}

# ================================================================== #
# DRAW CIRCUIT FOR ONE TROTTER STEP                                    #
# ================================================================== #
print("\nDrawing one Trotter step circuit...")
import pennylane as qml
import hybridlane as hqml

# Qudit level indices for Hybridlane (device-specific mapping)
# G=0, A=1, B=2, C=3 in our convention
# But Hybridlane uses the physical qudit encoding — check stage2_single_shot.py
# Legacy used _IDX_A=3, _IDX_B=1, _IDX_C=2 but that was for old convention.
# For our G=0,A=1,B=2,C=3 mapping, use level= directly.
_L_G, _L_A, _L_B, _L_C = 0, 1, 2, 3

# TAU in THz*ps units for Hybridlane (scale Hz*s = dimensionless -> THz*ps = dimensionless)
# Our H is in Hz, TAU in seconds. The product H*TAU is dimensionless (rad).
# Gate parameters are just the dimensionless angles.
tau_half = TAU / 2
tau_full = TAU

dev = qml.device('bosonicqiskit.hybrid', wires=['q','a','b','c','l'], max_fock_level=N)

@qml.qnode(dev)
def one_trotter_step():
    qml.PauliX(wires='q')  # prepare |A> state

    # === FORWARD HALF-STEP ===

    # H_free: 4 Rotation gates (unconditional, free on trapped-ion)
    hqml.Rotation(omega_ga * tau_half, wires='a')
    hqml.Rotation(omega_gb * tau_half, wires='b')
    hqml.Rotation(omega_gc * tau_half, wires='c')
    hqml.Rotation(omega_l  * tau_half, wires='l')

    # H_QCR: 6 QCR gates (3 local dispersive + 3 mode l per excited state)
    # QCR(theta, level) = exp[-i*(theta/2)*|l><l|*n], theta = -2*coeff*tau/2
    hqml.QCR(-2*chi_a   * tau_half, level=_L_A, wires=['q','a'])
    hqml.QCR(-2*omega_l  * tau_half, level=_L_A, wires=['q','l'])
    hqml.QCR(-2*chi_b   * tau_half, level=_L_B, wires=['q','b'])
    hqml.QCR(-2*omega_l  * tau_half, level=_L_B, wires=['q','l'])
    hqml.QCR(-2*chi_c   * tau_half, level=_L_C, wires=['q','c'])
    hqml.QCR(-2*omega_l  * tau_half, level=_L_C, wires=['q','l'])

    # H_QCD: 6 QCD gates (3 local modes + 3 mode l)
    # QCD(a, phi=pi/2, level): generator = -a*|l><l|*(r+r†)
    # For H term -coeff*PiR*xr, need a = coeff*tau/2
    hqml.QCD(omega_ea*np.sqrt(S_a) * tau_half, np.pi/2, level=_L_A, wires=['q','a'])
    hqml.QCD(omega_l*np.sqrt(S_l)  * tau_half, np.pi/2, level=_L_A, wires=['q','l'])
    hqml.QCD(omega_eb*np.sqrt(S_b) * tau_half, np.pi/2, level=_L_B, wires=['q','b'])
    hqml.QCD(omega_l*np.sqrt(S_l)  * tau_half, np.pi/2, level=_L_B, wires=['q','l'])
    hqml.QCD(omega_ec*np.sqrt(S_c) * tau_half, np.pi/2, level=_L_C, wires=['q','c'])
    hqml.QCD(omega_l*np.sqrt(S_l)  * tau_half, np.pi/2, level=_L_C, wires=['q','l'])

    # H_QPS: 6 QPS gates (3 local + 3 mode l)
    # QPS(theta, level) = exp[-i*theta*|l><l|], theta = coeff*tau/2
    hqml.QPS(omega_ea*S_a * tau_half, level=_L_A, wires='q')
    hqml.QPS(omega_l*S_l  * tau_half, level=_L_A, wires='q')
    hqml.QPS(omega_eb*S_b * tau_half, level=_L_B, wires='q')
    hqml.QPS(omega_l*S_l  * tau_half, level=_L_B, wires='q')
    hqml.QPS(omega_ec*S_c * tau_half, level=_L_C, wires='q')
    hqml.QPS(omega_l*S_l  * tau_half, level=_L_C, wires='q')

    # H_QT: 2 QT carrier gates (half tau)
    hqml.QT(J_AB0 * tau_half, level_i=_L_A, level_j=_L_B, wires='q')
    hqml.QT(J_AC0 * tau_half, level_i=_L_A, level_j=_L_C, wires='q')

    # === MIDDLE: H_QTSB full tau ===
    # 2 QT sideband gates
    hqml.QT(J_AB0*eta_AB * tau_full, level_i=_L_A, level_j=_L_B, wires=['q','l'], sideband=True)
    hqml.QT(J_AC0*eta_AC * tau_full, level_i=_L_A, level_j=_L_C, wires=['q','l'], sideband=True)

    # === BACKWARD HALF-STEP (reverse order) ===

    # H_QT: 2 QT carrier gates
    hqml.QT(J_AC0 * tau_half, level_i=_L_A, level_j=_L_C, wires='q')
    hqml.QT(J_AB0 * tau_half, level_i=_L_A, level_j=_L_B, wires='q')

    # H_QPS: 6 QPS gates (reverse)
    hqml.QPS(omega_l*S_l  * tau_half, level=_L_C, wires='q')
    hqml.QPS(omega_ec*S_c * tau_half, level=_L_C, wires='q')
    hqml.QPS(omega_l*S_l  * tau_half, level=_L_B, wires='q')
    hqml.QPS(omega_eb*S_b * tau_half, level=_L_B, wires='q')
    hqml.QPS(omega_l*S_l  * tau_half, level=_L_A, wires='q')
    hqml.QPS(omega_ea*S_a * tau_half, level=_L_A, wires='q')

    # H_QCD: 6 QCD gates (reverse)
    hqml.QCD(omega_l*np.sqrt(S_l)  * tau_half, np.pi/2, level=_L_C, wires=['q','l'])
    hqml.QCD(omega_ec*np.sqrt(S_c) * tau_half, np.pi/2, level=_L_C, wires=['q','c'])
    hqml.QCD(omega_l*np.sqrt(S_l)  * tau_half, np.pi/2, level=_L_B, wires=['q','l'])
    hqml.QCD(omega_eb*np.sqrt(S_b) * tau_half, np.pi/2, level=_L_B, wires=['q','b'])
    hqml.QCD(omega_l*np.sqrt(S_l)  * tau_half, np.pi/2, level=_L_A, wires=['q','l'])
    hqml.QCD(omega_ea*np.sqrt(S_a) * tau_half, np.pi/2, level=_L_A, wires=['q','a'])

    # H_QCR: 6 QCR gates (reverse)
    hqml.QCR(-2*omega_l  * tau_half, level=_L_C, wires=['q','l'])
    hqml.QCR(-2*chi_c   * tau_half, level=_L_C, wires=['q','c'])
    hqml.QCR(-2*omega_l  * tau_half, level=_L_B, wires=['q','l'])
    hqml.QCR(-2*chi_b   * tau_half, level=_L_B, wires=['q','b'])
    hqml.QCR(-2*omega_l  * tau_half, level=_L_A, wires=['q','l'])
    hqml.QCR(-2*chi_a   * tau_half, level=_L_A, wires=['q','a'])

    # H_free: 4 Rotation gates (reverse)
    hqml.Rotation(omega_l  * tau_half, wires='l')
    hqml.Rotation(omega_gc * tau_half, wires='c')
    hqml.Rotation(omega_gb * tau_half, wires='b')
    hqml.Rotation(omega_ga * tau_half, wires='a')

    return hqml.state(wires=['q','a','b','c','l'])

fig_circ, _ = hqml.draw_mpl(one_trotter_step, style='sketch')()
fig_circ.suptitle(f'Method A: One Trotter step  |  ({n_free} Rot + {n_QCR} QCR + {n_QCD} QCD + {n_QPS} QPS + {n_QT} QT) x2 + {n_QTSB} QTSB = {n_second_order} total',
                  fontsize=8)
fig_circ.tight_layout()
fig_circ.savefig(dp('circuit_method_a.png'), dpi=200, bbox_inches='tight')
fig_circ.savefig(dp('circuit_method_a.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig_circ)
print("  Saved circuit diagram to data/circuit_method_a.png")

# ================================================================== #
# RUN TROTTER                                                          #
# ================================================================== #
print(f"\nRunning Trotter ({N_STEPS} steps)...")
t0 = time.time()

tlist_tr = np.array([i*TAU*1e12 for i in range(N_STEPS+1)])  # in ps
psi = psi0.copy()
PA_tr = [float(qt.expect(PiA, psi))]
PB_tr = [float(qt.expect(PiB, psi))]
PC_tr = [float(qt.expect(PiC, psi))]

def trotter_step_sesolve(psi_in):
    """One symmetric Trotter step via sesolve sub-steps."""
    th = [0, TAU/2]; tf = [0, TAU]
    psi_in = qt.sesolve(H_free, psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QCR,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QCD,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QPS,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QT,   psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QTSB, psi_in, tf, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QT,   psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QPS,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QCD,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_QCR,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_free, psi_in, th, options=sesolve_opts).final_state
    return psi_in

for step in range(N_STEPS):
    if USE_DENSE:
        psi = U_step * psi
    else:
        psi = trotter_step_sesolve(psi)
    PA_tr.append(float(qt.expect(PiA, psi)))
    PB_tr.append(float(qt.expect(PiB, psi)))
    PC_tr.append(float(qt.expect(PiC, psi)))
    if (step+1) % 100 == 0:
        print(f"  Step {step+1}/{N_STEPS}  P_A={PA_tr[-1]:.3f}  ({time.time()-t0:.0f}s)")

PA_tr = np.array(PA_tr); PB_tr = np.array(PB_tr); PC_tr = np.array(PC_tr)
print(f"  done in {time.time()-t0:.1f}s")
print(f"  P_A(T)={PA_tr[-1]:.4f}  P_B={PB_tr[-1]:.4f}  P_C={PC_tr[-1]:.4f}")

# ================================================================== #
# EXACT SESOLVE REFERENCE                                              #
# ================================================================== #
print(f"\nRunning exact sesolve...")
t0 = time.time()
tlist_ex = np.linspace(0, T_MAX, 500)
tlist_ex_s = tlist_ex * 1e-12

res_ex = qt.sesolve(H_qq_A, psi0, tlist_ex_s, e_ops=[PiA, PiB, PiC],
                    options={'nsteps': 100000})
PA_ex = np.array(res_ex.expect[0])
PB_ex = np.array(res_ex.expect[1])
PC_ex = np.array(res_ex.expect[2])
print(f"  done in {time.time()-t0:.1f}s")
print(f"  P_A(T)={PA_ex[-1]:.4f}  P_B={PB_ex[-1]:.4f}  P_C={PC_ex[-1]:.4f}")

# ================================================================== #
# RMSE                                                                 #
# ================================================================== #
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

PA_ex_i = interp1d(tlist_ex, PA_ex)(tlist_tr)
PB_ex_i = interp1d(tlist_ex, PB_ex)(tlist_tr)
PC_ex_i = interp1d(tlist_ex, PC_ex)(tlist_tr)

rA = rmse(PA_tr, PA_ex_i)
rB = rmse(PB_tr, PB_ex_i)
rC = rmse(PC_tr, PC_ex_i)

print(f"\nRMSE (Trotter vs exact):  A={rA:.6f}  B={rB:.6f}  C={rC:.6f}")

# ================================================================== #
# SAVE DATA                                                            #
# ================================================================== #
out_npz = dp(f'trotter_method_a_fock{N}.npz')
np.savez(out_npz,
         t=tlist_tr, PA=PA_tr, PB=PB_tr, PC=PC_tr,
         PA_exact=PA_ex, PB_exact=PB_ex, PC_exact=PC_ex,
         t_exact=tlist_ex,
         gate_count=n_second_order,
         rmse=np.array([rA, rB, rC]))
print(f"Saved: {out_npz}")

# ================================================================== #
# LOAD VU TROTTER DATA                                                 #
# ================================================================== #
vu_path = dp('vu_reference/idealSim_Fock8_10k.out')
tlist_vu = None
if os.path.exists(vu_path):
    with open(vu_path) as f:
        lines = f.read().strip().split('\n')
    def parse_line(line):
        return np.array([complex(m.group()).real for m in re.finditer(r'\([^)]+\)', line)])
    tlist_vu = parse_line(lines[0])
    PA_vu = parse_line(lines[1])
    PB_vu = parse_line(lines[2])
    PC_vu = parse_line(lines[3])
    if tlist_vu[-1] < 1e-6:
        tlist_vu *= 1e12
    print(f"\nVu Trotter data: {len(tlist_vu)} points")

# ================================================================== #
# PLOT                                                                  #
# ================================================================== #
cols = {'A': '#C0392B', 'B': '#27AE60', 'C': '#2980B9'}
mk_map = {'A': 'o', 'B': 's', 'C': '^'}

fig, ax = plt.subplots(figsize=(10, 5))

sp_tr = max(1, N_STEPS // 60)
for ch, Pe, Pt in zip('ABC', [PA_ex,PB_ex,PC_ex], [PA_tr,PB_tr,PC_tr]):
    ax.plot(tlist_ex, Pe, color=cols[ch], lw=2, ls=['-','--','-.']['ABC'.index(ch)],
            label=f'Exact {ch}')
    ax.scatter(tlist_tr[::sp_tr], Pt[::sp_tr], facecolors='none', edgecolors=cols[ch],
               marker=mk_map[ch], s=25, lw=1.2, label=f'Trotter {ch}')

if tlist_vu is not None:
    sp_vu = max(1, len(tlist_vu) // 40)
    for ch, Pvu in zip('ABC', [PA_vu, PB_vu, PC_vu]):
        ax.scatter(tlist_vu[::sp_vu], Pvu[::sp_vu], facecolors='none',
                   edgecolors=cols[ch], marker='x', s=15, lw=0.6, alpha=0.3)

ax.set_xlabel('Time (ps)', fontsize=12); ax.set_ylabel('Population', fontsize=12)
ax.set_xlim(0, T_MAX+0.05); ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.25); ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_title(f'Method A Trotter vs exact  --  cutoff={N}, tau=5fs, {N_STEPS} steps\n'
             f'RMSE  A={rA:.4f}  B={rB:.4f}  C={rC:.4f}',
             fontsize=10, fontweight='bold')
plt.tight_layout()

for ext in ['png', 'pdf']:
    out = dp(f'trotter_method_a_fock{N}.{ext}')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close(fig)

# ================================================================== #
# GATE COUNT SUMMARY                                                    #
# ================================================================== #
print("\n" + "="*70)
print("GATE COUNT (per Trotter step)")
print("="*70)
print(f"  H_free:  {n_free} Rotation gates (unconditional, FREE on trapped-ion)")
print(f"  H_QCR:   {n_QCR} QCR gates (3 local dispersive + 3 mode l per excited state)")
print(f"  H_QCD:   {n_QCD} QCD gates (3 local modes + 3 mode l)")
print(f"  H_QPS:   {n_QPS} QPS gates (3 local + 3 mode l)")
print(f"  H_QT:    {n_QT} QT carrier gates")
print(f"  H_QTSB:  {n_QTSB} QT sideband gates")
print(f"  ---------")
print(f"  Half-step: {n_half_step} gates")
print(f"  Second-order: {n_second_order} gates per step")
print("="*70)
