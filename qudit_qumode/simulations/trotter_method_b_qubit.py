"""
trotter_method_b_qubit.py  --  Vu eqs 13-17 on 3-qubit + 4-qumode encoding
============================================================================
Implements H_cqed (Vu displaced/polaron frame) in the original qubit encoding:
  |G> = |000>, |A> = |100>, |B> = |010>, |C> = |001>

Hamiltonian: H = H_chi + H_cd + H_det + H_J + H_Jl
  (verbatim from 3_chromophore_exact.py signs)

Second-order Suzuki-Trotter splitting:
  [H_chi/2, H_cd/2, H_det/2, H_J, H_Jl, H_det/2, H_cd/2, H_chi/2]

Usage:
    python -m simulations.trotter_method_b_qubit       # cutoff=8
    python -m simulations.trotter_method_b_qubit 4     # cutoff=4
"""

import numpy as np
import re, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import qutip as qt
import os, time
from scipy.interpolate import interp1d

# Import parameters from single source of truth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import (omega_a, omega_b, omega_c, omega_l,
                    chi_a, chi_b, chi_c,
                    g_cd_a, g_cd_b, g_cd_c, g_cdl,
                    delta_ab, delta_ac,
                    g_ab, g_ac, g_abl, g_acl)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
def dp(f): return os.path.join(DATA_DIR, f)

# ================================================================== #
# SIMULATION SETTINGS                                                  #
# ================================================================== #
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TAU = 0.005e-12   # 5 fs in seconds
N_STEPS = 400
T_MAX = 2.0       # ps

print("="*70)
print(f"Trotter Method B (qubit)  --  cutoff={N}, tau={TAU*1e15:.0f}fs, {N_STEPS} steps")
print("="*70)

# ================================================================== #
# QUBIT + MODE OPERATORS                                                #
# Layout: qa x qb x qc x ma x mb x mc x ml                            #
# ================================================================== #
I2 = qt.qeye(2); IN = qt.qeye(N)
sx = qt.sigmax(); sy = qt.sigmay(); sz = qt.sigmaz()
sp = (sx + 1j*sy) / 2   # sigma_+ = |1><0|
sm = (sx - 1j*sy) / 2   # sigma_- = |0><1|

def q_op(op, idx):
    """Qubit operator on full Hilbert space."""
    ops = [I2]*3 + [IN]*4; ops[idx] = op; return qt.tensor(ops)

def m_op(op, idx):
    """Mode operator on full Hilbert space."""
    ops = [I2]*3 + [IN]*4; ops[3+idx] = op; return qt.tensor(ops)

# Qubit operators
sza = q_op(sz,0); szb = q_op(sz,1); szc = q_op(sz,2)
spa = q_op(sp,0); spb = q_op(sp,1); spc = q_op(sp,2)
sma = q_op(sm,0); smb = q_op(sm,1); smc = q_op(sm,2)
sxa = q_op(sx,0); sxb = q_op(sx,1); sxc = q_op(sx,2)
sya = q_op(sy,0); syb = q_op(sy,1); syc = q_op(sy,2)

# Mode operators
ma = m_op(qt.destroy(N),0); mb = m_op(qt.destroy(N),1)
mc = m_op(qt.destroy(N),2); ml = m_op(qt.destroy(N),3)
xa = ma+ma.dag(); xb = mb+mb.dag(); xc = mc+mc.dag(); xl = ml+ml.dag()
na = ma.dag()*ma; nb = mb.dag()*mb; nc = mc.dag()*mc; nl = ml.dag()*ml

# Qubit projectors for populations
Pe = (qt.qeye(2) - sz) / 2; Pg = (qt.qeye(2) + sz) / 2
def proj_qubit(exc, gnd):
    ops = [I2]*3 + [IN]*4; ops[exc] = Pe
    for g in gnd: ops[g] = Pg
    return qt.tensor(ops)
PA_q = proj_qubit(0,[1,2]); PB_q = proj_qubit(1,[0,2]); PC_q = proj_qubit(2,[0,1])

# Initial state: |A> = |100> x |0,0,0,0>
psi0 = qt.tensor(qt.basis(2,1), qt.basis(2,0), qt.basis(2,0),
                 qt.basis(N,0), qt.basis(N,0), qt.basis(N,0), qt.basis(N,0))

# ================================================================== #
# BUILD H_cqed (Vu eqs 13-17, VERBATIM paper signs)                   #
# ================================================================== #

# H0 (eq 14): free oscillators + detuning
H0_cqed = (omega_a*na + omega_b*nb + omega_c*nc + omega_l*nl
          - delta_ab*szb/2 - delta_ac*szc/2)

# H1 (eq 15): dispersive + conditional displacement
H1_cqed = (- chi_a*na*sza/2 - chi_b*nb*szb/2 - chi_c*nc*szc/2
           + g_cd_a*xa*sza/2 + g_cd_b*xb*szb/2 + g_cd_c*xc*szc/2)

# H2XX (eq 16)
H2xx = (g_cdl*xl*sza/4
      + g_ab*sxa*sxb/2 + g_ac*sxa*sxc/2
      + g_abl*sxa*sxb*xl/2 + g_acl*sxa*sxc*xl/2)

# H2YY (eq 17)
H2yy = (g_cdl*xl*sza/4
      + g_ab*sya*syb/2 + g_ac*sya*syc/2
      + g_abl*sya*syb*xl/2 + g_acl*sya*syc*xl/2)

H_cqed = H0_cqed + H1_cqed + H2xx + H2yy

print(f"\nH_cqed built: {H_cqed.shape}")

# ================================================================== #
# TROTTER SPLITTING                                                     #
# Split into physically meaningful blocks:                              #
#   H_chi: dispersive coupling (chi_r * sigma_z * n_r)                 #
#   H_cd:  conditional displacement (g_cd * sigma_z * x_r + g_cdl)     #
#   H_det: detuning (delta * sigma_z)                                   #
#   H_J:   carrier coupling (sigma_+ sigma_- + h.c.)                   #
#   H_Jl:  sideband coupling (sigma_+ sigma_- * (l+l†) + h.c.)        #
# ================================================================== #

# H_chi: dispersive (from H1, eq 15)
H_chi = (- chi_a*na*sza/2 - chi_b*nb*szb/2 - chi_c*nc*szc/2)

# H_cd: conditional displacement (from H1 + H2XX/YY g_cdl terms)
# H1 displacement: +g_cd_r*xr*szr/2
# H2XX+H2YY g_cdl: 2*(g_cdl*xl*sza/4) = g_cdl*xl*sza/2
H_cd = (g_cd_a*xa*sza/2 + g_cd_b*xb*szb/2 + g_cd_c*xc*szc/2
      + g_cdl*xl*sza/2)

# H_det: detuning + free oscillators
H_det = (omega_a*na + omega_b*nb + omega_c*nc + omega_l*nl
       - delta_ab*szb/2 - delta_ac*szc/2)

# H_J: carrier coupling (from H2XX+H2YY, sxsx+sysy terms)
# sxa*sxb + sya*syb = 2*(spa*smb + sma*spb) in qubit encoding
H_J = (g_ab*(sxa*sxb + sya*syb)/2 + g_ac*(sxa*sxc + sya*syc)/2)

# H_Jl: sideband coupling
H_Jl = (g_abl*(sxa*sxb + sya*syb)*xl/2 + g_acl*(sxa*sxc + sya*syc)*xl/2)

# Verify splitting
H_sum = H_chi + H_cd + H_det + H_J + H_Jl
diff_diag = (H_cqed - H_sum).diag()
print(f"Splitting check: max|diag(H_cqed - H_sum)| = {np.max(np.abs(diff_diag)):.2e}")

# ================================================================== #
# GATE COUNTS                                                          #
# ================================================================== #
n_chi  = 3   # 3 dispersive (one per qubit-mode pair)
n_cd   = 4   # 3 local conditional displacement + 1 g_cdl
n_det  = 6   # 4 free rotations + 2 detuning phase shifts
n_J    = 2   # 2 carrier coupling (AB, AC)
n_Jl   = 2   # 2 sideband coupling (ABl, ACl)

n_first_order  = n_chi + n_cd + n_det + n_J + n_Jl
n_half_step    = n_chi + n_cd + n_det + n_J
n_second_order = 2 * n_half_step + n_Jl

print(f"Gate count -- first order:  {n_first_order}")
print(f"Gate count -- second order: {n_second_order}")

# ================================================================== #
# PRE-COMPUTE PROPAGATORS OR USE SESOLVE                               #
# ================================================================== #
dim = 8 * N**4
USE_DENSE = (dim <= 2048)

if USE_DENSE:
    print(f"\nPre-computing dense propagators (dim={dim})...")
    t0 = time.time()
    from scipy.linalg import expm as dense_expm
    def make_U(H, dt):
        return qt.Qobj(dense_expm(-1j * H.full() * dt), dims=H.dims)
    U_chi_h = make_U(H_chi, TAU/2)
    U_cd_h  = make_U(H_cd,  TAU/2)
    U_det_h = make_U(H_det, TAU/2)
    U_J_h   = make_U(H_J,   TAU/2)
    U_Jl_f  = make_U(H_Jl,  TAU)
    # Symmetric: chi*cd*det*J * Jl * J*det*cd*chi
    U_fwd_h = U_chi_h * U_cd_h * U_det_h * U_J_h
    U_bwd_h = U_J_h * U_det_h * U_cd_h * U_chi_h
    U_step = U_fwd_h * U_Jl_f * U_bwd_h
    print(f"  done in {time.time()-t0:.1f}s")
else:
    print(f"\nUsing sesolve sub-steps (dim={dim}, too large for dense expm)")
    sesolve_opts = {'nsteps': 5000}

# ================================================================== #
# RUN TROTTER                                                          #
# ================================================================== #
print(f"\nRunning Trotter ({N_STEPS} steps)...")
t0 = time.time()

tlist_tr = np.array([i*TAU*1e12 for i in range(N_STEPS+1)])
psi = psi0.copy()
PA_tr = [float(qt.expect(PA_q, psi))]
PB_tr = [float(qt.expect(PB_q, psi))]
PC_tr = [float(qt.expect(PC_q, psi))]

def trotter_step_sesolve(psi_in):
    th = [0, TAU/2]; tf = [0, TAU]
    psi_in = qt.sesolve(H_chi, psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_cd,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_det, psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_J,   psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_Jl,  psi_in, tf, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_J,   psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_det, psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_cd,  psi_in, th, options=sesolve_opts).final_state
    psi_in = qt.sesolve(H_chi, psi_in, th, options=sesolve_opts).final_state
    return psi_in

for step in range(N_STEPS):
    if USE_DENSE:
        psi = U_step * psi
    else:
        psi = trotter_step_sesolve(psi)
    PA_tr.append(float(qt.expect(PA_q, psi)))
    PB_tr.append(float(qt.expect(PB_q, psi)))
    PC_tr.append(float(qt.expect(PC_q, psi)))
    if (step+1) % 100 == 0:
        print(f"  Step {step+1}/{N_STEPS}  P_A={PA_tr[-1]:.3f}  ({time.time()-t0:.0f}s)")

PA_tr = np.array(PA_tr); PB_tr = np.array(PB_tr); PC_tr = np.array(PC_tr)
print(f"  done in {time.time()-t0:.1f}s")
print(f"  P_A(T)={PA_tr[-1]:.4f}  P_B={PB_tr[-1]:.4f}  P_C={PC_tr[-1]:.4f}")

# ================================================================== #
# EXACT SESOLVE REFERENCE                                              #
# ================================================================== #
print(f"\nRunning exact sesolve on H_cqed...")
t0 = time.time()
tlist_ex = np.linspace(0, T_MAX, 500)
tlist_ex_s = tlist_ex * 1e-12

res_ex = qt.sesolve(H_cqed, psi0, tlist_ex_s, e_ops=[PA_q, PB_q, PC_q],
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
out_npz = dp(f'trotter_method_b_qubit_fock{N}.npz')
np.savez(out_npz,
         t=tlist_tr, PA=PA_tr, PB=PB_tr, PC=PC_tr,
         PA_exact=PA_ex, PB_exact=PB_ex, PC_exact=PC_ex,
         t_exact=tlist_ex,
         gate_count=n_second_order,
         rmse=np.array([rA, rB, rC]))
print(f"\nSaved: {out_npz}")

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
    print(f"Vu Trotter data: {len(tlist_vu)} points")

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
ax.set_title(f'Method B (qubit) Trotter vs exact  --  cutoff={N}, tau=5fs, {N_STEPS} steps\n'
             f'RMSE  A={rA:.4f}  B={rB:.4f}  C={rC:.4f}  |  {n_second_order} gates/step',
             fontsize=10, fontweight='bold')
plt.tight_layout()

for ext in ['png', 'pdf']:
    out = dp(f'trotter_method_b_qubit_fock{N}.{ext}')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close(fig)

# ================================================================== #
# GATE COUNT SUMMARY                                                    #
# ================================================================== #
print("\n" + "="*70)
print("GATE COUNT (per Trotter step)")
print("="*70)
print(f"  H_chi:   {n_chi} gates (chi_R * sigma_z^R * n_R)")
print(f"  H_cd:    {n_cd} gates (g_cd_R * sigma_z^R * x_R + g_cdl)")
print(f"  H_det:   {n_det} gates (4 free rotations + 2 detuning)")
print(f"  H_J:     {n_J} QT carrier gates")
print(f"  H_Jl:    {n_Jl} QT sideband gates")
print(f"  ---------")
print(f"  Half-step: {n_half_step} gates")
print(f"  Second-order: {n_second_order} gates per step")
print("="*70)
