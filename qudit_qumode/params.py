"""params.py -- Single source of truth for all physical parameters.

Table 1 (Vu et al. JCTC 2025) and derived Table 3 values.
All values in Hz (angular frequency, hbar=1).
"""

import numpy as np

# ================================================================== #
# Table 1: Raw parameters                                              #
# ================================================================== #
omega_ga = 4.95e13;  omega_ea = 4.63e13
omega_gb = 4.98e13;  omega_eb = 4.62e13
omega_gc = 4.92e13;  omega_ec = 4.65e13
omega_l  = 6.00e12
J_AB0 = 3.00e12;  J_AC0 = 2.70e12
eta_AB = -0.1;     eta_AC = 0.15
S_a = 0.005;  S_b = 0.004;  S_c = 0.006;  S_l = 0.05

# ================================================================== #
# Table 2/3: Derived parameters (exactly as 3_chromophore_exact.py)    #
# ================================================================== #
omega_a = (omega_ga + omega_ea) / 2
omega_b = (omega_gb + omega_eb) / 2
omega_c = (omega_gc + omega_ec) / 2

chi_a = omega_ea - omega_ga
chi_b = omega_eb - omega_gb
chi_c = omega_ec - omega_gc

g_a = np.sqrt(S_a) * omega_ea
g_b = np.sqrt(S_b) * omega_eb
g_c = np.sqrt(S_c) * omega_ec

g_cd_a = g_a * omega_ga / omega_a
g_cd_b = g_b * omega_gb / omega_b
g_cd_c = g_c * omega_gc / omega_c
g_cdl  = np.sqrt(S_l) * omega_l

g_ab  = J_AB0;   g_ac  = J_AC0
g_abl = J_AB0 * eta_AB
g_acl = J_AC0 * eta_AC

omega_qa = (omega_ga*S_a + chi_a/2 + chi_a*g_a**2/(4*omega_a**2)
            - g_a**2/omega_a + omega_l*S_l - g_cdl**2/omega_l**2)
omega_qb = (omega_gb*S_b + chi_b/2 + chi_b*g_b**2/(4*omega_b**2)
            - g_b**2/omega_b)
omega_qc = (omega_gc*S_c + chi_c/2 + chi_c*g_c**2/(4*omega_c**2)
            - g_c**2/omega_c)

delta_ab = omega_qa - omega_qb
delta_ac = omega_qa - omega_qc

# ================================================================== #
# Table 2: Lindblad rates (Hz, angular frequency)                      #
# ================================================================== #
# "All-equal" baseline values (Vu Table 2)
gamma_amp_all = 3.15e12
gamma_dep_all = 9.00e11

# Per-chromophore amplitude damping (default = all-equal baseline)
gamma_amp_a = gamma_amp_all
gamma_amp_b = gamma_amp_all
gamma_amp_c = gamma_amp_all

# Per-chromophore dephasing (default = all-equal baseline)
gamma_dep_a = gamma_dep_all
gamma_dep_b = gamma_dep_all
gamma_dep_c = gamma_dep_all

# Per-chromophore excitation rate (~0 at room temperature in Vu Table 1)
gamma_exc_a = 0.0
gamma_exc_b = 0.0
gamma_exc_c = 0.0
