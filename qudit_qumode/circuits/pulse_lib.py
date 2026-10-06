"""Shared pulse generation for the qudit and qubit pulse figures.

DOES NOT TOUCH THE DATA PATH.

GATE NAMES.  The manuscript renamed the five ISA primitives.  Code
identifiers, dict keys and results-file columns still use the old names, so
this mapping travels with every script that touches them:

    paper name      code / data name      what it is
    Z_d             QPS                   qudit phase shift
    X_E             QT                    qudit transition (carrier)
    C_{\\Pi_z}D      QCD                   level-conditioned displacement
    C_{\\Pi}R        QCR                   level-conditioned mode rotation
    C_{\\Pi_x}D      QT-SB, QTsb           transition-conditioned displacement

Multitone forms write the transition set as plain E.  CD and CR keep their
names: they are the qubit-reference gates, not qudit ISA primitives.  No
identifier below is renamed.

Every device parameter is read from pulses_meth/anchor_params.json, never
retyped.  Every GRAPE waveform is re-propagated independently afterwards
(plain qutip, no qtrl in the loop) so the reported fidelity is not the
optimiser's own claim.

ONE MECHANISM PER GATE.  The control operators handed to GRAPE are the ones
the physical drive actually provides, and no gate is optimised against a
control that IS its own target:

  carrier      sigma_x, sigma_y                 bound Omega_c
               resonant drive on one transition

  sigma_z SDF  sigma_z (x) X, sigma_z (x) P     bound eta*Omega_c*(J1+J3)
               Bazavan: two tones near the CARRIER, omega_0 +- delta, beat
               note 2*delta locked to the axial mode, ANTISYMMETRIC phase.
               The (J1+J3) Bessel derating is intrinsic to this drive --
               the carrier-order terms cancel and only the recoil-order
               piece survives.

  sideband     sigma_x (x) X, sigma_y (x) X     bound eta*Omega_c
               RESONANT red+blue sideband tones, SYMMETRIC phase.  No Bessel
               factor: this sits on the sidebands, not at 2*delta = nu.

  dispersive   NOT GRAPE-able.  Pi_l (x) n emerges at SECOND order in
               Omega/delta and is not in the span of the first-order
               controls; handed a raw sideband control GRAPE finds a fast
               non-dispersive path that beats chi and is not the gate
               (pulses_meth/appendix_table.md FINDING 6).  Flat analytic
               segment, T = theta/chi, plus Choi's AC-Stark echo.

  virtual      no pulse exists.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.special import jv
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
ANCHOR = ROOT / "pulses_meth" / "anchor_params.json"
CUTOFF = 8
N_TS = 200


# ---------------------------------------------------------------- device ---
def device():
    d = json.loads(ANCHOR.read_text(encoding="utf-8"))
    hw = d["hardware_bounds_NOT_in_meth"]
    D = dict(
        species=d["ion"]["species"]["value"],
        manifold=d["ion"]["qudit_manifold"]["value"],
        wz=2 * np.pi * d["trap"]["axial_com_frequency_over_2pi_MHz"]["value"],
        wz_MHz=d["trap"]["axial_com_frequency_over_2pi_MHz"]["value"],
        beam_deg=d["trap"]["beam_angle_to_trap_axis_deg"]["value"],
        eta=d["coupling"]["lamb_dicke_eta"]["value"],
        bsb_kHz=d["coupling"]["demonstrated_bsb_rabi_over_2pi_kHz"]["value"],
        Oc_kHz=hw["carrier_omega_max_over_2pi_kHz"]["value"],
        sb_kHz=hw["sideband_omega_max_over_2pi_kHz"]["value"],
        awg=hw["awg_sample_rate_MSa_per_s"]["value"],
        rise_ns=hw["rise_fall_ns"]["value"],
        T1=d["coherence"]["T1_D5_2_s"]["value"],
        T2_ms=d["coherence"]["T2_optical_ms"]["value"],
        tau_mot_ms=d["coherence"]["motional_coherence_tau_ms"]["value"],
        heat=d["coherence"]["heating_rate_phonons_per_s"]["value"],
    )
    D["Oc"] = 2 * np.pi * D["Oc_kHz"] * 1e-3          # rad/us
    D["x_bessel"] = 4 * D["Oc"] / D["wz"]
    D["bessel"] = float(jv(1, D["x_bessel"]) + jv(3, D["x_bessel"]))
    D["amp_carrier"] = D["Oc"]
    D["amp_sideband"] = D["eta"] * D["Oc"]
    D["amp_sdf_z"] = D["eta"] * D["Oc"] * D["bessel"]
    # chi, three ways.  CORRECTED -- an earlier comment here claimed the
    # adopted value was "traceable to Choi's measured dispersive shift".
    # THAT WAS WRONG.  Choi's MEASURED chi/2pi = 471 Hz gives T = 5.407 us,
    # not 25.465 (pulses_meth/appendix_table.md, the CHOI_MEASURED rows).
    # Choi's role is to VALIDATE the formula chi = Omega_sb^2/(4 delta), not
    # to supply the number.
    #
    #   chi           ADOPTED.  chi = Omega_sb^2/(4 delta) at delta/2pi =
    #                 40 kHz, which is ASSUMED in this work and appears as
    #                 such in the manuscript's parameter table.  Gives
    #                 chi/2pi = 100 Hz and T = theta/chi = 25.465 us at
    #                 theta = 0.016.  This is the paper's number.
    #   chi_halfwz    the same formula evaluated at delta = wz/2 = 385 kHz
    #                 instead of 40 kHz.  Gives T = 233 us.  It is NOT an
    #                 alternative anchoring of chi -- it is one point off the
    #                 end of the detuning sweep the paper already publishes
    #                 (results/qcr_delta_ratio_34.txt).  Kept only so the
    #                 figure note can say what it is.
    #   chi_choi      Choi's measured 471 Hz on 171Yb+.  Different platform,
    #                 sensitivity check only, never the headline.
    D["chi"] = 0.016 / 25.465
    D["chi_halfwz"] = (D["eta"] * D["Oc"]) ** 2 / (4 * D["wz"] / 2)
    D["chi_choi"] = 2 * np.pi * 471.0 * 1e-6                      # rad/us
    # kept under the old key so nothing downstream breaks; see chi_halfwz
    D["chi_derived"] = D["chi_halfwz"]
    return D


def floor_us(D, n_slots=None):
    """Shortest pulse this hardware can actually emit.

    The GRAPE durations below are set by the AMPLITUDE BOUND alone.  At the
    vibronic circuit's angles some gates come out shorter than the pulse can
    physically be shaped, in which case the amplitude bound is not what sets
    the duration:

        rise + fall                       the envelope has to get there
        n_slots / AWG sample rate         the waveform has to be representable

    appendix_table.md already flags this for QT-SB -- "lower bound; rise time
    and per-pulse overhead dominate".  It applies to QT as well.
    """
    f = 2 * D["rise_ns"] * 1e-3
    if n_slots:
        f = max(f, n_slots / D["awg"])
    return f


def hardware_lines(D):
    return [
        f"ion {D['species']}   {D['manifold']}",
        f"axial COM $\\omega_z/2\\pi$ = {D['wz_MHz']} MHz"
        f"    beam angle {D['beam_deg']}$^\\circ$"
        f"    Lamb-Dicke $\\eta$ = {D['eta']}",
        f"carrier $\\Omega_c/2\\pi \\leq$ {D['Oc_kHz']:.0f} kHz"
        f"    sideband $\\leq$ {D['sb_kHz']:.0f} kHz"
        f"    demonstrated BSB {D['bsb_kHz']} kHz",
        f"AWG {D['awg']:.0f} MSa/s, rise/fall {D['rise_ns']:.0f} ns"
        f"    $T_1$ = {D['T1']} s,  $T_2^{{opt}}$ = {D['T2_ms']} ms,"
        f"  $\\tau_{{mot}}$ = {D['tau_mot_ms']} ms,  heating {D['heat']}/s",
        f"amplitude bounds (rad/$\\mu$s):  carrier {D['amp_carrier']:.4e}"
        f"    sideband {D['amp_sideband']:.4e}"
        f"    $\\sigma_z$ SDF {D['amp_sdf_z']:.4e}"
        f"  [$\\times (J_1+J_3)({D['x_bessel']:.3f}) = {D['bessel']:.4f}$]",
    ]


# ---------------------------------------------------------------- controls -
def controls(kind):
    a = qt.destroy(CUTOFF)
    X = (a + a.dag()) / 2.0
    P = (a - a.dag()) / (2.0j)
    IN, I2 = qt.qeye(CUTOFF), qt.qeye(2)
    if kind == "carrier":
        return [qt.tensor(qt.sigmax(), IN), qt.tensor(qt.sigmay(), IN)], \
               r"$\sigma_x\!\otimes\!\mathbb{1},\ \sigma_y\!\otimes\!\mathbb{1}$"
    if kind == "sdf_z":
        return [qt.tensor(qt.sigmaz(), X), qt.tensor(qt.sigmaz(), P)], \
               r"$\sigma_z\!\otimes\!X,\ \sigma_z\!\otimes\!P$"
    if kind == "sideband":
        return [qt.tensor(qt.sigmax(), X), qt.tensor(qt.sigmay(), X)], \
               r"$\sigma_x\!\otimes\!X,\ \sigma_y\!\otimes\!X$"
    raise KeyError(kind)


def replay(H_c, u, T, U_targ):
    n, dim = u.shape[0], U_targ.shape[0]
    dt = T / n
    Hc = [h.full() for h in H_c]
    U = np.eye(dim, dtype=complex)
    for k in range(n):
        U = expm(-1j * sum(u[k, j] * Hc[j] for j in range(len(Hc))) * dt) @ U
    return float(abs(np.trace(U_targ.full().conj().T @ U)) ** 2 / dim ** 2)


def grape(kind, U_targ, T, amp, n_ts=N_TS, tries=(50, 200, 100, 50, 60, 40, 50)):
    """Best of several GRAPE attempts, scored by INDEPENDENT replay.

    qutip-qtrl reports "function converged" on plateaus, and how often it
    lands on one depends on the random seed and on how many slots it is
    given -- over-parameterising a simple target makes it worse, not better
    (a bare carrier rotation stalls at 5e-05 with 200 slots and reaches
    4e-11 with 50).  Rather than hand-tune the slot count per gate, run a
    few and keep whichever the replay likes best.
    """
    from qutip_qtrl.pulseoptim import optimize_pulse_unitary
    C, clabel = controls(kind)
    dim = U_targ.shape[0]
    t0 = time.time()
    best = None
    for attempt, nts in enumerate(tries):
        # RND init makes qutip-qtrl nondeterministic, and whether it lands on
        # a plateau is seed-dependent -- the same gate came out at 2e-11 and
        # 5e-05 on consecutive runs.  Seed each attempt so the figure is
        # reproducible.
        np.random.seed(20260910 + 17 * attempt)
        r = optimize_pulse_unitary(
            qt.Qobj(np.zeros((dim, dim)), dims=U_targ.dims), list(C),
            qt.Qobj(np.eye(dim), dims=U_targ.dims), U_targ, nts, T,
            amp_lbound=-amp, amp_ubound=amp, fid_err_targ=1e-12,
            max_iter=3000, max_wall_time=40, init_pulse_type="RND",
            gen_stats=False)
        u = np.asarray(r.final_amps, dtype=float)
        F = replay(C, u, T, U_targ)
        if best is None or F > best[1]:
            best = (u, F, nts)
        if 1 - F < 1e-10:
            break
    u, F, nts = best
    return dict(u=u, T=T, amp=amp, ctrl=clabel, kind=kind, F=F,
                wall=time.time() - t0, n_ts=nts,
                method="GRAPE (qutip-qtrl)")


def analytic_flat(kind, T, amp, ctrl, note):
    """A flat segment for the gates GRAPE cannot honestly optimise."""
    u = np.zeros((N_TS, 2))
    u[:, 0] = amp
    return dict(u=u, T=T, amp=amp, ctrl=ctrl, kind=kind, F=None,
                wall=0.0, method="analytic (flat)", note=note)


def virtual(ctrl, note):
    return dict(u=None, T=0.0, amp=0.0, ctrl=ctrl, kind="virtual",
                F=None, wall=0.0, method="virtual (no pulse)", note=note)
