"""Reproduce the three GRAPE pulses through HyPulse, from Frank's repo.

Targets and hardware parameters are the ones the stored runs used, read
from the same device spec.  THE STORED WAVEFORMS ARE NOT TOUCHED -- this
script writes its own file and compares.

    C_{Pi_z}D (QCD)   alpha = 0.5,   T = 160.4  us
    CD                alpha = 0.5,   T = 39.79  us
    CR                theta = 0.016, T = 24.192 us

A NOTE ON WHAT "THROUGH HYPULSE" CAN MEAN HERE.  Two of the three stored
runs went through hypulse.core.compile_pulse:

    QCD, CD   pulses_meth/hypulse_validation/step_hypulse_grape_v3.py and
              step_two_points_v4.py, both calling compile_pulse(...)

The third did not.  step_qcr_cr_two_points_v2.py, which produced the
24.192 us CR, calls qutip.control.pulseoptim.optimize_pulse_unitary
DIRECTLY against an imposed Pi_l (x) n generator, deliberately bypassing
the CR plugin.  appendix_table.md FINDING 6 records why: the plugin drives
a raw blue sideband with a default amp_bound of 62.83 rad/us, about 100x
Meth's carrier, so GRAPE finds a fast non-dispersive solution that beats
chi and is not the gate.  The plugin itself was never modified.

So CR is run BOTH ways here and both are reported: through the plugin, and
through the imposed-generator path the stored run used.  Anything else
would be comparing two different things and calling it a reproduction.

FIDELITIES REPORTED
  self-consistency  slot-by-slot expm of the produced waveform under the
                    gate's own control operator against its own target,
                    with plain qutip.  No qtrl in the loop.  Same protocol
                    as results/pulse_fidelity_common.txt, so the numbers
                    are directly comparable to the stored column.
  physical (QCD)    the produced waveform propagated under the full
                    time-dependent Bazavan Eq. 1 with only the optical RWA,
                    reusing step_qcd_bazavan_full_lab.py unchanged.

Output: results/hypulse_reproduce_three.txt
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.linalg import expm
from scipy.special import jv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pulses_meth" / "hypulse_validation"))

from hypulse.core import GateSpec, ModelSpec, compile_pulse   # noqa: E402
from hypulse.calibrations import get_device_spec              # noqa: E402

CUTOFF = 8
NUM_TSLOTS = 200
ETA = 0.041
OMEGA_Z_2PI_KHZ = 770.0
OMEGA_C_2PI_KHZ = 97.56          # = 4 kHz / eta, the Meth operating point
DEVICE = "innsbruck_meth_2025"

L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def commit(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse",
                               "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


# ------------------------------------------------------------------ bounds --
def amp_bounds():
    Oc = 2 * math.pi * OMEGA_C_2PI_KHZ * 1e3        # rad/s
    x = 4 * OMEGA_C_2PI_KHZ / OMEGA_Z_2PI_KHZ       # = 2 Omega_c / (omega_z/2)
    bess = float(jv(1, x) + jv(3, x))
    return dict(
        x=x, bessel=bess,
        QCD=ETA * Oc * bess / 1e6,                  # rad/us
        CD=ETA * Oc / 1e6,
    )


def chi_rad_per_us(delta_2pi_kHz=40.0):
    """chi = Omega_sb^2 / (4 delta), the App C formula."""
    osb = 2 * math.pi * ETA * OMEGA_C_2PI_KHZ * 1e3   # rad/s
    delta = 2 * math.pi * delta_2pi_kHz * 1e3
    return osb ** 2 / (4 * delta) / 1e6


# ------------------------------------------------------- self-consistency --
def sdf_controls():
    a = qt.destroy(CUTOFF)
    X = (a + a.dag()) / 2.0
    P = (a - a.dag()) / (2.0j)
    sz = qt.sigmaz()
    return [qt.tensor(sz, X), qt.tensor(sz, P)]


def replay(H_c, u, T, U_targ):
    """Slot-by-slot expm, trace fidelity |Tr(V^dag U)|^2 / D^2."""
    n, dim = u.shape[0], U_targ.shape[0]
    dt = T / n
    Hc = [h.full() for h in H_c]
    U = np.eye(dim, dtype=complex)
    for k in range(n):
        U = expm(-1j * sum(u[k, j] * Hc[j] for j in range(len(Hc))) * dt) @ U
    return float(abs(np.trace(U_targ.full().conj().T @ U)) ** 2 / dim ** 2)


def sdf_target(alpha):
    a = qt.destroy(CUTOFF)
    A = alpha * a.dag() - np.conj(alpha) * a
    return (qt.tensor(qt.sigmaz(), qt.qeye(CUTOFF))
            * qt.tensor(qt.qeye(2), A)).expm()


# ------------------------------------------------------------- the runs -----
def run_compile(gate_name, alpha, T_us, amp_bound, device):
    gate = GateSpec(
        name=gate_name,
        params={"alpha_re": float(alpha), "alpha_im": 0.0,
                "Delta_rad_per_s": 0.0,
                "duration_us": float(T_us),
                "num_tslots": int(NUM_TSLOTS),
                "amp_bound": float(amp_bound)},
        targets=(0, 0), meta={})
    model = ModelSpec(cutoff=CUTOFF, options={})
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        pulse = compile_pulse(gate, device, model,
                              strategy="synthesize", library_root=tmp)
        wall = time.time() - t0
    ch = pulse.channels or {}
    meta = (pulse.meta or {}).get("optimizer", {}).get("result", {})
    ux = np.asarray(ch.get("Omega_x", []), dtype=float)
    up = np.asarray(ch.get("Omega_p", []), dtype=float)
    return dict(u=np.column_stack([ux, up]), F_grape=meta.get("fidelity"),
                wall=wall, termination=meta.get("termination_reason"))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    hp = Path(get_device_spec(DEVICE).__class__.__module__ and
              "C:/Users/mhakimi/Documents/hypulse")
    W("=" * 104)
    W("REPRODUCING THE THREE GRAPE PULSES THROUGH HYPULSE")
    W("=" * 104)
    W(f"  qudit_qumode commit   {commit(ROOT)}")
    W(f"  hypulse commit        {commit(hp)}   ({hp})")
    W(f"  device spec           {DEVICE}")
    W(f"  cutoff {CUTOFF}, {NUM_TSLOTS} slots")
    W()
    W("  STORED WAVEFORMS ARE NOT TOUCHED.  This run writes only this file.")
    W()

    b = amp_bounds()
    chi = chi_rad_per_us()
    W("  AMPLITUDE BOUNDS, recomputed from the operating point")
    W(f"    x = 4 Omega_c/omega_z = {b['x']:.4f}   (J1+J3)(x) = {b['bessel']:.6f}")
    W(f"    QCD  eta*Omega_c*(J1+J3) = {b['QCD']:.6f} rad/us")
    W(f"    CD   eta*Omega_c         = {b['CD']:.6f} rad/us")
    W(f"    CR   chi = Omega_sb^2/4delta at delta/2pi=40 kHz = "
      f"{chi:.6e} rad/us")
    W()

    device = get_device_spec(DEVICE)

    # --------------------------------------------------------------- QCD ---
    W("=" * 104)
    W("1  C_{Pi_z}D  (QCD)   alpha = 0.5, T = 160.4 us")
    W("=" * 104)
    r = run_compile("QCD", 0.5, 160.4, b["QCD"], device)
    F_self = replay(sdf_controls(), r["u"], 160.4, sdf_target(0.5))
    W(f"    HyPulse F_grape        {r['F_grape']}")
    W(f"    self-consistency 1-F   {1-F_self:.6e}   (independent replay)")
    W(f"    termination            {r['termination']}   ({r['wall']:.1f} s)")
    stored = json.loads(
        (ROOT / "pulses_meth" / "hypulse_validation"
         / "final_v3.json").read_text(encoding="utf-8"))
    sx = np.asarray(stored["headline"]["Omega_x"], dtype=float)
    sp = np.asarray(stored["headline"]["Omega_p"], dtype=float)
    su = np.column_stack([sx, sp])
    F_stored = replay(sdf_controls(), su, 160.4, sdf_target(0.5))
    W(f"    stored waveform, same replay 1-F = {1-F_stored:.6e}")
    W(f"    max |u_new - u_stored| = {np.abs(r['u']-su).max():.6e} rad/us")
    W(f"    max|u| new {np.abs(r['u']).max():.6e}  stored "
      f"{np.abs(su).max():.6e}  bound {b['QCD']:.6e}")
    W()

    np.savez(ROOT / "data" / "hypulse_reproduce_qcd.npz",
             u=r["u"], T=160.4, amp_bound=b["QCD"])

    # ------------------------------------------- QCD physical replay ------
    W("  PHYSICAL REPLAY under the full Bazavan Eq. 1, optical RWA only.")
    W("  step_qcd_bazavan_full_lab.py is reused unchanged; only the module's")
    W("  waveform global is repointed at the freshly compiled pulse, so the")
    W("  physics and the fits are the stored run's own.")
    try:
        import step_qcd_bazavan_full_lab as BZ       # noqa: E402
        for tag, uu in (("HyPulse rerun", r["u"][:, 1]),
                        ("stored", su[:, 1])):
            BZ.Omega_c_slot_rad_per_s = (uu * 1e6) / (BZ.ETA * BZ.J1_PLUS_J3)
            n_cut = 8
            U_lab, _ = BZ.propagate_U_and_quadratures(n_cut)
            U_QCD = BZ.build_U_QCD(n_cut)
            F_raw = BZ.F_full(U_lab, U_QCD, n_cut)
            su2 = BZ.fit_su2_joint(U_lab, U_QCD, n_cut)
            W(f"    {tag:<14} N_cut=8   1-F raw {1-F_raw:.4e}   "
              f"1-F after frame fit {1-su2['F_max']:.4e}")
    except Exception as e:
        W(f"    physical replay FAILED: {type(e).__name__}: {str(e)[:120]}")
    W("    appendix_table.md records 0.998-0.999 unramped at N_cut 8/10/12,")
    W("    i.e. 1-F of order 1e-3 to 2e-3.")
    W()

    # ---------------------------------------------------------------- CD ---
    W("=" * 104)
    W("2  CD   alpha = 0.5, T = 39.79 us")
    W("=" * 104)
    r2 = run_compile("CD", 0.5, 39.79, b["CD"], device)
    F2 = replay(sdf_controls(), r2["u"], 39.79, sdf_target(0.5))
    W(f"    HyPulse F_grape        {r2['F_grape']}")
    W(f"    self-consistency 1-F   {1-F2:.6e}")
    W(f"    termination            {r2['termination']}   ({r2['wall']:.1f} s)")
    cd_stored = json.loads(
        (ROOT / "pulses_meth" / "hypulse_validation"
         / "cd_waveform_persist.json").read_text(encoding="utf-8"))
    key = "Omega_x" if "Omega_x" in cd_stored else None
    if key is None:
        for k in ("channels", "headline"):
            if k in cd_stored and "Omega_x" in cd_stored[k]:
                cd_stored = cd_stored[k]
                break
    cu = np.column_stack([np.asarray(cd_stored["Omega_x"], dtype=float),
                          np.asarray(cd_stored["Omega_p"], dtype=float)])
    F2s = replay(sdf_controls(), cu, 39.79, sdf_target(0.5))
    W(f"    stored waveform, same replay 1-F = {1-F2s:.6e}")
    W(f"    max |u_new - u_stored| = {np.abs(r2['u']-cu).max():.6e} rad/us")
    W()
    np.savez(ROOT / "data" / "hypulse_reproduce_cd.npz",
             u=r2["u"], T=39.79, amp_bound=b["CD"])

    # ---------------------------------------------------------------- CR ---
    W("=" * 104)
    W("3  CR   theta = 0.016, T = 24.192 us   -- TWO PATHS, BOTH REPORTED")
    W("=" * 104)
    W("    The stored 24.192 us CR did NOT come through the plugin.  See")
    W("    this file's docstring and appendix_table.md FINDING 6.")
    W()
    W("  (a) through the HyPulse CR plugin")
    try:
        gate = GateSpec(name="CR",
                        params={"theta": 0.016, "duration_us": 24.192,
                                "num_tslots": NUM_TSLOTS,
                                "amp_bound": float(chi),
                                "Delta_rad_per_s": 0.0,
                                "eta": ETA, "phi0": 0.0},
                        targets=(0, 0), meta={})
        model = ModelSpec(cutoff=CUTOFF, options={})
        with tempfile.TemporaryDirectory() as tmp:
            p = compile_pulse(gate, device, model, strategy="synthesize",
                              library_root=tmp)
        m = (p.meta or {}).get("optimizer", {}).get("result", {})
        W(f"      channels {list((p.channels or {}).keys())}")
        W(f"      F_grape  {m.get('fidelity')}   "
          f"termination {m.get('termination_reason')}")
    except Exception as e:
        W(f"      FAILED: {type(e).__name__}: {str(e)[:150]}")
        W("      Reported, not worked around.")
    W()
    W("  (b) the imposed-generator path the stored run used")
    from step_qcr_cr_two_points_v2 import compile_grape           # noqa: E402
    rc = compile_grape("CR", 0.016, chi, 24.192, "LIN")
    u = np.asarray(rc["u"], dtype=float)
    W(f"      F_grape  {rc['F_grape']:.10f}   ({rc['walltime_s']:.1f} s)")
    W(f"      int u dt = {np.trapezoid(u, dx=24.192/len(u)):+.6f}  "
      f"(target {0.016:+.6f})")
    stored_cr = json.loads(
        (ROOT / "pulses_meth" / "hypulse_validation"
         / "step_qcr_cr_two_points_v2_results.json").read_text(
             encoding="utf-8"))
    row = next(r for r in stored_cr["summary"]
               if r["gate"] == "CR" and r["point"] == "A_meth"
               and r["theta"] == 0.016)
    W(f"      stored   F_grape  {row['F_grape']:.10f}   "
      f"T {row['T_grape_us']} us")
    W()

    W("=" * 104)
    W("READ ONLY.  Stored waveforms untouched; no adoption.")
    out = ROOT / "results" / "hypulse_reproduce_three.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
