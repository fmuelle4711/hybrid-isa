"""figures/fig_pulses_cd.pdf -- two panels, both synthesized through HyPulse.

  a.  CzD   alpha = 0.008,  T = 2.5664 us   (C_{Pi_z}D at the circuit angle)
  b.  CxD   theta = 0.0015, T = 0.649351 us (C_{Pi_x}D, the half-period floor)

Each panel is synthesized by hypulse.core.compile_pulse against the
'innsbruck_meth_2025' device spec, then replayed under its own PHYSICAL
drive and scored against its own target.

THE TWO PHYSICAL REPLAYS, AND HOW EACH IS VALIDATED BEFORE USE
  CzD   Bazavan Eq. 1 in the lab frame with only the optical RWA, the same
        model as pulses_meth/hypulse_validation/step_qcd_bazavan_full_lab.py.
        Amplitude mapping Omega_c(t) = |u(t)| / (eta * (J1+J3)), the inverse
        of czd_amp_bound().  VALIDATION GATE: the stored alpha=0.5 waveform
        rescaled to this duration must reproduce 3.669830e-04, the no-ramp
        entry of results/qcd_circuit_angle_physical.txt.
  CxD   the resonant bichromatic sideband of scripts/qtsb_mechanism.py,
        propagated by scripts/qtsb_grape.propagate -- the same function that
        produced the stored pulse, not a reimplementation.  Amplitude
        mapping c(t) = |u(t)| / (2 eta).  VALIDATION GATE: the stored
        waveform in data/qtsb_qtrl_working.npz must reproduce its own
        recorded 1-F_vz = 3.452390e-09.

  Both gates run FIRST.  If either misses, the script refuses to plot.

ON FLATNESS
  The flat pulse is the physically intended one in both cases and is what
  the stored CxD waveform is.  It is constructed here analytically and
  plotted next to the GRAPE output, because the GRAPE output is NOT flat --
  see the report.  Nothing is tuned to make it flat; the two are shown side
  by side and the measurement is printed.

Output: figures/pulses/fig_pulses_cd.{pdf,png}
        figures/pulses/fig_pulses_cd.txt
        figures/pulses/fig_pulses_cd_sources.txt
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.special import jv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HYPULSE = Path("C:/Users/mhakimi/Documents/hypulse")
for p in (ROOT, HERE):
    sys.path.insert(0, str(p))

from hypulse.core import GateSpec, ModelSpec, compile_pulse      # noqa: E402
from hypulse.calibrations import get_device_spec                 # noqa: E402
from hypulse.sim.hamiltonians.czd import czd_amp_bound           # noqa: E402
from hypulse.sim.hamiltonians.cxd import (cxd_amp_bound,         # noqa: E402
                                          cxd_min_duration)

import qtsb_mechanism as QM                                      # noqa: E402
import qtsb_grape as GR                                          # noqa: E402

DEVICE = "innsbruck_meth_2025"
CUTOFF, NSLOTS = 8, 200

ALPHA_CZD, T_CZD = 0.008, 2.5664
THETA_CXD, T_CXD = 0.0015, 0.6493506493506493

# Bazavan lab-frame constants, as in step_qcd_bazavan_full_lab.py
ETA = 0.041
OMEGA_Z = 2 * math.pi * 0.77e6
DELTA_BAZ = OMEGA_Z / 2
OMEGA_C_NOM = 2 * math.pi * 97.56e3
J1P3 = float(jv(1, 2 * OMEGA_C_NOM / DELTA_BAZ)
             + jv(3, 2 * OMEGA_C_NOM / DELTA_BAZ))

STORED_CZD = 3.669830e-04      # results/qcd_circuit_angle_physical.txt, no ramp
STORED_CZD_SU2 = 5.168066e-06  # same file, SU2+R column
FINAL_V3 = ROOT / "pulses_meth" / "hypulse_validation" / "final_v3.json"
QTSB_NPZ = ROOT / "data" / "qtsb_qtrl_working.npz"

L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def commit(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


# ------------------------------------------------------------- synthesis ----
def synth(name, params, T, bound, device):
    p = dict(params)
    p.update({"Delta_rad_per_s": 0.0, "duration_us": float(T),
              "num_tslots": NSLOTS, "amp_bound": float(bound)})
    gate = GateSpec(name=name, params=p, targets=(0, 0), meta={})
    with tempfile.TemporaryDirectory() as tmp:
        pulse = compile_pulse(gate, device, ModelSpec(cutoff=CUTOFF,
                                                      options={}),
                              strategy="synthesize", library_root=tmp)
    ch = pulse.channels or {}
    ux = np.asarray(ch["Omega_x"], dtype=float)
    up = np.asarray(ch["Omega_p"], dtype=float)
    res = (pulse.meta or {}).get("optimizer", {}).get("result", {})
    return np.column_stack([ux, up]), res


def flatness(u, bound):
    """Per-channel and envelope spread, as a percentage of the bound."""
    env = np.hypot(u[:, 0], u[:, 1])
    return {
        "env_mean_over_bound": float(env.mean() / bound),
        "env_ripple_pct": float((env.max() - env.min()) / bound * 100.0),
        "x_ripple_pct": float((u[:, 0].max() - u[:, 0].min()) / bound * 100.0),
        "p_ripple_pct": float((u[:, 1].max() - u[:, 1].min()) / bound * 100.0),
        "env_max_over_bound": float(env.max() / bound),
    }


# --------------------------------------------------------- CzD physical ----
def baz_replay(omega_c_slots, T_us, alpha, n_cut=CUTOFF):
    """Bazavan Eq. 1, optical RWA only.  omega_c_slots in rad/s."""
    N = len(omega_c_slots)
    dt_us = T_us / N

    def om(t):
        i = int(t * 1e6 / dt_us)
        return omega_c_slots[min(max(i, 0), N - 1)]

    a = qt.destroy(n_cut)
    I = qt.qeye(n_cut)
    H = [[qt.tensor(-qt.sigmay(), I),
          lambda t, *_: om(t) * math.cos(DELTA_BAZ * t)],
         [qt.tensor(qt.sigmax(), a),
          lambda t, *_: ETA * om(t) * math.cos(DELTA_BAZ * t)
          * np.exp(-1j * OMEGA_Z * t)],
         [qt.tensor(qt.sigmax(), a.dag()),
          lambda t, *_: ETA * om(t) * math.cos(DELTA_BAZ * t)
          * np.exp(+1j * OMEGA_Z * t)]]
    tl = np.linspace(0.0, T_us * 1e-6, int(T_us * 1e-6 / 30e-9) + 1)
    U = qt.sesolve(H, qt.tensor(qt.qeye(2), I), tl,
                   options={"atol": 1e-10, "rtol": 1e-8,
                            "nsteps": 500000}).states[-1]
    P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
    P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
    V = (qt.tensor(P0, qt.displace(n_cut, -alpha))
         + qt.tensor(P1, qt.displace(n_cut, +alpha)))
    d = 2 * n_cut
    return 1.0 - abs(complex((V.dag() * U).tr())) ** 2 / d ** 2


def czd_gate():
    """Rescaled stored waveform must reproduce the recorded no-ramp 1-F."""
    up = np.asarray(json.loads(FINAL_V3.read_text())["headline"]["Omega_p"],
                    dtype=float)
    got = baz_replay((up * 1e6) / (ETA * J1P3), T_CZD, ALPHA_CZD)
    return got, abs(got - STORED_CZD) / STORED_CZD


# --------------------------------------------------------- CxD physical ----
def cxd_replay(u_hypulse, n_coarse=24, n_refine=18):
    """c(t) = |u|/(2 eta), with the global DRIVE PHASE fitted.

    A global phase on c(t) is a laser-phase frame choice, calibrated in any
    real run, and the qtsb control basis and the HyPulse CxD control basis
    do not share its origin: the stored flat pulse sits at -2 theta/T in the
    p quadrature, not +.  Scanning it is the same kind of frame absorption
    as the virtual-Z used everywhere else in this paper, not a fit
    parameter of the gate.  The fitted angle is reported.
    """
    c = (u_hypulse[:, 0] + 1j * u_hypulse[:, 1]) / (2.0 * ETA)
    V = QM.U_paper(THETA_CXD)

    def score(phi):
        cc = c * np.exp(1j * phi)
        u = np.column_stack([cc.real, cc.imag])
        f, _ = QM.fid_vz(GR.propagate(u, T_CXD, GR.SUBSTEPS), V)
        return 1.0 - f

    # coarse sweep, then a golden-section refine on the winning bracket.
    grid = np.linspace(0.0, 2 * np.pi, n_coarse, endpoint=False)
    vals = [score(p_) for p_ in grid]
    k = int(np.argmin(vals))
    step = grid[1] - grid[0]
    lo, hi = grid[k] - step, grid[k] + step
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = score(a_), score(b_)
    for _ in range(n_refine):
        if fa < fb:
            hi, b_, fb = b_, a_, fa
            a_ = hi - gr * (hi - lo)
            fa = score(a_)
        else:
            lo, a_, fa = a_, b_, fb
            b_ = lo + gr * (hi - lo)
            fb = score(b_)
    best_phi = a_ if fa < fb else b_
    best_val = min(fa, fb, vals[k])
    if vals[k] <= best_val:
        best_phi = float(grid[k])
    return float(best_val), float(best_phi % (2 * np.pi))


def cxd_gate():
    d = np.load(QTSB_NPZ, allow_pickle=True)
    stored = 1.0 - float(d["F_vz"])
    U = GR.propagate(d["u"], float(d["T"]), GR.SUBSTEPS)
    f, _ = QM.fid_vz(U, QM.U_paper(float(d["theta"])))
    got = 1.0 - f
    return got, stored, abs(got - stored) / stored


# ------------------------------------------------------------------ main ----
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    device = get_device_spec(DEVICE)
    bz = czd_amp_bound(device)
    bx = cxd_amp_bound(device)
    b_czd = bz["amp_bound_rad_per_us"]
    b_cxd = bx["amp_bound_rad_per_us"]
    floor = cxd_min_duration(device)

    W("=" * 104)
    W("fig_pulses_cd -- CzD AND CxD THROUGH HYPULSE")
    W("=" * 104)
    W(f"  qudit_qumode commit {commit(ROOT)}    hypulse commit "
      f"{commit(HYPULSE)}")
    W(f"  device spec {DEVICE}, cutoff {CUTOFF}, {NSLOTS} slots")
    W()
    W("  AMPLITUDE BOUNDS, from the device alone")
    W(f"    CzD  eta*Omega_c*(J1+J3)(4 Omega_c/nu) = {b_czd:.9f} rad/us")
    W(f"         eta {bz['eta']}  Omega_c {bz['Omega_c_rad_per_us']:.9f}  "
      f"J1+J3 {bz['J1_plus_J3']:.9f}")
    W(f"    CxD  eta*Omega_c (NO Bessel)           = {b_cxd:.9f} rad/us")
    W(f"         half-period floor {floor['min_duration_us']:.6f} us; "
      f"T = {T_CXD:.6f} us sits on it")
    W()

    # ---- validation gates BEFORE anything is plotted --------------------
    W("  VALIDATION GATES -- the replays must reproduce stored numbers first")
    g1, r1 = czd_gate()
    W(f"    CzD Bazavan replay of the rescaled stored waveform")
    W(f"      got {g1:.6e}   stored {STORED_CZD:.6e}   "
      f"rel {r1:.2e}  {'PASS' if r1 < 1e-3 else 'FAIL'}")
    g2, s2, r2 = cxd_gate()
    W(f"    CxD sideband replay of the stored waveform")
    W(f"      got {g2:.6e}   stored {s2:.6e}   "
      f"rel {r2:.2e}  {'PASS' if r2 < 1e-3 else 'FAIL'}")
    if r1 >= 1e-3 or r2 >= 1e-3:
        W("    A GATE FAILED.  Refusing to synthesize or plot.")
        (ROOT / "figures" / "pulses" / "fig_pulses_cd.txt").write_text(
            "\n".join(L) + "\n", encoding="utf-8")
        return 1
    W()

    # ---- synthesis ------------------------------------------------------
    W("  SYNTHESIS THROUGH hypulse.core.compile_pulse")
    u_z, res_z = synth("CzD", {"alpha_re": ALPHA_CZD, "alpha_im": 0.0},
                       T_CZD, b_czd, device)
    u_x, res_x = synth("CxD", {"theta": THETA_CXD},
                       T_CXD, b_cxd, device)
    for nm, res in (("CzD", res_z), ("CxD", res_x)):
        W(f"    {nm}  1-F_grape {1 - float(res.get('fidelity')):.6e}   "
          f"{res.get('termination_reason')}")
    W()

    # ---- the analytic flat pulse, for comparison ------------------------
    # CzD: int u dt = 2 alpha  ->  u_flat = 2 alpha / T
    u_z_flat = np.column_stack([np.zeros(NSLOTS),
                                np.full(NSLOTS, 2 * ALPHA_CZD / T_CZD)])
    u_x_flat = np.column_stack([np.zeros(NSLOTS),
                                np.full(NSLOTS, 2 * THETA_CXD / T_CXD)])

    W("  FLATNESS, as a percentage of each gate's own bound")
    W(f"    {'':<22}{'|u|/bound':>11}{'|u| ripple':>13}"
      f"{'Om_x ripple':>13}{'Om_p ripple':>13}")
    rows = []
    for nm, u, bd in (("CzD HyPulse GRAPE", u_z, b_czd),
                      ("CzD analytic flat", u_z_flat, b_czd),
                      ("CxD HyPulse GRAPE", u_x, b_cxd),
                      ("CxD analytic flat", u_x_flat, b_cxd)):
        f = flatness(u, bd)
        rows.append((nm, f))
        W(f"    {nm:<22}{f['env_mean_over_bound']:>11.4f}"
          f"{f['env_ripple_pct']:>12.4f}%{f['x_ripple_pct']:>12.4f}%"
          f"{f['p_ripple_pct']:>12.4f}%")
    W()
    flat_ok = all(f["env_ripple_pct"] < 1.0 for nm, f in rows
                  if "GRAPE" in nm)
    W("    FLAT WITHIN 1% OF THE BOUND?  "
      + ("yes, both" if flat_ok else "NO -- neither GRAPE waveform is flat."))
    if not flat_ok:
        W("    The GRAPE outputs swing across the full +/- bound.  At these")
        W("    angles the target is nearly the identity and the 1e-5 goal is")
        W("    met by a large family of solutions, so nothing in the")
        W("    optimizer prefers the flat one.  qutip offers no CONST seed")
        W("    ('No option for pulse_type CONST'), and LIN does not help")
        W("    either.  Nothing was tuned; the flat pulse is constructed")
        W("    analytically above and both are plotted.")
    W()

    # ---- physical replay ------------------------------------------------
    W("  PHYSICAL REPLAY, 1-F next to the stored value")
    W(f"    {'':<22}{'1-F replay':>14}{'stored':>14}  note")
    env_z = np.hypot(u_z[:, 0], u_z[:, 1])
    fz_grape = baz_replay((env_z * 1e6) / (ETA * J1P3), T_CZD, ALPHA_CZD)
    fz_flat = baz_replay((u_z_flat[:, 1] * 1e6) / (ETA * J1P3),
                         T_CZD, ALPHA_CZD)
    fx_grape, phi_g = cxd_replay(u_x)
    fx_flat, phi_f = cxd_replay(u_x_flat)
    W(f"    {'CzD HyPulse GRAPE':<22}{fz_grape:>14.6e}"
      f"{STORED_CZD:>14.6e}  envelope mapping, see below")
    W(f"    {'CzD analytic flat':<22}{fz_flat:>14.6e}"
      f"{STORED_CZD:>14.6e}  exact, single quadrature")
    W(f"    {'CxD HyPulse GRAPE':<22}{fx_grape:>14.6e}{s2:>14.6e}"
      f"  drive phase {phi_g:+.4f} rad")
    W(f"    {'CxD analytic flat':<22}{fx_flat:>14.6e}{s2:>14.6e}"
      f"  drive phase {phi_f:+.4f} rad")
    W()
    W("    The stored CzD column is the no-ramp entry of")
    W("    results/qcd_circuit_angle_physical.txt (its SU2+R column, the")
    W(f"    frame-fitted one, is {STORED_CZD_SU2:.6e}); the stored CxD")
    W("    column is F_vz from data/qtsb_qtrl_working.npz.")
    W("    The Bazavan model carries ONE quadrature, so a two-quadrature")
    W("    GRAPE waveform enters it through its envelope |u(t)| and its")
    W("    relative quadrature phase is dropped.  That mapping is exact for")
    W("    the flat pulse and approximate for the GRAPE one; the GRAPE row")
    W("    is therefore an upper bound on how well that waveform could do.")
    W()

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for k, (a, u, uf, bd, ttl, T, sub) in enumerate((
            (ax[0], u_z, u_z_flat, b_czd,
             r"(a)  $C_{\Pi_z}D$   $\alpha=0.008$",
             T_CZD, r"$T=2.566\,\mu$s,  bound $\eta\Omega_c(J_1{+}J_3)$"),
            (ax[1], u_x, u_x_flat, b_cxd,
             r"(b)  $C_{\Pi_x}D$   $\theta=0.0015$",
             T_CXD, r"$T=0.649\,\mu$s,  bound $\eta\Omega_c$"))):
        t = np.arange(len(u)) * T / len(u)
        a.step(t, u[:, 0] / bd, where="post", lw=1.0, color="#1f77b4",
               label=r"$\Omega_x$ (GRAPE)")
        a.step(t, u[:, 1] / bd, where="post", lw=1.0, color="#d62728",
               label=r"$\Omega_p$ (GRAPE)")
        a.plot(t, uf[:, 1] / bd, lw=1.6, color="k", ls="--",
               label="flat, analytic")
        a.axhline(1.0, color="0.5", lw=0.7, ls=":")
        a.axhline(-1.0, color="0.5", lw=0.7, ls=":")
        a.set_xlim(0, T)
        a.set_xlabel(r"$t$ ($\mu$s)")
        a.set_title(ttl + "\n" + sub, fontsize=8)
        if k == 0:
            a.set_ylabel(r"amplitude / bound")
        a.legend(fontsize=6, loc="lower right", framealpha=0.9)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    figs = ROOT / "figures" / "pulses"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"fig_pulses_cd.{ext}", dpi=200)
    plt.close(fig)
    W(f"  wrote {figs / 'fig_pulses_cd.pdf'} (+ .png)")
    W()
    W("  READ ONLY.  No adoption.")

    (ROOT / "figures" / "pulses" / "fig_pulses_cd.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8")

    # ---- rewrite the sources file ---------------------------------------
    S = ["=" * 100,
         "PULSE FIGURE SOURCES -- ONE ENTRY PER PANEL",
         "=" * 100,
         f"  generated by scripts/make_fig_pulses_cd.py at commit "
         f"{commit(ROOT)}",
         "",
         "  figures/fig_pulses_cd.pdf   2 panels: CzD, CxD",
         "",
         "  fig_pulses_qudit.pdf and fig_pulses_qubit.pdf are SUPERSEDED and",
         "  have been removed from the paper manifest.  The files remain on",
         "  disk; they are simply no longer shipped or referenced.",
         "",
         "  Z_d is a virtual frame update and has no waveform, so it has no",
         "  panel.  MS HAS NO WAVEFORM ANYWHERE IN THIS REPOSITORY: it is",
         "  taken from Meth SI Note II p.3 as 0.2 ms per gate, angle",
         "  independent, and that quoted number is the whole of the",
         "  Molmer-Sorensen cost in the runtime model.  It cannot be",
         "  plotted, and Appendix B should not say the saved waveforms",
         "  cover it.",
         "=" * 100,
         f"  Omega_c = {bz['Omega_c_rad_per_us']:.9f} rad/us  "
         f"(device spec '{DEVICE}', amp_bound_rad_per_us)",
         "",
         "-" * 100,
         "  PANEL a: CzD  (HyPulse, hypulse.core.compile_pulse)",
         f"    gate       alpha = {ALPHA_CZD}, T = {T_CZD} us, "
         f"{NSLOTS} slots, cutoff {CUTOFF}",
         f"    bound      {b_czd:.9f} rad/us = eta*Omega_c*(J1+J3)"
         f"(4 Omega_c/nu), czd_amp_bound()",
         f"    1-F_grape  {1 - float(res_z.get('fidelity')):.6e}  "
         f"({res_z.get('termination_reason')})",
         f"    1-F_phys   {fz_grape:.6e}  Bazavan Eq. 1 lab frame, optical "
         f"RWA only, envelope mapping",
         f"    stored     {STORED_CZD:.6e}  raw / {STORED_CZD_SU2:.6e} SU2+R,"
         f" results/qcd_circuit_angle_physical.txt (no ramp)",
         f"    flatness   |u| ripple {rows[0][1]['env_ripple_pct']:.4f}% of "
         f"the bound -- NOT flat",
         f"    flat ref   analytic 2*alpha/T, 1-F_phys {fz_flat:.6e}",
         "-" * 100,
         "  PANEL b: CxD  (HyPulse, hypulse.core.compile_pulse)",
         f"    gate       theta = {THETA_CXD}, T = {T_CXD:.6f} us, "
         f"{NSLOTS} slots, cutoff {CUTOFF}",
         f"    bound      {b_cxd:.9f} rad/us = eta*Omega_c, NO Bessel "
         f"(FINDING 4), cxd_amp_bound()",
         f"    duration   half-period floor "
         f"{floor['min_duration_us']:.6f} us; T sits on it, cxd_min_duration()",
         f"    1-F_grape  {1 - float(res_x.get('fidelity')):.6e}  "
         f"({res_x.get('termination_reason')})",
         f"    1-F_phys   {fx_grape:.6e}  resonant bichromatic sideband, "
         f"scripts/qtsb_grape.propagate",
         f"    stored     {s2:.6e}  F_vz, data/qtsb_qtrl_working.npz "
         f"(seed 'analytic', i.e. the flat pulse)",
         f"    flatness   |u| ripple {rows[2][1]['env_ripple_pct']:.4f}% of "
         f"the bound -- NOT flat",
         f"    flat ref   analytic 2*theta/T, 1-F_phys {fx_flat:.6e}",
         "-" * 100,
         "",
         "  BOTH REPLAYS WERE GATED BEFORE USE.  The CzD replay reproduces",
         f"  the stored {STORED_CZD:.6e} to {r1:.1e} relative, and the CxD",
         f"  replay reproduces the stored {s2:.6e} to {r2:.1e} relative.",
         "",
         "  NEITHER HyPulse GRAPE WAVEFORM IS FLAT.  The request was to",
         "  confirm flatness within 1% of the bound; the measurement is",
         "  above and it does not hold. Nothing was tuned to force it.",
         ]
    (ROOT / "figures" / "pulses" / "fig_pulses_cd_sources.txt").write_text(
        "\n".join(S) + "\n", encoding="utf-8")
    print("\nwrote figures/pulses/fig_pulses_cd_sources.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
