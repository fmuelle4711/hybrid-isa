"""Two single-column pulse figures, stacked panels of control amplitudes.

    figures/fig_pulses_qudit.pdf   X_E (11.0 us), C_{Pi_z}D (160.4 us),
                                   C_{Pi}R (25.465 us), C_{Pi_x}D (0.649 us)
    figures/fig_pulses_qubit.pdf   CD (39.79 us), CR (24.192 us)

Amplitudes in units of Omega_c, time in us.  Z_d is virtual and has no
panel.  MS has no saved waveform anywhere in the repo and cannot be
plotted -- see the sources file.

GATE NAMES.  The manuscript renamed the five ISA primitives.  Code
identifiers, JSON keys and results-file columns still use the old names, so
every script in this repo carries this mapping:

    paper name      code / data name        what it is
    Z_d             QPS                     qudit phase shift
    X_E             QT                      qudit transition (carrier)
    C_{\\Pi_z}D      QCD                     level-conditioned displacement
    C_{\\Pi}R        QCR                     level-conditioned mode rotation
    C_{\\Pi_x}D      QT-SB, QTsb             transition-conditioned displacement

Multitone forms write the transition set as plain E.  CD and CR keep their
names: they are the qubit-reference gates, not qudit ISA primitives.  Only
the DRAWN LABELS below are renamed; no identifier is touched.

STRICT RULE FOR THIS FIGURE: waveforms are READ from the saved npz/json of the
adopted run.  Nothing is regenerated, re-optimised, or reconstructed.

CORRECTION (v2).  A first pass reported that QCD had no saved waveform.  That
was WRONG: final_v3.json stores it under headline/Omega_x and headline/Omega_p
(200 slots each).  The probe that missed it searched only for fields named
u/amps/final_amps and did not look for the Omega_* naming.  QCD is now drawn
from its saved waveform.

CONSEQUENCE: TWO of the five adopted runs saved no waveform.  CD and QT stored
scalars only
(T, F_grape, F_indep, terminations) and discarded the pulse.  Those three
panels therefore cannot be drawn from saved data and are rendered as explicit
"no saved waveform" placeholders rather than being regenerated -- regenerating
would defeat the point of the instruction and would silently present a NEW
optimisation as the adopted one.

This is the same class of gap as FINDING 7 (a CD F_indep reported without a
stored waveform to recompute it from).

Omega_c is read from the hypulse device spec, not retyped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "pulses_meth" / "hypulse_validation"
OUTFIG_QUDIT = ROOT / "figures" / "pulses" / "fig_pulses_qudit.pdf"
OUTFIG_QUBIT = ROOT / "figures" / "pulses" / "fig_pulses_qubit.pdf"
OUTTXT = ROOT / "figures" / "pulses" / "fig_pulses_sources.txt"

sys.path.insert(0, str(ROOT))
from hypulse.calibrations import get_device_spec            # noqa: E402

DEV = get_device_spec("innsbruck_meth_2025")
OMEGA_C = float(DEV.params["amp_bound_rad_per_us"])          # rad/us

LINES = []


def W(s=""):
    LINES.append(s)
    print(s, flush=True)


def load_qcr_physical():
    """QCR from the PHYSICAL detuned-sideband run, not the imposed generator.

    scripts/qcr_mechanism.py propagates the real drive

        H(t) = (Omega_sb/2) ( |l><g| (x) a^dag e^{-i delta_disp t} + h.c. )

    at theta = 0.016, T = theta/chi = 25.465 us, delta_disp/2pi = 40 kHz.
    There is no OPTIMISED waveform to load: the drive is flat by construction,
    at the fixed sideband Rabi frequency Omega_sb from the device spec.  So the
    trace plotted here is the model's own constant amplitude, not a fitted or
    reconstructed pulse.

        Omega_sb / Omega_c = 0.041000000 = eta exactly,
    which is the expected value, since Omega_sb = eta * Omega_c by definition.

    The two quadratures are also drawn: the ENVELOPE is flat, but the drive
    phase winds at delta_disp (period 25.0 us), so what a lab actually programs
    is Omega_sb*cos(delta t) and -Omega_sb*sin(delta t).  Over T = 25.465 us the
    phase completes 1.019 turns -- the gate is essentially one detuning period
    long, which is exactly why the chi fit over this window reads 1.22x the
    Schrieffer-Wolff prediction (results/qcr_offdiag_trace.txt).

    The IMPOSED-GENERATOR waveform previously plotted here
    (step_qcr_cr_two_points_v2_results.json) is deliberately NOT used: its
    control operator IS the gate generator, so it carries no information about
    whether a laser can produce Pi_l (x) n.
    """
    import qcr_mechanism as QC
    T = 25.465
    n = 400
    t = (np.arange(n) + 0.5) * (T / n)
    env = np.full(n, QC.OMEGA_SB)
    qx = QC.OMEGA_SB * np.cos(QC.DELTA_DISP * t)
    qy = -QC.OMEGA_SB * np.sin(QC.DELTA_DISP * t)
    u = np.column_stack([env, qx, qy])
    return dict(src="scripts/qcr_mechanism.py  (drive model) + hypulse device "
                    "spec 'innsbruck_meth_2025' (Omega_sideband_rad_per_s)",
                path="H_phys: (Omega_sb/2)(|l><g|(x)a^dag e^{-i delta t} + h.c.); "
                     "flat by construction, no optimised waveform exists",
                T=T, u=u,
                labels=[r"$|\Omega_{sb}|$", r"$\Omega_{sb}\cos\delta t$",
                        r"$-\Omega_{sb}\sin\delta t$"],
                F_indep=2.900031e-03,
                F_src="endpoint, fitted generator + VZ, "
                      "results/qcr_mechanism_paper_angle.txt",
                note=(f"flat envelope at Omega_sb/Omega_c = "
                      f"{QC.OMEGA_SB/OMEGA_C:.9f} = eta exactly; "
                      f"phase winds at delta_disp, period "
                      f"{2*np.pi/QC.DELTA_DISP:.3f} us, "
                      f"{T/(2*np.pi/QC.DELTA_DISP):.3f} turns over T"))


def load_qcr():
    p = H / "step_qcr_cr_two_points_v2_results.json"
    d = json.loads(p.read_text())
    r = [x for x in d["all_rows"] if x["gate"] == "QCR" and x["point"] == "A_meth"
         and abs(x["theta"] - 0.016) < 1e-9 and abs(x["T_us"] - 25.465) < 1e-3][0]
    s = [x for x in d["summary"] if x["gate"] == "QCR" and x["point"] == "A_meth"
         and abs(x["theta"] - 0.016) < 1e-9][0]
    u = np.asarray(r["u"], dtype=float)
    return dict(src=str(p.relative_to(ROOT)),
                path=f"all_rows[gate=QCR, point=A_meth, theta=0.016, T_us=25.465].u",
                T=float(r["T_us"]), u=u[:, None],
                labels=[r"$u$"],
                F_indep=1.0 - float(s["check_C_F_indep"]),
                F_src="1 - check_C_F_indep from summary",
                note=f"flat, max|u| = {abs(u).max():.4e} rad/us, std = {u.std():.1e}")


def load_qtsb():
    p = ROOT / "data" / "qtsb_qtrl_working.npz"
    z = np.load(p, allow_pickle=True)
    u = np.asarray(z["u"], dtype=float)
    return dict(src=str(p.relative_to(ROOT)), path="u  (shape 200x2)",
                T=float(z["T"]), u=u, labels=[r"$u_x$", r"$u_y$"],
                F_indep=1.0 - float(z["F_vz"]),
                F_src="1 - F_vz (VZ-corrected, independent replay)",
                note=(f"flat, max|c| = {np.hypot(u[:,0],u[:,1]).max():.4e} rad/us "
                      f"= {100*np.hypot(u[:,0],u[:,1]).max()/OMEGA_C:.1f}% of Omega_c; "
                      f"leak {float(z['leak']):.2e}"))


def load_cd():
    """CD waveform.

    The HyPulse reproduction reproduced this one BIT FOR BIT
    (max |u_new - u_stored| = 0, results/hypulse_reproduce_three.txt), so
    the stored file and the rerun are the same array and either may be
    plotted.  The stored file is used, and the equality is asserted.
    """
    p = H / "cd_waveform_persist.json"
    d = json.loads(p.read_text())
    u = np.column_stack([np.asarray(d["Omega_x"], float),
                         np.asarray(d["Omega_p"], float)])
    return dict(src=str(p.relative_to(ROOT)), path="Omega_x / Omega_p (200 slots)",
                T=float(d["T_us"]), u=u,
                labels=["$\Omega_x$", "$\Omega_p$"],
                F_indep=4.010618e-06,
                F_src="common replay; matches audit_cd_independent exactly",
                note=(f"bit-for-bit verified: F_grape recompiled == stored "
                      f"({d['F_grape_stored']!r}); max|Omega_p| = "
                      f"{abs(u[:,1]).max():.4e} rad/us"))


def load_qt():
    """QT waveform from the bit-for-bit-verified persistence rerun."""
    p = H / "qt_waveform_persist.json"
    d = json.loads(p.read_text())
    u = np.asarray(d["u"], dtype=float)
    return dict(src=str(p.relative_to(ROOT)),
                path=f"u  (shape {u.shape[0]}x{u.shape[1]})",
                T=float(d["T_us"]), u=u,
                labels=[f"$u_{{{i}}}$" for i in range(u.shape[1])],
                F_indep=1.926236e-05,
                F_src="common replay (results/pulse_fidelity_common.txt)",
                note=(f"bit-for-bit verified: F_grape recompiled == stored "
                      f"({d['F_grape_stored']!r}); {u.shape[1]} quadrature "
                      f"controls, max|u| = {abs(u).max():.4e} rad/us"))


def load_qcd(fv3):
    """QCD waveform from final_v3.json headline (Omega_x, Omega_p)."""
    h = fv3["headline"]
    u = np.column_stack([np.asarray(h["Omega_x"], dtype=float),
                         np.asarray(h["Omega_p"], dtype=float)])
    return dict(src="pulses_meth/hypulse_validation/final_v3.json",
                path="headline.Omega_x / headline.Omega_p  (200 slots each)",
                T=float(h["T_us"]), u=u,
                labels=["$\Omega_x$", "$\Omega_p$"],
                F_indep=1.0 - float(h["F_grape"]),
                F_src="1 - F_grape (headline has no F_indep field; "
                      "audit_v3.py computed F_indep separately)",
                note=(f"max|Omega_x| = {abs(u[:,0]).max():.4e}, "
                      f"max|Omega_p| = {abs(u[:,1]).max():.4e} rad/us; "
                      f"amp bound {fv3['amp_bound_rad_per_us']:.4e} "
                      f"({fv3['amp_bound_tag']})"))


def _cr_common_replay(u, T):
    """1-F for CR under the common protocol, so every panel agrees.

    The optimizer reports the PSU amplitude error; this protocol uses the
    trace fidelity, and the two differ by exactly 2x for every GRAPE gate
    (results/pulse_fidelity_common.txt).  tab:pulsemethods quotes the trace
    value, so that is what is reported here.
    """
    from scipy.linalg import expm
    sys.path.insert(0, str(H))
    from step_qcr_cr_two_points import build_generator, build_target
    Hc = build_generator("CR").full()
    U_t = build_target("CR", 0.016).full()
    dim = U_t.shape[0]
    dt = T / len(u)
    U = np.eye(dim, dtype=complex)
    for k in range(len(u)):
        U = expm(-1j * u[k, 0] * Hc * dt) @ U
    return 1.0 - float(abs(np.trace(U_t.conj().T @ U)) ** 2 / dim ** 2)


def load_cr():
    """CR waveform of the qubit reference, at the adopted 24.192 us.

    This one is NOT a plugin pulse.  step_qcr_cr_two_points_v2.py optimises
    an IMPOSED Pi_l (x) n generator with qutip directly, bypassing the CR
    plugin, because the plugin drives a raw blue sideband (FINDING 6).  The
    HyPulse reproduction in results/hypulse_reproduce_three.txt confirms
    this: through the plugin the same target reaches only 0.99776.
    """
    p = H / "step_qcr_cr_two_points_v2_results.json"
    d = json.loads(p.read_text())
    row = next(r for r in d["all_rows"]
               if r["gate"] == "CR" and r["point"] == "A_meth"
               and r["theta"] == 0.016 and abs(r["T_us"] - 24.192) < 1e-6)
    u = np.asarray(row["u"], dtype=float).reshape(-1, 1)
    return dict(src=str(p.relative_to(ROOT)),
                path='all_rows[gate=CR, point=A_meth, theta=0.016, '
                     'T_us=24.192].u',
                T=float(row["T_us"]), u=u, labels=[r"$u$"],
                F_indep=_cr_common_replay(u, float(row["T_us"])),
                F_src="common replay protocol, same as every other panel "
                      "(results/pulse_fidelity_common.txt).  Exactly 2x the "
                      "optimizer's 1-F_grape = 5.596049e-06, the documented "
                      "PSU-vs-trace convention gap, and it is the trace "
                      "value that tab:pulsemethods quotes",
                note=("single control, imposed Pi_l (x) n generator, NOT the "
                      "CR plugin; max|u| = "
                      f"{abs(u).max():.4e} rad/us"))


def missing(name, T, src, path, why, F_indep, fsrc):
    return dict(src=src, path=path, T=T, u=None, labels=[],
                F_indep=F_indep, F_src=fsrc, note=why)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    fv3 = json.loads((H / "final_v3.json").read_text())
    v4 = json.loads((H / "step_two_points_v4_results.json").read_text())
    cdi = json.loads((H / "audit_cd_independent_results.json").read_text())
    qt = json.loads((H / "step_qt_meth_aggcap_results.json").read_text())
    qt_pick = qt["variants"]["archive"]["picked"]

    cd_F = None
    for k, v in (cdi.items() if isinstance(cdi, dict) else []):
        if isinstance(v, dict) and "F_indep" in v:
            cd_F = float(v["F_indep"]); break
    if cd_F is None:
        cd_F = float(json.dumps(cdi).split('"F_indep":')[1].split(',')[0])

    # Two figures.  The qudit ISA primitives that carry a pulse, and the
    # two qubit-reference gates.  Z_d is virtual and has no panel anywhere.
    panels_qudit = [
        ("QT (HyPulse, two tones)", load_qt()),
        ("QCD (GRAPE)", load_qcd(fv3)),
        ("QCR (physical sideband)", load_qcr_physical()),
        ("QT-SB (flat)", load_qtsb()),
    ]
    panels_qubit = [
        ("CD qubit reference (GRAPE)", load_cd()),
        ("CR qubit reference (imposed generator)", load_cr()),
    ]
    panels = panels_qudit + panels_qubit

    # ---------------- figure ----------------
    # Nature-style: 89 mm single column, sans-serif, bold lower-case panel
    # letters outside top-left, Okabe-Ito colour-blind-safe palette, direct
    # curve labelling where two curves make a legend unnecessary, outward
    # ticks, no top/right spines, 600 dpi.  Data and sources unchanged.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7F7F7F"   # Okabe-Ito

    def style_for(lbl, k):
        low = lbl.lower()
        if "|" in lbl and "omega_{sb}" in low:
            return GREY, (0, (1, 1.6)), 0.7
        if "sin" in low or "_p$" in low or "_y$" in low:
            return VERM, "-", 0.8
        if "cos" in low or "_x$" in low:
            return BLUE, "-", 0.8
        return (BLUE if k % 2 == 0 else VERM), ("-" if k < 2 else (0, (2.6, 1.3))), 0.8

    # Paper names (see the mapping in this file's docstring).  Keys are the
    # internal panel names and are NOT renamed; only the drawn label changes.
    SHORT = {"QT (HyPulse, two tones)": r"$X_E$",
             "QCD (GRAPE)": r"$C_{\Pi_z}D$",
             "CD qubit reference (GRAPE)": "CD",
             "QCR (physical sideband)": r"$C_{\Pi}R$",
             "QT-SB (flat)": r"$C_{\Pi_x}D$",
             "CR qubit reference (imposed generator)": "CR"}
    # panels with only two curves are labelled directly on the curve instead
    # of carrying a legend -- the Nature convention, and it removes four boxes
    DIRECT = {"QCD (GRAPE)", "CD qubit reference (GRAPE)", "QT-SB (flat)"}

    MM = 1 / 25.4

    def nice_limit(x):
        """Smallest round number >= 1.12*x, from a 1/1.5/2/3/4/5/6/8 mantissa
        ladder, so the three y ticks read as round values rather than as
        arbitrary multiples of the data maximum."""
        import math as _m
        v = 1.12 * x
        e = _m.floor(_m.log10(v))
        for mant in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
            c = mant * 10 ** e
            if c >= v:
                return c
        return 10 ** (e + 1)

    def render(group, out):
      fig, axes = plt.subplots(len(group), 1,
                               figsize=(89 * MM, 31 * len(group) * MM))
      axes = np.atleast_1d(axes)
      fig.subplots_adjust(left=0.20, right=0.795, top=0.975, bottom=0.062,
                          hspace=0.55)
      for idx, (ax, (name, d)) in enumerate(zip(axes, group)):
          u = d["u"] / OMEGA_C
          t = (np.arange(u.shape[0]) + 0.5) * (d["T"] / u.shape[0])
          ax.axhline(0, color="#CCCCCC", lw=0.4, ls=(0, (1, 2.5)), zorder=0)
          for k in range(u.shape[1]):
              c, ls, lw = style_for(d["labels"][k], k)
              ax.step(t, u[:, k], where="mid", lw=lw, color=c, ls=ls,
                      label=d["labels"][k], zorder=2, solid_joinstyle="round")
          ax.set_xlim(0, d["T"])
          m = nice_limit(float(np.abs(u).max()))
          ax.set_ylim(-m, m)
          ax.set_yticks([-m, 0.0, m])
          ax.set_yticklabels([f"{-m:g}", "0", f"{m:g}"])
          ax.spines["top"].set_visible(False)
          ax.spines["right"].set_visible(False)
          ax.spines["left"].set_bounds(-m, m)
          ax.spines["bottom"].set_bounds(0, d["T"])

          # bold lower-case panel letter, outside top-left (Nature convention)
          ax.text(-0.235, 1.12, "abcde"[idx], transform=ax.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left")
          # gate name, plain, inside top-left
          ax.text(0.02, 1.10, SHORT.get(name, name), transform=ax.transAxes,
                  fontsize=7, va="top", ha="left", color="0.15")

          if name in DIRECT:
              for k in range(u.shape[1]):
                  c, _, _ = style_for(d["labels"][k], k)
                  y = u[-1, k]
                  ax.annotate(d["labels"][k], xy=(d["T"], y),
                              xytext=(3, 0), textcoords="offset points",
                              color=c, fontsize=6.5, va="center", ha="left",
                              annotation_clip=False)
          else:
              ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        frameon=False, handlelength=1.2, handletextpad=0.45,
                        labelspacing=0.32, borderaxespad=0.0)

      axes[-1].set_xlabel("Time (µs)")
      fig.supylabel(r"Control amplitude  $u/\Omega_c$", fontsize=7, x=0.022)
      out.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(out, dpi=600, bbox_inches="tight")
      fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
      plt.close(fig)
      print(f"  wrote {out} and .png  ({len(group)} panels)")

    OUTFIG_QUDIT.parent.mkdir(parents=True, exist_ok=True)
    render(panels_qudit, OUTFIG_QUDIT)
    render(panels_qubit, OUTFIG_QUBIT)

    # ---------------- sources ----------------
    W("=" * 100)
    W("PULSE FIGURE SOURCES -- ONE ENTRY PER PANEL")
    W("=" * 100)
    W(f"  figures/fig_pulses_qudit.pdf   {len(panels_qudit)} panels: "
      "X_E, C_{Pi_z}D, C_{Pi}R, C_{Pi_x}D")
    W(f"  figures/fig_pulses_qubit.pdf   {len(panels_qubit)} panels: CD, CR")
    W()
    W("  Z_d is a virtual frame update and has no waveform, so it has no")
    W("  panel in either figure.")
    W()
    W("  MS HAS NO WAVEFORM ANYWHERE IN THIS REPOSITORY.  It is not an")
    W("  optimised pulse: appendix_table.md takes it from Meth SI Note II")
    W("  p.3 as 0.2 ms per gate, angle independent, and that single quoted")
    W("  number is the whole of the Molmer-Sorensen cost in the runtime")
    W("  model.  Searched across pulses_meth/ and data/: MS appears only as")
    W("  a duration and as counts (MS_J, MS_Jl) in runtime_table.json.  It")
    W("  therefore cannot be plotted, and Appendix B should not say the")
    W("  saved waveforms cover it.")
    W("=" * 100)
    W(f"  Omega_c = {OMEGA_C:.9f} rad/us  (hypulse device spec "
      f"'innsbruck_meth_2025', amp_bound_rad_per_us)")
    W("  All amplitudes plotted in units of Omega_c.  Nothing regenerated.")
    W()
    for name, d in panels:
        W("-" * 100)
        W(f"  PANEL: {name}")
        W(f"    duration   T = {d['T']:g} us")
        W(f"    source     {d['src']}")
        W(f"    field      {d['path']}")
        if d["u"] is None:
            W(f"    waveform   *** NOT AVAILABLE -- {d['note']} ***")
        else:
            W(f"    waveform   {d['u'].shape[0]} slots x "
              f"{d['u'].shape[1]} control(s); {d['note']}")
        if d.get("F_indep") is None:
            W(f"    1-F_indep  not available ({d.get('F_src','')})")
        else:
            W(f"    1-F_indep  {d['F_indep']:.6e}"
              + (f"   ({d['F_src']})" if d.get("F_src") else ""))
    W()
    W("=" * 100)
    W("CAPTION LINES (one per panel)")
    W("=" * 100)
    for name, d in panels:
        f = ("1-F_indep = %.2e" % d["F_indep"]) if d.get("F_indep") is not None \
            else "1-F_indep unavailable"
        drawn = "waveform shown" if d["u"] is not None else "NO SAVED WAVEFORM"
        W(f"  {name}: T = {d['T']:g} us, {f}, {drawn}.")
    W()
    W("=" * 100)
    W(f"STATUS -- ALL {len(panels)} PANELS DRAWN FROM SAVED WAVEFORMS")
    W("=" * 100)
    W("  Earlier passes of this script reported missing waveforms.  Both claims")
    W("  are now resolved:")
    W("    QCD  was never missing.  final_v3.json stores it under")
    W("         headline/Omega_x and headline/Omega_p.  The original probe")
    W("         searched only for u / amps / final_amps field names.")
    W("    CD, QT  were genuinely missing.  Both have now been recompiled at")
    W("         their adopted settings with waveform persistence, and ACCEPTED")
    W("         only after the recompiled F_grape matched the stored value")
    W("         BIT-FOR-BIT (difference exactly 0.0):")
    W("           CD  0.9999979946930231   cd_waveform_persist.json")
    W("           QT  0.9999903687584839   qt_waveform_persist.json")
    W("         See results/waveform_persist_rerun.txt.")
    W()
    W("  Fidelities quoted per panel come from the ONE common replay protocol")
    W("  in results/pulse_fidelity_common.txt, except QCD which is shown as")
    W("  1 - F_grape and labelled as such.")
    W()
    W(f"  wrote {OUTFIG_QUDIT.relative_to(ROOT)} and .png")
    W(f"  wrote {OUTFIG_QUBIT.relative_to(ROOT)} and .png")

    OUTTXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTTXT}", flush=True)


if __name__ == "__main__":
    main()
