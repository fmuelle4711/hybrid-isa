"""One table: gate, test angle, test duration, circuit angle, scaling rule,
circuit duration used.  Nc = 3.  All read from files, no reruns.
GATE NAMES.  The manuscript renamed the five ISA primitives.  Code
identifiers, dict keys and results-file columns still use the old names, so
this mapping travels with every script that touches them:

    paper name      code / data name      what it is
    Z_d             QPS                   qudit phase shift
    X_E             QT                    qudit transition (carrier)
    C_{\Pi_z}D      QCD                   level-conditioned displacement
    C_{\Pi}R        QCR                   level-conditioned mode rotation
    C_{\Pi_x}D      QT-SB, QTsb           transition-conditioned displacement

Multitone forms write the transition set as plain E.  CD and CR keep their
names: they are the qubit-reference gates, not qudit ISA primitives.  No
identifier is renamed anywhere below.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "pulses_meth" / "hypulse_validation"
OUT = ROOT / "results" / "duration_rule_table.txt"
L = []
def W(s=""): L.append(s); print(s, flush=True)

def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    rt = json.loads((ROOT/"pulses_meth"/"runtime_table.json").read_text())
    g = rt["gate_times_us"]; qlo, qhi = rt["conventions"]["qt_bounds_us"]
    r3 = rt["rows"][0]

    # test values, read from the originating runs
    v4 = json.loads((H/"step_two_points_v4_results.json").read_text())
    cd = [x for x in v4["sweep_rows"] if x["gate"]=="CD" and x["point"]=="A_meth"
          and x["amp_variant"]=="CD_main" and abs(x["T_us"]-39.79)<1e-9][0]
    qcr = json.loads((H/"step_qcr_cr_two_points_v2_results.json").read_text())
    cr = [s for s in qcr["summary"] if s["gate"]=="CR" and s["point"]=="A_meth"
          and abs(s["theta"]-0.016)<1e-9][0]
    fv3 = json.loads((H/"final_v3.json").read_text())["headline"]
    qt = json.loads((H/"step_qt_meth_aggcap_results.json").read_text())
    qtp = qt["variants"]["archive"]["picked"]

    rows = [
      ("QT",    "pi/4 = 0.7854", f"{qtp['T_us']:.3f}", "0.0075",
       "linear in angle (bracket)", f"[{qlo}, {qhi}]"),
      ("QCD",   "alpha = 0.5", f"{fv3['T_us']:.1f}", "alpha = 0.008",
       "linear in alpha: T x 0.008/0.5", f"{g['QCD']:.4f}"),
      ("CD",    "alpha = 0.5", f"{cd['T_us']:.2f}", "alpha = 0.008",
       "own run UNUSED; takes QCD [1]", f"{g['QCD']:.4f}"),
      ("CR",    "theta = 0.016", f"{cr['T_grape_us']:.3f}", "theta = 0.016",
       "own run UNUSED; takes QCR [1]", f"{g['QCR']:.2f}"),
      ("QCR",   "theta = 0.016", "25.465", "theta = 0.016",
       "T=theta/chi, then echo 2T+t_pi [2]", f"{g['QCR']:.2f}"),
      ("QT-SB", "0.0015 / 0.002025", "0.0597", "same as test",
       "NOT angle-based: 0.5 x 2pi/nu [3]", f"{g['QT_SB']:.6f}"),
      ("MS",    "none", "200", "none",
       "angle-independent (Meth) [4]", f"{g['MS']:.1f}"),
    ]
    W("="*128)
    W("DURATION RULES, Nc = 3   (all values read from files; no reruns)")
    W("="*128)
    W(f"{'gate':>6} {'test angle':>19} {'test T (us)':>12} {'circuit angle':>15} "
      f"{'scaling rule':<36} {'circuit T used (us)':>21}")
    W("-"*128)
    for a,b,c,d,e,f in rows:
        W(f"{a:>6} {b:>19} {c:>12} {d:>15} {e:<36} {f:>21}")
    W("-"*128)
    W()
    W(f"  counts at Nc=3:  qudit {r3['qudit_counts']}")
    W(f"                   qubit {r3['qubit_counts']}  + {r3['ms_windows']} MS windows")
    W()
    W("  [1] CD and CR have their OWN pulse runs but the table does not use them;")
    W("      both read the qudit keys.  Scaled to the circuit angle their own")
    W(f"      runs give CD {cd['T_us']*0.016:.4f} us (table uses {g['QCD']:.4f}) and")
    W(f"      CR {cr['T_grape_us']:.3f} us (table uses {g['QCR']:.2f}).")
    W("  [2] chi = Omega_sb^2/(4 delta_disp); t_pi = 5 us is ASSUMED.")
    W(f"      2 x 25.465 + 5 = {g['QCR']:.2f}.")
    W("  [3] the angle sets the AMPLITUDE, not the duration.  The old angle-based")
    W("      0.0597 us is 0.046 motional periods and does not function.")
    W("  [4] Meth quote 0.2 ms per gate independent of angle; the entire runtime")
    W("      asymmetry rests on that simplification.")
    W()
    W("  NOTES")
    W("  1. CD and CR have their own pulse runs but the table does not use them;")
    W("     both read the qudit keys.  Both substitutions charge the qubit MORE")
    W(f"     (CR 55.93 vs 24.192 = 2.31x; CD 2.5664 vs 0.6366 = 4.03x), so the")
    W("     reported advantage is conservative with respect to them.")
    W("  2. Only QCR/CR was tested AT the circuit angle.  QCD/CD were tested at")
    W("     62.5x the circuit alpha and QT at 105x its circuit angle, then")
    W("     extrapolated down linearly.")
    W("  3. QT-SB is the one gate whose duration is NOT set by its angle -- the")
    W("     old angle-based value (0.0597 us) was 0.046 motional periods and did")
    W("     not function; see results/qtsb_duration_sweep.txt.")
    W("  4. MS is the only gate with no angle dependence at all, by Meth's own")
    W("     framing; the entire runtime asymmetry rests on that simplification.")
    OUT.write_text("\n".join(L)+"\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

main()
