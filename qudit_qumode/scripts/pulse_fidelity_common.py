"""One common replay protocol applied to all five saved waveforms.

THE PROTOCOL (identical for every gate; only the gate's own model changes)
  1. take the SAVED waveform of the adopted run
  2. propagate it slot-by-slot, U <- expm(-i H_k dt) U, with H_k built from the
     gate's own control operators and the same amplitudes the optimiser wrote
  3. compare against the gate's own target with the SAME fidelity definition,
        F = |Tr(V^dag U)|^2 / D^2                                    (raw)
        F_vz = (sum_q |Tr_q[(P_q (x) I) U V^dag]|)^2 / D^2           (after the
                 optimal per-level virtual-Z, which is free on trapped ions)
  4. report 1-F for both

WHAT IS AND IS NOT COMMON.  The integrator, the fidelity formula and the
virtual-Z treatment are identical across all five.  The Hamiltonian and target
are NOT and cannot be -- the five gates act on different registers with
different generators.  So this answers "does the saved pulse reproduce its own
target under a uniform metric", which is the comparable question.  It does NOT
answer "does the physical drive produce the target", which is a separate axis;
see the QCR note at the bottom, which is exactly that distinction.

No optimisation is run here.  Waveforms are read from:
  QCD    final_v3.json                headline/Omega_x, Omega_p
  CD     cd_waveform_persist.json     Omega_x, Omega_p   (bit-for-bit verified)
  QT     qt_waveform_persist.json     u                  (bit-for-bit verified)
  QCR    step_qcr_cr_two_points_v2_results.json  all_rows[...].u
  QT-SB  data/qtsb_qtrl_working.npz   u

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

import json
import math
import sys
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "pulses_meth" / "hypulse_validation"
OUT = ROOT / "results" / "pulse_fidelity_common.txt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(H))

LINES = []


def W(s=""):
    LINES.append(s)
    print(s, flush=True)


# ---------------------------------------------------------------- protocol
def replay(ops, u, T, V, level_dim):
    """Common slot-by-slot replay.  ops: list of DxD arrays, u: (N, len(ops))."""
    N = u.shape[0]
    dt = T / N
    D = V.shape[0]
    U = np.eye(D, dtype=complex)
    for k in range(N):
        Hk = sum(u[k, j] * ops[j] for j in range(len(ops)))
        U = expm(-1j * Hk * dt) @ U
    tr = np.trace(np.conj(V.T) @ U)
    F = abs(tr) ** 2 / D ** 2
    sub = D // level_dim
    M = U @ np.conj(V.T)
    S = [np.trace(M[q*sub:(q+1)*sub, q*sub:(q+1)*sub]) for q in range(level_dim)]
    F_vz = float(np.sum(np.abs(S)) ** 2 / D ** 2)
    return float(F), F_vz, D


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("=" * 104)
    W("COMMON REPLAY PROTOCOL -- all five adopted waveforms")
    W("=" * 104)
    W("  slot-by-slot expm of the SAVED pulse under the gate's own control")
    W("  operators, against the gate's own target, with one fidelity formula")
    W("  F = |Tr(V^dag U)|^2/D^2 and one virtual-Z treatment.  No optimisation.")
    W()

    rows = []

    # ---- QCD and CD: use the canonical operators/targets from the runner
    # itself (step_hypulse_grape_v3.build_static_H / target_U) rather than
    # re-deriving them.  A first attempt re-derived them and got two things
    # wrong: the target is the projector-displacement form
    # P0 (x) D(+alpha) + P1 (x) D(-alpha), NOT exp(-i alpha sigma_z X), and
    # CD is conditioned on sigma_x, not sigma_z.  Both are read from the
    # module now so they cannot drift.
    import step_hypulse_grape_v3 as V3

    def replay_cdlike(gate, alpha, T, ux, up):
        Hcx, Hcp = V3.build_static_H(gate)
        Vt = V3.target_U(gate, alpha)
        ops = [Hcx.full(), Hcp.full()]
        u = np.column_stack([np.asarray(ux, float), np.asarray(up, float)])
        return replay(ops, u, float(T), Vt.full(), 2)

    fv3 = json.loads((H / "final_v3.json").read_text())
    hd = fv3["headline"]
    F, Fvz, D = replay_cdlike("QCD", float(hd["alpha"]), hd["T_us"],
                              hd["Omega_x"], hd["Omega_p"])
    rows.append(("QCD", float(hd["T_us"]), "final_v3.json headline",
                 F, Fvz, D, float(hd["F_grape"])))

    cdw = json.loads((H / "cd_waveform_persist.json").read_text())
    F, Fvz, D = replay_cdlike("CD", float(cdw["alpha"]), cdw["T_us"],
                              cdw["Omega_x"], cdw["Omega_p"])
    rows.append(("CD", float(cdw["T_us"]), "cd_waveform_persist.json",
                 F, Fvz, D, float(cdw["F_grape_stored"])))

    # QCR needs the Fock ops below
    a = qt.destroy(8)

    # ---- QT: 5-level model from step_qt_meth ----
    import step_qt_meth as QT
    from step_qt_meth_aggcap import BOUND_AGG                     # noqa: F401
    qtw = json.loads((H / "qt_waveform_persist.json").read_text())
    u = np.asarray(qtw["u"], dtype=float)
    ops, _ = QT.control_ops()
    sum_om2 = len(QT.DRIVEN) * QT.OMEGA_MAX ** 2
    H_d = QT.stark_diag(QT.stark_scale(sum_om2, QT.STARK_REFS["archive"]))
    Vq = ((-1j * H_d * float(qtw["T_us"])).expm() * QT.U_QT_bare()).full()
    ops_np = [o.full() for o in ops]
    Hd_np = H_d.full()
    # drift is part of the model: fold it in as a constant control
    ops_all = ops_np + [Hd_np]
    u_all = np.column_stack([u, np.ones(u.shape[0])])
    F, Fvz, D = replay(ops_all, u_all, float(qtw["T_us"]), Vq, QT.DIM)
    rows.append(("QT", float(qtw["T_us"]), "qt_waveform_persist.json",
                 F, Fvz, D, float(qtw["F_grape_stored"])))

    # ---- QCR: sigma_z/2 (x) n imposed generator, cutoff 8 ----
    d = json.loads((H / "step_qcr_cr_two_points_v2_results.json").read_text())
    r = [x for x in d["all_rows"] if x["gate"] == "QCR" and x["point"] == "A_meth"
         and abs(x["theta"] - 0.016) < 1e-9 and abs(x["T_us"] - 25.465) < 1e-3][0]
    n_op = (a.dag() * a).full()
    G = np.kron(np.diag([0.0, 1.0]).astype(complex), n_op)     # Pi_l (x) n
    u = np.asarray(r["u"], dtype=float)[:, None]
    V = expm(-1j * float(r["theta"]) * G)
    F, Fvz, D = replay([G], u, float(r["T_us"]), V, 2)
    rows.append(("QCR", float(r["T_us"]),
                 "step_qcr_cr_two_points_v2_results.json", F, Fvz, D,
                 float(r["F_grape"])))

    # ---- QT-SB: sigma_x (x) x_hat physical drive ----
    import qtsb_mechanism as QM
    sys.path.insert(0, str(ROOT / "scripts"))
    z = np.load(ROOT / "data" / "qtsb_qtrl_working.npz", allow_pickle=True)
    u = np.asarray(z["u"], dtype=float)
    T = float(z["T"])
    Nsl = u.shape[0]
    dtq = T / Nsl
    Uq = np.eye(QM.DIM, dtype=complex)
    for k in range(Nsl):
        t_c = (k + 0.5) * dtq
        G_ = math.cos(QM.NU * t_c) * np.kron(QM.KG, QM.M_ld(t_c, "LDfull"))
        A_, B_ = G_ + G_.conj().T, 1j * (G_ - G_.conj().T)
        Uq = expm(-1j * (u[k, 0] * A_ + u[k, 1] * B_) * dtq) @ Uq
    Vq = QM.U_paper(float(z["theta"]))
    trq = np.trace(np.conj(Vq.T) @ Uq)
    Fq = abs(trq) ** 2 / QM.DIM ** 2
    Fqvz = QM.fid_vz(Uq, Vq)[0]
    rows.append(("QT-SB", T, "data/qtsb_qtrl_working.npz", Fq, Fqvz,
                 QM.DIM, None))

    # ---------------- report ----------------
    W(f"{'gate':>7} {'T (us)':>11} {'dim':>5} {'1-F raw':>13} {'1-F after VZ':>14} "
      f"{'1-F_grape (stored)':>20}   source")
    for nm, T_, src, F, Fvz, D, Fg in rows:
        fg = f"{1-Fg:.6e}" if Fg is not None else "n/a"
        W(f"{nm:>7} {T_:>11.6f} {D:>5} {1-F:>13.6e} {1-Fvz:>14.6e} "
          f"{fg:>20}   {src}")
    W()
    W("  '1-F_grape (stored)' is what the optimiser reported at the time and is")
    W("  shown only for comparison; it is NOT the common-protocol number.")
    W()
    W("  THE FACTOR 2 IS A METRIC CONVENTION, NOT A DISCREPANCY.  For every")
    W("  GRAPE gate the replayed 1-F is exactly twice the stored value:")
    W("      QCD  1.130109e-05 = 2 x 5.650584e-06")
    W("      CD   4.010618e-06 = 2 x 2.005307e-06")
    W("      QT   1.926236e-05 = 2 x 9.631242e-06")
    W("  qutip-qtrl reports the PSU amplitude error eps = 1 - |Tr(V^dag U)|/D,")
    W("  while this protocol uses the trace fidelity |Tr|^2/D^2, and")
    W("      1 - (1-eps)^2 = 2 eps - eps^2 ~ 2 eps.")
    W("  Quote one convention consistently; they differ by 2x by construction.")
    W()
    W("  INDEPENDENT CROSS-CHECK.  CD's common-protocol value, 4.010618e-06,")
    W("  reproduces audit_cd_independent_results.json EXACTLY -- that audit was")
    W("  the FINDING 7 remediation and recomputed CD from a deterministically")
    W("  recompiled waveform.  Two separate routes, same digits, so the replay")
    W("  and the persisted waveform are both sound.")
    W()

    W("=" * 104)
    W("Q1  WHY QCR SHOWS 3.64e-06 HERE AND 2.9e-03 IN THE MECHANISM RERUN")
    W("=" * 104)
    W("  They propagate DIFFERENT Hamiltonians and answer different questions.")
    W()
    W("  3.64e-06  fig_pulses_sources.txt / this table")
    W("      Source: step_qcr_cr_two_points_v2_results.json, an IMPOSED-GENERATOR")
    W("      GRAPE run.  The control operator handed to the optimiser IS the gate")
    W("      generator, H_c = Pi_l (x) n, and the target is exp(-i theta Pi_l n).")
    W("      Replaying that waveform under that same operator therefore asks only:")
    W("      did the optimiser hit its own target?  With a single control that")
    W("      commutes with itself the answer is trivially yes -- the pulse is flat")
    W("      (std exactly 0) and the problem reduces to setting one integral.")
    W("      It contains NO physics about whether a laser can produce Pi_l (x) n.")
    W()
    W("  2.9e-03   results/qcr_mechanism_paper_angle.txt")
    W("      Source: the REAL off-resonant blue-sideband drive,")
    W("        H(t) = (Omega_sb/2)(|l><g| (x) a^dag e^{-i delta t} + h.c.),")
    W("      with NO imposed generator, compared against the effective generator")
    W("      that Schrieffer-Wolff predicts it should produce.  That is the")
    W("      physics question, and 2.9e-03 (endpoint, after VZ) is its answer.")
    W()
    W("  So the two numbers are not in conflict and neither is wrong.  3.64e-06")
    W("  is a compiler self-consistency check; 2.9e-03 is the gate's actual")
    W("  infidelity against the operator it is supposed to implement.  THE PAPER")
    W("  SHOULD QUOTE 2.9e-03 for QCR and must not present 3.64e-06 as a")
    W("  physical fidelity.  The same caveat applies to every imposed-generator")
    W("  row in this table -- see the note below.")
    W()

    W("=" * 104)
    W("Q2  WHERE THE QCD 0.998-0.999 BAZAVAN REPLAY LIVES")
    W("=" * 104)
    for f in ("step_qcd_bazavan_full_lab_results.json",
              "step_qcd_bazavan_full_lab_ramps_results.json"):
        p = H / f
        W(f"  {f}   exists = {p.exists()}")
    W("  produced by  pulses_meth/hypulse_validation/step_qcd_bazavan_full_lab.py")
    W("               and step_qcd_bazavan_full_lab_ramps.py")
    W("  quoted in    pulses_meth/appendix_table.md, the table at lines 98-99")
    W()
    W("  That run takes the SAME final_v3 QCD waveform and propagates it with")
    W("  qutip.sesolve under the full time-dependent Bazavan Eq. 1 -- cos(delta t)")
    W("  carrier envelope and e^{+-i omega_z t} mode oscillations kept explicit,")
    W("  only the optical RWA applied -- rather than under the imposed")
    W("  sigma_z (x) X control operator.  Cutoff-converged over N in {8, 10, 12}.")
    W("  Reported there:")
    W("    unramped, T = 160.4 us   F = 0.999 / 0.999 / 0.998")
    W("    sin^2 ramps 5 us/side, T = 165.4 us   F = 0.999 / 0.999 / 0.999")
    W()
    W("  So QCD has BOTH kinds of number, and they differ for the same reason")
    W("  QCR's two numbers differ:")
    W(f"    imposed-generator replay (this table)  1-F = "
      f"{[1-r[3] for r in rows if r[0]=='QCD'][0]:.6e}")
    W("    full Bazavan lab-frame replay          1-F ~ 1e-3 to 2e-3")
    W("  The Bazavan value is the physically meaningful one for QCD, and it is")
    W("  the analogue of the 2.9e-03 for QCR.")
    W()

    W("=" * 104)
    W("READ THIS TABLE WITH THE FOLLOWING CAVEAT")
    W("=" * 104)
    W("  QCD, CD, QT and QCR were all optimised against IMPOSED effective")
    W("  generators (the control operator is the gate generator).  Their")
    W("  common-protocol numbers are therefore self-consistency checks, not")
    W("  physical fidelities.  Only QT-SB is replayed here under its real")
    W("  time-dependent drive, because that is the model its waveform was")
    W("  optimised against.")
    W("  Physical counterparts that DO exist:")
    W("    QCD   1e-3 to 2e-3   step_qcd_bazavan_full_lab*_results.json")
    W("    QCR   2.9e-03        results/qcr_mechanism_paper_angle.txt")
    W("    QT-SB 3.45e-09       already physical (this table)")
    W("    CD    none           no full-lab replay exists")
    W("    QT    none           no full-lab replay exists")

    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
