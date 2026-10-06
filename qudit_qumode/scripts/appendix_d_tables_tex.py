"""Emit the two per-instance LaTeX tables for Appendix D.

Every row is read from a compiled tape, so the tables cannot drift from the
circuits.  Convention: J_l is priced at ONE Molmer-Sorensen window, and the
Hadamards, CNOTs and conditional displacements inside its sandwiches are
listed but carry no separate charge.

Output: results/appendix_d_tables.tex
        results/appendix_d_tables.txt   (the same numbers, readable)
"""
from __future__ import annotations

import collections
import math
import subprocess
import sys
from pathlib import Path

from scipy.special import jv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulations"))
sys.path.insert(0, str(ROOT / "circuits" / "trapped_ion_qubit"))

TAU = 0.005e-12
ETA, OMEGA_C = 0.041, 0.612993689
NU = 2 * math.pi * 0.77
XE_FRAC, T_ROT, T_MS = 0.07, 5.0, 200.0
X_B = 4 * OMEGA_C / NU
U_CD = ETA * OMEGA_C * float(jv(1, X_B) + jv(3, X_B))
CHI = (2 * math.pi * ETA * OMEGA_C / (2 * math.pi)) ** 2  # placeholder
CHI = (2 * math.pi * 4.0e-3) ** 2 / (4 * 2 * math.pi * 0.040)

# R2: every dispersive gate runs for ONE DETUNING PERIOD, 1/delta = 25.000 us,
# and the compiled angle is delivered by calibrating the drive AMPLITUDE
# instead of the duration.  results/cpir_amplitude_calibration.txt.
#
# Why not calibrate the duration.  The population driven out of |l> returns on
# the detuning period, so endpoint leakage is near zero at integer multiples of
# it and bad in between.  Solving for a phase-matched DURATION walks off that
# null and costs a factor of fifteen in leakage
# (results/cpir_replay_phase_matched.txt).  Holding T on the null and solving
# the amplitude instead lands every angle on ratio 1.000 with LOWER leakage
# than the published operating point.
T_R2 = 25.0000
AMP_FACTOR = {0.0135: 0.942332, 0.0160: 1.029035, 0.0180: 1.094148}


def t_r2(theta_sz):
    """One detuning period for every instance; refuse an uncalibrated angle."""
    for k in AMP_FACTOR:
        if abs(abs(theta_sz) - k) < 1e-9:
            return T_R2
    raise KeyError(
        f"C_Pi R / CR angle {theta_sz} has no calibrated amplitude. "
        f"Solve it with scripts/cpir_amplitude_calibration.py and add it to "
        f"AMP_FACTOR.")


T_CPIXD = (2 * math.pi / NU) / 2
N_JL = 2                      # J_l gates per step at Nc = 3

# rule id -> (short label for the table, full statement)
RULES = {
    "R1": ("R1", r"Unconditional mode rotation, $5.0~\mu$s, a scheduling "
                 r"allowance chosen in this work and charged on both sides. "
                 r"At $N_c=3$ it is $40~\mu$s of the qudit total and "
                 r"$40~\mu$s of the qubit total."),
    # R2 wording is the author's, synced from mas.tex.
    "R2": ("R2",
           r"$T=2\pi/\delta=25.000~\mu$s, one detuning period, a single "
           r"flat pulse, for every dispersive gate on both sides. "
           r"Population driven out of the addressed level mostly returns "
           r"on that period, so the leakage has a minimum there. The "
           r"compiled angle is delivered by calibrating the sideband "
           r"amplitude at this fixed duration, $\Omega_{\rm "
           r"sb}/2\pi=3.77$, $4.12$, and $4.38$~kHz for the three angles, "
           r"each verified under the physical drive to deliver its angle "
           r"to $1.000$ (Appendix~\ref{app:pulses}). The angle column is "
           r"as compiled."),
    "R3": ("R3", r"$T=2|\alpha|/u$ with $u=\eta\Omega_c(J_1{+}J_3)"
                 r"(4\Omega_c/\nu)=6.23\times10^{-3}$~rad/$\mu$s. "
                 r"The qubit reference's CD is the same physical gate and "
                 r"uses the same bound."),
    "R4": ("R4", r"Flat two-tone carrier block, "
                 r"$T=2|\theta|/(0.07\,\Omega_c)$ "
                 r"(Appendix~\ref{app:pulses})."),
    "R5": ("R5", r"Not angle-based: half a motional period, "
                 r"$\pi/\nu=0.649~\mu$s, below which the quadrature "
                 r"selection does not exist."),
    "R6": ("R6", r"Virtual frame update, $T=0$."),
    "R7": ("R7", r"M{\o}lmer--S{\o}rensen window, $200~\mu$s, taken as "
                 r"angle independent following Ref.~\cite{meth2025}."),
    "R8": ("R8", r"Inside a $J_l$ sandwich; $J_l$ is priced at one "
                 r"M{\o}lmer--S{\o}rensen window, so these carry no "
                 r"separate charge."),
}

TEX = {"Rotation": r"$R$", "C_Pi R": r"$C_{\Pi}R$", "CR": r"CR",
       "C_Pi_z D": r"$C_{\Pi_z}D$", "CD": r"CD", "Z_d": r"$Z_d$",
       "r_z": r"$r_z$", "X_E": r"$X_E$", "C_Pi_x D": r"$C_{\Pi_x}D$",
       "MS XX": r"MS$_{XX}$", "MS YY": r"MS$_{YY}$", "H": r"H$^{\dagger}$",
       "CNOT": r"CNOT$^{\dagger}$"}

L = []


def W(s=""):
    L.append(s)


def commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def classify(op, in_sandwich):
    n, p = op.name, [float(x) for x in op.parameters]
    hyp = getattr(op, "hyperparameters", {}) or {}
    if n == "Rotation":
        return "Rotation", p[0], "R1", T_ROT
    if n in ("QuditConditionalRotation", "ConditionalRotation"):
        # Both tapes are priced at the SAME duration for the reference
        # angle, 25.465 us.  That is an EMPIRICAL result, not a convention
        # conversion: scripts/qcr_duration_scan.py propagates the real
        # detuned-sideband drive and scans T for the best match to each
        # encoding's own target.  Both peak at 25.465 us, ratio 1.000, with
        # neighbouring points an order of magnitude worse
        # (results/qcr_duration_scan.txt).  Counting arguments about Pi_l
        # against sigma_z were tried first and are NOT the reason -- they
        # returned factors of two in both directions depending on which
        # convention was held fixed, and the scan refutes them.  The factor
        # below is what puts each tape's own angle on that common scale.
        if n.startswith("Qudit"):
            # qudit tape is Pi_l convention: theta_sigma_z = 2|theta|
            return "C_Pi R", p[0], "R2", t_r2(2 * abs(p[0]))
        return "CR", p[0], "R2", t_r2(abs(p[0]))
    if n in ("QuditConditionalDisplacement", "ConditionalDisplacement"):
        c = "C_Pi_z D" if n.startswith("Qudit") else "CD"
        if in_sandwich:
            return c, p[0], "R8", 0.0
        return c, p[0], "R3", 2 * abs(p[0]) / U_CD
    if n == "QuditPhaseShift":
        return "Z_d", p[0], "R6", 0.0
    if n == "RZ":
        return "r_z", p[0], "R6", 0.0
    if n == "QuditTransition":
        if hyp.get("sideband"):
            return "C_Pi_x D", p[0], "R5", T_CPIXD
        return "X_E", p[0], "R4", 2 * abs(p[0]) / (XE_FRAC * OMEGA_C)
    if n in ("IsingXX", "IsingYY"):
        return "MS " + n[-2:], p[0], "R7", T_MS
    if n == "Hadamard":
        return "H", None, "R8", 0.0
    if n == "CNOT":
        return "CNOT", None, "R8", 0.0
    raise KeyError(n)


def sandwich_mask(ops):
    """True for every op between the first and last H of each J_l sandwich."""
    idx = [i for i, o in enumerate(ops) if o.name in ("Hadamard", "CNOT")]
    mask = [False] * len(ops)
    if not idx:
        return mask
    # a sandwich runs from an H to the matching H; mark everything in the
    # span of consecutive H/CNOT blocks, including the CD between them
    runs, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b - a <= 3:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    for r in runs:
        for i in range(min(r), max(r) + 1):
            mask[i] = True
    return mask


def build(ops, label):
    mask = sandwich_mask(ops)
    rows = []
    for i, o in enumerate(ops):
        rows.append(classify(o, mask[i]))
    return rows


def latex_table(rows, caption, lab, extra_line=None, total_note=""):
    """Two side-by-side column groups so the table fits one page."""
    n = len(rows)
    half = (n + 1) // 2
    out = [r"\begin{table*}[t]", r"  \centering", r"  \footnotesize",
           r"  \setlength{\tabcolsep}{3pt}",
           r"  \begin{tabular}{r l r l r @{\hspace{0.9em}} r l r l r}",
           r"    \toprule",
           r"    \# & gate & angle & rule & $T$ ($\mu$s) & "
           r"\# & gate & angle & rule & $T$ ($\mu$s) \\",
           r"    \midrule"]
    for k in range(half):
        cells = []
        for j in (k, k + half):
            if j < n:
                cls, ang, rule, T = rows[j]
                a = "--" if ang is None else f"${ang:+.4f}$"
                cells.append(f"{j+1} & {TEX[cls]} & {a} & {rule} & {T:.3f}")
            else:
                cells.append(" & & & & ")
        out.append("    " + " & ".join(cells) + r" \\")
    out.append(r"    \midrule")
    tot = sum(r[3] for r in rows)
    if extra_line:
        out.append(r"    \multicolumn{10}{l}{" + extra_line + r"} \\")
        tot += N_JL * T_MS
    out.append(r"    \multicolumn{10}{r}{\textbf{total\quad "
               + f"{tot:.1f}" + r"$~\mu$s}} \\")
    out += [r"    \bottomrule", r"  \end{tabular}",
            r"  \caption{" + caption + total_note + r"}",
            r"  \label{" + lab + r"}", r"\end{table*}", ""]
    return out, tot


def multitone(rows):
    """Reprice the qudit rows under the 27-control-block multitone schedule.

    Only two classes change.  X_E: the sequential A<->B and A<->C tones of
    each half step are driven at once as ONE flat block whose duration is
    set by the AGGREGATE field, T = 2 sqrt(sum theta^2)/(0.07 Omega_c)
    (results/xe_multitone_flat.txt, results/runtime_xe_flat.txt).
    C_Pi_x D: the Nc-1 sideband terms fuse into one summed exponential,
    one block, same duration.  Z_d fuses too but is virtual either way.
    """
    xe = [abs(a) for c, a, _, _ in rows if c == "X_E"]
    cx = [T for c, _, _, T in rows if c == "C_Pi_x D"]
    pairs = [xe[i:i + 2] for i in range(0, len(xe), 2)]
    xe_blocks = [2 * math.sqrt(sum(t * t for t in q)) / (XE_FRAC * OMEGA_C)
                 for q in pairs]
    cx_blocks = [cx[0]] if cx else []
    base = sum(T for c, _, _, T in rows if c not in ("X_E", "C_Pi_x D"))
    return (base + sum(xe_blocks) + sum(cx_blocks),
            xe_blocks, cx_blocks,
            sum(T for c, _, _, T in rows if c == "X_E"), sum(cx))


def main():
    import pennylane as qml
    import circuit_fig6_hqml as QD
    import circuit_qubit_hybridlane as QB

    with qml.tape.QuantumTape() as t:
        QD.one_trotter_step(TAU)
    qd_ops = list(t.operations)
    with qml.tape.QuantumTape() as t:
        QB.one_hot_trotter_step()
    qb_ops = list(t.operations)

    qd = build(qd_ops, "qudit")
    qb = build(qb_ops, "qubit")

    W("% generated by scripts/appendix_d_tables_tex.py at commit "
      + commit())
    W("% rows read from the compiled tapes; do not hand-edit")
    W("")
    W(r"\paragraph{Pricing rules.} Each gate instance is priced from its own")
    W(r"angle. The rules are:")
    W(r"\begin{description}\setlength{\itemsep}{0pt}")
    for k in sorted(RULES):
        short, full = RULES[k]
        W(r"  \item[\textbf{" + short + r"}] " + full)
    W(r"\end{description}")
    W("")
    W(r"\noindent Both tables price the \emph{reference} circuits: the "
      r"$34$-operation qudit ISA circuit and the $32$-operation qubit ISA "
      r"circuit, compiled to $36$ and $70$ tape operations respectively. "
      r"Neither table prices the multitone qudit schedule; that total is "
      r"given below them.")
    W("")

    a, tqd = latex_table(
        qd,
        r"Per-instance timing of the qudit reference circuit, one "
        r"second-order Trotter step at $N_c=3$. Angles are read from the "
        r"compiled circuit. The $36$ rows are the $34$ reference ISA "
        r"invocations plus the two $Z_d$ that fuse; both are virtual, so "
        r"the total is unchanged. ",
        "tab:perinstance_qudit")
    W("\n".join(a))

    b, tqb = latex_table(
        qb,
        r"Per-instance timing of the qubit reference circuit, one "
        r"second-order Trotter step at $N_c=3$, compiled to $70$ "
        r"operations. The Hadamards, CNOTs and conditional displacements "
        r"of the two $J_l$ sandwiches are listed for completeness but "
        r"carry no separate charge: $J_l$ is priced at one "
        r"M{\o}lmer--S{\o}rensen window each, added below the rows. ",
        "tab:perinstance_qubit",
        extra_line=r"$J_l$, $2\times$ one MS window (rule R7) --- "
                   r"\emph{optimistic: each window holds $8$~H and $4$~CNOT, "
                   r"all charged as one MS gate} \hfill $2\times200.0$",
        total_note=r" $^{\dagger}$Inside a $J_l$ sandwich; charged as part "
                   r"of that term's single $200~\mu$s M{\o}lmer--S{\o}rensen "
                   r"window, not separately.")
    W("\n".join(b))

    W(r"\noindent The projected ratio is $T_{\rm qubit}/T_{\rm qudit} = "
      + f"{tqb:.1f}/{tqd:.1f} = {tqb/tqd:.1f}" + r"$.")
    W("")

    global MT
    MT = multitone(qd)
    mt = MT[0]
    W(r"\noindent Under the multitone schedule of "
      r"Table~\ref{tab:gatecounts} the same step is $27$ control blocks: "
      r"the two $X_E$ pairs are each driven as one flat block at the "
      r"aggregate field, and the $C_{\Pi_x}D$ terms fuse into a single "
      r"summed exponential. Only those two classes change. The qudit total "
      + f"becomes ${mt:.1f}~" + r"\mu$s and the ratio "
      + f"${tqb:.1f}/{mt:.1f} = {tqb/mt:.1f}" + r"$.")

    out = ROOT / "results" / "appendix_d_tables.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    mt, xeb, cxb, xe_seq, cx_seq = MT
    txt = [f"qudit total {tqd:.4f} us over {len(qd)} rows",
           f"  multitone 27-block qudit total {mt:.4f} us",
           f"    X_E   {len(xeb)} blocks {sum(xeb):.4f} us "
           f"(was {xe_seq:.4f} over {len([1 for c,_,_,_ in qd if c=='X_E'])})",
           f"    C_PixD {len(cxb)} block {sum(cxb):.4f} us "
           f"(was {cx_seq:.4f} over {len([1 for c,_,_,_ in qd if c=='C_Pi_x D'])})",
           f"  multitone ratio {tqb/mt:.4f}  -> {tqb/mt:.1f}",
           f"qubit total {tqb:.4f} us over {len(qb)} rows "
           f"(incl. {N_JL} x {T_MS} us J_l windows)",
           f"ratio {tqb/tqd:.4f}  -> {tqb/tqd:.1f}", ""]
    # ---- per-instance tables, the same rows the LaTeX tables carry ----
    for nm, rows, lbl in (("QUDIT", qd, "tab:perinstance_qudit"),
                          ("QUBIT", qb, "tab:perinstance_qubit")):
        txt.append(f"PER-INSTANCE, {nm}   ({lbl})")
        txt.append(f"  {'#':>3}  {'gate':<12} {'angle':>9}  {'rule':<6} "
                   f"{'T (us)':>9}")
        sub = 0.0
        for i, (c, ang, rule, T) in enumerate(rows, 1):
            a = "--" if ang is None else f"{ang:+.4f}"
            sub += T
            txt.append(f"  {i:>3}  {c:<12} {a:>9}  {rule:<6} {T:>9.3f}")
        if nm == "QUBIT":
            txt.append(f"  {'':>3}  {'J_l, ' + str(N_JL) + ' x one MS window':<12}"
                       f" {'--':>9}  {'R7':<6} {N_JL * T_MS:>9.3f}")
            sub += N_JL * T_MS
        txt.append(f"  {'':>3}  {'TOTAL':<12} {'':>9}  {'':<6} {sub:>9.3f}")
        txt.append("")

    for nm, rows in (("QUDIT", qd), ("QUBIT", qb)):
        by = collections.defaultdict(lambda: [0, 0.0])
        for c, _, _, T in rows:
            by[c][0] += 1
            by[c][1] += T
        txt.append(nm)
        for c, (k, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
            txt.append(f"  {c:<10} {k:>3}  {s:>10.4f} us")
        txt.append("")
    (ROOT / "results" / "appendix_d_tables.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
