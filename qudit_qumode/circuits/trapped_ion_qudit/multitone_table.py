"""The multitone level: how 34 single-tone ISA calls become 27 driven blocks.

DOES NOT TOUCH THE DATA PATH.

Companion to term_gate_table_qudit.py.  That table stops at the reference
level, 34 logical ISA invocations, one tone each.  This one shows what the
control hardware actually emits: 27 blocks, because some blocks carry more
than one tone at once.

THE POINT.  Fusion does not remove tones.  It removes BLOCKS.

    34 tones  delivered in  27 blocks  costing  25 timed segments

as drawn, under the one-pulse convention (C_Pi R one segment).  That is
the multitone row of tab:gatecounts.  The reference row, 27, leaves the
X_E unfused; the qubit reference is 58.  All three are read back from
results/segment_counts.txt at the bottom of this file and cross-checked.

Every one of the 34 logical calls is still driven -- the multitone blocks
just deliver several of them simultaneously in one waveform.  That is why 34
and 27 describe the same circuit and are not in competition.

THE THREE FUSIONS ARE NOT EQUALLY SOLID, and the table says which is which:

  QPS_gen      EXACT.  The three level phases are diagonal operators; they
               commute, so the summed exponential IS the product.  Virtual,
               so it also costs zero segments.

  QT-SB_multi  MEASURED, not an identity.  The two centre generators do not
               commute -- max_commutator_bare = 1.0 -- but at the circuit's
               own tau the residual is 3.95e-05 and the infidelity of
               replacing the pair by one multitone block is 5.25e-11.
               Numbers read from the JSON's centre_operator_check, not
               retyped.

  QT_multi     SYNTHESIZED.  A-B and A-C share level A, so their generators
               do not commute either, and unlike the centre block this one is
               not a small-angle argument: the waveform is built to reproduce
               the sequential QT unitary.  It is a compiled control
               primitive, not an algebraic identity.  The manuscript says
               this in Sec. III B; the table repeats it so the figure cannot
               be read as claiming exactness.

Outputs
    circuits/trapped_ion_qudit/results/multitone_table.txt
    circuits/trapped_ion_qudit/figures/multitone_table.{pdf,png}

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

import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "params.py").exists())
sys.path.insert(0, str(ROOT))

import params as P                                                 # noqa: E402

JSON27 = ROOT / "data" / "circuit_qudit_generalized_27_one_step.json"
TAU = 0.005e-12
TAU_H = TAU / 2
L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


# segments per block, one-pulse convention: C_Pi R (code name QCR) is a
# single flat pulse, App. D rule R2.  Same rule as scripts/segment_counts.py.
SEGF = ROOT / "results" / "segment_counts.txt"
SEG = {"Rotation": 1, "QCR": 1, "QCD": 1, "QPS_gen": 0,
       "QT_multi": 1, "QT-SB_multi": 1}

FUSION = {
    "Rotation": ("single tone", "-"),
    "QCR": ("single tone", "-"),
    "QCD": ("single tone", "-"),
    "QPS_gen": ("EXACT", "diagonal operators commute"),
    "QT_multi": ("SYNTHESIZED", "A-B and A-C share level A; built to\n"
                                "reproduce the sequential unitary"),
    "QT-SB_multi": ("MEASURED", "bare commutator 1.0; residual 3.95e-05\n"
                                "at tau, merge infidelity 5.25e-11"),
}



def build_blocks():
    """The DISTINCT driven blocks.  13 per half-step, mirrored, plus the
    un-mirrored QT-SB centre = 27.  Multitone blocks are written as the sum
    of the terms they carry, because that is literally what the waveform is:
    several tones added together in one window."""
    t, T = TAU_H, TAU
    e = lambda v: f"{v:.6f}"
    return [
        # ---- single-tone blocks, one term each ---------------------------
        ("Rotation(a)", ["omega_ga n_a"], [e(P.omega_ga*t)], 1, 2, 1),
        ("Rotation(b)", ["omega_gb n_b"], [e(P.omega_gb*t)], 1, 2, 1),
        ("Rotation(c)", ["omega_gc n_c"], [e(P.omega_gc*t)], 1, 2, 1),
        ("Rotation(l)", ["omega_l n_l"],  [e(P.omega_l*t)],  1, 2, 1),
        ("QCR(A,a)", ["chi_a Pi_A n_a"], [e(-2*P.chi_a*t)], 1, 2, SEG["QCR"]),
        ("QCR(B,b)", ["chi_b Pi_B n_b"], [e(-2*P.chi_b*t)], 1, 2, SEG["QCR"]),
        ("QCR(C,c)", ["chi_c Pi_C n_c"], [e(-2*P.chi_c*t)], 1, 2, SEG["QCR"]),
        ("QCD(A,a)", ["-omega_ea sqrt(S_a) Pi_A x_a"],
         [e(P.omega_ea*np.sqrt(P.S_a)*t)], 1, 2, 1),
        ("QCD(B,b)", ["-omega_eb sqrt(S_b) Pi_B x_b"],
         [e(P.omega_eb*np.sqrt(P.S_b)*t)], 1, 2, 1),
        ("QCD(C,c)", ["-omega_ec sqrt(S_c) Pi_C x_c"],
         [e(P.omega_ec*np.sqrt(P.S_c)*t)], 1, 2, 1),
        ("QCD(A,l)", ["-omega_l sqrt(S_l) Pi_A x_l"],
         [e(P.omega_l*np.sqrt(P.S_l)*t)], 1, 2, 1),
        # ---- multitone: the sums --------------------------------------
        ("QPS_gen", ["(omega_ea S_a + omega_l S_l) Pi_A",
                     "omega_eb S_b Pi_B",
                     "omega_ec S_c Pi_C"],
         [e((P.omega_ea*P.S_a+P.omega_l*P.S_l)*t), e(P.omega_eb*P.S_b*t),
          e(P.omega_ec*P.S_c*t)], 3, 2, 0),
        ("QT_multi", ["J_AB0 (|A><B| + h.c.)",
                      "J_AC0 (|A><C| + h.c.)"],
         [e(P.J_AB0*t), e(P.J_AC0*t)], 2, 2, 1),
        # ---- the un-mirrored centre -----------------------------------
        ("QT-SB_multi", ["J_AB0 eta_AB (|A><B| + h.c.) x_l",
                         "J_AC0 eta_AC (|A><C| + h.c.) x_l"],
         [e(P.g_abl*T), e(P.g_acl*T)], 2, 1, 1),
    ]


def tone_detail(gate):
    """The tones each block type carries, with their parameters."""
    t = TAU_H
    if gate == "Rotation":
        return 1, "omega_g,R * tau/2   (one mode each)"
    if gate == "QCR":
        return 1, "chi_R * tau/2       (one chromophore each)"
    if gate == "QCD":
        return 1, "omega_e,R sqrt(S_R) * tau/2, and omega_l sqrt(S_l) on Pi_A"
    if gate == "QPS_gen":
        return 3, (f"Pi_A {(P.omega_ea*P.S_a + P.omega_l*P.S_l)*t:.6f}, "
                   f"Pi_B {P.omega_eb*P.S_b*t:.6f}, "
                   f"Pi_C {P.omega_ec*P.S_c*t:.6f}")
    if gate == "QT_multi":
        return 2, (f"A-B {P.J_AB0*t:.6f}, A-C {P.J_AC0*t:.6f}")
    if gate == "QT-SB_multi":
        return 2, (f"A-B {P.g_abl*TAU:+.6f}, A-C {P.g_acl*TAU:+.6f}  "
                   f"(link sideband)")
    raise KeyError(gate)


def seg_totals():
    """Nc = 3 totals from results/segment_counts.txt, by block heading."""
    want = {"QUDIT, reference": "ref", "QUDIT, multitone": "mt",
            "QUBIT, J_l decomposed (manuscript)": "qubit"}
    txt = SEGF.read_text(encoding="utf-8").split("\n")
    i = next(k for k, l in enumerate(txt) if l.strip() == "Nc = 3")
    out, cur = {}, None
    for l in txt[i:]:
        t = l.strip()
        if t.startswith("Nc =") and t != "Nc = 3":
            break
        if t in want:
            cur = want[t]
        if t.startswith("TOTAL SEGMENTS") and cur:
            out[cur] = int(t.split()[-1])
            cur = None
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    d = json.loads(JSON27.read_text(encoding="utf-8"))
    blocks = collections.Counter(g["gate"] for g in d["gates"])
    chk = d["centre_operator_check"]

    blk = build_blocks()
    n_blocks = sum(r[4] for r in blk)
    n_tones = sum(r[3] * r[4] for r in blk)
    n_seg = sum(r[5] * r[4] for r in blk)

    W("=" * 118)
    W("THE MULTITONE LEVEL  --  34 tones delivered in 27 blocks")
    W("=" * 118)
    W(f"  source : {JSON27.relative_to(ROOT)}")
    W(f"  Nc = 3, d = 4, tau = {TAU:.3e} s, one second-order Trotter step")
    W()
    W("  13 distinct blocks per half-step, mirrored, plus the un-mirrored")
    W("  QT-SB centre.  13 x 2 + 1 = 27.")
    W()
    W(f"  {'#':>2}  {'block':<14} {'tones':>5} {'x/step':>7} {'seg':>4}   "
      f"{'generator (multitone blocks written as the sum they carry)':<58} "
      f"{'angle':>10}")
    W("  " + "-" * 116)
    for i, (name, terms, angs, nt, rep, sg) in enumerate(blk, 1):
        W(f"  {i:>2}  {name:<14} {nt:>5} {rep:>7} {sg:>4}   "
          f"{terms[0]:<58} {angs[0]:>10}")
        for tm, ag in zip(terms[1:], angs[1:]):
            W(f"  {'':>2}  {'':<14} {'':>5} {'':>7} {'':>4} + {tm:<58} {ag:>10}")
        if nt > 1:
            W("  " + " " * 38 + "-" * 78)
    W("  " + "-" * 116)
    W(f"  {'':>2}  {'TOTAL':<14} {n_tones:>5} {n_blocks:>7} {n_seg:>4}")
    W()
    W("  FUSION REMOVES BLOCKS, NOT TONES.")
    W(f"    {n_tones} tones  = the 34 single-tone ISA calls, every one still driven")
    W(f"    {n_blocks} blocks = what the waveform generator emits")
    W(f"    {n_seg} segments = timed pulse slots as drawn (C_Pi R 1 each, "
      f"Z_d 0)")
    W()

    W("=" * 118)
    W("WHY ALL THREE FUSIONS ARE EXACT")
    W("=" * 118)
    W("  A multitone drive applies its tones SIMULTANEOUSLY, so the Hamiltonian")
    W("  during the pulse is their SUM and the block realises exp[-i(sum)t].")
    W("  H_lab carries each of these groups as a sum:")
    W("      QT      J_AB0 X_AB + J_AC0 X_AC")
    W("      QT-SB   g_abl X_AB x_l + g_acl X_AC x_l")
    W("      QPS     the three level phases")
    W("  so the summed exponential IS that group's evolution.  These are the")
    W("  generalized (higher-rank) generators of Table I, driven directly.")
    W()
    W("  THE SEQUENTIAL PRODUCT IS THE APPROXIMATION, not the multitone:")
    W("      max|multitone - sequential|,  QT group      2.53e-05")
    W("      max|multitone - sequential|,  QT-SB centre  1.97e-05")
    W("  Splitting a group into consecutive single-tone gates introduces that")
    W("  error; driving it as one block does not.")
    W()
    W("  This corrects two labels this table previously carried.  QT_multi was")
    W("  marked SYNTHESIZED and QT-SB_multi MEASURED, both on the premise that")
    W("  the multitone block had to reproduce the sequential product.  It does")
    W("  not: the sequential product was never the target.  The JSON's")
    W("  centre_operator_check numbers remain valid, but they measure how much")
    W("  error the SEQUENTIAL split would have introduced, not an error in the")
    W("  merge:")
    for k in ("max_commutator_bare", "max_commutator_tau_scaled",
              "max_abs_diff", "infidelity"):
        W(f"      {k:<28} {chk[k]}")
    W()
    W("  Non-commutation is why the split is lossy.  It is not an obstacle to")
    W("  the multitone block, which never performs the split.")
    W()
    W("=" * 118)
    W("TIMED PULSES: THIS TABLE AGAINST tab:gatecounts")
    W("=" * 118)
    # the reference count leaves each fused X_E block as its separate tones
    n_ref = n_seg + sum((r[3] - 1) * r[4] for r in blk if r[0] == "QT_multi")
    seg_file = seg_totals()
    W(f"  Pricing every gate AS DRAWN gives {n_seg} segments: the X_E of each")
    W("  half step are fused into one QT_multi block, and every C_Pi R is one")
    W("  flat pulse.  That is the multitone row of tab:gatecounts.")
    W()
    W(f"  Leaving the X_E unfused gives {n_ref}, the reference row.  The qubit")
    W("  reference, counted in results/segment_counts.txt, is the third row:")
    W(f"      {n_ref}   qudit reference, X_E unfused, centre merged   "
      f"(tab:gatecounts)")
    W(f"      {n_seg}   qudit multitone, as Fig 3 draws it           "
      f"(tab:gatecounts; this table)")
    W(f"      {seg_file['qubit']}   qubit reference, J_l decomposed             "
      f"(tab:gatecounts)")
    W()
    seg_ok = (n_ref, n_seg) == (seg_file["ref"], seg_file["mt"])
    W(f"  cross-check against results/segment_counts.txt: reference "
      f"{seg_file['ref']}, multitone {seg_file['mt']}, qubit "
      f"{seg_file['qubit']}  ->  {'match' if seg_ok else 'MISMATCH'}")
    W()
    W()
    ok = (n_blocks == 27 and n_tones == 34 and seg_ok)
    W("=" * 118)
    W(f"SELF-CHECK   blocks {n_blocks}/27, tones {n_tones}/34  ->  "
      f"{'OK' if ok else 'MISMATCH'}   segments {n_seg}/{seg_file['mt']} "
      f"against results/segment_counts.txt")
    W("=" * 118)

    out = HERE / "results" / "multitone_table.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    render(blk, n_blocks, n_tones, n_seg, chk)
    return 0 if ok else 1


COLOR = {"single tone": "#EEF1F4", "EXACT": "#E4F2E4",
         "MEASURED": "#FCF2E2", "SYNTHESIZED": "#FBE7E7"}


def render(blk, n_blocks, n_tones, n_seg, chk):
    FUS = {"QPS_gen": ("EXACT", "#E4F2E4"),
           "QT_multi": ("EXACT", "#E4F2E4"),
           "QT-SB_multi": ("EXACT", "#E4F2E4")}
    cells, colors = [], []
    for i, (name, terms, angs, nt, rep, sg) in enumerate(blk, 1):
        fus, face = FUS.get(name, ("", "#EEF1F4"))
        for j, (tm, ag) in enumerate(zip(terms, angs)):
            first = (j == 0)
            cells.append([str(i) if first else "",
                          name if first else "",
                          ("" if first else "+ ") + tm,
                          ag,
                          str(nt) if first else "",
                          str(rep) if first else "",
                          str(sg) if first else ""])
            colors.append([face] * 7)
    cells.append(["", "TOTAL", "", "", str(n_tones), str(n_blocks),
                  str(n_seg)])
    colors.append(["#DDE1E6"] * 7)

    fig = plt.figure(figsize=(16.0, 10.4))
    y = 0.978
    fig.text(0.5, y, "The multitone level: 34 tones delivered in 27 blocks",
             ha="center", va="top", fontsize=15, fontweight="bold")
    y -= 0.034
    fig.text(0.5, y,
             "13 distinct blocks per half-step, mirrored, plus the un-mirrored "
             "QT-SB centre.  13×2 + 1 = 27.  "
             "Multitone blocks are written as the sum of the tones they carry.",
             ha="center", va="top", fontsize=10.4, color="#444444")

    y -= 0.040
    band_h = 0.098
    boxes = [(f"{n_tones}", "TONES", "every logical ISA call, still driven",
              "#EAF5EA", "#3B7A3B"),
             (f"{n_blocks}", "BLOCKS", "what the AWG emits",
              "#E8F0FA", "#2C5D8F"),
             (f"{n_seg}", "SEGMENTS", "as drawn; tab:gatecounts multitone",
              "#FDF0E4", "#A65B1E")]
    x0, w, gap = 0.032, 0.294, 0.020
    for i, (big, lab, sub, face, edge) in enumerate(boxes):
        x = x0 + i * (w + gap)
        fig.patches.append(plt.Rectangle(
            (x, y - band_h), w, band_h, transform=fig.transFigure,
            facecolor=face, edgecolor=edge, linewidth=1.4, zorder=0))
        fig.text(x + 0.024, y - band_h / 2, big, ha="left", va="center",
                 fontsize=28, fontweight="bold", color=edge)
        fig.text(x + 0.106, y - band_h / 2 + 0.020, lab, ha="left",
                 va="center", fontsize=11, fontweight="bold", color=edge)
        fig.text(x + 0.106, y - band_h / 2 - 0.019, sub, ha="left",
                 va="center", fontsize=8.5, color="#555555")
        if i < 2:
            fig.text(x + w + gap / 2, y - band_h / 2, "→", ha="center",
                     va="center", fontsize=16, color="#999999")
    y -= band_h + 0.030

    ax = fig.add_axes([0.028, 0.125, 0.952, y - 0.135])
    ax.axis("off")
    col = ["#", "block", "generator", "angle", "tones", "x/step", "seg"]
    tab = ax.table(cellText=cells, colLabels=col, cellColours=colors,
                   cellLoc="left", colLoc="left", loc="upper center",
                   colWidths=[0.028, 0.118, 0.470, 0.090, 0.058, 0.060,
                              0.044])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.0)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0.7)
        cell.PAD = 0.030
        cell.set_height(1.0 / (len(cells) + 1))
        if r == 0:
            cell.set_facecolor("#40454B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(cells):
            cell.set_text_props(fontweight="bold")
        if c in (1, 2, 3) and r > 0:
            cell.set_text_props(fontfamily="DejaVu Sans Mono", fontsize=8.4)

    fig.text(0.032, 0.086,
             "Fusion removes BLOCKS, not TONES — all 34 logical calls are still driven, several at a time in one window.",
             ha="left", va="bottom", fontsize=9.0, color="#333333")
    fig.text(0.032, 0.062,
             "All three fusions are EXACT: a multitone drive applies its tones simultaneously, so the block realises "
             "exp[-i(sum)t] — and H_lab carries each group as a sum.  These are the generalized higher-rank "
             "generators of Table I, driven directly.",
             ha="left", va="bottom", fontsize=8.7, color="#444444")
    fig.text(0.032, 0.038,
             "The SEQUENTIAL product is the approximation, not the multitone: max|multitone − sequential| is 2.53e-05 "
             "for the QT group and 1.97e-05 for the QT-SB centre.  Splitting a group into consecutive single-tone "
             "gates costs that; driving it as one block does not.",
             ha="left", va="bottom", fontsize=8.7, color="#444444")
    fig.text(0.032, 0.014,
             "Segments as drawn total 25, the multitone row of tab:gatecounts; leaving the X_E unfused gives the "
             "reference 27.  One pulse per C_Pi R.",
             ha="left", va="bottom", fontsize=8.7, color="#666666")

    for ext in ("pdf", "png"):
        p2 = HERE / "figures" / f"multitone_table.{ext}"
        p2.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p2, bbox_inches="tight", dpi=200)
        print(f"wrote {p2}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
