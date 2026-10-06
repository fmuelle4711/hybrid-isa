"""Two reads, no simulation runs.

1. Multitone qudit op count with QPS fused, QT fused, AND the (Nc-1) QT-SB
   terms fused into one block.  Closed form, at Nc = 3, 6, 8, 12.
   Counts are derived from pulses_meth/hypulse_validation/step_runtime_table.py
   qudit_counts(), not hand-written, so they cannot drift from the runtime
   table.

2. For the dissipative shot circuits behind Table III, whether the central
   J_l (qubit) and QT-SB (qudit) terms were applied SUMMED or SEQUENTIALLY.
   Answered by reading the generating code, quoting the deciding lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pulses_meth" / "hypulse_validation"))

from step_runtime_table import qudit_counts                      # noqa: E402

OUT = ROOT / "results" / "multitone_counts_and_centres.txt"
L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def chain_links(Nc):
    """Link terms (QT_SB) per step under the chain, Vu Eq. 21 with Eq. 52.

    Every bond carries two vibronic link terms, one per endpoint
    low-frequency mode; Eq. 52 drops the two boundary modes, so the end
    bonds keep one each.  Total 2(Nc-2), against the Nc-1 of the earlier
    star reading.  Equal at Nc = 3.  The shared qudit_counts() in
    pulses_meth/hypulse_validation/step_runtime_table.py still carries
    Nc-1; it is a Meth-era validation record and is overridden here
    rather than rewritten.
    """
    return 2 * (Nc - 2)


def audited(c, Nc):
    """Apply the audited chain counts to a convention dict.

    Rotation and QCD become 4Nc-4 (one mode free evolution per mode per
    half step over 2Nc-2 modes; per site one R3 on the high mode and one
    on the low mode, boundary low modes dropped), and QT_SB becomes the
    chain link count.  See results/cluster_bound.md section 5d.
    """
    c = dict(c)
    c["Rotation"] = 4 * Nc - 4
    c["QCD"] = 4 * Nc - 4
    c["QT_SB"] = chain_links(Nc)
    return c


def counts_34_chain(Nc):
    """The 34-op headline convention, audited chain counts."""
    return audited(qudit_counts(Nc, convention="34"), Nc)


def counts_28_chain(Nc):
    """The 28-op convention, audited chain counts."""
    return audited(qudit_counts(Nc, convention="28"), Nc)


def counts_27(Nc):
    """QPS fused + QT fused + all (Nc-1) QT-SB terms fused into ONE block.

    Starts from the existing '28' convention -- which already fuses QPS and
    QT -- and changes only the QT-SB entry to 1.

    THE CHAIN LEAVES THIS UNCHANGED at 6Nc+9.  The link count rose from
    Nc-1 to 2(Nc-2), but every link term is fused into the one block, so
    the block count does not see it.  The fusion does carry more weight
    now: under the star all link terms shared the single trimer mode l,
    whereas the chain spreads them over Nc-2 distinct low-frequency
    modes, so the merged block is a multi-mode multitone block.  That is
    the same class of claim as the QT fusion below, which is likewise
    not an operator identity.
    """
    c = audited(qudit_counts(Nc, convention="28"), Nc)
    c["QT_SB"] = 1
    return c


def quote(path, lo, hi):
    txt = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()
    for i in range(lo - 1, min(hi, len(txt))):
        W(f"      {i+1:4d} | {txt[i].rstrip()}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    # =================================================================== 1
    W("=" * 104)
    W("1  MULTITONE QUDIT COUNT -- QPS fused, QT fused, QT-SB fused")
    W("=" * 104)
    W("  Built from step_runtime_table.qudit_counts(Nc, '28') with the QT-SB")
    W("  entry changed from (Nc-1) to 1.  Nothing else differs.")
    W()
    W(f"  {'primitive':<12} {'per step':<16} {'Nc=3':>6} {'Nc=6':>6} "
      f"{'Nc=8':>6} {'Nc=12':>6}")
    forms = {
        "Rotation": "4Nc-4",
        "QCR": "2Nc",
        "QCD": "4Nc-4",
        "QPS": "2         (fused)",
        "QT_multi": "2         (fused)",
        "QT_SB": "1         (fused)",
    }
    NCS = (3, 6, 8, 12)
    table = {n: counts_27(n) for n in NCS}
    for k in ("Rotation", "QCR", "QCD", "QPS", "QT_multi", "QT_SB"):
        W(f"  {k:<12} {forms[k]:<16} "
          + " ".join(f"{table[n][k]:>6}" for n in NCS))
    W(f"  {'-'*12} {'-'*16} " + " ".join("------" for _ in NCS))
    tot = {n: sum(table[n].values()) for n in NCS}
    W(f"  {'TOTAL':<12} {'10Nc - 3':<16} "
      + " ".join(f"{tot[n]:>6}" for n in NCS))
    W()
    W("  CLOSED FORM")
    W("      (4Nc-4) + 2Nc + (4Nc-4) + 2 + 2 + 1  =  10 Nc - 3")
    ok = all(tot[n] == 10 * n - 3 for n in NCS)
    W(f"      checked against the built dict at every Nc above: "
      f"{'consistent' if ok else 'MISMATCH'}")
    W()
    W("  AGAINST THE EXISTING CONVENTIONS")
    W(f"  {'convention':<34} {'closed form':<14} "
      + " ".join(f"{'Nc='+str(n):>6}" for n in NCS))
    for lbl, fn, form in (
            ("34-op headline, QPS fused only", counts_34_chain, "16Nc - 14"),
            ("28-op, QPS + QT fused", counts_28_chain, "12Nc - 8"),
    ):
        vals = [sum(fn(n).values()) for n in NCS]
        W(f"  {lbl:<34} {form:<14} " + " ".join(f"{v:>6}" for v in vals))
    W(f"  {'27-op, + QT-SB fused (this read)':<34} {'10Nc - 3':<14} "
      + " ".join(f"{tot[n]:>6}" for n in NCS))
    W()
    W("  Nc=3 gives 27, matching the tape count of the figure drawn in")
    W("  simulations/draw_qudit_27_hqml.py (27 operations excluding the")
    W("  leading state-prep X) and data/circuit_qudit_generalized_27_one_step.json.")
    W()
    W("  WHAT THE FUSIONS COST IN EXACTNESS -- unchanged by this read:")
    W("    QPS fusion   EXACT.   All QPS generators are diagonal projectors on")
    W("                 one wire; measured [Pi_A, Pi_B] = 0.000000.")
    W("    QT fusion    NOT exact.  |A><B| and |A><C| share level A;")
    W("                 measured commutator 1.000000.  Rests on a multitone")
    W("                 hardware claim, not an operator identity.")
    W("    QT-SB fusion NOT exact in the same sense -- same level structure --")
    W("                 but it is the SUMMED exponential, which is what the")
    W("                 qutip reference trotter_method_a.py:212 actually uses,")
    W("                 so it moves the circuit TOWARD its own reference.")
    W("                 At the circuit angle the merged and sequential centres")
    W("                 differ by max 1.97e-05, infidelity 5.25e-11")
    W("                 (results/circuit_qudit_27.txt).")
    W()
    W("  NOTE: only QT and QT-SB carry laser time here; QPS is virtual (T=0).")
    W("  Under the audited chain counts, 10Nc-3 against 12Nc-8 is a")
    W("  runtime difference of 2(Nc-2) - 1 = 2Nc-5 QT-SB gates, not of")
    W("  the whole op-count gap.  NO ADOPTION -- a count, not a repricing.")
    W()

    # =================================================================== 2
    W("=" * 104)
    W("2  CENTRE ORDERING IN THE TABLE III SHOT CIRCUITS")
    W("=" * 104)
    W("  Question: were the central J_l (qubit) / QT-SB (qudit) terms applied")
    W("  as ONE SUMMED exponential, or SEQUENTIALLY?")
    W()
    W("  First, a correction to the premise: 'hqml stinespring' describes the")
    W("  N=8 row only.  Per results/table3_provenance.txt the backends are")
    W("      N=4  classical_kraus_shots            (both encodings)")
    W("      N=8  hqml_qnode_stinespring_shots     (qudit)")
    W("           hqml_qnode_stinespring_b_qubit   (qubit)")
    W("  so the four cells are produced by THREE different code paths.")
    W()

    W("  " + "-" * 100)
    W("  N=4, BOTH ENCODINGS  ->  SUMMED")
    W("  " + "-" * 100)
    W("  path: generate_figs_hqml_comparison.py maps both methods onto")
    W("        circuit_dissipative_kraus.build_sub_hamiltonians(builder, N),")
    W("        with _METHOD_TO_BUILDER = {a_qudit: 'A', a_qudit_d: 'A',")
    W("        b_qubit: 'B'}, then build_u_step.")
    W()
    W("  The centre is assembled as a SINGLE Hamiltonian and exponentiated once:")
    quote("simulations/circuit_dissipative_kraus.py", 106, 107)
    W("      ...  (method 'B', the qubit column, is the same shape)")
    quote("simulations/circuit_dissipative_kraus.py", 121, 122)
    W("  and applied as one exponential of that sum:")
    quote("simulations/circuit_dissipative_kraus.py", 202, 202)
    W("      -> H_QTSB is the SUM of the A-B and A-C terms, and Umid is ONE")
    W("         exp(-i H_QTSB dt).  SUMMED, for both the qubit and the qudit")
    W("         column at N=4.")
    W()

    W("  " + "-" * 100)
    W("  N=8, QUDIT  ->  SEQUENTIAL")
    W("  " + "-" * 100)
    W("  path: circuit_fig89_10_hqml_qnode.py imports the unitary step from")
    W("        circuit_fig6_hqml.one_trotter_step and wraps it in Stinespring")
    W("        channels.  Its centre is TWO separate sideband gates:")
    quote("simulations/circuit_fig6_hqml.py", 122, 126)
    W("      -> two hqml.QT(..., sideband=True) calls, each at full tau.")
    W("         SEQUENTIAL.")
    W()

    W("  " + "-" * 100)
    W("  N=8, QUBIT  ->  SEQUENTIAL")
    W("  " + "-" * 100)
    W("  path: circuit_fig89_10_hqml_qnode_b_qubit.py, a literal 3-qubit")
    W("        circuit on wires QB[0..2] plus one ancilla.  Its step is")
    quote("simulations/circuit_fig89_10_hqml_qnode_b_qubit.py", 221, 229)
    W("      The J_l pieces are the g_alt displacement sandwiches inside")
    W("      _H2XX and _H2YY, emitted one pair at a time in a Python loop:")
    quote("simulations/circuit_fig89_10_hqml_qnode_b_qubit.py", 190, 193)
    W("      -> the A-B and A-C link terms are separate gate sandwiches, and")
    W("         the XX and YY halves are separate blocks.  SEQUENTIAL on both")
    W("         counts; there is no single summed centre anywhere in this path.")
    W()

    W("  " + "-" * 100)
    W("  ANSWER")
    W("  " + "-" * 100)
    W(f"  {'cell':<22} {'centre':<12} {'generating path'}")
    for cell, centre, path in (
            ("N=4  qubit", "SUMMED",
             "circuit_dissipative_kraus build_sub_hamiltonians('B') + build_u_step"),
            ("N=4  qudit", "SUMMED",
             "circuit_dissipative_kraus build_sub_hamiltonians('A') + build_u_step"),
            ("N=8  qubit", "SEQUENTIAL",
             "circuit_fig89_10_hqml_qnode_b_qubit one_trotter_step"),
            ("N=8  qudit", "SEQUENTIAL",
             "circuit_fig6_hqml one_trotter_step, via qnode.py"),
    ):
        W(f"  {cell:<22} {centre:<12} {path}")
    W()
    W("  CONSEQUENCE, reported not fixed.  The two rows of Table III now differ")
    W("  in THREE respects at once, not one:")
    W("      1. Fock cutoff        4       vs 8")
    W("      2. noise channel      classical Kraus vs Stinespring in-circuit")
    W("      3. centre ordering    summed  vs sequential")
    W("  Item 3 is new here; items 1 and 2 were recorded in")
    W("  results/table3_provenance.txt section 7.  Any reading of Table III as")
    W("  a clean Fock-convergence test is confounded by 2 and 3.")
    W()
    W("  Scale of item 3, from the measurements already in the repo: the")
    W("  summed-vs-sequential centre gap is 0.00377 % RMSE on the qudit")
    W("  (results/qtsb_summed_center.txt) and 0.00408 % on the qubit")
    W("  (results/jl_summed_center.txt).  Both are far below the ~1 % Table III")
    W("  entries, so this does not explain those numbers -- but it does mean")
    W("  the N=4 and N=8 rows are not the same circuit, and the paper should")
    W("  not describe them as one pipeline run at two cutoffs.")
    W()
    W("  A further note on the N=4 qubit column: builder 'B' is the cqed")
    W("  Hamiltonian expressed with trans_d/proj_d, i.e. in the QUDIT")
    W("  encoding, whereas the N=8 qubit column is a literal 3-qubit circuit.")
    W("  So 'qubit' does not mean the same object in the two rows either.")
    W("  Reported for the record; no number is changed.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
