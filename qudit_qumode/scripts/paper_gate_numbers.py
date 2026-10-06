"""Every gate number the manuscript quotes, checked against its source.

DOES NOT TOUCH THE DATA PATH.  Reads existing artifacts and counts tapes.

The manuscript (abstract, Sec. resources, tab:gatecounts) quotes, at Nc = 3:
    34   qudit operations per Trotter step
    32   qubit operations per Trotter step
    27   qudit timed pulses, reference circuit
    25   qudit timed pulses, multitone schedule
    58   qubit timed pulses
    96   Vu, cited from their paper -- NOT ours to verify

Each is regenerated here from the file that owns it, so a number that has
drifted shows up as a MISMATCH rather than being quoted forward.

TIMED PULSES are read directly from results/segment_counts.txt, which uses
the one-pulse convention: C_Pi R and CR one segment each, the C_Pi_x D
centre merged into one block, J_l decomposed into 8 H + 4 CNOT + 2 CD.
An earlier version of that file priced the dispersive gate as a
three-segment echo, and this script subtracted an "echo surcharge" to get
from its 39 / 70 to the paper's 27 / 58.  That correction is gone because
the file no longer needs it.

THE QUBIT'S TWO 58s.  The segment count above and the drawn tape's timed
operations (70 tape ops, 12 of them virtual RZ) are different counts of
different things that agree at 58 under the one-pulse convention.  The
breakdowns still differ term by term: segments price a decomposed J_l at
14, the tape prices every operation at 1.

Output: results/paper_gate_numbers.txt

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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "circuits" / "trapped_ion_qubit"))

SEGF = ROOT / "results" / "segment_counts.txt"
J27 = ROOT / "data" / "circuit_qudit_generalized_27_one_step.json"
J28 = ROOT / "data" / "circuit_qudit_generalized_28_one_step.json"
JVU = ROOT / "data" / "circuit_vu_one_step.json"

L = []
FAIL = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def check(label, got, want, note=""):
    ok = (got == want)
    if not ok:
        FAIL.append(label)
    W(f"  {label:<46} {got:>6}   expected {want:<6} "
      f"{'OK' if ok else '<-- MISMATCH'}   {note}")
    return ok


def seg_block(name):
    """Pull one 'TOTAL SEGMENTS' block out of results/segment_counts.txt."""
    txt = SEGF.read_text(encoding="utf-8").split("\n")
    i = next(k for k, l in enumerate(txt) if l.strip() == "Nc = 3")
    out, cur = {}, None
    for l in txt[i:]:
        s = l.strip()
        if s.startswith("Nc =") and s != "Nc = 3":
            break
        if s in ("QUDIT, reference", "QUDIT, multitone",
                 "QUBIT, J_l decomposed (manuscript)",
                 "QUBIT, J_l as one MS window (alternative)"):
            cur = s
        m = re.match(r"TOTAL SEGMENTS\s+(\d+)", s)
        if m and cur:
            out[cur] = int(m.group(1))
    return out.get(name)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("=" * 92)
    W("EVERY GATE NUMBER IN THE MANUSCRIPT, CHECKED AGAINST ITS SOURCE")
    W("=" * 92)
    W("  Nc = 3.  DOES NOT TOUCH THE DATA PATH.")
    W()

    # ---------------- qudit ----------------
    d27 = json.loads(J27.read_text(encoding="utf-8"))
    d28 = json.loads(J28.read_text(encoding="utf-8"))
    drawn = len(d27["gates"])
    logical = collections.Counter()
    for g in d27["gates"]:
        gate = g["gate"]
        logical["QPS" if gate == "QPS_gen" else
                "QT-SB" if gate.startswith("QT-SB") else
                "QT" if gate.startswith("QT") else gate] += (
            3 if gate == "QPS_gen" else 2 if gate in ("QT_multi", "QT-SB_multi")
            else 1)
    n_log = sum(logical.values())
    seg_qudit_ref = seg_block("QUDIT, reference")
    seg_qudit_mt = seg_block("QUDIT, multitone")

    W("QUDIT   (source: data/circuit_qudit_generalized_27_one_step.json,")
    W("                results/segment_counts.txt)")
    check("blocks drawn, multitone-fused (Fig 3)", drawn, 27)
    check("blocks drawn before the centre merge", len(d28["gates"]), 28)
    check("LOGICAL ISA operations  <- paper's 34", n_log, 34,
          "Rot 8, QCR 6, QCD 8, QPS 6, QT 4, QT-SB 2")
    check("timed pulses, reference  <- paper's 27", seg_qudit_ref, 27,
          "one-pulse C_{Pi}R, merged C_{Pi_x}D centre")
    check("timed pulses, multitone  <- paper's 25", seg_qudit_mt, 25,
          "X_E fused to one block per half step")
    W()
    W("  TWO DIFFERENT 27s.  The first row is 27 scheduled CONTROL BLOCKS")
    W("  (multitone), the last is 27 TIMED PULSES of the 34-invocation")
    W("  reference circuit.  Unrelated arithmetic, same digits.  The")
    W("  manuscript keeps them in separate columns and never in one")
    W("  sentence; do the same when quoting this file.")
    W()

    # ---------------- qubit ----------------
    import pennylane as qml
    import circuit_qubit_hybridlane as CMP
    import circuit_qubit_logical32 as LOG
    with qml.tape.QuantumTape() as t_log:
        LOG.logical_trotter_step()
    with qml.tape.QuantumTape() as t_cmp:
        CMP.one_hot_trotter_step()
    n_qb_log = len(t_log.operations)
    c_cmp = collections.Counter(o.name for o in t_cmp.operations)
    n_tape = sum(c_cmp.values())
    n_virt = c_cmp["RZ"]
    seg_qubit_1ms = seg_block("QUBIT, J_l as one MS window (alternative)")
    seg_qubit_dec = seg_block("QUBIT, J_l decomposed (manuscript)")

    W("QUBIT   (source: circuits/trapped_ion_qubit tapes,")
    W("                results/segment_counts.txt)")
    check("LOGICAL operations  <- paper's 32", n_qb_log, 32, "11*Nc - 1")
    check("timed pulses  <- paper's 58", seg_qubit_dec, 58,
          "one-pulse CR, J_l decomposed")
    check("timed pulses, J_l as 1 MS window", seg_qubit_1ms, 32,
          "the OPTIMISTIC J_l convention, not the paper's")
    W()
    W("  the drawn circuit, a SEPARATE counting:")
    check("tape operations", n_tape, 70, "the OLD segment total, coincidence")
    check("of which virtual (RZ)", n_virt, 12)
    check("timed tape operations", n_tape - n_virt, 58,
          "equals the paper's 58 under one-pulse pricing; see note below")
    W()

    # ---------------- Vu ----------------
    dvu = json.loads(JVU.read_text(encoding="utf-8"))
    cvu = collections.Counter(g["gate"] for g in dvu["gates"])
    W("VU cQED (source: data/circuit_vu_one_step.json, verified against Vu's")
    W("                own script by circuits/cqed_vu/verify_vu_132.py)")
    check("c2qa primitives, measured", len(dvu["gates"]), 132,
          "148 emitted - 16 barriers")
    check("abstract count  <- paper's 96", dvu["gate_count_total"], 96,
          "CITED from Vu, never recounted by us")
    check("of which SWAP (pure routing)", cvu["swap"], 24, "qudit/qubit: 0")
    check("of which rz (virtual)", cvu["rz"], 20)
    check("timed primitives", len(dvu["gates"]) - cvu["rz"], 112)
    W()

    # ---------------- the comparison the paper makes ----------------
    W("=" * 92)
    W("THE LIKE-FOR-LIKE COMPARISON THE ABSTRACT MAKES")
    W("=" * 92)
    W(f"  logical operations   qudit {n_log:>3}   vs   qubit {n_qb_log:>3}"
      f"      (paper: '34 and 32')")
    W(f"  timed pulses         qudit {seg_qudit_ref:>3}   vs   qubit "
      f"{seg_qubit_dec:>3}      (paper: '27 against 58')")
    W()
    W("  Both rows compare the SAME quantity on both sides, under the")
    W("  one-pulse C_{Pi}R convention the manuscript now uses.")
    W()
    W("  WHY THE QUBIT'S TWO 58s NOW AGREE.  The drawn tape has 58 timed")
    W("  operations and the segment total is also 58.  Under the OLD echo")
    W("  pricing these were 58 and 70 and had to be kept apart.  Charging")
    W("  the dispersive gate one segment instead of three removes exactly")
    W(f"  the {2 * c_cmp['ConditionalRotation']} that separated them (two per CR, "
      f"{c_cmp['ConditionalRotation']} CR gates), so the two counts coincide for a")
    W("  reason, not by accident.  They are still different quantities.")
    W()
    W("  58 remains a FLOOR: Appendix B prices each CNOT as one primitive,")
    W("  and a CNOT on trapped ions is itself an MS gate plus rotations.")
    W()

    W("=" * 92)
    if FAIL:
        W(f"RESULT: {len(FAIL)} MISMATCH(ES) -- {', '.join(FAIL)}")
    else:
        W("RESULT: every number in the manuscript matches its source.")
    W("=" * 92)

    out = ROOT / "results" / "paper_gate_numbers.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
