"""
vibronic_scaling.py -- generalized-Nc vibronic Trotter step gate counting.

ADDITIVE ONLY.  Does not modify any existing module.  Alongside
trotter_method_a.py (Nc = 3, qudit, 36 gates), trotter_method_a_fused_qps.py
(Nc = 3, qudit_reduced, 34 gates), and trotter_method_b_qubit.py (Nc = 3,
one-hot, 32 gates), this module builds the same second-order Suzuki-Trotter
step at arbitrary Nc for four encodings and counts gates programmatically.

Model (single-excitation manifold):
    Nc chromophores + G (electronic register of dimension Nc + 1)
    each chromophore R has one local high-frequency mode  (Nc modes total)
    chromophore A additionally couples to one low-frequency link mode
    excitation-transfer couplings A <-> R for R != A, modulated to first
        order in the link-mode position

Encodings:
    qudit          : single qudit d = Nc + 1, native ISA
                     {QPS, QT, QT-SB, QCR, QCD, Rotation}
    qudit_reduced  : same qudit ISA, with the two QPS gates on Pi_A
                     (local + link Huang-Rhys) fused into one
    one_hot        : Nc qubits (single-excitation manifold, Vu-style)
                     native gates {chi, cd, rz, Rotation, J, Jl}
    compact        : m = ceil(log2(Nc + 1)) qubits, binary encoding
                     each Hamiltonian term realized as multi-controlled
                     Qiskit gate on the register, then transpile to
                     basis=['cx', 'u']

Reproduces at Nc = 3 (SANITY CHECK):
    qudit         -> 36
    qudit_reduced -> 34
    one_hot       -> 32
If any of these does not match, the module aborts with SystemExit(1)
before writing any output.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


# =====================================================================
# Symbolic circuit representation (no Qiskit needed for qudit / one-hot)
# =====================================================================

@dataclass
class SymGate:
    name: str
    wires: Tuple[str, ...]
    entangling: bool = False


class SymCircuit:
    """
    Programmatic gate list -- the count is len(gates), never patched by hand.
    depth() uses a greedy layer-packer over wires.
    """

    def __init__(self) -> None:
        self.gates: List[SymGate] = []

    def append(self, name: str, wires: Iterable[str], entangling: bool = False) -> None:
        self.gates.append(SymGate(name, tuple(wires), entangling))

    def total(self) -> int:
        return len(self.gates)

    def n_entangling(self) -> int:
        return sum(1 for g in self.gates if g.entangling)

    def register_wires(self) -> List[str]:
        seen = []
        for g in self.gates:
            for w in g.wires:
                if w not in seen:
                    seen.append(w)
        return seen

    def depth(self) -> int:
        """Simple layer depth: each gate takes the max of its wires' current times + 1."""
        wire_time: dict = {}
        d = 0
        for g in self.gates:
            t = max((wire_time.get(w, 0) for w in g.wires), default=0) + 1
            for w in g.wires:
                wire_time[w] = t
            if t > d:
                d = t
        return d


# =====================================================================
# QUDIT builder (Method A / Method A fused)
# =====================================================================
#
# Term structure inherited from simulations/trotter_method_a.py:
#     Free rotations:  Nc + 1  (Nc local modes + link mode; unconditional)
#     QCR (entang.):   Nc      (chi_R * Pi_R * n_R^local, local modes only)
#     QCD (entang.):   Nc + 1  (Nc local + 1 link-mode for A only)
#     QPS:             Nc + 1  (Nc local + 1 link-mode for A only)
#     QT carrier (e.): Nc - 1  (A <-> R for R != A)
#     QT-SB (entang.): Nc - 1  (sideband; middle full-tau block)
#
# Second-order symmetric Trotter step:
#     U = [free QCR QCD QPS QT]_forward  QT-SB_middle  [QT QPS QCD QCR free]_backward
# Per half-step: (Nc+1) + Nc + (Nc+1) + (Nc+1) + (Nc-1) = 5Nc + 2
# Total:         2 * (5Nc + 2) + (Nc - 1) = 11 Nc + 3
#
# Fused variant: the two QPS gates on Pi_A (local + link) are commuting
# diagonal operators on the same projector and combine into one gate
# without approximation.  Save 1 gate per half-step -> 2 total.
# Total:         11 Nc + 1

def build_qudit_trotter_step(Nc: int, fused: bool = False) -> SymCircuit:
    if Nc < 2:
        raise ValueError("Nc must be >= 2 (need at least one A-B pair)")
    c = SymCircuit()
    q = 'q'
    local = [f'mL{r}' for r in range(Nc)]
    link = 'link'

    def free_rotations():
        for m in local + [link]:
            c.append('Rotation', [m], entangling=False)

    def qcr_forward():
        for r in range(Nc):
            c.append('QCR', [q, local[r]], entangling=True)

    def qcr_reverse():
        for r in reversed(range(Nc)):
            c.append('QCR', [q, local[r]], entangling=True)

    def qcd_forward():
        for r in range(Nc):
            c.append('QCD', [q, local[r]], entangling=True)
        c.append('QCD', [q, link], entangling=True)  # A + link

    def qcd_reverse():
        c.append('QCD', [q, link], entangling=True)  # A + link (reversed)
        for r in reversed(range(Nc)):
            c.append('QCD', [q, local[r]], entangling=True)

    def qps_forward():
        if fused:
            # A (chromophore 0) combines local + link into one QPS
            c.append('QPS_fusedA', [q], entangling=False)
            for r in range(1, Nc):
                c.append('QPS', [q], entangling=False)
        else:
            for r in range(Nc):
                c.append('QPS', [q], entangling=False)
            c.append('QPS', [q], entangling=False)  # A + link separately

    def qps_reverse():
        if fused:
            for r in reversed(range(1, Nc)):
                c.append('QPS', [q], entangling=False)
            c.append('QPS_fusedA', [q], entangling=False)
        else:
            c.append('QPS', [q], entangling=False)  # A + link separately
            for r in reversed(range(Nc)):
                c.append('QPS', [q], entangling=False)

    def qt_forward():
        for r in range(1, Nc):
            c.append('QT', [q], entangling=True)  # A - R carrier

    def qt_reverse():
        for r in reversed(range(1, Nc)):
            c.append('QT', [q], entangling=True)

    def qtsb_middle():
        for r in range(1, Nc):
            c.append('QT-SB', [q, link], entangling=True)

    # forward half-step
    free_rotations()
    qcr_forward()
    qcd_forward()
    qps_forward()
    qt_forward()
    # middle (full tau) sideband
    qtsb_middle()
    # backward half-step (reverse order)
    qt_reverse()
    qps_reverse()
    qcd_reverse()
    qcr_reverse()
    free_rotations()  # commuting free rotations still counted per half-step
    return c


# =====================================================================
# ONE-HOT builder (Method B)
# =====================================================================
#
# Term structure inherited from simulations/trotter_method_b_qubit.py:
#     H_chi (entang.):  Nc       (sigma_z^R * n_R^local, one per pair)
#     H_cd (entang.):   Nc + 1   (Nc local + 1 link-mode for A)
#     H_det:            2 Nc     ((Nc+1) mode free rotations + (Nc-1) rz)
#     H_J (entang.):    Nc - 1   (2-qubit XY carriers A <-> R)
#     H_Jl (entang.):   Nc - 1   (2-qubit + link sideband; middle full-tau)
#
# Per half-step:  Nc + (Nc+1) + 2Nc + (Nc-1) = 5 Nc
# Total:          2 * 5 Nc + (Nc - 1) = 11 Nc - 1

def build_one_hot_trotter_step(Nc: int) -> SymCircuit:
    if Nc < 2:
        raise ValueError("Nc must be >= 2 (need at least one A-B pair)")
    c = SymCircuit()
    qubits = [f'q{r}' for r in range(Nc)]
    local = [f'mL{r}' for r in range(Nc)]
    link = 'link'

    def chi_forward():
        for r in range(Nc):
            c.append('chi', [qubits[r], local[r]], entangling=True)

    def chi_reverse():
        for r in reversed(range(Nc)):
            c.append('chi', [qubits[r], local[r]], entangling=True)

    def cd_forward():
        for r in range(Nc):
            c.append('cd', [qubits[r], local[r]], entangling=True)
        c.append('cd', [qubits[0], link], entangling=True)  # A + link

    def cd_reverse():
        c.append('cd', [qubits[0], link], entangling=True)  # A + link
        for r in reversed(range(Nc)):
            c.append('cd', [qubits[r], local[r]], entangling=True)

    def det_forward():
        for m in local + [link]:
            c.append('Rotation', [m], entangling=False)
        for r in range(1, Nc):
            c.append('rz', [qubits[r]], entangling=False)

    def det_reverse():
        for r in reversed(range(1, Nc)):
            c.append('rz', [qubits[r]], entangling=False)
        for m in reversed(local + [link]):
            c.append('Rotation', [m], entangling=False)

    def j_forward():
        for r in range(1, Nc):
            c.append('J', [qubits[0], qubits[r]], entangling=True)

    def j_reverse():
        for r in reversed(range(1, Nc)):
            c.append('J', [qubits[0], qubits[r]], entangling=True)

    def jl_middle():
        for r in range(1, Nc):
            c.append('Jl', [qubits[0], qubits[r], link], entangling=True)

    chi_forward()
    cd_forward()
    det_forward()
    j_forward()
    jl_middle()
    j_reverse()
    det_reverse()
    cd_reverse()
    chi_reverse()
    return c


# =====================================================================
# COMPACT (log2) builder -- real Qiskit circuit, transpile to cx+u
# =====================================================================
#
# m = ceil(log2(Nc + 1)) qubits, binary encoding
#     b(R) = m-bit representation of R (R in {0, 1, ..., Nc})
#
# The compact register carries the same Nc+1 electronic states as the qudit
# encoding, so the Hamiltonian term structure is IDENTICAL to qudit:
#     Nc + 1  free rotations   (mode-only; 0 register CNOTs)
#     Nc      QCR-type          (site-projector, register-side = MCPhase(m-1))
#     Nc + 1  QCD-type          (site-projector, register-side = MCPhase(m-1))
#     Nc + 1  QPS-type          (site-projector diagonal, MCPhase(m-1))
#     Nc - 1  QT carrier        (pairwise exchange = C^{m-1}Ry after X-conj)
#     Nc - 1  QT-SB sideband    (register-side same as QT carrier)
#
# All register operations are emitted as real Qiskit gates and the whole
# thing is transpiled to basis=['cx', 'u'] at optimization_level=1.
# This yields an honest CNOT count -- no estimated / asymptotic numbers.
#
# Site-projector operations that in principle need mode-conditioning are
# emitted here as the register-side stub (MCPhase); the mode-conditioned
# qumode gate is 0 CNOTs on the register and is out of scope for the
# CNOT metric.

def _bit_pattern(R: int, m: int) -> List[int]:
    return [(R >> i) & 1 for i in range(m)]


def build_compact_trotter_step_qiskit(Nc: int):
    """
    Return (raw_qc, transpiled_qc) for a single second-order Trotter step
    of the compact (log2) encoding.  transpiled_qc is in cx+u basis.

    Uses Qiskit's MCPhaseGate for site-diagonal terms and RYGate.control()
    for pairwise electronic exchanges.  All angles are placeholder floats
    (0.1 rad) since only gate counts matter here -- content of the circuit
    is structure-preserving.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import MCPhaseGate, RYGate
    except ImportError as exc:
        raise ImportError(
            "Qiskit required for compact encoding (task 2c). "
            "Install with: pip install qiskit"
        ) from exc

    if Nc < 2:
        raise ValueError("Nc must be >= 2")
    m = max(1, math.ceil(math.log2(Nc + 1)))
    qc = QuantumCircuit(m, name=f'compact_Nc{Nc}')

    THETA = 0.1  # placeholder rotation angle -- content-preserving

    def x_conjugate_to_R(R: int):
        """Emit X gates on positions where b(R) is 0, so that |R> -> |1..1>."""
        bits = _bit_pattern(R, m)
        for i, bit in enumerate(bits):
            if bit == 0:
                qc.x(i)

    def site_projector_op(R: int, angle: float):
        """
        Realize register-side of exp(-i angle |R><R|) as MCPhase.
        MCPhaseGate applies exp(i angle) on the all-ones state; conjugate
        by X on 0-bits of b(R) to redirect the projector to |R>.
        """
        x_conjugate_to_R(R)
        if m == 1:
            # Single-qubit case: MCPhase becomes a plain phase (rz + global)
            qc.p(-angle, 0)
        else:
            qc.append(MCPhaseGate(-angle, num_ctrl_qubits=m - 1),
                      list(range(m)))
        x_conjugate_to_R(R)  # uncompute

    def pairwise_exchange(R: int, Rp: int, angle: float):
        """
        Realize register-side of exp(-i angle (|R><Rp| + |Rp><R|)) as a
        C^{m-1} Ry(2 angle) on a chosen pivot qubit after conjugating by
        X's that reduce the b(R)-vs-b(Rp) difference to that pivot bit.
        """
        bR = _bit_pattern(R, m)
        bRp = _bit_pattern(Rp, m)
        diff = [i for i in range(m) if bR[i] != bRp[i]]
        if not diff:
            return  # R == Rp -- shouldn't happen
        pivot = diff[0]

        # Post-conjugation goal: |R> and |Rp> differ only in the pivot bit,
        # and all non-pivot bits are 1 (so controls trigger on |1>).
        # For every non-pivot bit i: apply X iff bR[i] == 0.  This makes
        # non-pivot bits equal to 1 in both |R> and |Rp> (since bR == bRp
        # for i not in diff, and bR flips through X to 1; for i in diff \
        # {pivot}, bR[i] and bRp[i] differ, so applying X where bR[i]==0
        # sets bR[i] -> 1 and bRp[i] -> 1 too via the same X since they
        # differ by one -- WAIT.  This is subtle, so let's derive:
        #
        #   For i in diff \ {pivot}: bR[i] != bRp[i].  After X on both
        #   states equally, they'd still differ, so we can't make BOTH
        #   equal to 1 by applying a single X.  Fix: apply an X iff
        #   bR[i] == 0 AND i not in diff (non-diff bits).  For i in
        #   diff \ {pivot}, apply a CNOT with pivot as control and i as
        #   target -- this equalises the two states on that qubit.
        #
        # We use the standard "gray-code" reduction: two CNOTs per extra
        # differing bit (pivot as control, that bit as target) -- one to
        # equalise, one to unequalise after the multi-controlled Ry.

        # Step 1: X-conjugate NON-DIFFERING 0-bits to 1
        non_diff = [i for i in range(m) if i not in diff]
        for i in non_diff:
            if bR[i] == 0:
                qc.x(i)

        # Step 2: gray-code CNOTs to reduce diff to pivot only
        extra_diff = [i for i in diff if i != pivot]
        for i in extra_diff:
            qc.cx(pivot, i)
        # Now on |R>-|Rp> subspace, non-pivot bits agree between the two
        # states.  But those agreed values might be 0 (if bR[i]=0=bRp[i]
        # after CNOT) rather than 1.  Fix: for each extra_diff bit, the
        # post-CNOT value equals bR[pivot] XOR bR[i].  Both states have
        # the same post-CNOT value only if we compute it consistently.
        # We further X-conjugate each extra_diff bit if the post-CNOT
        # value happens to be 0 for |R>.
        for i in extra_diff:
            post = bR[pivot] ^ bR[i]
            if post == 0:
                qc.x(i)

        # Step 3: apply C^{m-1} Ry(2 angle) on pivot with all other qubits
        #         as controls (they are now all 1 on both |R> and |Rp>)
        controls = [i for i in range(m) if i != pivot]
        if m == 1:
            qc.ry(2 * angle, 0)
        else:
            qc.append(RYGate(2 * angle).control(m - 1),
                      controls + [pivot])

        # Uncompute Step 3 => none needed (Ry is the intended op)
        # Uncompute Step 2 (extra-diff X-conjugation)
        for i in extra_diff:
            post = bR[pivot] ^ bR[i]
            if post == 0:
                qc.x(i)
        # Uncompute the gray-code CNOTs (they are self-inverse)
        for i in reversed(extra_diff):
            qc.cx(pivot, i)
        # Uncompute non-diff X-conjugation
        for i in non_diff:
            if bR[i] == 0:
                qc.x(i)

    A = 1  # chromophore A
    chromophores = list(range(1, Nc + 1))  # R = 1..Nc (R=0 is ground)

    # ============ FORWARD HALF-STEP ============
    # (1) free rotations -- mode-only, 0 register ops
    # (2) QCR: site-projector for each chromophore (register-side stub)
    for R in chromophores:
        site_projector_op(R, THETA)
    # (3) QCD: Nc local + 1 link-mode for A
    for R in chromophores:
        site_projector_op(R, THETA)
    site_projector_op(A, THETA)  # A + link
    # (4) QPS: Nc local + 1 link-mode for A
    for R in chromophores:
        site_projector_op(R, THETA)
    site_projector_op(A, THETA)  # A + link
    # (5) QT carriers A <-> R for R != A
    for R in chromophores:
        if R != A:
            pairwise_exchange(A, R, THETA)

    # ============ MIDDLE (full-tau) QT-SB ============
    for R in chromophores:
        if R != A:
            pairwise_exchange(A, R, 2 * THETA)  # full tau

    # ============ BACKWARD HALF-STEP ============
    for R in reversed(chromophores):
        if R != A:
            pairwise_exchange(A, R, THETA)
    site_projector_op(A, THETA)
    for R in reversed(chromophores):
        site_projector_op(R, THETA)
    site_projector_op(A, THETA)
    for R in reversed(chromophores):
        site_projector_op(R, THETA)
    for R in reversed(chromophores):
        site_projector_op(R, THETA)

    # Transpile to standard basis for real CNOT count
    qc_t = transpile(qc, basis_gates=['cx', 'u'], optimization_level=1)
    return qc, qc_t


def compact_row(Nc: int) -> dict:
    """Build compact circuit at given Nc and read gate counts from Qiskit."""
    _, qc_t = build_compact_trotter_step_qiskit(Nc)
    counts = qc_t.count_ops()
    cx = counts.get('cx', 0)
    u = counts.get('u', 0)
    total = cx + u
    m = max(1, math.ceil(math.log2(Nc + 1)))
    return dict(
        nc=Nc,
        encoding='compact',
        register_size=m,
        total_gates=total,
        entangling_gates=cx,   # cx is the only entangling gate in cx+u basis
        cnots=cx,
        swaps=0,               # no SWAPs on all-to-all connectivity (A6)
        depth=qc_t.depth(),
    )


# =====================================================================
# Sanity check at Nc = 3
# =====================================================================

EXPECTED_Nc3 = {
    'qudit': 36,
    'qudit_reduced': 34,
    'one_hot': 32,
}


def sanity_check_Nc3() -> None:
    """Verify the three builders reproduce the Nc=3 counts from the pinned code."""
    got = {
        'qudit': build_qudit_trotter_step(3, fused=False).total(),
        'qudit_reduced': build_qudit_trotter_step(3, fused=True).total(),
        'one_hot': build_one_hot_trotter_step(3).total(),
    }
    problems = [f"  {enc}: expected {EXPECTED_Nc3[enc]}, got {got[enc]}"
                for enc in EXPECTED_Nc3 if got[enc] != EXPECTED_Nc3[enc]]
    if problems:
        print("SANITY CHECK FAILED at Nc=3:")
        for p in problems:
            print(p)
        sys.exit(1)
    print("Sanity check PASSED at Nc=3:  qudit=36, qudit_reduced=34, one_hot=32.")


# =====================================================================
# Uniform row emission
# =====================================================================

CSV_FIELDS = ['nc', 'encoding', 'register_size',
              'total_gates', 'entangling_gates',
              'cnots', 'swaps', 'depth']


def symbolic_row(Nc: int, encoding: str) -> dict:
    if encoding == 'qudit':
        c = build_qudit_trotter_step(Nc, fused=False)
        reg = Nc + 1
    elif encoding == 'qudit_reduced':
        c = build_qudit_trotter_step(Nc, fused=True)
        reg = Nc + 1
    elif encoding == 'one_hot':
        c = build_one_hot_trotter_step(Nc)
        reg = Nc
    else:
        raise ValueError(encoding)
    return dict(
        nc=Nc,
        encoding=encoding,
        register_size=reg,
        total_gates=c.total(),
        entangling_gates=c.n_entangling(),
        cnots='',          # native ISA; not CNOT-compiled
        swaps=0,
        depth=c.depth(),
    )


def build_all_rows(Ncs: Iterable[int]) -> List[dict]:
    rows: List[dict] = []
    for Nc in Ncs:
        for enc in ('qudit', 'qudit_reduced', 'one_hot'):
            rows.append(symbolic_row(Nc, enc))
        rows.append(compact_row(Nc))
    return rows


def emit_csv(rows: List[dict], path: str) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in CSV_FIELDS})


def emit_md(rows: List[dict], path: str) -> None:
    """Human-readable table + blunt trend paragraph."""
    from collections import defaultdict
    by_key = {(r['nc'], r['encoding']): r for r in rows}
    Ncs = sorted({r['nc'] for r in rows})
    encodings = ['qudit', 'qudit_reduced', 'one_hot', 'compact']

    lines: List[str] = []
    lines.append('# Vibronic Trotter-step gate counts (Nc = 3..12)\n')
    lines.append('Generated by `simulations/vibronic_scaling.py`.  Every '
                 'count is read from a programmatically-constructed '
                 'circuit; no hand-tallied or extrapolated numbers.\n')
    lines.append('Sanity check at Nc = 3 (matches the pinned '
                 '`trotter_method_a.py` / `trotter_method_a_fused_qps.py` / '
                 '`trotter_method_b_qubit.py`):\n')
    lines.append('- qudit          : 36 native gates\n'
                 '- qudit_reduced  : 34 native gates\n'
                 '- one_hot        : 32 native gates\n')

    # Total-gate table
    lines.append('\n## Total gates per second-order Trotter step\n')
    hdr = ['Nc'] + encodings
    lines.append('| ' + ' | '.join(hdr) + ' |')
    lines.append('|' + '|'.join(['---'] * len(hdr)) + '|')
    for Nc in Ncs:
        row = [str(Nc)] + [str(by_key[(Nc, e)]['total_gates']) for e in encodings]
        lines.append('| ' + ' | '.join(row) + ' |')

    # Entangling / CNOT table
    lines.append('\n## Entangling gates per step\n')
    lines.append('For the qudit / qudit_reduced / one_hot columns this is '
                 'the count of native two-body / two-DoF gates '
                 '(QCR/QCD/QT/QT-SB or chi/cd/J/Jl).  For compact it is the '
                 'transpiled CNOT count on the log2 qubit register '
                 '(basis `cx + u`, opt_level=1).\n')
    lines.append('| ' + ' | '.join(hdr) + ' |')
    lines.append('|' + '|'.join(['---'] * len(hdr)) + '|')
    for Nc in Ncs:
        row = [str(Nc)] + [str(by_key[(Nc, e)]['entangling_gates']) for e in encodings]
        lines.append('| ' + ' | '.join(row) + ' |')

    # Register size and depth
    lines.append('\n## Register size and depth\n')
    lines.append('| Nc | ' + ' | '.join(f'{e} reg' for e in encodings) + ' | '
                 + ' | '.join(f'{e} depth' for e in encodings) + ' |')
    lines.append('|' + '|'.join(['---'] * (1 + 2 * len(encodings))) + '|')
    for Nc in Ncs:
        regs = [str(by_key[(Nc, e)]['register_size']) for e in encodings]
        deps = [str(by_key[(Nc, e)]['depth']) for e in encodings]
        lines.append('| ' + ' | '.join([str(Nc)] + regs + deps) + ' |')

    # Blunt trend paragraph -- compute from the actual numbers
    lines.append('\n## What the numbers actually show\n')
    q3  = by_key[(3, 'qudit')]['total_gates']
    q12 = by_key[(12, 'qudit')]['total_gates']
    o3  = by_key[(3, 'one_hot')]['total_gates']
    o12 = by_key[(12, 'one_hot')]['total_gates']
    c3  = by_key[(3, 'compact')]['entangling_gates']
    c12 = by_key[(12, 'compact')]['entangling_gates']
    e_q12 = by_key[(12, 'qudit')]['entangling_gates']
    e_o12 = by_key[(12, 'one_hot')]['entangling_gates']

    # slopes
    qudit_growth  = (q12 - q3) / (12 - 3)
    onehot_growth = (o12 - o3) / (12 - 3)

    # Compact CNOT ratchet at m boundaries -- surface it explicitly
    ratchets = []
    for i in range(1, len(Ncs)):
        prev = by_key[(Ncs[i - 1], 'compact')]['register_size']
        curr = by_key[(Ncs[i], 'compact')]['register_size']
        if curr != prev:
            ratchets.append((Ncs[i], prev, curr,
                             by_key[(Ncs[i - 1], 'compact')]['cnots'],
                             by_key[(Ncs[i], 'compact')]['cnots']))
    ratchet_txt = '; '.join(
        f'Nc={nc}: m {p}->{c} jumps CNOTs {cp}->{cc}'
        for nc, p, c, cp, cc in ratchets)

    para = (
        f'**Qudit total gates**: {q3} at Nc=3 -> {q12} at Nc=12, exactly '
        f'`11*Nc + 3` (slope {qudit_growth:.1f}).  '
        f'**One-hot total gates**: {o3} -> {o12}, exactly `11*Nc - 1` '
        f'(slope {onehot_growth:.1f}).  '
        f'The qudit sits `+4` above one-hot at *every* Nc -- a pure '
        f'constant, not a scaling advantage.  '
        f'\n\n'
        f'**Entangling gates are literally IDENTICAL** between qudit and '
        f'one-hot: both `7*Nc - 1` ({e_q12} at Nc=12).  The '
        f'four-gate total-count gap is entirely non-entangling '
        f'(mode Rotation + qps bookkeeping).  In the metric that actually '
        f'sets error budget -- two-body gate count -- qudit and one-hot '
        f'in this vibronic model are **the same**.  '
        f'\n\n'
        f'**Compact (log2 qubit) CNOTs**: {c3} at Nc=3, {c12} at Nc=12, '
        f'so {c12 / max(1, e_q12):.1f}x the qudit entangling-gate count '
        f'at Nc=12.  The compact curve jumps in ratchets whenever '
        f'`m = ceil(log2(Nc+1))` increments ({ratchet_txt}).  Between '
        f'ratchets the growth is linear at fixed m.  The pairwise '
        f'`C^(m-1) Ry` decomposition and the per-site MCPhase both scale '
        f'with `m` and dominate the CNOT budget.  '
        f'\n\n'
        f'**BLUNT VERDICT.**  In native-ISA units, qudit vs one-hot for '
        f'this A-star single-excitation vibronic model is a wash: identical '
        f'entangling counts, +4 constant on totals, forever.  The qudit '
        f'advantage in the *simulation* is not "fewer gates than one-hot"; '
        f'it is register-footprint (1 ion vs Nc qubits), no MS-gate '
        f'shuttling between ions, and no need for the ancilla / decoding '
        f'infrastructure that any compact encoding demands.  The qudit '
        f'vs qubit gap that IS large ({c12 / max(1, e_q12):.1f}x at Nc=12) '
        f'exists only against the compact log2 encoding, which no '
        f'reasonable qubit implementation of a single-excitation model '
        f'would actually pick.  Reporting "qudit crushes qubits" without '
        f'saying which qubit encoding is being crushed is misleading.\n'
    )
    lines.append(para)

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# =====================================================================
# Main
# =====================================================================

def _results_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'results')
    os.makedirs(d, exist_ok=True)
    return d


def main() -> None:
    sanity_check_Nc3()

    Ncs = list(range(3, 13))
    rows = build_all_rows(Ncs)

    outdir = _results_dir()
    csv_path = os.path.join(outdir, 'gate_counts.csv')
    md_path  = os.path.join(outdir, 'gate_counts.md')

    emit_csv(rows, csv_path)
    emit_md(rows, md_path)

    # Print a summary table
    from collections import defaultdict
    by = {(r['nc'], r['encoding']): r for r in rows}
    encodings = ['qudit', 'qudit_reduced', 'one_hot', 'compact']
    print()
    print(f"{'Nc':>3s}  " + '  '.join(f'{e:>15s}' for e in encodings))
    for Nc in Ncs:
        cells = []
        for e in encodings:
            r = by[(Nc, e)]
            if e == 'compact':
                cells.append(f'total={r["total_gates"]:>4d} cx={r["cnots"]:>4d}')
            else:
                cells.append(f'total={r["total_gates"]:>4d} ent={r["entangling_gates"]:>3d}')
        print(f'{Nc:>3d}  ' + '  '.join(f'{c:>15s}' for c in cells))

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == '__main__':
    main()
