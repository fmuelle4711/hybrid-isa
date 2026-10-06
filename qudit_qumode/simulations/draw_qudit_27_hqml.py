"""Render the 27-op circuit using HybridLane's draw_mpl, style='sketch' --
the same renderer and style as draw_qudit_28_hqml.py, which is what actually
produced figures/circuit_qudit_generalized_28_one_step.{png,pdf,svg}.

The ONLY difference from the 28-op circuit is the centre: the two sideband
QT calls

    hqml.QT(J_AB0*eta_AB*TAU, level_i=A, level_j=B, wires=['q','l'], sideband=True)
    hqml.QT(J_AC0*eta_AC*TAU, level_i=A, level_j=C, wires=['q','l'], sideband=True)

are replaced by ONE multitone sideband block on the link mode, giving 27
operations instead of 28.  Everything else -- the X on the qudit, the four
Rotations, three QCR, four QCD, the fused QPS box and the fused QT box, and
the mirrored backward half-step -- is byte-for-byte the same as the 28-op
script.

PARAMETER CONVENTION.  As in draw_qudit_28_hqml.py, a fused box carries the
SUMMED coefficient with a placeholder level pair, because hqml.QT renders one
transition per box.  The 28-op script already does this for its fused QPS box
(sum of all four Stark terms, level=A placeholder) and its fused QT box
(J_AB0 + J_AC0, A-B placeholder).  The merged QT-SB box follows the same
convention: coefficient (J_AB0*eta_AB + J_AC0*eta_AC)*TAU with an A-B
placeholder.  The drawn parameter is therefore a LABEL for the multitone
block, not the literal generator -- the true generator is
Sigma_AB + Sigma_AC, which is not a single A-B sideband.  The operator-level
consequences of the merge are quantified in simulations/circuit_diagrams_27.py
and results/circuit_qudit_27.txt.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pennylane as qml
import hybridlane as hqml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import params as P                                               # noqa: E402

TAU = 0.005e-12
TAU_H = TAU / 2
_G, _A, _B, _C = 0, 1, 2, 3


def make_qudit_27():
    dev = qml.device('bosonicqiskit.hybrid', wires=['q', 'a', 'b', 'c', 'l'],
                     max_fock_level=4)

    @qml.qnode(dev)
    def one_step():
        qml.PauliX(wires='q')
        # forward half-step
        hqml.Rotation(P.omega_ga * TAU_H, wires='a')
        hqml.Rotation(P.omega_gb * TAU_H, wires='b')
        hqml.Rotation(P.omega_gc * TAU_H, wires='c')
        hqml.Rotation(P.omega_l * TAU_H, wires='l')
        hqml.QCR(P.chi_a * TAU_H, level=_A, wires=['q', 'a'])
        hqml.QCR(P.chi_b * TAU_H, level=_B, wires=['q', 'b'])
        hqml.QCR(P.chi_c * TAU_H, level=_C, wires=['q', 'c'])
        hqml.QCD(P.omega_ea*np.sqrt(P.S_a) * TAU_H, np.pi/2, level=_A, wires=['q', 'a'])
        hqml.QCD(P.omega_eb*np.sqrt(P.S_b) * TAU_H, np.pi/2, level=_B, wires=['q', 'b'])
        hqml.QCD(P.omega_ec*np.sqrt(P.S_c) * TAU_H, np.pi/2, level=_C, wires=['q', 'c'])
        hqml.QCD(P.omega_l * np.sqrt(P.S_l) * TAU_H, np.pi/2, level=_A, wires=['q', 'l'])
        # ONE fused QPS_gen  (rendered as a QPS box, level_A placeholder)
        hqml.QPS((P.omega_ea*P.S_a + P.omega_l*P.S_l + P.omega_eb*P.S_b
                  + P.omega_ec*P.S_c) * TAU_H, level=_A, wires='q')
        # ONE fused multi-transition QT  (rendered as a QT box, A-B placeholder)
        hqml.QT((P.J_AB0 + P.J_AC0) * TAU_H, level_i=_A, level_j=_B, wires='q')
        # centre: ONE generalised multitone QT-SB on the link mode
        # (the 28-op circuit has two separate sideband QT calls here)
        hqml.QT((P.J_AB0*P.eta_AB + P.J_AC0*P.eta_AC) * TAU,
                level_i=_A, level_j=_B, wires=['q', 'l'], sideband=True)
        # backward half-step (reverse order)
        hqml.QT((P.J_AB0 + P.J_AC0) * TAU_H, level_i=_A, level_j=_B, wires='q')
        hqml.QPS((P.omega_ea*P.S_a + P.omega_l*P.S_l + P.omega_eb*P.S_b
                  + P.omega_ec*P.S_c) * TAU_H, level=_A, wires='q')
        hqml.QCD(P.omega_l * np.sqrt(P.S_l) * TAU_H, np.pi/2, level=_A, wires=['q', 'l'])
        hqml.QCD(P.omega_ec*np.sqrt(P.S_c) * TAU_H, np.pi/2, level=_C, wires=['q', 'c'])
        hqml.QCD(P.omega_eb*np.sqrt(P.S_b) * TAU_H, np.pi/2, level=_B, wires=['q', 'b'])
        hqml.QCD(P.omega_ea*np.sqrt(P.S_a) * TAU_H, np.pi/2, level=_A, wires=['q', 'a'])
        hqml.QCR(P.chi_c * TAU_H, level=_C, wires=['q', 'c'])
        hqml.QCR(P.chi_b * TAU_H, level=_B, wires=['q', 'b'])
        hqml.QCR(P.chi_a * TAU_H, level=_A, wires=['q', 'a'])
        hqml.Rotation(P.omega_l * TAU_H, wires='l')
        hqml.Rotation(P.omega_gc * TAU_H, wires='c')
        hqml.Rotation(P.omega_gb * TAU_H, wires='b')
        hqml.Rotation(P.omega_ga * TAU_H, wires='a')
        return hqml.state(wires=['q', 'a', 'b', 'c', 'l'])

    fig, _ = hqml.draw_mpl(one_step, style='sketch')()
    relabel_merged_sideband(fig)
    relabel_to_paper_names(fig)
    # one Trotter step is followed by more steps, not by a readout
    from circuit_fig_common import strip_measurements
    strip_measurements(fig, n_wires=5)
    return fig, one_step


# --------------------------------------------------------------------------
# PAPER GATE NAMES.  The manuscript renamed the five ISA primitives.  Code
# identifiers, HybridLane class names and results-file columns still use the
# old names, so this repo carries the mapping everywhere it draws or prints:
#
#     paper name      code / data name      what it is
#     Z_d             QPS                   qudit phase shift
#     X_E             QT                    qudit transition (carrier)
#     C_{\Pi_z}D      QCD                   level-conditioned displacement
#     C_{\Pi}R        QCR                   level-conditioned mode rotation
#     C_{\Pi_x}D      QT-SB, QTsb           transition-conditioned displacement
#
# Multitone forms write the transition set as plain E.  Only the DRAWN TEXT
# is renamed below; hqml.QPS/QT/QCD/QCR calls above are untouched.
#
# Level indices also become letters, matching the paper's |G>,|A>,|B>,|C>:
#     0 -> G   1 -> A   2 -> B   3 -> C
# --------------------------------------------------------------------------
_LEVEL = {'0': 'G', '1': 'A', '2': 'B', '3': 'C'}

_PAPER_NAME = [
    ('QTsb', r'C_{\Pi_x}D'),   # longest first: QTsb before QT
    ('QCD', r'C_{\Pi_z}D'),
    ('QCR', r'C_{\Pi}R'),
    ('QPS', r'Z_{d}'),
    ('QT', r'X_{E}'),
]


def relabel_to_paper_names(fig):
    """Rewrite every gate box label from the code name to the paper name.

    Matches `NAME_{args}` exactly, rewrites NAME by the table above and maps
    the numeric level indices to letters.  Anything that is not a gate label
    (wire names, the leading X, the Rotation boxes) is left alone.  Returns
    the number of labels rewritten so the caller can assert on it.
    """
    import re
    pat = re.compile(r'^(QTsb|QCD|QCR|QPS|QT)_\{([0-9,;]+)\}$')
    n = 0
    for ax in fig.axes:
        for t in ax.texts:
            m = pat.match(t.get_text())
            if not m:
                continue
            code, args = m.group(1), m.group(2)
            paper = dict(_PAPER_NAME)[code]
            args = ''.join(_LEVEL.get(ch, ch) for ch in args)
            t.set_text(f'${paper}({args})$')
            n += 1
    return n


def relabel_merged_sideband(fig, frm='QTsb_{1,2}', to='QTsb_{1,2;1,3}'):
    """Rename the merged centre box so it does not read as the A-B sideband.

    hqml.QT labels its box from level_i/level_j, and the merged block carries
    an A-B placeholder pair (see the module docstring).  Left alone the centre
    would render as 'QTsb_{1,2}', which a reader would take to mean the A-C
    sideband had been DROPPED rather than merged into it.  The 27-op circuit
    has exactly one sideband QT, so exactly one such label must exist; if that
    is not the case the figure is not what this script thinks it is and we
    stop rather than relabel the wrong box.
    """
    hits = [t for ax in fig.axes for t in ax.texts if t.get_text() == frm]
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one '{frm}' label to rename, found {len(hits)}")
    hits[0].set_text(to)


def count_ops(qnode):
    """Operation count from the tape, excluding the leading state-prep X."""
    tape = qml.workflow.construct_tape(qnode)()
    ops = list(tape.operations)
    prep = 1 if ops and ops[0].name in ("PauliX", "X") else 0
    kinds = {}
    for op in ops[prep:]:
        kinds[op.name] = kinds.get(op.name, 0) + 1
    return len(ops) - prep, kinds, len(ops)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    fig, qnode = make_qudit_27()
    n, kinds, n_with_prep = count_ops(qnode)
    print(f"operations (excluding the leading PauliX state prep): {n}")
    print(f"  by kind: {dict(sorted(kinds.items()))}")
    print(f"  boxes drawn including the X: {n_with_prep}")
    if n != 27:
        print(f"  ERROR: expected 27, got {n} -- figure NOT written")
        sys.exit(1)

    out = os.path.join(ROOT, 'figures', 'circuit_qudit_generalized_27_one_step')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'saved {out}.{{png,pdf,svg}}')
