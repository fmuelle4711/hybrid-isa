"""Render the 32-operation qubit reference step in the SAME style as the
qudit figure, figures/circuit_qudit_generalized_27_one_step.pdf.

That figure is drawn by simulations/draw_qudit_27_hqml.py with
hqml.draw_mpl(..., style='sketch').  The existing qubit drawing,
circuits/trapped_ion_qubit/circuit_qubit_logical32.py, uses the DEFAULT
style with show_wire_types and a suptitle, so the two do not sit together
on a page.  This script reuses that module's circuit unchanged and only
re-renders it, so there is exactly one definition of the 32-operation step
in the repo.

GATE NAMES.  The qubit reference gates are NOT renamed by the manuscript:
CD and CR are the two-level gates, not qudit ISA primitives, and keep their
names.  Only the drawn labels are mapped to the words Table tab:qubit32
uses, so the figure and that table agree:

    drawn label                          paper wording
    ConditionalRotation / CR             dispersive  chi sigma_z n
    ConditionalDisplacement / CD         conditional displacement
    IsingXY                              exchange J
    JlGate                               link exchange J_l
    Rotation / R                         mode rotation
    RZ                                   detuning r_z (virtual)

Output: figures/circuit_qubit_32_one_step.{pdf,png,svg}
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import pennylane as qml                                        # noqa: E402
import hybridlane as hqml                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "circuits", "trapped_ion_qubit"))

import circuit_qubit_logical32 as LOG                          # noqa: E402

EXPECTED = 32


# The sketch renderer draws BOTH Rotation and ConditionalRotation as "R",
# and ConditionalDisplacement as "D".  Left alone this circuit shows 14
# boxes reading "R" for what Table tab:qubit32 splits into 8 mode rotations
# and 6 dispersive gates, which a reader cannot separate.  Patching
# Operator.label does not help -- the renderer does not go through it.
#
# So the relabel is driven from the TAPE instead.  The renderer emits
# exactly one text artist per operation, in tape order, which is checked
# two ways below: the counts must match, and every artist must already
# carry the label its operation is expected to produce.  A misaligned zip
# therefore raises instead of silently mislabelling the figure.
_RENAME = {
    "Rotation":                 ("R", "R"),        # expected, final
    "ConditionalRotation":      ("R", "CR"),
    "ConditionalDisplacement":  ("D", "CD"),
    "RZ":                       ("RZ", "RZ"),
    "IsingXY":                  ("IsingXY", "J"),
    "JlGate":                   ("Jl", "Jl"),
}


def make_qubit_32():
    """The 32-operation step, reusing circuit_qubit_logical32 unchanged."""
    dev = qml.device("default.qubit", wires=LOG.WIRE_ORDER)

    @qml.qnode(dev)
    def one_step():
        LOG.logical_trotter_step()
        return qml.state()

    fig, _ = hqml.draw_mpl(one_step, wire_order=LOG.WIRE_ORDER,
                           style="sketch")()
    relabel_from_tape(fig, one_step)
    # one Trotter step is followed by more steps, not by a readout
    from circuit_fig_common import strip_measurements
    strip_measurements(fig, n_wires=len(LOG.WIRE_ORDER))
    check_labels(fig)
    return fig, one_step


def relabel_from_tape(fig, qnode):
    """Rewrite each drawn label using the operation it belongs to.

    The renderer lays operations out in layers, not in tape order, so the
    two cannot simply be zipped.  Instead each text artist is matched to an
    operation by its position: the x coordinate is the layer index and the
    y coordinate falls within the span of that operation's wires.  Within a
    layer the operations occupy disjoint wires, so the match is unique.

    Every match is then checked against the label the operation is expected
    to have produced, so a wrong match raises rather than mislabelling.
    """
    from pennylane.drawer.drawable_layers import drawable_layers

    tape = qml.workflow.construct_tape(qnode)()
    ops = list(tape.operations)
    wire_map = {w: i for i, w in enumerate(LOG.WIRE_ORDER)}
    layers = drawable_layers(ops, wire_map=wire_map)

    wires = set(str(w) for w in LOG.WIRE_ORDER)
    arts = [t for ax in fig.axes for t in ax.texts
            if t.get_text() not in wires]
    if len(arts) != len(ops):
        raise RuntimeError(
            f"{len(arts)} drawn labels for {len(ops)} operations; the "
            "renderer is not emitting one text per op")

    for art in arts:
        x, y = art.get_position()
        layer = layers[int(round(x))]
        hits = [o for o in layer
                if min(wire_map[w] for w in o.wires) - 0.01 <= y
                <= max(wire_map[w] for w in o.wires) + 0.01]
        if len(hits) != 1:
            raise RuntimeError(
                f"label at (x={x}, y={y}) reading {art.get_text()!r} matched "
                f"{len(hits)} operations in its layer; cannot relabel safely")
        op = hits[0]
        if op.name not in _RENAME:
            raise RuntimeError(f"no label rule for operation {op.name!r}")
        expected, final = _RENAME[op.name]
        got = art.get_text()
        if not got.startswith(expected):
            raise RuntimeError(
                f"label/op mismatch: operation {op.name!r} expected a label "
                f"starting {expected!r} but the artist reads {got!r}")
        art.set_text(re.sub(r"^" + re.escape(expected), final, got))


# What the figure must show, straight off Table tab:qubit32.
WANT = {"R": 8, "CR": 6, "CD": 8, "RZ": 4, "J": 4, "Jl": 2}


def check_labels(fig):
    """Assert the drawn boxes match Table tab:qubit32 row for row."""
    import collections
    wires = set(str(w) for w in LOG.WIRE_ORDER)
    got = collections.Counter(
        t.get_text() for ax in fig.axes for t in ax.texts
        if t.get_text() not in wires)
    if dict(got) != WANT:
        raise RuntimeError(
            f"drawn labels {dict(got)} do not match tab:qubit32 {WANT}")
    print(f"  label audit: {dict(sorted(got.items()))}  matches tab:qubit32")


def count_ops(qnode):
    tape = qml.workflow.construct_tape(qnode)()
    ops = list(tape.operations)
    kinds = {}
    for op in ops:
        kinds[op.name] = kinds.get(op.name, 0) + 1
    return len(ops), kinds


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    fig, qnode = make_qubit_32()
    n, kinds = count_ops(qnode)
    print(f"operations: {n}")
    print(f"  by kind: {dict(sorted(kinds.items()))}")
    if n != EXPECTED:
        print(f"  ERROR: expected {EXPECTED}, got {n} -- figure NOT written")
        return 1

    out = os.path.join(ROOT, "figures", "circuit_qubit_32_one_step")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out}.{{png,pdf,svg}}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
