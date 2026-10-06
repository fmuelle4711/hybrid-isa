"""Render the compiled qubit step -- 58 timed pulses plus 12 virtual -- in
the same style as figures/circuit_qubit_32_one_step.pdf.

This is the 32-operation logical step of draw_qubit_32_hqml.py after the
two exchange classes are expanded, as Appendix B describes:

    each J    -> two Molmer-Sorensen windows, one XX and one YY
                 (exact, since [XX, YY] = 0)
    each J_l  -> a sandwich per half: 8 Hadamard, 4 CNOT, 2 conditional
                 displacements, plus 4 virtual Z updates
    RZ        -> virtual, drawn but costing no time

The circuit is NOT redefined here.  It is
circuits/trapped_ion_qubit/circuit_qubit_hybridlane.py::one_hot_trotter_step,
reused unchanged, so the repo keeps one definition of the compiled step.

WIDTH.  One row is 51 inches at 7 wires and 48 layers, roughly seven times
textwidth, so the step is split into two rows at its centre.  Each half is
drawn separately and the two axes are then moved into a single figure.

LABELS.  As in draw_qubit_32_hqml.py, the renderer draws both Rotation and
ConditionalRotation as "R" and ConditionalDisplacement as "D", and does not
go through Operator.label -- drawer/tape_mpl.py substitutes a different
operation before handing it to the drawer.  Labels are therefore matched to
operations by layer and wire span and rewritten, with every match checked
against the label its operation should have produced.

CNOT draws as a control dot and a target cross with NO text, so 70
operations produce only 62 text artists.  The audit counts CNOT from the
tape and everything else from the figure.

Output: figures/circuit_qubit_58_one_step.{pdf,png}
"""
from __future__ import annotations

import collections
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import pennylane as qml                                        # noqa: E402
import hybridlane as hqml                                      # noqa: E402
from circuit_fig_common import strip_measurements               # noqa: E402
from pennylane.drawer.drawable_layers import drawable_layers    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "circuits", "trapped_ion_qubit"))

import circuit_qubit_hybridlane as CMP                          # noqa: E402

WIRES = list(CMP.WIRE_ORDER)
WMAP = {w: i for i, w in enumerate(WIRES)}

# expected drawn label -> final label.  CNOT carries no text at all.
_RENAME = {
    "Rotation":                ("R", "R"),
    "ConditionalRotation":     ("R", "CR"),
    "ConditionalDisplacement": ("D", "CD"),
    "RZ":                      ("RZ", "RZ"),
    "Hadamard":                ("H", "H"),
    "IsingXX":                 ("IsingXX", "MS-XX"),
    "IsingYY":                 ("IsingYY", "MS-YY"),
}
_NO_TEXT = {"CNOT"}

# Appendix B, the full native histogram.
WANT_TIMED = {"H": 16, "CD": 12, "R": 8, "MS-XX": 4, "MS-YY": 4,
              "CNOT": 8, "CR": 6}
WANT_VIRTUAL = {"RZ": 12}


def step_ops():
    with qml.tape.QuantumTape() as tape:
        CMP.one_hot_trotter_step()
    return list(tape.operations)


def draw_ops(ops):
    """Draw a list of operations in the shared style, relabelled."""
    dev = qml.device("default.qubit", wires=WIRES)

    @qml.qnode(dev)
    def node():
        for op in ops:
            qml.apply(op)
        return qml.state()

    fig, _ = hqml.draw_mpl(node, wire_order=WIRES, style="sketch")()
    relabel(fig, ops)
    return fig


def relabel(fig, ops):
    """Match each text artist to its operation by layer and wire span."""
    layers = drawable_layers(ops, wire_map=WMAP)
    wire_names = set(str(w) for w in WIRES)
    arts = [t for ax in fig.axes for t in ax.texts
            if t.get_text() not in wire_names]

    expect = sum(1 for o in ops if o.name not in _NO_TEXT)
    if len(arts) != expect:
        raise RuntimeError(
            f"{len(arts)} text artists for {expect} labelled operations "
            f"({len(ops)} total, {len(ops)-expect} drawn without text)")

    for art in arts:
        x, y = art.get_position()
        layer = layers[int(round(x))]
        hits = [o for o in layer
                if o.name not in _NO_TEXT
                and min(WMAP[w] for w in o.wires) - 0.01 <= y
                <= max(WMAP[w] for w in o.wires) + 0.01]
        if len(hits) != 1:
            raise RuntimeError(
                f"label {art.get_text()!r} at (x={x}, y={y}) matched "
                f"{len(hits)} operations in its layer")
        op = hits[0]
        if op.name not in _RENAME:
            raise RuntimeError(f"no label rule for {op.name!r}")
        exp, final = _RENAME[op.name]
        got = art.get_text()
        if not got.startswith(exp):
            raise RuntimeError(
                f"label/op mismatch: {op.name!r} expected {exp!r}, "
                f"artist reads {got!r}")
        art.set_text(re.sub(r"^" + re.escape(exp), final, got))


def split_at_centre(ops):
    """Split the symmetric step into two halves at its centre layer."""
    layers = drawable_layers(ops, wire_map=WMAP)
    cut = len(layers) // 2
    first = {id(o) for L in layers[:cut] for o in L}
    a = [o for o in ops if id(o) in first]
    b = [o for o in ops if id(o) not in first]
    return a, b, cut, len(layers)


def audit(fig, ops):
    """Box counts: text labels from the figure, CNOT from the tape."""
    wire_names = set(str(w) for w in WIRES)
    got = collections.Counter(
        t.get_text() for ax in fig.axes for t in ax.texts
        if t.get_text() not in wire_names)
    got["CNOT"] = sum(1 for o in ops if o.name == "CNOT")

    timed = {k: v for k, v in got.items() if k not in WANT_VIRTUAL}
    virtual = {k: v for k, v in got.items() if k in WANT_VIRTUAL}
    if timed != WANT_TIMED or virtual != WANT_VIRTUAL:
        raise RuntimeError(
            f"box counts {dict(got)} do not match Appendix B "
            f"(timed {WANT_TIMED}, virtual {WANT_VIRTUAL})")
    return timed, virtual


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ops = step_ops()
    a, b, cut, n_layers = split_at_centre(ops)
    print(f"operations: {len(ops)}   layers: {n_layers}")
    print(f"  split at layer {cut}: {len(a)} operations in row 1, "
          f"{len(b)} in row 2")

    f1, f2 = draw_ops(a), draw_ops(b)
    n1 = strip_measurements(f1, n_wires=len(WIRES))
    n2 = strip_measurements(f2, n_wires=len(WIRES))
    print(f"  readout meters removed: row 1 {n1} artists, row 2 {n2} "
          f"({len(WIRES)} per row); a Trotter step is followed by more "
          f"steps, not a measurement")
    w1, _ = f1.get_size_inches()
    w2, _ = f2.get_size_inches()
    print(f"  row widths: {w1:.1f} in, {w2:.1f} in "
          f"(one row would be {w1+w2:.0f} in)")

    # combined audit over both rows
    both = plt.figure()
    got = collections.Counter()
    wire_names = set(str(w) for w in WIRES)
    for f in (f1, f2):
        got.update(t.get_text() for ax in f.axes for t in ax.texts
                   if t.get_text() not in wire_names)
    got["CNOT"] = sum(1 for o in ops if o.name == "CNOT")
    plt.close(both)

    timed = {k: v for k, v in got.items() if k not in WANT_VIRTUAL}
    virtual = {k: v for k, v in got.items() if k in WANT_VIRTUAL}
    if timed != WANT_TIMED or virtual != WANT_VIRTUAL:
        print(f"  ERROR: box counts {dict(got)} do not match Appendix B")
        print(f"         timed   want {WANT_TIMED}")
        print(f"         virtual want {WANT_VIRTUAL}")
        return 1
    print(f"  label audit timed   : {dict(sorted(timed.items()))} "
          f"= {sum(timed.values())}")
    print(f"  label audit virtual : {dict(sorted(virtual.items()))} "
          f"= {sum(virtual.values())}")
    print(f"  total boxes {sum(got.values())} "
          f"= {sum(timed.values())} timed + {sum(virtual.values())} virtual")

    # --- compose the two rows into one figure -----------------------------
    h = max(f1.get_size_inches()[1], f2.get_size_inches()[1])
    wide = max(w1, w2)
    out = plt.figure(figsize=(wide, 2.15 * h))
    for i, f in enumerate((f1, f2)):
        ax = f.axes[0]
        frac = f.get_size_inches()[0] / wide
        ax.remove()
        ax.figure = out
        out.add_axes(ax)
        ax.set_position([0.0, 0.52 - 0.50 * i, frac, 0.46])
        plt.close(f)

    base = os.path.join(ROOT, "figures", "circuit_qubit_58_one_step")
    for ext in ("pdf", "png"):
        out.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {base}.{{pdf,png}}")
    plt.close(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
