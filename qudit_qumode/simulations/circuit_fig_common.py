"""Shared helpers for the three circuit figures.

    figures/circuit_qudit_generalized_27_one_step.{pdf,png,svg}
    figures/circuit_qubit_32_one_step.{pdf,png,svg}
    figures/circuit_qubit_58_one_step.{pdf,png}

All three draw one Trotter step, which is followed by more steps and not by
a measurement, so none of them should end in a readout meter.  The renderer
adds one per wire because each circuit is drawn as a qnode and a qnode has
to return something.  strip_measurements removes them.
"""
from __future__ import annotations


def strip_measurements(fig, n_wires, *, required=True):
    """Remove the readout meters the renderer puts at the end of a circuit.

    A meter is three artists per wire: a rounded box, an arc, and a needle.
    They are identified by type and by sitting together in the rightmost
    column, and every count is checked against ``n_wires`` BEFORE anything
    is deleted, so a renderer change cannot silently strip gate boxes
    instead -- it raises.

    Returns the number of artists removed.  With ``required=False`` a figure
    that has no meters is left alone and 0 is returned, which is what a
    continued row needs after its meters have already gone.
    """
    from matplotlib.patches import Arc, FancyArrow, FancyBboxPatch

    ax = fig.axes[0]
    arcs = [p for p in ax.patches if isinstance(p, Arc)]
    needles = [p for p in ax.patches if isinstance(p, FancyArrow)]

    if not arcs and not needles:
        if required:
            raise RuntimeError("no meters found; expected one per wire")
        return 0

    if len(arcs) != n_wires or len(needles) != n_wires:
        raise RuntimeError(
            f"expected {n_wires} arcs and {n_wires} needles for {n_wires} "
            f"wires, found {len(arcs)} and {len(needles)}; not stripping")

    boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    if not boxes:
        raise RuntimeError("no boxes at all; the figure is not what "
                           "this helper expects")
    meter_x = max(round(p.get_x(), 2) for p in boxes)
    meter_boxes = [p for p in boxes if round(p.get_x(), 2) == meter_x]
    if len(meter_boxes) != n_wires:
        raise RuntimeError(
            f"{len(meter_boxes)} boxes in the rightmost column at "
            f"x={meter_x}, expected {n_wires}; not stripping")

    for p in arcs + needles + meter_boxes:
        p.remove()

    # pull the right edge in so the removed column leaves no whitespace
    ax.set_xlim(ax.get_xlim()[0], meter_x - 0.25)

    left = [p for p in ax.patches if isinstance(p, (Arc, FancyArrow))]
    if left:
        raise RuntimeError(f"{len(left)} meter artists survived the strip")
    return len(arcs) + len(needles) + len(meter_boxes)
