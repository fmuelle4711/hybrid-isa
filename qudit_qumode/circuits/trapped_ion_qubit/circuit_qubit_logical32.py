"""One-hot QUBIT-encoding Trotter step at the LOGICAL level -- 32 operations.

Companion to circuits/trapped_ion_qubit/circuit_qubit_hybridlane.py, which draws the same step at
the COMPILED level (70 ops = 58 timed + 12 virtual).  Same physics, same gate
order, same parameters; the only difference is how deep the drawing goes.

    logical   (this file)              32 = 11*Nc - 1   at Nc = 3
    compiled  (circuit_qubit_hybridlane) 70 = 58 timed + 12 virtual

WHERE THE 32 COMES FROM -- 11*Nc - 1, term by term at Nc = 3

    chi       2*Nc      = 6     ConditionalRotation, the -chi_r n_r sigma_z / 2
    cd        2*(Nc+1)  = 8     ConditionalDisplacement, Nc local + 1 link
    Rotation  2*(Nc+1)  = 8     free mode evolution, Nc local + 1 link
    rz        2*(Nc-1)  = 4     per-chromophore detuning frame update
    J         2*(Nc-1)  = 4     exchange, ONE IsingXY per pair per half-step
    Jl          (Nc-1)  = 2     link-modulated exchange, centre block only
                          ----
                            32

The leading 2's are the symmetric second-order Suzuki mirror: every block is
applied at tau/2 in the forward half and again in the reverse half.  Jl carries
no 2 because it is the CENTRE block, applied once at full tau -- that single
un-mirrored block is the "-1" in 11*Nc - 1.

WHAT "LOGICAL" MEANS HERE, AND WHAT IT DOES NOT

  This is a picture of the Hamiltonian's term structure, not of the pulse
  sequence.  Two of the six rows above are drawn as single boxes that no
  trapped-ion machine can execute in one shot:

    J  is drawn as ONE qml.IsingXY per pair.  A bichromatic Molmer-Sorensen
       window produces sigma_phi (x) sigma_phi for a SINGLE phase setting --
       XX or YY, never their sum.  The hardware therefore needs two windows
       per pair.  The split is EXACT, since [XX, YY] = 0: it is forced by the
       hardware, not by Trotterisation, and it costs no accuracy.
       4 logical IsingXY  ->  8 MS windows.

    Jl is drawn as ONE opaque box per chromophore pair.  It has no native
       primitive at all.  Each of its FOUR terms (g_abl and g_acl, each
       appearing once in H2XX and once in H2YY) needs its own
       H-CNOT-CD-CNOT-H sandwich, because one sandwich rotates a single
       Pauli axis into sigma_z.
       2 logical Jl boxes  ->  4 sandwiches  ->  28 pulses.

  So this figure MUST NOT be used for any timing, gate-count, or fidelity
  claim -- those come from the compiled figure and from Appendix B.  Its
  legitimate uses are connectivity, term structure, and Trotter ordering.

  This is exactly the trap that the deleted _noh / _norz_noh variants fell
  into (see the VARIANTS note in circuit_qubit_hybridlane.py, removed
  2026-09-09): a reduced drawing is only safe when it is LABELLED as one and
  the reduction is a whole logical gate, never a piece of one.  Dropping the
  Hadamards from inside a Jl sandwich replaced H_Jl with a different
  Hamiltonian; collapsing the entire sandwich into a box labelled "Jl" does
  not, because nothing is claimed about its interior.

Output: figures/circuit_qubit_logical32_one_step.{png,pdf,svg}
        results/circuit_qubit_logical32.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

HERE = Path(__file__).resolve().parent
# Repo root = the directory holding params.py.  Found by walking up rather
# than by counting levels, so moving this file between subfolders (as on
# 2026-09-09, scripts/ -> circuits/ -> circuits/trapped_ion_qubit/) does not
# silently break the imports.
ROOT = next(p for p in HERE.parents if (p / "params.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import hybridlane as hqml                                          # noqa: E402

# Reuse the compiled script's constants verbatim -- same Nc, same wires, same
# parameters, same tau.  Importing rather than re-declaring is deliberate: the
# two figures must not be able to drift apart.
from circuit_qubit_hybridlane import (                             # noqa: E402
    NC, QB, ML, LINK, WIRE_ORDER, TAU, TAU_H,
    CHI, OMEGA_E, S_LOC, OMEGA_M, J0, ETA_J, OUTDIR, TXTDIR,
)
import params as P                                                 # noqa: E402


class JlGate(hqml.ops.Hybrid, qml.operation.Operation):
    """The link-modulated exchange (XX + YY) (x) x_link, as one opaque box.

    The DRAWING shows this as one box: qml.draw and hqml.draw_mpl render the
    tape as constructed and do not expand it, so the figure stays at 32.

    The decomposition below exists only so the circuit can be EXECUTED, which
    is what verify_logical_vs_compiled.py needs.  It is the same four
    H-CNOT-CD-CNOT-H sandwiches the compiled circuit emits, so agreement on
    the Jl block is true BY CONSTRUCTION and is not independent evidence --
    the verification script says so in its own output.  The content of the
    equivalence check is the J block, where the two levels use genuinely
    different gates (one IsingXY against IsingXX then IsingYY).

    Inherits hqml.ops.Hybrid so the static-analysis pass types the third wire
    as a Qumode.  Without it hqml.sa infers every wire of an unrecognised
    operator as a Qubit and aborts, since `link` is already a Qumode
    elsewhere in the tape.  The mixin's convention is qubits first, then
    qumodes -- which is why the wire order is [QB[0], QB[r], LINK].
    """

    num_wires = 3
    num_params = 1
    num_qumodes = 1
    grad_method = None

    @staticmethod
    def compute_decomposition(g, wires):
        # g is the SIGNED coupling, exactly as the compiled jl_block computes
        # it.  The displacement takes (|g|/2, arg g); arg g is pi when g < 0,
        # so passing abs(g) alone and hardcoding phase 0 would silently drop a
        # sign.  Kept signed so the two levels cannot disagree on it.
        q0, qr, link = wires[0], wires[1], wires[2]
        alpha, phi = abs(g) / 2.0, float(np.angle(g + 0j))
        ops = []
        for axis in ("xx", "yy"):
            if axis == "yy":
                ops += [qml.RZ(-np.pi / 2, wires=q0), qml.RZ(-np.pi / 2, wires=qr)]
            ops += [qml.Hadamard(wires=q0), qml.Hadamard(wires=qr),
                    qml.CNOT(wires=[qr, q0]),
                    hqml.ConditionalDisplacement(alpha, phi, wires=[q0, link]),
                    qml.CNOT(wires=[qr, q0]),
                    qml.Hadamard(wires=qr), qml.Hadamard(wires=q0)]
            if axis == "yy":
                ops += [qml.RZ(np.pi / 2, wires=qr), qml.RZ(np.pi / 2, wires=q0)]
        return ops

    def label(self, decimals=None, base_label=None, cache=None):
        if decimals is None:
            return "Jl"
        return f"Jl({self.data[0]:.{decimals}f})"


# Test-only switch; see jl_block's docstring.  Must stay False for the
# figure -- True emits the compiled Jl block and breaks the 32-op count.
JL_MATCH_COMPILED_ORDER = False


def logical_trotter_step():
    """The same step as circuit_qubit_hybridlane.one_hot_trotter_step, drawn
    at the logical level: J stays a single IsingXY, Jl stays a single box."""

    def chi_block(rev=False):
        rng = reversed(range(NC)) if rev else range(NC)
        for r in rng:
            hqml.ConditionalRotation(-2 * CHI[r] * TAU_H, wires=[QB[r], ML[r]])

    def cd_block(rev=False):
        if not rev:
            for r in range(NC):
                hqml.ConditionalDisplacement(
                    OMEGA_E[r] * np.sqrt(S_LOC[r]) * TAU_H, np.pi / 2,
                    wires=[QB[r], ML[r]])
            hqml.ConditionalDisplacement(
                P.omega_l * np.sqrt(P.S_l) * TAU_H, np.pi / 2,
                wires=[QB[0], LINK])
        else:
            hqml.ConditionalDisplacement(
                P.omega_l * np.sqrt(P.S_l) * TAU_H, np.pi / 2,
                wires=[QB[0], LINK])
            for r in reversed(range(NC)):
                hqml.ConditionalDisplacement(
                    OMEGA_E[r] * np.sqrt(S_LOC[r]) * TAU_H, np.pi / 2,
                    wires=[QB[r], ML[r]])

    def det_block(rev=False):
        modes = ML + [LINK]
        if not rev:
            for m, w in zip(modes, OMEGA_M + [P.omega_l]):
                hqml.Rotation(w * TAU_H, wires=m)
            for r in range(1, NC):
                qml.RZ(OMEGA_E[r] * S_LOC[r] * TAU_H, wires=QB[r])
        else:
            for r in reversed(range(1, NC)):
                qml.RZ(OMEGA_E[r] * S_LOC[r] * TAU_H, wires=QB[r])
            for m, w in zip(reversed(modes), reversed(OMEGA_M + [P.omega_l])):
                hqml.Rotation(w * TAU_H, wires=m)

    def j_block(rev=False):
        """ONE IsingXY per pair -- the logical exchange gate.

        The hardware needs two MS windows per box; see the module docstring.

        ANGLE.  The two conventions do NOT share an argument:

            IsingXX(t) = exp(-i t/2 XX),  IsingYY(t) = exp(-i t/2 YY)
            IsingXY(p) = exp(+i p/4 (XX + YY))

        so IsingXX(t) IsingYY(t) = exp(-i t/2 (XX+YY)) = IsingXY(-2t), and the
        logical box must carry -2 * J0 * TAU_H to match the compiled pair at
        J0 * TAU_H.  Passing the same argument to both is wrong by a factor of
        -2; verify_logical_vs_compiled.py measures this identity rather than
        trusting it.  The pre-b1f8ee6 figure made exactly that mistake, on top
        of emitting only two Jl sandwiches.
        """
        rng = reversed(range(1, NC)) if rev else range(1, NC)
        for r in rng:
            qml.IsingXY(-2.0 * J0[r - 1] * TAU_H, wires=[QB[0], QB[r]])

    def jl_block():
        """ONE box per pair, applied once at full tau as the centre block.

        ORDERING.  Each box expands to its own xx sandwich then its own yy
        sandwich, so the four sandwiches come out grouped BY PAIR:

            xx1 yy1 xx2 yy2          (this file, Nc-1 boxes)

        The compiled circuit groups them BY AXIS, matching the production
        qnode's H2XX-then-H2YY structure:

            xx1 xx2 yy1 yy2          (circuit_qubit_hybridlane.py)

        Those are different operators: the xx sandwich for A-B carries
        sigma_x on qubit A and the yy sandwich for A-C carries sigma_y on the
        same qubit A, and they anticommute.  The difference is O(tau^2) --
        i.e. of the same order as the Trotter error the step already has --
        but it is not zero, and verify_logical_vs_compiled.py measures it.

        Set JL_MATCH_COMPILED_ORDER = True to emit the four sandwiches
        directly in by-axis order instead of as boxes.  That destroys the
        32-op count (it is the compiled Jl block), so it is for the
        equivalence test ONLY, never for the figure.
        """
        if not JL_MATCH_COMPILED_ORDER:
            for r in range(1, NC):
                g = J0[r - 1] * ETA_J[r - 1] * TAU
                JlGate(g, wires=[QB[0], QB[r], LINK])
            return
        for axis in ("xx", "yy"):
            for r in range(1, NC):
                g = J0[r - 1] * ETA_J[r - 1] * TAU
                q0, qr = QB[0], QB[r]
                alpha, phi = abs(g) / 2.0, float(np.angle(g + 0j))
                if axis == "yy":
                    qml.RZ(-np.pi / 2, wires=q0); qml.RZ(-np.pi / 2, wires=qr)
                qml.Hadamard(wires=q0); qml.Hadamard(wires=qr)
                qml.CNOT(wires=[qr, q0])
                hqml.ConditionalDisplacement(alpha, phi, wires=[q0, LINK])
                qml.CNOT(wires=[qr, q0])
                qml.Hadamard(wires=qr); qml.Hadamard(wires=q0)
                if axis == "yy":
                    qml.RZ(np.pi / 2, wires=qr); qml.RZ(np.pi / 2, wires=q0)

    det_block(); chi_block(); cd_block(); j_block()
    jl_block()
    j_block(rev=True); cd_block(rev=True); chi_block(rev=True); det_block(rev=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    import collections
    with qml.tape.QuantumTape() as tape:
        logical_trotter_step()
    counts = collections.Counter(op.name for op in tape.operations)
    total = sum(counts.values())
    expect = 11 * NC - 1

    dev = qml.device("default.qubit", wires=WIRE_ORDER)
    node = qml.QNode(lambda: (logical_trotter_step(), qml.state())[1], dev)

    txt = qml.draw(node, wire_order=WIRE_ORDER, max_length=200)()
    hdr = [
        "One-hot QUBIT-encoding Trotter step -- LOGICAL level",
        f"Nc = {NC}   wires = {WIRE_ORDER}",
        "",
        f"  logical operations drawn here      : {total}   (11*Nc - 1 = {expect})",
        "  compiled operations, same step     : 70   = 58 timed + 12 virtual",
        "  see figures/circuit_qubit_hybridlane_one_step.* and Appendix B",
        "",
        "  breakdown:",
        f"    chi       2*Nc      = {2*NC:<3d}  ConditionalRotation",
        f"    cd        2*(Nc+1)  = {2*(NC+1):<3d}  ConditionalDisplacement",
        f"    Rotation  2*(Nc+1)  = {2*(NC+1):<3d}  free mode evolution",
        f"    rz        2*(Nc-1)  = {2*(NC-1):<3d}  detuning frame update",
        f"    J         2*(Nc-1)  = {2*(NC-1):<3d}  IsingXY, one per pair per half",
        f"    Jl          (Nc-1)  = {NC-1:<3d}  centre block, applied once at full tau",
        "",
        "  TWO of these boxes are NOT executable in one shot:",
        "    J  -> 2 MS windows each (XX then YY; one bichromatic window gives",
        "          a single Pauli axis, never XX+YY).  4 logical -> 8 windows.",
        "    Jl -> 4 H-CNOT-CD-CNOT-H sandwiches (g_abl and g_acl, each in both",
        "          H2XX and H2YY).  2 logical boxes -> 4 sandwiches -> 28 pulses.",
        "  Use this figure for connectivity, term structure and Trotter order",
        "  ONLY -- never for a timing, gate-count, or fidelity claim.",
        "",
        "  measured tape counts:",
    ]
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        hdr.append(f"    {k:28s} {v}")
    hdr.append(f"    {'TOTAL':28s} {total}")
    if total != expect:
        hdr.append(f"    MISMATCH: expected 11*Nc-1 = {expect}, drew {total}")
    hdr.append("")

    out_txt = TXTDIR / "circuit_qubit_logical32.txt"
    out_txt.write_text("\n".join(hdr) + txt + "\n", encoding="utf-8")
    print("\n".join(hdr))
    print(txt)
    print(f"\nwrote {out_txt}")

    fig, ax = hqml.draw_mpl(node, wire_order=WIRE_ORDER,
                            show_wire_types=True, decimals=2)()
    fig.suptitle(
        f"One-hot qubit encoding, one Trotter step (Nc={NC})  "
        f"— LOGICAL level, {total} operations",
        fontsize=10)
    for ext in ("png", "pdf", "svg"):
        p = OUTDIR / f"circuit_qubit_logical32_one_step.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
