"""Where does 132 come from?  Count it out of VU'S OWN SCRIPT.

WHY THIS EXISTS.  data/circuit_vu_one_step.json says gate_count_actual = 132,
and it is tempting to describe that file as "an enumeration of the primitives
Vu's script emits".  It is not.  simulations/circuit_diagrams.py::make_vu does
not read, import, or execute Vu's file at all -- it BUILDS ITS OWN c2qa circuit
and maintains a parallel hand-written list beside it:

        ckt.cv_r(omega_t[i] / 2, qmr[i]);  add('cv_r', [f'm_{i}'], ...)
        ^ the real gate                    ^ the JSON entry, written by hand

So the JSON is our re-implementation of Vu's algorithm, and 132 counts OUR
circuit.  That is a weaker claim than it looks, and it was asserted here
before it was checked.  This script closes the gap with two independent
comparisons.

CHECK 1 -- is the hand-maintained list faithful to the circuit it mirrors?
    Capture the CVCircuit make_vu constructs and compare len(ckt.data) and the
    per-primitive breakdown against the JSON.  Catches an `add` that was
    forgotten next to a `ckt.` call, or vice versa.

CHECK 2 -- does our circuit match VU'S PUBLISHED CODE?
    This is the one that actually answers the question.  Load Vu's vendored
    script, execute ONLY its prelude (parameters, register setup, and the four
    H0/H1/H2XX/H2YY function definitions), then call those functions ourselves
    for exactly one second-order step in Vu's own order, taken from their
    driver loop:

        H0 H1 H2XX H2YY | H2YY' H2XX' H1' H0'      (' = reverse=True)

    and count the instructions Vu's own code appended.  Nothing is
    transcribed: the gates are emitted by Vu's functions.

    The prelude is cut at `for i in range(sim_steps)` so that the driver loop
    -- 401 steps at 10000 shots, with damping and dephasing channels -- never
    runs.  We want the gates, not the simulation.

TWO THINGS THAT ARE NOT GATES
    barrier   Vu's functions emit 16 barriers per step.  A barrier is a
              scheduling directive to the transpiler; it compiles to nothing
              and occupies no pulse slot.  Vu's raw instruction count is
              therefore 148, and 148 - 16 = 132.
    96        Vu's paper's own "96 gates per step" is an abstract operator
              count, hardcoded at simulations/circuit_diagrams.py:545.  It is
              a citation, not a measurement, and this script does not check it.

Output: circuits/cqed_vu/results/verify_vu_132.txt
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "params.py").exists())
sys.path.insert(0, str(ROOT))

VU_SRC = ROOT / ("legacy/vu_reference/cqed-vibronic-simulation-main/"
                 "3_chromophore_vibronic_simulation.py")
VU_JSON = ROOT / "data" / "circuit_vu_one_step.json"

# c2qa registers its CV gates under short internal names.
ALIAS = {"R": "cv_r", "cD": "cv_c_d", "cR": "cv_c_r"}
NOT_A_GATE = {"barrier"}

# Vu's per-step ordering, read from their driver loop at
# 3_chromophore_vibronic_simulation.py:296-303.
VU_ORDER = [("H0", False), ("H1", False), ("H2XX", False), ("H2YY", False),
            ("H2YY", True), ("H2XX", True), ("H1", True), ("H0", True)]

L = []


def W(s=""):
    L.append(s)
    print(s, flush=True)


def norm(counter):
    out = collections.Counter()
    for k, v in counter.items():
        out[ALIAS.get(k, k)] += v
    return out


def check1_json_vs_our_circuit():
    """Is the hand-maintained JSON list faithful to the circuit make_vu builds?"""
    W("=" * 96)
    W("CHECK 1  --  the JSON list vs the circuit make_vu actually builds")
    W("=" * 96)
    import bosonic_qiskit as c2qa

    built = []
    orig = c2qa.CVCircuit.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        built.append(self)

    c2qa.CVCircuit.__init__ = spy
    try:
        from simulations.circuit_diagrams import make_vu
        _fig, gates, abstract, _srcs = make_vu()
    finally:
        c2qa.CVCircuit.__init__ = orig

    ckt = built[0]
    ours = collections.Counter(g["gate"] for g in gates)
    real = norm(collections.Counter(i.operation.name for i in ckt.data))
    real_gates = collections.Counter({k: v for k, v in real.items()
                                      if k not in NOT_A_GATE})

    W(f"  hand-maintained json_gates : {sum(ours.values())}")
    W(f"  circuit instructions       : {sum(real.values())}")
    W(f"  abstract count returned    : {abstract}   (Vu's paper figure, not measured)")
    W()
    W(f"  {'gate':<10} {'JSON':>6} {'circuit':>8}")
    ok = True
    for k in sorted(set(ours) | set(real_gates)):
        a, b = ours.get(k, 0), real_gates.get(k, 0)
        hit = a == b
        ok &= hit
        W(f"  {k:<10} {a:>6} {b:>8}   {'' if hit else '<-- DIFFER'}")
    W(f"  {'TOTAL':<10} {sum(ours.values()):>6} {sum(real_gates.values()):>8}")
    W()
    W(f"  -> the JSON faithfully describes the circuit make_vu builds: {ok}")
    W("     (this says nothing yet about whether that circuit is VU'S)")
    W()
    return ok, ours


def check2_vu_own_script():
    """Execute Vu's own H functions for one step and count what THEY emit."""
    W("=" * 96)
    W("CHECK 2  --  one Trotter step emitted by VU'S OWN CODE")
    W("=" * 96)
    if not VU_SRC.exists():
        W(f"  VU SOURCE NOT FOUND: {VU_SRC}")
        W("  Cannot verify.  Reporting as NOT RUN rather than as a pass.")
        W()
        return None, None

    src = VU_SRC.read_text(encoding="utf-8").split("\n")
    try:
        cut = next(i for i, l in enumerate(src)
                   if l.startswith("for i in range(sim_steps)"))
    except StopIteration:
        W("  could not locate the driver loop; Vu's script may have changed.")
        W()
        return None, None

    W(f"  source : {VU_SRC.relative_to(ROOT)}")
    W(f"  running lines 1-{cut} only (parameters, registers, and the four H")
    W(f"  function definitions).  The driver loop at line {cut + 1} -- "
      f"401 steps")
    W("  at 10000 shots with damping and dephasing -- is NOT executed.")
    W()

    saved_argv = sys.argv
    sys.argv = ["3_chromophore_vibronic_simulation.py"]
    g = {"__name__": "vu_ref"}
    try:
        exec(compile("\n".join(src[:cut]), str(VU_SRC), "exec"), g)
    finally:
        sys.argv = saved_argv

    ckt, ts = g["circuit"], g["timestep"]
    n0 = len(ckt.data)
    W(f"  timestep = {ts} ps;  {n0} instruction(s) present before the step "
      f"(Vu's initial circuit.x)")
    W(f"  applying Vu's own order: "
      f"{' '.join(n + ('rev' if r else '') for n, r in VU_ORDER)}")
    for name, rev in VU_ORDER:
        g[name](ts / 2, reverse=True) if rev else g[name](ts / 2)

    step = list(ckt.data)[n0:]
    raw = norm(collections.Counter(i.operation.name for i in step))
    bar = sum(v for k, v in raw.items() if k in NOT_A_GATE)
    gates = collections.Counter({k: v for k, v in raw.items()
                                 if k not in NOT_A_GATE})
    W()
    W(f"  raw instructions emitted by Vu's code : {sum(raw.values())}")
    W(f"  of which barriers (not gates)         : {bar}")
    W(f"  GATES                                 : {sum(gates.values())}")
    W()
    return gates, bar


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    W("WHERE 132 COMES FROM")
    W()

    ok1, ours = check1_json_vs_our_circuit()
    vu, bar = check2_vu_own_script()

    W("=" * 96)
    W("SIDE BY SIDE  --  Vu's own script vs our re-implementation")
    W("=" * 96)
    if vu is None:
        W("  Check 2 did not run; the comparison cannot be made.")
        ok2 = None
    else:
        W(f"  {'gate':<10} {'VU':>6} {'ours':>6}")
        ok2 = True
        for k in sorted(set(vu) | set(ours)):
            a, b = vu.get(k, 0), ours.get(k, 0)
            hit = a == b
            ok2 &= hit
            W(f"  {k:<10} {a:>6} {b:>6}   {'' if hit else '<-- DIFFER'}")
        W(f"  {'TOTAL':<10} {sum(vu.values()):>6} {sum(ours.values()):>6}")
        W()
        W(f"  -> our circuit reproduces Vu's gate for gate: {ok2}")
    W()

    W("=" * 96)
    W("ANSWER")
    W("=" * 96)
    if ok2:
        W(f"  132 is not a number we assigned.  Vu's own published script emits")
        W(f"  {sum(vu.values()) + bar} instructions per second-order Trotter step, of which {bar}")
        W(f"  are barriers -- transpiler scheduling directives that compile to")
        W(f"  nothing and occupy no pulse slot.  {sum(vu.values()) + bar} - {bar} = {sum(vu.values())}.")
        W()
        W("  Every primitive type matches our re-implementation exactly, so the")
        W("  JSON, the HybridLane port, and Vu's own code all agree.")
        W()
        W("  What 132 is NOT: it is not Vu's paper's count.  Their stated '96")
        W("  gates per step' is an abstract operator decomposition, hardcoded")
        W("  at simulations/circuit_diagrams.py:545 as a citation.  We have")
        W("  never recounted 96 and this script does not check it.")
    else:
        W("  NOT ESTABLISHED -- see the differing rows above.")
    W()

    out = HERE / "results" / "verify_vu_132.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
