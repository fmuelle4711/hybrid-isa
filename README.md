# hybrid_qudit_oscillator_isa

Code and data accompanying

**"A Hardware-Native Trapped-Ion Hybrid Qudit–Oscillator Instruction Set"**
Masoud Hakimi Heris, Yuan Liu, Frank Mueller

This is a *flattened* snapshot — no submodules, no `.git` history in the
sub-trees. For the full development history of either sub-tree, see the
origin repositories below.

---

## The manuscript is not here yet

`mas.tex` is **not** in this repository. The current draft still carries
review markup, and it is added at acceptance. Until then this repository
holds only the code, data and figures behind the numbers.

---

## What is in here, and why

This repository contains exactly the files the manuscript depends on, and
nothing else. Every file is listed in

    qudit_qumode/results/paper_manifest.txt

with the reason it is there — the figure it draws, the table it backs, or
the script that imports it. The manifest is generated, not hand-written,
and anything absent from it has been removed.

To check a claim in the paper, find its artifact in
[REPRODUCE.md](REPRODUCE.md), run the one command listed, and read the one
file it writes.

```
pip install -r requirements-frozen.txt
cd qudit_qumode
python scripts/paper_repo_build_check.py
```

That runs every command in REPRODUCE.md and checks each rewrote the file it
is supposed to produce.

---

## Contents

### `qudit_qumode/`

The paper's own code: circuit builders, figure generators, the pulse
analyses, and the results files that every quoted number traces to.

- Origin: <https://github.com/mhakimih1988/qudit_qumode>
- Synced from commit `09d4293b6f176b949c3184d9bcbbf509a79cbdae`
- License: MIT (see top-level `LICENSE`)

Note that `writeup/` is no longer carried here. The manuscript is written
in Overleaf and arrives at acceptance; the superseded LaTeX fragments that
used to live in this tree were not what the paper is built from.

### `hybridlane/`

Our fork of the PNNL HybridLane oscillator–qubit programming framework,
vendored at the version used for the paper. It adds the five hybrid
qudit–oscillator primitives the circuit builders call.

The manuscript names those primitives $Z_d$, $X_E$, $C_{\Pi_z}D$,
$C_{\Pi}R$ and $C_{\Pi_x}D$. The class names and drawn labels keep the
shorter `QPS`, `QT`, `QCD`, `QCR`, `QTsb`, and both spellings are
importable — see the mapping table in `hybridlane/README.md`.

- Origin: <https://github.com/mhakimih1988/hybridlane>
- Synced from commit `faf1178917432cfba560146956399cb80dfbf6ee`
- Upstream: <https://github.com/pnnl/hybridlane>
- **Upstream commit we forked from:** `31ddf4c41d5fbe1fde361d12d744fd240cf9395a`
  (upstream tag/context: pre-v0.5.0-alpha, dated 2026-03-20)
- License: **BSD-2-Clause** (© Battelle Memorial Institute 2025).
  The original `LICENSE.txt` and `LICENSES/BSD-2-Clause.txt` are preserved
  verbatim in `hybridlane/`. Our modifications are released under the
  same BSD-2-Clause license, consistent with the upstream project's
  REUSE metadata.

**Our modifications to hybridlane/:**

| Commit  | Summary                                                                        |
|---------|--------------------------------------------------------------------------------|
| c61932b | Add qudit-level-selective hybrid gates: `QCR`, `QCD`, `QPS`, `QT`              |
| 160a48b | Full qudit support: parametric ops, level-selective observables, register map  |
| da43401 | Fix qubit/qumode order in `apply_gate` (qumode LSB, qudit MSB, kron convention)|
| 5d3f494 | Sample-mode `QuditLevelProjector` measurements on `bosonicqiskit.hybrid`        |
| 44213c9 | Flag-ancilla Stinespring channels for 4-level qudits (amp damping + dephasing) |
| 7aa2603 | Document the paper gate names on the qudit primitives and in the README         |
| faf1178 | Add the paper names as aliases: `Zd`, `XE`, `CPizD`, `CPiR`, `CPixD`            |

The rest of the upstream tree is carried over unchanged from `31ddf4c…`.

### `tools/`

`verify_reproduce.sh`, a thin wrapper over the build check.

---

## Two things this repository does not contain

**Vu *et al.*'s published simulation script.** Appendix B's count of 132
primitives per Trotter step was obtained by executing their released code
and counting what it emitted. That code arrives with no license file, so it
is not redistributed here. `circuits/cqed_vu/verify_vu_132.py` is included
so the method is auditable, and its stored output is the result from when
the source was present, but the command cannot be re-run from this
repository alone. REPRODUCE.md says how to restore it.

**A Mølmer–Sørensen waveform.** There is none, anywhere. The MS gate is not
optimised in this work: its 200 µs duration is taken from Meth *et al.*
SI Note II p.3 as a measured, angle-independent value, and that single
quoted number carries the entire MS cost of the runtime model. Every other
timed gate in the paper has a saved waveform; this one does not, and
`qudit_qumode/results/fig_pulses_sources.txt` says so in as many words.

One figure is also absent: `figures/chrome_a.png`, the three-chromophore
schematic, is a hand-drawn figure that exists only in the manuscript
source. The manifest lists it as missing rather than quietly dropping it.
