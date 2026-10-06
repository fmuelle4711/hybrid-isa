"""Build the paper manifest: every file the manuscript depends on.

Seeded from mas.tex and from the results files whose numbers the manuscript
quotes, then closed transitively over what those scripts import and read.
One line per file with the reason it is in.

The manifest is what gets copied into the submission repository.  Anything
not in it is removed there, so a file missing here becomes a build failure
in that repo rather than a silent gap -- which is the point.

Output: results/paper_manifest.txt
"""
from __future__ import annotations

import re
import subprocess
import sys
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- seeds ----
# (path, reason).  Figures come from mas.tex \includegraphics; results files
# are the ones this session traced manuscript numbers to.
FIGURES = {
    "figures/circuit_qudit_generalized_27_one_step.pdf":
        ("Fig. circuit_qudit, 27-control-block qudit step",
         "simulations/draw_qudit_27_hqml.py"),
    "figures/circuit_qubit_32_one_step.pdf":
        ("Fig. circuit_qubit, 32-operation qubit reference",
         "simulations/draw_qubit_32_hqml.py"),
    "figures/circuit_qubit_58_one_step.pdf":
        ("Fig. circuit_qubit_58, compiled to 58 timed pulses",
         "simulations/draw_qubit_58_hqml.py"),
    # SUPPLEMENTARY -- not referenced by mas.tex.  Kept in the repo so the
    # waveform and pulse figures for every gate ship with the paper.
    "figures/pulses/fig_pulses_cd.pdf":
        ("Supplementary, CzD and CxD through HyPulse (not used by mas.tex)",
         "scripts/make_fig_pulses_cd.py"),
    "figures/pulses/fig_pulses_qudit.pdf":
        ("Supplementary, four qudit primitives (not used by mas.tex)",
         "scripts/make_fig_pulses.py"),
    "figures/pulses/fig_pulses_qubit.pdf":
        ("Supplementary, CD and CR (not used by mas.tex)",
         "scripts/make_fig_pulses.py"),
    "figures/fig_convergence_trotter.pdf":
        ("Fig. conv_trotter, Trotter convergence",
         "figures/fig_convergence_trotter.py"),
    "figures/fig6_three_way.pdf":
        ("Fig. closed, closed-system three-way comparison",
         "figures/fig6_three_way.py"),
}

# Figures the manuscript includes that NO script in this repo produces.
UNPRODUCED = {
    "figures/fig1.png": "Fig. landscape, hand-drawn schematic",
    "figures/fig5_amplitude_final.png": "Fig. amp, dissipative amplitude damping",
    "figures/fig6_final.png": "Fig. dep, dissipative dephasing",
    "figures/fig7_final.png": "Fig. combined, dissipative combined",
}

RESULTS = {
    "results/paper_gate_numbers.txt":
        ("every gate count in the manuscript, checked against source",
         "scripts/paper_gate_numbers.py"),
    "results/cpir_one_pulse_regen.txt":
        ("tab:gatecounts, tab:runtime, tab:deltasweep under one-pulse C_Pi R",
         "scripts/cpir_one_pulse_regen.py"),
    "results/closed_qubit_vs_reference_401.txt":
        ("Sec. results_closed, 0.031% and 1.6%",
         "scripts/closed_qubit_vs_reference_401.py"),
    "results/hcqed_four_piece_split.txt":
        ("App. B, H_cqed four-piece split and the 32 count",
         "scripts/hcqed_four_piece_split.py"),
    "results/hypulse_reproduce_three.txt":
        ("App. C, the three GRAPE pulses reproduced through HyPulse",
         "scripts/hypulse_reproduce_three.py"),
    "figures/pulses/fig_pulses_cd_sources.txt":
        ("Supplementary, source/bound/flatness/replay of the CzD and CxD "
         "panels", "scripts/make_fig_pulses_cd.py"),
    "figures/pulses/fig_pulses_cd.txt":
        ("Supplementary, CzD/CxD synthesis, flatness and physical replay",
         "scripts/make_fig_pulses_cd.py"),
    "figures/pulses/fig_pulses_sources.txt":
        ("Supplementary, source and fidelity of the six legacy pulse panels",
         "scripts/make_fig_pulses.py"),
    "results/segment_counts.txt":
        ("tab:gatecounts, timed pulses per step",
         "scripts/segment_counts.py"),
    "results/qcr_delta_ratio_34.txt":
        ("tab:deltasweep, detuning sweep",
         "scripts/qcr_delta_ratio_34.py"),
    "results/table3_cross_recomputed.txt":
        ("tab:rmse, all six dissipative RMSE values",
         "scripts/table3_cross_recompute.py"),
    "results/pulse_fidelity_common.txt":
        ("tab:pulsemethods, one replay protocol for every gate",
         "scripts/pulse_fidelity_common.py"),
    "results/qubit_ladder_58.txt":
        ("App. B, 32 logical to 58 timed pulses",
         "scripts/appendix_b_qubit_ladder.py"),
    "results/entangling_only_counts.txt":
        ("tab:gatecounts, entangling column",
         "scripts/entangling_only_counts.py"),
    "results/multitone_counts_and_centres.txt":
        ("Sec. methodA, 27 control blocks and 6Nc+9",
         "scripts/multitone_counts_and_centres.py"),
    "results/universality_degree_vs_bandwidth.txt":
        ("App. A, rank sequence 12/29/105/375 and bandwidth 3",
         "scripts/universality_degree_vs_bandwidth.py"),
    # The primary run behind Sec. V. Ships because it is the only stored
    # evidence for two claims the derived .txt above does not carry: the
    # iteration-4 saturation on u(4N), and the unconditional-displacement
    # control that holds bandwidth at 1. Not in paper_repo_build_check:
    # the three cutoffs take ~310 s and the file has not changed since July.
    "results/universality_closure_check.log":
        ("Sec. V, Lie closure per iteration, u(4N) saturation, and the "
         "Gaussian control", "verification/universality_closure_check.py"),
    "results/duration_rule_table.txt":
        ("App. D, per-gate duration scaling rules",
         "scripts/duration_rule_table.py"),
    "circuits/cqed_vu/results/verify_vu_132.txt":
        ("App. B, 132 primitives against Vu's own script",
         "circuits/cqed_vu/verify_vu_132.py"),
    "circuits/trapped_ion_qudit/results/multitone_table.txt":
        ("Sec. methodA, 34 tones / 27 blocks / 37 segments",
         "circuits/trapped_ion_qudit/multitone_table.py"),
    # ---- Appendix D, declared explicitly.  These were previously shipped
    # only because the crawler followed split_benchmark.py's imports.
    "results/appendix_d_tables.txt":
        ("App. D, rules R1-R8, the per-instance qudit and qubit tables "
         "(tab:perinstance_qudit / tab:perinstance_qubit) and the per-class "
         "totals 210.5 / 2207.9 us, ratio 10.5",
         "scripts/appendix_d_tables_tex.py"),
    "results/runtime_final_regen.tex":
        ("App. D, tab:runtime and tab:deltasweep", "scripts/runtime_final_regen.py"),
    "results/runtime_final_regen.txt":
        ("App. D, N_c projection and detuning sweep, readable",
         "scripts/runtime_final_regen.py"),
    "results/cpir_amplitude_calibration.txt":
        ("App. D rule R2, C_Pi R amplitudes 3.77 / 4.12 / 4.38 kHz at T = 25 us",
         "scripts/cpir_amplitude_calibration.py"),
    "results/cpir_amplitude_sweep.txt":
        ("App. D tab:deltasweep rows, amplitude solved at each detuning",
         "scripts/cpir_amplitude_sweep.py"),
    "results/qcr_delta_sweep.txt":
        ("App. D, the detuning grid read by runtime_final_regen.py; its 1-F "
         "and leakage columns are superseded", "scripts/qcr_delta_sweep.py"),
    "results/split_benchmark.md":
        ("multi-ion split of the qudit register, crossing X_E at two MS "
         "windows", "scripts/split_benchmark.py"),
    "results/split_benchmark.json":
        ("multi-ion split, machine-readable", "scripts/split_benchmark.py"),
    "results/cluster_bound.md":
        ("cluster speedup S, its bound and the (N_c-1)/(N_c-K) limit, "
         "re-derived from the edge costs with cluster size K free",
         "scripts/split_benchmark.py"),
    "results/cluster_bound.json":
        ("cluster speedup, bound and limit, machine-readable",
         "scripts/split_benchmark.py"),
}

# Modules that are imported rather than run, and data the scripts read.
# Discovered by the crawler below, but these are named explicitly because a
# crawler cannot see them all.
EXTRA = {
    "params.py": "vibronic model parameters, imported by nearly everything",
    "scripts/qcr_mechanism.py":
        "detuned-sideband drive for C_Pi R, imported by the App. D "
        "amplitude calibration and sweep",
    "pulses_meth/anchor_params.json": "Meth hardware anchors",
    "pulses_meth/runtime_table.json": "gate times and counts, machine readable",
    "pulses_meth/appendix_table.md": "provenance for every pulse number",
    "pulses_meth/runtime_table.md": "runtime table, human readable twin",
    "circuits/pulse_lib.py": "shared device and GRAPE helpers",
    "simulations/circuit_fig_common.py": "shared circuit-figure helpers",
    "simulations/vibronic_scaling.py":
        "one-hot qubit Trotter step, the 32-operation builder",
    "simulations/circuit_diagrams.py":
        "Vu cQED circuit, recounted for the 132 primitives",
    "verification/universality_closure_check.py":
        "Lie-closure rank sequence for App. A",
    "scripts/paper_manifest.py": "builds this manifest",
    "scripts/paper_repo_sync.py": "syncs the submission repo to this manifest",
    "scripts/paper_repo_build_check.py":
        "runs every manifest script and checks it reproduces its artifact",
}

# Data the scripts open through f-strings or globs, which the crawler
# cannot see.  Discovered from the build check, not guessed.
DATA_GLOBS = [
    ("data/fig[89]_hqml_*_N[48]_10000shots.npz",
     "dissipative shot trajectories for tab:rmse"),
    ("data/fig10_hqml_*_N[48]_10000shots.npz",
     "dissipative shot trajectories for tab:rmse"),
    ("data/vu_reference/*.out",
     "published reference trajectories of Ref. vu2025"),
    ("data/convergence_trotter_*_N8.npz",
     "Trotter convergence sweep behind the fitted slopes"),
    ("data/trotter_method_*_fock8.npz",
     "closed-system reference trajectories"),
    ("data/circuit_qudit_generalized_2[78]_one_step.json",
     "drawn qudit circuits, counted by paper_gate_numbers"),
    ("data/circuit_vu_one_step.json",
     "Vu circuit, 132 primitives"),
    ("data/qtsb_qtrl_working.npz", "C_Pi_x D waveform"),
]

SKIP_DIRS = ("__pycache__", ".git", "archive", "legacy")

# Figures the manuscript includes from figures/ but which live elsewhere in
# this repo.  The manifest records where they actually are; the sync copies
# them to the path mas.tex expects.
SOURCE_OVERRIDE = {
    # hand-drawn; kept at the repo root, copied into figures/ by the sync
    "figures/chrome_a.png": "chrome_a.png",
    "figures/fig1.png": "figures/paper/fig1.png",
    "figures/fig5_amplitude_final.png":
        "figures/final_paper_figures/fig5_amplitude_final.png",
    "figures/fig6_final.png": "figures/final_paper_figures/fig6_final.png",
    "figures/fig7_final.png": "figures/final_paper_figures/fig7_final.png",
}


# The HybridLane package, which every circuit script imports.  The paper
# repository vendors it rather than pinning a version, so the manifest
# covers its source tree too.  Everything under hybridlane/ in the manifest
# is copied from the working checkout at cvdv_app/hybridlane.
HYBRIDLANE_SRC = ROOT.parent / "hybridlane"
HYBRIDLANE_KEEP = ("src", "test", "pyproject.toml", "README.md",
                   "LICENSE.txt", "LICENSES", "REUSE.toml", "conftest.py",
                   "pytest.toml", "CONTRIBUTING.md", "DISCLAIMER.txt")


def hybridlane_files():
    """Every file of the HybridLane package that the paper repo needs."""
    out = {}
    for keep in HYBRIDLANE_KEEP:
        p = HYBRIDLANE_SRC / keep
        if p.is_file():
            out[f"hybridlane/{keep}"] = "HybridLane package"
        elif p.is_dir():
            for q in sorted(p.rglob("*")):
                if not q.is_file():
                    continue
                if any(d in q.parts for d in SKIP_DIRS):
                    continue
                if q.suffix in (".pyc", ".pyo"):
                    continue
                rel = q.relative_to(HYBRIDLANE_SRC).as_posix()
                out[f"hybridlane/{rel}"] = (
                    "HybridLane qudit primitives and tests"
                    if "qudit" in rel else "HybridLane package")
    return out


def commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


# ------------------------------------------------------------- crawler ----
PATH_RE = re.compile(r'["\']([\w/\\.\-]+\.(?:py|txt|json|npz|md|out))["\']')
IMPORT_RE = re.compile(r'^\s*(?:from|import)\s+([\w.]+)', re.M)


def crawl(seed_scripts):
    """Close over what each script imports and the repo files it names."""
    seen, out, queue = set(), {}, list(seed_scripts)
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        seen.add(rel)
        p = ROOT / rel
        if not p.exists() or p.suffix != ".py":
            continue
        s = p.read_text(encoding="utf-8", errors="replace")

        for mod in IMPORT_RE.findall(s):
            top = mod.split(".")[0]
            dotted = mod.replace(".", "/") + ".py"      # simulations.foo -> simulations/foo.py
            for cand in (dotted, f"{top}.py", f"simulations/{top}.py",
                         f"verification/{top}.py",
                         f"scripts/{top}.py", f"circuits/{top}.py",
                         f"pulses_meth/hypulse_validation/{top}.py",
                         f"circuits/trapped_ion_qubit/{top}.py",
                         f"circuits/trapped_ion_qudit/{top}.py"):
                if (ROOT / cand).exists():
                    out.setdefault(cand, f"imported by {rel}")
                    queue.append(cand)
                    break

        for m in PATH_RE.findall(s):
            m = m.replace("\\", "/")
            if any(d in m for d in SKIP_DIRS):
                continue
            for cand in (m, f"data/{m}", f"results/{m}",
                         f"pulses_meth/hypulse_validation/{m}"):
                q = ROOT / cand
                if q.exists() and q.is_file():
                    if cand.endswith(".py"):
                        out.setdefault(cand, f"imported by {rel}")
                        queue.append(cand)
                    else:
                        out.setdefault(cand, f"read by {rel}")
                    break
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    entries = {}

    def add(rel, reason):
        entries.setdefault(rel, reason)

    for f, (why, prod) in FIGURES.items():
        add(f, why)
        add(prod, f"produces {f}")
    for f, (why, prod) in RESULTS.items():
        add(f, why)
        add(prod, f"produces {f}")
    for f, why in UNPRODUCED.items():
        add(f, why + " (no generator in this repo)")
    for f, why in EXTRA.items():
        add(f, why)

    producers = [v[1] for v in FIGURES.values()] + [v[1] for v in RESULTS.values()]
    for rel, why in crawl(producers).items():
        add(rel, why)
    for rel, why in hybridlane_files().items():
        add(rel, why)
    # The manifest must be closed under what these scripts WRITE, not only
    # what they read: several emit .png and .svg companions beside the .pdf,
    # and the manifest lists itself.  Without this the sync removes files the
    # build check has just produced, and the two oscillate.
    add("results/paper_manifest.txt", "this manifest")
    for fig in list(FIGURES) + ["circuits/trapped_ion_qudit/figures/"
                                "multitone_table.pdf"]:
        base = fig.rsplit(".", 1)[0]
        for ext in ("png", "svg"):
            q = ROOT / f"{base}.{ext}"
            if q.exists():
                add(f"{base}.{ext}",
                    "companion raster/vector written beside "
                    f"{pathlib.Path(fig).name}")
    add("circuits/trapped_ion_qudit/figures/multitone_table.pdf",
        "multitone schedule table, written by multitone_table.py")

    for pattern, why in DATA_GLOBS:
        for q in sorted(ROOT.glob(pattern)):
            add(q.relative_to(ROOT).as_posix(), why)

    def src_path(rel):
        if rel.startswith("hybridlane/"):
            return HYBRIDLANE_SRC / rel[len("hybridlane/"):]
        return ROOT / SOURCE_OVERRIDE.get(rel, rel)

    missing = [r for r in entries if not src_path(r).exists()]

    lines = []
    lines.append("# paper manifest -- every file the manuscript depends on")
    lines.append(f"# built by scripts/paper_manifest.py at commit {commit()}")
    lines.append("# seeded from mas.tex figures and the results files whose")
    lines.append("# numbers the manuscript quotes, then closed over imports")
    lines.append("# and file reads.  One line per file: path, then reason.")
    lines.append("#")
    lines.append(f"# files: {len(entries)}   missing from this repo: "
                 f"{len(missing)}")
    lines.append("")
    width = max(len(r) for r in entries)
    for rel in sorted(entries):
        src = SOURCE_OVERRIDE.get(rel)
        flag = "" if src_path(rel).exists() else "   *** MISSING ***"
        extra = f"  [copied from {src}]" if src else ""
        lines.append(f"{rel:<{width}}  {entries[rel]}{extra}{flag}")

    if missing:
        lines.append("")
        lines.append("# MISSING -- referenced by the manuscript but not in "
                     "this repository:")
        for m in sorted(missing):
            lines.append(f"#   {m}")

    out = ROOT / "results" / "paper_manifest.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nwrote {out}   ({len(entries)} files, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
