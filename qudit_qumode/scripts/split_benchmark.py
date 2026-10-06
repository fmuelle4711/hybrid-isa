"""Multi-ion split of the qudit register: timing when the manifold does
not fit in one ion.

PRICING OF CROSSING GATES.  A gate whose two levels sit on different
ions becomes a two-ion gate, priced in Molmer-Sorensen windows (rule
R7, 200 us each) with the same convention the qubit reference uses:
  crossing X_E        TWO windows, 400 us.  It is an exchange between
                      two ions, and the qubit reference prices every
                      exchange J as two windows because XX and YY do
                      not fuse.
  crossing C_{Pi_x}D  ONE window, 200 us.  It is the link-modulated
                      exchange, priced like the qubit reference's J_l.
Everything else keeps its R1-R6 price.  The one-window-per-X_E variant
is reported only as a side note.

PLATFORM CAPACITY.  An ion with d_eff usable levels spends one on the
reference state G (ion 1 only; see below) and n_aux on auxiliary levels
reserved for the level-selective mode operations, so it holds
    capacity = d_eff - 1 - n_aux
excited states.  The generic case is n_aux = 1, which gives the
d_eff - 2 of the manuscript's condition N_c + 2 <= d_eff.  The 40Ca+
case is d_eff = 8 with n_aux = 2, capacity 5.

Part A.  Same molecule, smaller ion.  N_c = 3 at d_eff = 4, n_aux = 1:
ion 1 holds G, A, B and ion 2 holds C.  At N_c = 3 the crossing gates are
the two X_E on A<->C and the C_{Pi_x}D on A<->C in the central block.
Rows are read from the same compiled tape and classifier as Table D1
(scripts/appendix_d_tables_tex.py), so the unsplit column reproduces
tab:perinstance_qudit exactly.

Part B.  Bigger molecule.  Ion 1 holds G plus `capacity` excited states;
every further ion holds `capacity` excited states.
N_ion = ceil(N_c / capacity).  Coupling is the model's star: all N_c - 1
edges leave A, which sits on ion 1, so an edge crosses ions exactly when
its other end is not on ion 1, giving
    crossing edges e = max(0, N_c - capacity).
Each edge carries two X_E (one per half step) and one C_{Pi_x}D (central
block), the per-edge counts of the closed forms 2(N_c-1) and N_c-1, so a
crossing edge costs 2 x 2 + 1 = 5 windows.  Everything else is charged
at its N_c = 3 mean instance time, the same rule as tab:runtime
(scripts/runtime_final_regen.py).  The qubit reference is already
multi-ion and is unchanged.

CLUSTER SPEEDUP CHECK.  The Part B ratio above is computed from a
capacity that follows from d_eff.  The cluster check re-derives the same
quantity from the edge costs alone, with the cluster size K as a free
parameter, so the bound and the large-N_c limit can be evaluated without
reference to any particular ion.  It splits each step total into an
edge-free remainder T0 and the per-edge terms, then reassembles

    S = [T0_qubit + E c_out] / [T0_qudit + E_in c_in + E_out c_out]

with E = N_c - 1 and E_out set by the placement.  T0 is derived on each
side independently and the two are reported side by side; they are NOT
assumed equal.

TWO PLACEMENTS are priced from the same T0 and the same edge costs:
  star   all N_c - 1 edges leave the hub A, which shares its ion with
         K - 1 neighbours, so E_in = min(K-1, N_c-1).  This is the
         model the manuscript's timing section uses (mas.tex:2010).
  chain  nearest-neighbour links only, the 1D array of N chromophores
         of Ref. vu2025 Sec. 2.4 and Eq. 52 (that equation maps an
         N-chromophore line onto 2N-2 transmons and 2N-2 cavities).
         K consecutive chromophores per ion cut the line into
         ceil(N_c/K) blocks, so E_out = ceil(N_c/K) - 1 and
         E_in = E - E_out = N_c - ceil(N_c/K).
The chain leaves strictly fewer crossing edges once N_c > K, and its
window-dominated ratio tends to K rather than to 1.

Output: results/split_benchmark.md, results/split_benchmark.json,
        results/cluster_bound.md, results/cluster_bound.json
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "scripts", "simulations", "circuits/trapped_ion_qubit"):
    sys.path.insert(0, str(ROOT / sub))

import appendix_d_tables_tex as A                              # noqa: E402
import runtime_final_regen as RF                               # noqa: E402

LEVEL = {0: "G", 1: "A", 2: "B", 3: "C"}
SPLIT_A = {0: 1, 1: 1, 2: 1, 3: 2}          # G, A, B on ion 1; C on ion 2

# generic platform: n_aux = 1, the manuscript's N_c + 2 <= d_eff
GENERIC = dict(n_aux=1, d_list=(4, 6, 8), nc_list=(3, 6, 8, 12))
# 40Ca+: eight usable levels, two reserved auxiliaries
CA40 = dict(label="40Ca+", d_eff=8, n_aux=2, nc_list=(3, 5, 6, 8, 12))

# cluster speedup check: cluster sizes K and the N_c grid it sweeps
CLUSTER_K = (2, 4, 6)
CLUSTER_K_CA = 5
CLUSTER_NC = (3, 5, 6, 8, 12)

# MS windows per crossing instance.  X_E follows the qubit reference's
# exchange J (two windows); C_{Pi_x}D follows its J_l (one window).
WIN = {"X_E": 2, "C_Pi_x D": 1}
WIN_SIDE = {"X_E": 1, "C_Pi_x D": 1}        # side note only


def by_rule(rows):
    """(class, rule) -> (n, sum T, mean T) for a compiled tape."""
    out = {}
    for cls, _, rule, T in rows:
        n, t = out.get((cls, rule), (0, 0.0))
        out[(cls, rule)] = (n + 1, t + T)
    return {k: (n, t, t / n) for k, (n, t) in out.items()}


# Non-edge class counts per Trotter step, derived from Vu Eq. 18-21 for the
# chain with 2N_c-2 modes, second-order Trotter, two half steps.
#   R1  one mode free evolution per mode per half step
#       modes = N_c high + (N_c-2) low = 2N_c-2      -> 2(2N_c-2) = 4N_c-4
#   R2  one dispersive sigma_z (x) n per chromophore (Eq. 20, first term)
#                                                    -> 2N_c
#   R3  one conditional displacement per site on its HIGH mode (Eq. 20,
#       second term) plus one on its LOW mode (Eq. 21, term 1); the two
#       boundary low modes are dropped with Eq. 52, so per half step
#       N_c + (N_c-2) = 2N_c-2                       -> 4N_c-4
#   virtual classes carry no time; their counts are decomposition
#   bookkeeping, not fixed by the Hamiltonian, and are left as they are.
NONEDGE_NEW = {
    "R1": (lambda n: 4 * n - 4, "4N_c-4", "one per mode per half step"),
    "R2": (lambda n: 2 * n, "2N_c", "one per chromophore, Eq. 20 term 1"),
    "R3": (lambda n: 4 * n - 4, "4N_c-4",
           "high mode per site + low mode per site, boundaries dropped"),
}
NONEDGE_OLD = {
    "R1": (lambda n: 2 * (n + 1), "2(N_c+1)"),
    "R2": (lambda n: 2 * n, "2N_c"),
    "R3_qudit": (lambda n: 2 * (n + 1), "2(N_c+1)"),
    "R3_qubit": (lambda n: 4 * n, "4N_c"),
}


def chain_bonds(nc, K=None):
    """Bonds of the N_c chain, termwise from Vu Eq. 18 -> Eq. 21.

    Each chromophore xi carries a high-frequency mode xi_0 and a
    low-frequency mode xi_1 (Vu Sec. 2.1.4).  Bond i joins sites i and
    i+1 and collects, from H_2^(xi=i) of Eq. 21:
      terms 2 and 3  two MODE-INDEPENDENT nearest-neighbour electronic
                     exchanges                             -> n_exchange = 2
      terms 4 and 5  two MODE-DEPENDENT vibronic link terms, the first
                     modulated by i_1, the second by (i+1)_1
                                                           -> n_link = 2
    Eq. 52 drops the boundary low-frequency modes, "assuming negligible
    low-frequency modes at the boundaries", which here are 1_1 and
    N_c_1.  A link term whose mode is dropped goes with it, so the two
    END bonds keep one link term each and every INTERIOR bond keeps
    two.  Surviving modes: N_c high-frequency plus N_c - 2 low-frequency
    = 2N_c - 2, the count of Eq. 52 and of mas.tex:1039.

    Totals: exchange 2(N_c - 1), link 2(N_c - 2).  The link total equals
    the earlier one-per-bond count N_c - 1 only at N_c = 3, where both
    give 2 and both end bonds share the single surviving mode 2_1 --
    the shared link mode of the trimer.

    With K given, bond i also carries `crossing`: blocks are
    [1..K], [K+1..2K], ..., so bond i is severed iff i % K == 0.
    """
    out = []
    for i in range(1, nc):
        modes = [m for m in (i, i + 1) if m not in (1, nc)]
        kind = "end" if i in (1, nc - 1) else "interior"
        assert (kind == "end") == (len(modes) == 1), \
            f"N_c={nc} bond {i}: kind {kind} against {len(modes)} link(s)"
        out.append(dict(bond=i, sites=(i, i + 1), kind=kind,
                        n_exchange=2, n_link=len(modes),
                        link_modes=[f"{m}_1" for m in modes],
                        crossing=(None if K is None else i % K == 0)))
    return out


def capacity(d_eff, n_aux):
    """Excited states one ion can hold: d_eff minus G minus the auxiliaries."""
    return d_eff - 1 - n_aux


def commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                               "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def levels_of(op):
    h = getattr(op, "hyperparameters", {}) or {}
    if "level_i" in h:
        return {int(h["level_i"]), int(h["level_j"])}
    if "level" in h:
        return {int(h["level"])}
    return set()


def pair_label(op):
    h = getattr(op, "hyperparameters", {}) or {}
    if "level_i" in h:
        return f"{LEVEL[int(h['level_i'])]}-{LEVEL[int(h['level_j'])]}"
    if "level" in h:
        return LEVEL[int(h["level"])]
    return ""


def cluster_block(cq, cb, t_single, t_qubit, t_single_star, t_qubit_star,
                  grid, ca, ca_capacity, qd_rows, qb_rows):
    """Speedup, bound and limit from the edge costs alone, cluster size K.

    Returns (markdown lines, json dict, ok).  Every number is derived
    here from the same class means and closed forms Part B uses; the
    existing Part B ratio is looked up, not recomputed, so a drift
    between the two routes shows up as a nonzero difference.
    """
    M, ok = [], True

    # ---- 1. per-step costs ------------------------------------------------
    t_MS = A.T_MS
    t_X_loc = cq["X_E"][2]
    t_L_loc = cq["C_Pi_x D"][2]
    c_out = 5 * t_MS
    c_in = 2 * t_X_loc + t_L_loc

    M.append("# Cluster speedup check")
    M.append("")
    M.append(f"Generated by `scripts/split_benchmark.py` at commit "
             f"{commit()}.")
    M.append("")
    M.append("## Assumptions")
    M.append("")
    M.append("a. Gate times are class means at N_c = 3, a model "
             "approximation.")
    M.append("b. The link-modulated term costs one MS window on the qubit "
             "side, an assumption of the timing model.")
    M.append("c. Hub placement is optimal, A shares its ion with K-1 "
             "neighbours. Star placement only.")
    M.append("d. Chain placement takes nearest-neighbour links only, the "
             "1D array of N chromophores of Ref. vu2025 Sec. 2.4 and "
             "Eq. 52, with K consecutive chromophores per ion.")
    M.append("")
    M.append("## 1. Per-step costs from the existing rules")
    M.append("")
    M.append("| symbol | value (us) | rule | source |")
    M.append("|---|---|---|---|")
    M.append(f"| t_MS | {t_MS:.4f} | R7 | "
             "`scripts/appendix_d_tables_tex.py:29` (value), `:102` (rule) |")
    M.append(f"| t_X,loc | {t_X_loc:.4f} | R4 | class mean over "
             f"{cq['X_E'][0]} instances, `scripts/runtime_final_regen.py:86` "
             "(count 2(Nc-1), rule), `:115` (mean) |")
    M.append(f"| t_L,loc | {t_L_loc:.4f} | R5 | class mean over "
             f"{cq['C_Pi_x D'][0]} instance(s), "
             "`scripts/runtime_final_regen.py:87` (count (Nc-1), rule), "
             "`:115` (mean) |")
    M.append("")
    M.append("One edge per step is two exchanges plus one link term.")
    M.append("")
    M.append(f"- qubit edge cost  c_out = 5 t_MS = {c_out:.4f} us")
    M.append(f"- qudit local edge c_in  = 2 t_X,loc + t_L,loc = "
             f"2 x {t_X_loc:.4f} + {t_L_loc:.4f} = {c_in:.4f} us")
    M.append(f"- c_in <= c_out: {'yes' if c_in <= c_out else 'NO'} "
             f"(ratio {c_out / c_in:.2f})")
    M.append("")

    # the qubit's 5 windows per edge are 2 MS XX + 2 MS YY + 1 J_l window
    w_xx, w_yy = cb["MS XX"][2], cb["MS YY"][2]
    flat = abs(w_xx - t_MS) < 1e-9 and abs(w_yy - t_MS) < 1e-9
    ok = ok and flat
    M.append(f"The qubit's five windows per edge are 2 MS XX + 2 MS YY "
             f"(counts 2(Nc-1) each, `scripts/runtime_final_regen.py:95-96`) "
             f"plus one J_l window (assumption b). MS XX and MS YY price at "
             f"{w_xx:.4f} and {w_yy:.4f} us; flat at t_MS: "
             f"{'PASS' if flat else 'FAIL'}.")
    M.append("")

    # ---- 2. T0 on each side, independently --------------------------------
    M.append("## 2. T0 derived on each side independently")
    M.append("")
    M.append("T0_qudit = qudit step total minus all edge terms, "
             "2(N_c-1) exchanges at t_X,loc and 2(N_c-2) link terms at "
             "t_L,loc. T0_qubit = qubit step total minus its "
             "4(N_c-1) + 2(N_c-2) = 6N_c-8 MS windows. Equality is not "
             "imposed. T0 is the edge-free remainder and so does not depend "
             "on the bond model; the star and chain sections below share it.")
    M.append("")
    M.append("| N_c | E | qudit step (us) | qudit edges (us) | "
             "T0_qudit (us) | qubit step (us) | qubit edges (us) | "
             "T0_qubit (us) | T0_qubit - T0_qudit (us) |")
    M.append("|---|---|---|---|---|---|---|---|---|")
    T0 = {}
    for nc in CLUSTER_NC:
        E = nc - 1
        sq, sb = t_single(nc), t_qubit(nc)
        # chain edge terms: 2(Nc-1) exchanges and 2(Nc-2) link terms
        eq = 2 * E * t_X_loc + 2 * (nc - 2) * t_L_loc
        eb = (4 * E + 2 * (nc - 2)) * t_MS
        t0q, t0b = sq - eq, sb - eb
        T0[nc] = (t0q, t0b)
        M.append(f"| {nc} | {E} | {sq:.4f} | {eq:.4f} | {t0q:.4f} | "
                 f"{sb:.4f} | {eb:.4f} | {t0b:.4f} | "
                 f"{t0b - t0q:+.4f} |")
    M.append("")
    M.append("T0 is edge-free, not N_c-independent: the qudit keeps its "
             "2(Nc+1) Rotation, 2Nc C_Pi R, 2(Nc+1) C_Pi_z D and 2(Nc+1) "
             "Z_d, the qubit its 2(Nc+1) Rotation, 2Nc CR, 4Nc CD, 6(Nc-1) "
             "r_z, 8(Nc-1) H and 4(Nc-1) CNOT.")
    M.append("")
    M.append("The two are now equal at EVERY N_c, not only at N_c = 3. "
             "Under the audited counts (section 5d) both sides carry "
             "4N_c-4 Rotation, 2N_c dispersive and 4N_c-4 R3 conditional "
             "displacements at identical means, so T0 is the same "
             "expression on both sides:")
    M.append("")
    M.append("| term | qudit | qubit |")
    M.append("|---|---|---|")
    M.append(f"| Rotation | 4Nc-4 x {cq['Rotation'][2]:.4f} | "
             f"4Nc-4 x {cb['Rotation'][2]:.4f} |")
    M.append(f"| C_Pi R / CR | 2Nc x {cq['C_Pi R'][2]:.4f} | "
             f"2Nc x {cb['CR'][2]:.4f} |")
    M.append(f"| Z_d / r_z, H, CNOT, CD_Jl | {cq['Z_d'][2]:.4f} each | "
             f"{cb['r_z'][2]:.4f}, {cb['H'][2]:.4f}, {cb['CNOT'][2]:.4f}, "
             f"{cb['CD_Jl'][2]:.4f} each |")
    M.append(f"| C_Pi_z D / CD | 4Nc-4 x {cq['C_Pi_z D'][2]:.8f} | "
             f"4Nc-4 x {cb['CD'][2]:.8f} |")
    M.append("")
    M.append("Every row cancels term by term at every N_c. The earlier "
             "model differed only because it counted the qubit CD as 4Nc "
             "at a mean blended over the R3 and R8 instances; splitting "
             "them by rule removes the difference entirely. The residual "
             "below is now identically zero rather than the dispersive "
             "term it used to be.")
    M.append("")
    dis_ok = True
    M.append("| N_c | T0_qubit - T0_qudit (us) | predicted (us) | "
             "residual (us) |")
    M.append("|---|---|---|---|")
    for nc in CLUSTER_NC:
        got = T0[nc][1] - T0[nc][0]
        pred = 0.0
        dis_ok = dis_ok and abs(got - pred) < 1e-9
        M.append(f"| {nc} | {got:+.4e} | {pred:+.4f} | "
                 f"{got - pred:+.3e} |")
    M.append("")
    M.append(f"Difference zero to 1e-9 on every row: "
             f"{'PASS' if dis_ok else 'FAIL'}. T0 is the same closed form "
             "on both sides.")
    ok = ok and dis_ok
    M.append("")

    # ---- 3 to 5. speedup against the existing table ------------------------
    have = {(r["N_c"], r["capacity"]): r["ratio"] for r in grid}
    have.update({(r["N_c"], ca_capacity): r["ratio"] for r in ca})

    M.append("## 3 to 5. Speedup, star placement (SUPERSEDED)")
    M.append("")
    M.append("The star is the trimer-generalized reading, N_c-1 link terms, "
             "one per bond. The chain of section 5b replaces it as the "
             "adopted extension; this section and sections 6 and 7 are kept "
             "so the Part B table of `results/split_benchmark.md`, which is "
             "still priced on the star, has something to check against. "
             "Star totals are rebuilt from the same T0 with star edge "
             "costs, so they stay self-consistent.")
    M.append("")
    M.append("S = [T0_qubit + E c_out] / [T0_qudit + E_in c_in + E_out "
             "c_out], E = N_c - 1, E_in = min(K-1, N_c-1), "
             "E_out = E - E_in.")
    M.append("")
    M.append("| N_c | K | E | E_in | E_out | S | Part B ratio | difference |")
    M.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for nc in CLUSTER_NC:
        for K, tag in [(k, "generic") for k in CLUSTER_K] + \
                      [(CLUSTER_K_CA, "40Ca+")]:
            E = nc - 1
            e_in = min(K - 1, nc - 1)
            e_out = E - e_in
            t0q, t0b = T0[nc]
            num = t0b + E * c_out
            den = t0q + e_in * c_in + e_out * c_out
            S = num / den
            ref = have.get((nc, K))
            diff = None if ref is None else S - ref
            rows.append(dict(N_c=nc, K=K, platform=tag, E=E, E_in=e_in,
                             E_out=e_out, S=S, part_B_ratio=ref, diff=diff))
            rtxt = "no row" if ref is None else f"{ref:.4f}"
            dtxt = "--" if diff is None else f"{diff:+.2e}"
            M.append(f"| {nc} | {K} ({tag}) | {E} | {e_in} | {e_out} | "
                     f"{S:.4f} | {rtxt} | {dtxt} |")
    M.append("")
    seen = [r for r in rows if r["diff"] is not None]
    worst = max((abs(r["diff"]) for r in seen), default=0.0)
    agree = worst < 1e-9
    ok = ok and agree
    M.append(f"{len(seen)} of {len(rows)} cluster rows have a Part B row at "
             f"the same capacity; largest absolute difference "
             f"{worst:.3e}: {'PASS' if agree else 'MISMATCH'} at 1e-9.")
    M.append("")
    missing = sorted({(r['N_c'], r['K']) for r in rows if r["diff"] is None})
    if missing:
        M.append("No Part B row exists for: "
                 + ", ".join(f"N_c={a} K={b}" for a, b in missing)
                 + ". The generic grid is N_c in "
                 + str(list(GENERIC['nc_list']))
                 + ", so N_c = 5 is priced only on 40Ca+. Reported as is, "
                   "not filled in.")
        M.append("")

    # ---- 5b. chain placement, termwise from Eq. 21 -------------------------
    def c_in_bond(bd):
        return bd["n_exchange"] * t_X_loc + bd["n_link"] * t_L_loc

    def c_out_bond(bd):
        return (bd["n_exchange"] * WIN["X_E"]
                + bd["n_link"] * WIN["C_Pi_x D"]) * t_MS

    M.append("## 5b. Chain placement, termwise from Vu Eq. 18 to 21")
    M.append("")
    M.append("Each chromophore xi carries a high-frequency mode xi_0 and a "
             "low-frequency mode xi_1 (Vu Sec. 2.1.4). Bond i joins sites i "
             "and i+1 and collects, from H_2^(xi=i) of Eq. 21: terms 2 and 3, "
             "two mode-independent nearest-neighbour electronic exchanges; "
             "and terms 4 and 5, two mode-dependent vibronic link terms, the "
             "first modulated by i_1 and the second by (i+1)_1.")
    M.append("")
    M.append("Eq. 52 drops the boundary low-frequency modes, assuming "
             "negligible low-frequency modes at the boundaries. Here those "
             "are 1_1 and N_c_1. A link term whose mode is dropped goes with "
             "it, so each END bond keeps one link term and each INTERIOR "
             "bond keeps two. Surviving modes: N_c high-frequency plus "
             "N_c - 2 low-frequency = 2N_c - 2, the count of Eq. 52 and of "
             "mas.tex:1039.")
    M.append("")
    M.append("### 1. Every term of H_2 for N_c = 4, bond by bond")
    M.append("")
    B4 = chain_bonds(4)
    M.append("Chromophore low-frequency modes 1_1, 2_1, 3_1, 4_1. Eq. 52 "
             "drops **1_1** and **4_1** (the boundary chromophores 1 and 4), "
             "leaving 2_1 and 3_1. Surviving modes: 4 high-frequency "
             "(1_0..4_0) + 2 low-frequency (2_1, 3_1) = 6 = 2N_c - 2.")
    M.append("")
    M.append("| bond | sites | kind | exchange terms (Eq. 21 terms 2,3) | "
             "link terms as written (terms 4,5) | modes those would use | "
             "dropped | link terms kept | mode(s) kept |")
    M.append("|---|---|---|---|---|---|---|---|---|")
    for bd in B4:
        i = bd["bond"]
        would = [f"{i}_1", f"{i + 1}_1"]
        gone = [m for m in would if m not in bd["link_modes"]]
        M.append(f"| {i} | ({bd['sites'][0]},{bd['sites'][1]}) | "
                 f"{bd['kind']} | {bd['n_exchange']} | 2 | "
                 f"{', '.join(would)} | {', '.join(gone) or '--'} | "
                 f"{bd['n_link']} | {', '.join(bd['link_modes'])} |")
    M.append("")
    M.append("The site term of Eq. 21, term 1, couples xi_1 to its own "
             "electronic state and is not a bond term. It too disappears for "
             "the boundary chromophores, leaving 2 of 4 site terms at "
             "N_c = 4 (modes 2_1 and 3_1). It does not enter c_in or c_out.")
    M.append("")
    x4 = sum(bd["n_exchange"] for bd in B4)
    l4 = sum(bd["n_link"] for bd in B4)
    M.append(f"Totals at N_c = 4: {x4} exchange terms = 2(N_c-1) = "
             f"{2 * 3}, and {l4} link terms = 2(N_c-2) = {2 * 2}. Without "
             f"the boundary rule there would be 2(N_c-1) = {2 * 3} link "
             f"terms; the end bonds lose one each.")
    M.append("")
    M.append("| N_c | bonds | end | interior | exchange 2(N_c-1) | "
             "link 2(N_c-2) | earlier one-per-bond count N_c-1 |")
    M.append("|---|---|---|---|---|---|---|")
    same3 = True
    for nc in CLUSTER_NC:
        Bn = chain_bonds(nc)
        ne = sum(1 for x in Bn if x["kind"] == "end")
        ni = len(Bn) - ne
        lk = sum(x["n_link"] for x in Bn)
        M.append(f"| {nc} | {len(Bn)} | {ne} | {ni} | "
                 f"{sum(x['n_exchange'] for x in Bn)} | {lk} | {nc - 1} |")
        if nc == 3:
            same3 = (lk == nc - 1)
    M.append("")
    ok = ok and same3
    M.append(f"The two link counts agree only at N_c = 3, where both give 2: "
             f"{'PASS' if same3 else 'FAIL'}. There both bonds are end bonds "
             "and the single surviving low-frequency mode is 2_1, which both "
             "link terms use -- the shared link mode of the trimer, and the "
             "'1 link' of mas.tex:1039's 3 local + 1 link at N_c = 3. For "
             "N_c >= 4 the earlier count is low: 2(N_c-2) against N_c-1.")
    M.append("")

    # ---- 2. per-bond cost --------------------------------------------------
    M.append("### 2. Per-bond cost")
    M.append("")
    M.append("c_in charges each term its class mean; c_out charges the MS "
             "windows of a two-ion gate, 2 per exchange and 1 per link term.")
    M.append("")
    M.append("| bond kind | exchange | link | c_in (us) | c_out (windows) | "
             "c_out (us) |")
    M.append("|---|---|---|---|---|---|")
    cost = {}
    for kind, nl in (("end", 1), ("interior", 2)):
        bd = dict(n_exchange=2, n_link=nl)
        ci, co = c_in_bond(bd), c_out_bond(bd)
        cost[kind] = (ci, co)
        M.append(f"| {kind} | 2 | {nl} | 2 x {t_X_loc:.4f} + {nl} x "
                 f"{t_L_loc:.4f} = {ci:.4f} | 2x2 + {nl}x1 = "
                 f"{int(co / t_MS)} | {co:.4f} |")
    M.append("")
    M.append(f"An interior bond costs {cost['interior'][1] - cost['end'][1]:.0f}"
             f" us more than an end bond when it crosses, one extra MS "
             f"window, and {cost['interior'][0] - cost['end'][0]:.4f} us more "
             f"when it does not. The earlier uniform figures, c_in = "
             f"{c_in:.4f} us and c_out = {c_out:.4f} us, are the END-bond "
             "values; they were applied to every bond.")
    M.append("")

    # ---- 3. S_chain --------------------------------------------------------
    M.append("### 3. S_chain with these counts")
    M.append("")
    M.append("Blocks are [1..K], [K+1..2K], ..., so bond i is severed iff "
             "i mod K = 0. T0 is unchanged: it is the edge-free remainder "
             "(Rotation, C_Pi R, C_Pi_z D, Z_d on the qudit; Rotation, CR, "
             "CD on the qubit), so it does not depend on the bond model.")
    M.append("")
    M.append("    S_chain = [T0_qubit + sum_all c_out(bond)]")
    M.append("            / [T0_qudit + sum_local c_in(bond) "
             "+ sum_cross c_out(bond)]")
    M.append("")
    M.append("| N_c | K | bonds | crossing | crossing kinds | links crossed | "
             "S chain (new) | S chain (old) | S star |")
    M.append("|---|---|---|---|---|---|---|---|---|")
    cnt_ok = True
    for r in rows:
        nc, K = r["N_c"], r["K"]
        Bn = chain_bonds(nc, K)
        cr_b = [x for x in Bn if x["crossing"]]
        lo_b = [x for x in Bn if not x["crossing"]]
        cnt_ok = cnt_ok and len(cr_b) == math.ceil(nc / K) - 1
        t0q, t0b = T0[nc]
        num = t0b + sum(c_out_bond(x) for x in Bn)
        den = (t0q + sum(c_in_bond(x) for x in lo_b)
               + sum(c_out_bond(x) for x in cr_b))
        Sc = num / den
        kinds = "/".join(x["kind"][0] for x in cr_b) or "--"
        # the superseded one-link-per-bond model, for the comparison column
        E_o = math.ceil(nc / K) - 1
        r["S_chain_old"] = ((t0b + (nc - 1) * c_out)
                            / (t0q + (nc - 1 - E_o) * c_in + E_o * c_out))
        r["S_chain"] = Sc
        r["chain_E_out"] = len(cr_b)
        r["chain_E_in"] = len(lo_b)
        r["chain_links_crossed"] = sum(x["n_link"] for x in cr_b)
        r["chain_qudit_us"], r["chain_qubit_us"] = den, num
        M.append(f"| {nc} | {K} ({r['platform']}) | {len(Bn)} | {len(cr_b)} | "
                 f"{kinds} | {r['chain_links_crossed']} | {Sc:.4f} | "
                 f"{r['S_chain_old']:.4f} | {r['S']:.4f} |")
    M.append("")
    ok = ok and cnt_ok
    up = [r for r in rows if r["S_chain"] > r["S_chain_old"] + 1e-12]
    dn = [r for r in rows if r["S_chain"] < r["S_chain_old"] - 1e-12]
    M.append(f"Crossing count equals ceil(N_c/K) - 1 on every row: "
             f"{'PASS' if cnt_ok else 'FAIL'}. Kinds: e = end bond, "
             "i = interior bond.")
    M.append("")
    M.append(f"The correction does not move S in one direction: "
             f"{len(up)} rows rise, {len(dn)} fall, "
             f"{len(rows) - len(up) - len(dn)} are unchanged. Which way a "
             "row goes depends on whether the severed bonds are end or "
             "interior. Every bond gains link terms, which raises the "
             "unsplit total on both sides and so raises the numerator; but "
             "a severed INTERIOR bond now costs 6 windows instead of 5, "
             "which raises the denominator harder. Rows whose cuts are all "
             "interior and numerous fall -- N_c = 6, 8, 12 at K = 2 and "
             "N_c = 6 at K = 4. Rows with no cut, or an end-bond cut, rise: "
             "the numerator grows and the denominator barely moves.")
    M.append("")

    # ---- the limit ---------------------------------------------------------
    M.append("### 3b. Large-N_c limit, recomputed")
    M.append("")
    ci_int, co_int = cost["interior"]
    M.append("Almost every bond is interior as N_c grows, so the per-bond "
             f"figures that survive are c_in = {ci_int:.4f} us and "
             f"c_out = {co_int:.0f} us = {int(co_int / t_MS)} t_MS.")
    M.append("")
    M.append("**Window-dominated.** sum_all c_out = 2 end at 5 t_MS + "
             "(N_c-3) interior at 6 t_MS = (6 N_c - 8) t_MS, and the "
             "crossing bonds are interior for large N_c, so")
    M.append("")
    M.append("    L_chain = (6 N_c - 8) / (6 (ceil(N_c/K) - 1))  ->  K")
    M.append("")
    M.append("still K, as in the one-link model. The per-bond window cost "
             "cancels whatever its value: the numerator grows as (window "
             "cost) x N_c and the denominator as (window cost) x N_c/K, so "
             "only the block size survives. Raising the interior bond from "
             "5 to 6 windows changes the finite-N_c values but not the "
             "limit. The star's counterpart (N_c-1)/(N_c-K) still tends "
             "to 1.")
    M.append("")
    M.append("**Full limit.** Taking slopes in N_c, with r, cr, cd, czd the "
             "class means:")
    M.append("")
    M.append("    lim S_chain = [2r + 2cr + 4cd + 6 t_MS]")
    M.append("                / [2r + 2cr + 2czd + (1 - 1/K)(2 t_X,loc + "
             "2 t_L,loc) + 6 t_MS / K]")
    M.append("")
    r_, cr_ = cq["Rotation"][2], cq["C_Pi R"][2]
    cd_, czd_ = cb["CD"][2], cq["C_Pi_z D"][2]
    nsl = 2 * r_ + 2 * cr_ + 4 * cd_ + co_int
    M.append("| K | window limit | full limit (new) | full limit (old) | "
             "S chain at N_c = 12 |")
    M.append("|---|---|---|---|---|")
    lim_rows = []
    for K in list(CLUSTER_K) + [CLUSTER_K_CA]:
        dsl = (2 * r_ + 2 * cr_ + 2 * czd_
               + (1 - 1 / K) * ci_int + co_int / K)
        full = nsl / dsl
        old = ((2 * r_ + 2 * cr_ + 4 * cd_ + c_out)
               / (2 * r_ + 2 * cr_ + 2 * czd_
                  + (1 - 1 / K) * c_in + c_out / K))
        at12 = next(x["S_chain"] for x in rows
                    if x["N_c"] == 12 and x["K"] == K)
        lim_rows.append(dict(K=K, window_limit=K, full_limit=full,
                             full_limit_one_link=old, S_chain_Nc12=at12))
        M.append(f"| {K} | {K} | {full:.4f} | {old:.4f} | {at12:.4f} |")
    M.append("")
    M.append("The full limit rises slightly against the one-link model: the "
             "window term per bond grew from 5 t_MS to 6 t_MS while the "
             "edge-free slope did not, so windows carry a larger share. It "
             "still sits below K for every K, and still above 1, so the "
             "conclusion is unchanged -- a chain keeps a constant-factor "
             "speedup as the molecule grows and a star keeps none.")
    M.append("")

    # ---- 5d. non-edge class audit ------------------------------------------
    br_qd, br_qb = by_rule(qd_rows), by_rule(qb_rows)
    m_r1 = br_qd[("Rotation", "R1")][2]
    m_r2 = br_qd[("C_Pi R", "R2")][2]
    m_r3_qd = br_qd[("C_Pi_z D", "R3")][2]
    m_r3_qb = br_qb[("CD", "R3")][2]
    n_r3_qb_r8 = br_qb[("CD", "R8")][0]

    M.append("## 5d. Non-edge class audit, chain Hamiltonian")
    M.append("")
    M.append("Counts derived from the Hamiltonian, Vu Eq. 18 to 21, for "
             "the chain with 2N_c-2 modes under a second-order Trotter "
             "step of two half steps. They are the adopted closed forms; "
             "this section is the derivation behind them.")
    M.append("")
    M.append("| class | rule | derivation | count |")
    M.append("|---|---|---|---|")
    M.append("| Rotation | R1 | one mode free evolution per mode per half "
             "step; modes = N_c high + (N_c-2) low = 2N_c-2 | **4N_c-4** |")
    M.append("| C_Pi R / CR | R2 | one dispersive sigma_z (x) n per "
             "chromophore, Eq. 20 term 1, per half step | **2N_c** |")
    M.append("| C_Pi_z D / CD | R3 | one conditional displacement per site "
             "on its high mode (Eq. 20 term 2) plus one on its low mode "
             "(Eq. 21 term 1), boundary low modes dropped: "
             "N_c + (N_c-2) per half step | **4N_c-4** |")
    M.append("| Z_d, r_z, H, CNOT | R6 / R8 | virtual, T = 0 | unchanged |")
    M.append("")
    M.append("**The qubit CD class splits by rule.** The compiled tape "
             f"carries {br_qb[('CD', 'R3')][0]} CD at R3, mean "
             f"{m_r3_qb:.8f} us, and {n_r3_qb_r8} CD at R8, mean 0, the "
             "ones inside the J_l sandwiches. The qudit's C_Pi_z D is "
             f"{br_qd[('C_Pi_z D', 'R3')][0]} at R3, mean {m_r3_qd:.8f} us. "
             "The two R3 means are identical, and the two R3 sums are "
             f"both {br_qd[('C_Pi_z D', 'R3')][1]:.6f} us.")
    M.append("")
    same_mean = abs(m_r3_qd - m_r3_qb) < 1e-12
    ok = ok and same_mean
    M.append(f"Identical R3 means on both sides: "
             f"{'PASS' if same_mean else 'FAIL'}. The superseded 4N_c CD "
             f"blended the R3 and R8 instances, giving a diluted mean of "
             f"{(br_qb[('CD', 'R3')][1]) / (4 * 3):.8f} us at N_c = 3, low "
             "by exactly 12/8 = 3/2. That dilution was the whole source of "
             "the nonzero T0 difference the model used to carry.")
    M.append("")

    M.append("### Adopted counts against the compiled tape")
    M.append("")
    M.append("These counts are the model. `scripts/runtime_final_regen.py`, "
             "`scripts/multitone_counts_and_centres.py` and "
             "`scripts/appendix_b_qubit_ladder.py` all carry them, so this "
             "section records the derivation and the check, not a "
             "comparison against a superseded set. The check is that each "
             "class reproduces the compiled tape's own time at N_c = 3.")
    M.append("")
    M.append("| class | side | count | mean (us) | count at N_c=3 | "
             "adopted time (us) | tape time (us) |")
    M.append("|---|---|---|---|---|---|---|")
    audit_ok = True
    rows_audit = []
    for key, side, tape_key, mean in (
            ("Rotation", "qudit", ("Rotation", "R1"), m_r1),
            ("Rotation", "qubit", ("Rotation", "R1"), m_r1),
            ("C_Pi R / CR", "both", ("C_Pi R", "R2"), m_r2),
            ("C_Pi_z D", "qudit", ("C_Pi_z D", "R3"), m_r3_qd),
            ("CD", "qubit", ("CD", "R3"), m_r3_qb)):
        fn, form, _ = NONEDGE_NEW[
            "R1" if key == "Rotation" else
            ("R2" if key.startswith("C_Pi R") else "R3")]
        src = br_qd if (side == "qudit" or key.startswith("C_Pi R")) else br_qb
        n3 = fn(3)
        adopted = n3 * mean
        tape = src[tape_key][1]
        good = abs(adopted - tape) < 1e-9
        audit_ok = audit_ok and good
        rows_audit.append(dict(cls=key, side=side, form=form, mean_us=mean,
                               count_Nc3=n3, adopted_time_us=adopted,
                               tape_time_us=tape, matches_tape=bool(good)))
        M.append(f"| {key} | {side} | **{form}** | {mean:.8f} | {n3} | "
                 f"{adopted:.6f} | {tape:.6f} |")
    M.append("")
    ok = ok and audit_ok
    M.append(f"Every class reproduces the tape at N_c = 3: "
             f"{'PASS' if audit_ok else 'FAIL'}.")
    M.append("")
    M.append("The superseded reading counted Rotation and C_Pi_z D as "
             "2(N_c+1) and the qubit CD as 4N_c at a mean blended over R3 "
             "and R8. All three agree with the adopted counts at N_c = 3 in "
             "TIME, which is why no N_c = 3 number moved; they diverge for "
             "N_c > 3.")
    M.append("")

    # ---- T0 as a closed form ----------------------------------------------
    def T0_new(nc):
        return ((4 * nc - 4) * m_r1 + 2 * nc * m_r2 + (4 * nc - 4) * m_r3_qd)

    slope = 4 * (m_r1 + m_r3_qd) + 2 * m_r2
    inter = -4 * (m_r1 + m_r3_qd)
    M.append("### T0 as a closed form")
    M.append("")
    M.append("Both sides carry the same non-edge classes at the same counts "
             "and the same means, so")
    M.append("")
    M.append("    T0_qudit(N_c) = T0_qubit(N_c)")
    M.append("      = (4N_c-4) <R1> + 2N_c <R2> + (4N_c-4) <R3>")
    M.append(f"      = (4N_c-4)({m_r1:.6f} + {m_r3_qd:.8f}) + "
             f"2N_c({m_r2:.4f})")
    M.append(f"      = {slope:.8f} N_c - {-inter:.8f}   us")
    M.append("")
    M.append("    T0_qubit - T0_qudit = 0   identically, slope 0")
    M.append("")
    M.append("| N_c | T0 (us), both sides |")
    M.append("|---|---|")
    t0_rows = []
    for nc in CLUSTER_NC:
        t0n = T0_new(nc)
        t0_rows.append(dict(N_c=nc, T0_us=t0n))
        M.append(f"| {nc} | {t0n:.6f} |")
    M.append("")
    t0_3_ok = abs(T0_new(3) - T0[3][0]) < 1e-9
    ok = ok and t0_3_ok
    M.append(f"At N_c = 3, {T0_new(3):.8f} us, matching section 2: "
             f"{'PASS' if t0_3_ok else 'FAIL'}.")
    M.append("")
    M.append("The earlier nonzero difference, <CD>_blended (N_c - 3), was "
             "an artefact of that blended mean. Its zero at N_c = 3 was "
             "where the 3/2 dilution cancelled the 4N_c against 2(N_c+1) "
             "count ratio, not a calibration.")
    M.append("")

    # ---- helpers kept for the N_c = 3 confirmation below -------------------
    def steps_new(nc):
        t0 = T0_new(nc)
        return (t0 + 2 * (nc - 1) * t_X_loc + 2 * (nc - 2) * t_L_loc,
                t0 + (6 * nc - 8) * t_MS)

    def isa_new(nc):
        return (4*nc-4) + 2*nc + (4*nc-4) + 2*nc + 2*(nc-1) + 2*(nc-2)

    def qubit_ops_new(nc):
        return (4*nc-4) + 2*nc + (4*nc-4) + 2*(nc-1) + 2*(nc-1) + 2*(nc-2)

    def mt_new(nc):
        return (4*nc-4) + 2*nc + (4*nc-4) + 2 + 2 + 1

    M.append("### Where the adopted numbers live")
    M.append("")
    M.append("Closed forms: qudit ISA **16N_c-14**, qubit logical "
             "**16N_c-16**, multitone **10N_c-3**, MS windows **6N_c-8**.")
    M.append("")
    M.append("- tab:runtime rows: `results/runtime_final_regen.txt` and "
             "`.tex`, regenerated from these counts.")
    M.append("- Chain S for every N_c and K: section 5b above.")
    M.append("- The large-N_c limits: section 5c above.")
    M.append("")
    M.append("Those tables are not repeated here. An earlier draft of this "
             "section carried its own copies under an old-against-new "
             "heading; once the counts were adopted the two columns became "
             "the same quantity and the comparison inverted, so they are "
             "gone.")
    M.append("")

    q3n, b3n = steps_new(3)
    conf = [("qudit ISA ops", isa_new(3), 34),
            ("qubit logical ops", qubit_ops_new(3), 32),
            ("multitone blocks", mt_new(3), 27),
            ("MS windows", 6 * 3 - 8, 10),
            ("Rotation count", 4 * 3 - 4, 8),
            ("C_Pi R / CR count", 2 * 3, 6),
            ("R3 count", 4 * 3 - 4, 8)]
    nc3_all = (all(a_ == b_ for _, a_, b_ in conf)
               and abs(q3n - 210.4971) < 5e-4
               and abs(b3n - 2207.8701) < 5e-4
               and abs(b3n / q3n - 10.4888) < 5e-4)
    ok = ok and nc3_all

    M.append("### N_c = 3 confirmation")
    M.append("")
    M.append("| quantity | adopted | expected |")
    M.append("|---|---|---|")
    for nm, a_, b_ in conf:
        M.append(f"| {nm} | {a_} | {b_} |")
    M.append(f"| T_qudit (us) | {q3n:.4f} | 210.4971 |")
    M.append(f"| T_qubit (us) | {b3n:.4f} | 2207.8701 |")
    M.append(f"| ratio | {b3n/q3n:.4f} | 10.4888 |")
    M.append(f"| T0 (us) | {T0_new(3):.4f} | 207.8701 |")
    M.append("")
    M.append(f"Every N_c = 3 quantity unchanged: "
             f"{'PASS' if nc3_all else 'FAIL'}.")
    M.append("")

    audit_json = dict(
        status="adopted; carried by runtime_final_regen.py, "
               "multitone_counts_and_centres.py and "
               "appendix_b_qubit_ladder.py",
        counts=dict(
            Rotation=dict(form="4N_c-4",
                          derivation="one mode free evolution per mode per "
                                     "half step, modes N_c high + (N_c-2) "
                                     "low = 2N_c-2"),
            C_Pi_R_CR=dict(form="2N_c",
                           derivation="one dispersive term per chromophore, "
                                      "Vu Eq. 20 term 1, per half step"),
            R3=dict(form="4N_c-4",
                    derivation="high mode per site (Eq. 20 term 2) plus low "
                               "mode per site (Eq. 21 term 1), boundary low "
                               "modes dropped",
                    note="the qubit CD splits by rule: 4N_c-4 at R3 and "
                         "4(N_c-2) zero-time at R8 inside the J_l")),
        means=dict(R1=m_r1, R2=m_r2, R3_qudit=m_r3_qd, R3_qubit=m_r3_qb,
                   R3_means_identical=bool(same_mean),
                   qubit_CD_R8_count=n_r3_qb_r8),
        tape_check=rows_audit,
        T0=dict(closed_form="(4N_c-4)(<R1>+<R3>) + 2N_c<R2>",
                slope_us_per_Nc=slope, intercept_us=inter,
                difference="0 identically", rows=t0_rows),
        closed_forms=dict(qudit_ISA="16N_c-14", qubit_logical="16N_c-16",
                          multitone="10N_c-3", MS_windows="6N_c-8"),
        see_also=dict(chain_S="section 5b", limits="section 5c",
                      tab_runtime="results/runtime_final_regen.txt"),
        Nc3_unchanged=bool(nc3_all))

    # ---- 4. N_c = 3 must be untouched --------------------------------------
    M.append("### 4. N_c = 3 against 210.5 us and 10.5")
    M.append("")
    B3 = chain_bonds(3)
    q3 = T0[3][0] + sum(c_in_bond(x) for x in B3)
    b3 = T0[3][1] + sum(c_out_bond(x) for x in B3)
    nc3_ok = (round(q3, 1) == 210.5 and round(b3 / q3, 1) == 10.5)
    ok = ok and nc3_ok
    M.append(f"At N_c = 3 both bonds are end bonds, so the link total is "
             f"2(N_c-2) = 2, the same as the earlier N_c-1 = 2. Nothing in "
             f"the unsplit totals moves.")
    M.append("")
    M.append("| quantity | value |")
    M.append("|---|---|")
    M.append(f"| T0_qudit | {T0[3][0]:.4f} us |")
    M.append(f"| sum c_in over 2 end bonds | "
             f"{sum(c_in_bond(x) for x in B3):.4f} us |")
    M.append(f"| qudit step total | **{q3:.4f} us** |")
    M.append(f"| T0_qubit | {T0[3][1]:.4f} us |")
    M.append(f"| sum c_out over 2 end bonds | "
             f"{sum(c_out_bond(x) for x in B3):.4f} us |")
    M.append(f"| qubit step total | **{b3:.4f} us** |")
    M.append(f"| ratio | **{b3 / q3:.4f}** |")
    M.append("")
    M.append(f"One decimal: {q3:.1f} us and {b3 / q3:.1f}. Target 210.5 and "
             f"10.5: {'PASS' if nc3_ok else 'FAIL'}. These also match the "
             f"section 2 totals, {t_single(3):.4f} us and "
             f"{t_qubit(3):.4f} us, to "
             f"{abs(q3 - t_single(3)):.1e} and "
             f"{abs(b3 - t_qubit(3)):.1e} us.")
    M.append("")

    # ---- 6. bounds ---------------------------------------------------------
    M.append("## 6. Bounds, star placement")
    M.append("")
    M.append("Using E_in <= min(K-1, N_c-1) and c_in <= c_out. "
             "For N_c <= K the bound is "
             "[T0_qubit + E c_out] / [T0_qudit + E c_in]; for N_c > K it is "
             "[T0_qubit + E c_out] / [T0_qudit + (K-1) c_in + "
             "(N_c-K) c_out].")
    M.append("")
    M.append("| N_c | K | case | S | bound | gap |")
    M.append("|---|---|---|---|---|---|")
    for r in rows:
        nc, K, E = r["N_c"], r["K"], r["E"]
        t0q, t0b = T0[nc]
        num = t0b + E * c_out
        if nc <= K:
            den_b, case = t0q + E * c_in, "N_c <= K"
        else:
            den_b = t0q + (K - 1) * c_in + (nc - K) * c_out
            case = "N_c > K"
        B = num / den_b
        r["case"], r["bound"] = case, B
        r["gap_bound"] = B - r["S"]
        M.append(f"| {nc} | {K} ({r['platform']}) | {case} | {r['S']:.4f} | "
                 f"{B:.4f} | {B - r['S']:+.3e} |")
    M.append("")
    tight = max(abs(r["gap_bound"]) for r in rows) < 1e-9
    ok = ok and tight
    M.append(f"Every gap is zero to 1e-9: {'PASS' if tight else 'FAIL'}. "
             "The bound is attained, not merely respected: optimal hub "
             "placement (assumption c) already sets E_in to its maximum "
             "min(K-1, N_c-1), so the inequality on E_in is saturated by "
             "construction and the two expressions coincide term by term. "
             "The separate inequality c_in <= c_out is what makes larger "
             "E_in the better choice, and it holds numerically above; it "
             "does not loosen the bound.")
    M.append("")

    # ---- 7. limit ----------------------------------------------------------
    M.append("## 7. Window-dominated limit, star placement")
    M.append("")
    M.append("(N_c-1)/(N_c-K), the ratio once the MS windows dominate both "
             "sides. Defined only for N_c > K. The chain counterpart is in "
             "section 5c.")
    M.append("")
    M.append("| N_c | K | S | (N_c-1)/(N_c-K) | S - limit |")
    M.append("|---|---|---|---|---|")
    for r in rows:
        nc, K = r["N_c"], r["K"]
        if nc <= K:
            r["limit"] = r["gap_limit"] = None
            M.append(f"| {nc} | {K} ({r['platform']}) | {r['S']:.4f} | "
                     "-- | -- |")
            continue
        lim = (nc - 1) / (nc - K)
        r["limit"], r["gap_limit"] = lim, r["S"] - lim
        M.append(f"| {nc} | {K} ({r['platform']}) | {r['S']:.4f} | "
                 f"{lim:.4f} | {r['S'] - lim:+.4f} |")
    M.append("")
    gaps = [r["gap_limit"] for r in rows if r["gap_limit"] is not None]
    below = all(g < 0 for g in gaps)
    ok = ok and below
    M.append(f"S lies BELOW the limit on all {len(gaps)} rows where the "
             f"limit is defined (largest gap {min(gaps):+.4f}, smallest "
             f"{max(gaps):+.4f}): {'PASS' if below else 'FAIL'}. The limit "
             "drops the non-window terms from both sides, and those terms "
             "are the larger share of the qudit denominator than of the "
             "qubit numerator, so discarding them overstates the ratio. "
             "The gap closes as N_c grows at fixed K, and widens with K at "
             "fixed N_c, since a larger cluster leaves fewer crossing "
             "edges and so less window time to dominate.")
    M.append("")

    # ---- checks ------------------------------------------------------------
    M.append("## Checks")
    M.append("")
    M.append(f"- MS XX and MS YY flat at t_MS = {t_MS:.4f} us: "
             f"{'PASS' if flat else 'FAIL'}.")
    M.append(f"- cluster S against the Part B ratio at equal capacity, "
             f"{len(seen)} rows, max |diff| {worst:.3e}: "
             f"{'PASS' if agree else 'FAIL'}.")
    M.append(f"- bound attained on all {len(rows)} rows: "
             f"{'PASS' if tight else 'FAIL'}.")
    M.append(f"- T0_qubit - T0_qudit is zero on every row: "
             f"{'PASS' if dis_ok else 'FAIL'}.")
    M.append(f"- S below the (N_c-1)/(N_c-K) limit on every row where it is "
             f"defined: {'PASS' if below else 'FAIL'}.")
    M.append(f"- chain link total 2(N_c-2) equals the earlier N_c-1 at "
             f"N_c = 3: {'PASS' if same3 else 'FAIL'}.")
    M.append(f"- chain crossing count = ceil(N_c/K) - 1 on every row: "
             f"{'PASS' if cnt_ok else 'FAIL'}.")
    M.append(f"- N_c = 3 chain totals reproduce 210.5 us and 10.5: "
             f"{'PASS' if nc3_ok else 'FAIL'}.")
    M.append(f"- qudit and qubit R3 means identical: "
             f"{'PASS' if same_mean else 'FAIL'}.")
    M.append(f"- every audited class equal at N_c = 3: "
             f"{'PASS' if audit_ok else 'FAIL'}.")
    M.append(f"- new T0 reproduces the old at N_c = 3: "
             f"{'PASS' if t0_3_ok else 'FAIL'}.")
    M.append(f"- every N_c = 3 quantity unchanged under the audit: "
             f"{'PASS' if nc3_all else 'FAIL'}.")
    M.append("")

    J = dict(assumptions=dict(
                 a="gate times are class means at N_c = 3, a model "
                   "approximation",
                 b="the link-modulated term costs one MS window on the "
                   "qubit side, an assumption of the timing model",
                 c="hub placement is optimal, A shares its ion with K-1 "
                   "neighbours; star placement only",
                 d="chain placement takes nearest-neighbour links only, "
                   "the 1D array of N chromophores of Ref. vu2025 "
                   "Sec. 2.4 and Eq. 52, K consecutive per ion"),
             placements=dict(
                 star=dict(E_out="max(0, N_c - K)",
                           E_in="min(K-1, N_c-1)",
                           window_limit="(N_c-1)/(N_c-K) -> 1",
                           source="mas.tex:2010"),
                 chain=dict(E_out="ceil(N_c/K) - 1",
                            E_in="N_c - ceil(N_c/K)",
                            window_limit="(6N_c-8)/(6(ceil(N_c/K)-1)) -> K",
                            full_limit="[2r+2cr+4cd+6t_MS] / "
                                       "[2r+2cr+2czd+(1-1/K)(2t_X+2t_L)"
                                       "+6t_MS/K]",
                            terms_per_bond="2 exchange (Eq. 21 terms 2,3); "
                                           "2 link (terms 4,5), one per "
                                           "endpoint low-frequency mode",
                            boundary_rule="Eq. 52 drops 1_1 and N_c_1, so "
                                          "end bonds keep 1 link, interior "
                                          "bonds 2",
                            totals="exchange 2(N_c-1), link 2(N_c-2)",
                            source="vu2025 Sec. 2.1.4, Eq. 18-21, Eq. 52")),
             costs=dict(t_MS_us=t_MS, t_X_loc_us=t_X_loc,
                        t_L_loc_us=t_L_loc, c_out_us=c_out, c_in_us=c_in,
                        c_in_le_c_out=bool(c_in <= c_out),
                        sources=dict(
                            t_MS="scripts/appendix_d_tables_tex.py:29 "
                                 "(value), :102 (rule R7)",
                            t_X_loc="scripts/runtime_final_regen.py:86 "
                                    "(count, rule R4), :115 (class mean)",
                            t_L_loc="scripts/runtime_final_regen.py:87 "
                                    "(count, rule R5), :115 (class mean)")),
             qubit_window_means=dict(MS_XX_us=w_xx, MS_YY_us=w_yy,
                                     flat_at_t_MS=bool(flat)),
             T0=[dict(N_c=nc, E=nc - 1, qudit_step_us=t_single(nc),
                      qubit_step_us=t_qubit(nc), T0_qudit_us=T0[nc][0],
                      T0_qubit_us=T0[nc][1],
                      difference_us=T0[nc][1] - T0[nc][0])
                 for nc in CLUSTER_NC],
             rows=rows, chain_limits=lim_rows, nonedge_audit=audit_json,
             checks=dict(windows_flat=bool(flat),
                         matches_part_B=bool(agree),
                         max_abs_diff_vs_part_B=worst,
                         rows_compared=len(seen), rows_total=len(rows),
                         bound_attained=bool(tight),
                         T0_difference_is_zero=bool(dis_ok),
                         S_below_limit=bool(below),
                         chain_link_count_matches_at_Nc3=bool(same3),
                         chain_crossing_count=bool(cnt_ok),
                         chain_Nc3_reproduces_210_5_and_10_5=bool(nc3_ok),
                         missing_part_B_rows=[dict(N_c=a, K=b)
                                              for a, b in missing]))
    return M, J, ok


def main():
    import pennylane as qml
    import circuit_fig6_hqml as QD
    import circuit_qubit_hybridlane as QB

    with qml.tape.QuantumTape() as t:
        QD.one_trotter_step(A.TAU)
    qd_ops = list(t.operations)
    with qml.tape.QuantumTape() as t:
        QB.one_hot_trotter_step()
    qb_ops = list(t.operations)
    qd = A.build(qd_ops, "qudit")
    qb_raw = A.build(qb_ops, "qubit")
    qb = RF.split_cd(qb_raw)      # CD at R8 -> CD_Jl, so the mean is not diluted

    ad = (ROOT / "results" / "appendix_d_tables.txt").read_text(encoding="utf-8")
    f_qb = float(re.search(r"qubit total ([\d.]+) us", ad).group(1))

    # ---- Part A -----------------------------------------------------------
    assert capacity(4, 1) == 2, "Part A layout assumes capacity 2 at d_eff=4"
    rows = []
    for i, (op, (cls, ang, rule, T)) in enumerate(zip(qd_ops, qd), 1):
        ions = {SPLIT_A[lv] for lv in levels_of(op)}
        cross = len(ions) > 1
        if cross and cls not in WIN:
            raise KeyError(f"row {i}: crossing {cls} has no window rule")
        w = WIN.get(cls, 0) if cross else 0
        w1 = WIN_SIDE.get(cls, 0) if cross else 0
        rows.append(dict(n=i, gate=cls, levels=pair_label(op),
                         angle=ang, rule_single=rule, T_single=T,
                         crossing=cross, windows=w,
                         rule=(("R7" if w == 1 else f"R7 x{w}") if cross
                               else rule),
                         T=(w * A.T_MS if cross else T),
                         T_side=(w1 * A.T_MS if cross else T)))
    T_unsplit = sum(r["T_single"] for r in rows)
    T_A = sum(r["T"] for r in rows)
    T_A_side = sum(r["T_side"] for r in rows)
    cross_rows = [r for r in rows if r["crossing"]]
    xe_removed = sum(r["T_single"] for r in cross_rows if r["gate"] == "X_E")

    # ---- Part B -----------------------------------------------------------
    cq, cb = RF.per_class(qd), RF.per_class(qb)
    mXE, mCX = cq["X_E"][2], cq["C_Pi_x D"][2]
    w_edge = 2 * WIN["X_E"] + WIN["C_Pi_x D"]                  # 5
    w_edge_side = 2 * WIN_SIDE["X_E"] + WIN_SIDE["C_Pi_x D"]   # 3

    # CHAIN totals.  runtime_final_regen now counts the link term
    # C_Pi_x D at 2(Nc-2), the chain count of Vu Eq. 21 with Eq. 52's
    # boundary rule, so these follow it.
    def t_single(nc):
        return sum(RF.QUDIT_N[c][0](nc) * cq[c][2] for c in RF.QUDIT_N)

    def t_qubit(nc):
        return (sum(RF.QUBIT_N[c][0](nc) * cb[c][2] for c in RF.QUBIT_N)
                + 2 * (nc - 2) * A.T_MS)

    # STAR totals, the superseded trimer-star reading with Nc-1 link
    # terms.  Retained only so Part B and the star placement sections
    # below stay internally consistent; the chain is the adopted
    # extension.  Equal to the chain totals at Nc = 3.
    def t_single_star(nc):
        return sum((nc - 1 if c == "C_Pi_x D" else RF.QUDIT_N[c][0](nc))
                   * cq[c][2] for c in RF.QUDIT_N)

    def t_qubit_star(nc):
        return (sum(RF.QUBIT_N[c][0](nc) * cb[c][2] for c in RF.QUBIT_N)
                + (nc - 1) * A.T_MS)

    def split_row(nc, d, n_aux):
        k = capacity(d, n_aux)
        if k < 1:
            raise ValueError(f"d_eff={d}, n_aux={n_aux} leaves no capacity")
        nion = math.ceil(nc / k)
        e = max(0, nc - k)
        base = t_single_star(nc) - e * (2 * mXE + mCX)
        T = base + w_edge * e * A.T_MS
        T1 = base + w_edge_side * e * A.T_MS
        Tq = t_qubit_star(nc)
        return dict(N_c=nc, d_eff=d, n_aux=n_aux, capacity=k, N_ion=nion,
                    edges=nc - 1, crossing_edges=e, windows=w_edge * e,
                    T_qudit=T, T_qubit=Tq, ratio=Tq / T,
                    T_qudit_one_window=T1, ratio_one_window=Tq / T1)

    grid = [split_row(nc, d, GENERIC["n_aux"])
            for nc in GENERIC["nc_list"] for d in GENERIC["d_list"]]
    ca = [split_row(nc, CA40["d_eff"], CA40["n_aux"])
          for nc in CA40["nc_list"]]

    g = {(r["N_c"], r["d_eff"]): r for r in grid}
    c1 = g[(3, 8)]
    chk1 = (round(c1["T_qudit"], 1) == 210.5 and round(c1["ratio"], 1) == 10.5)
    c2 = g[(3, 4)]
    chk2_diff = c2["T_qudit"] - T_A
    chk2 = abs(chk2_diff) < 0.05
    ca3 = next(r for r in ca if r["N_c"] == 3)
    chk3 = (round(ca3["T_qudit"], 1) == 210.5 and round(ca3["ratio"], 1) == 10.5)

    # ---- cluster speedup check -------------------------------------------
    CM, CJ, chk4 = cluster_block(cq, cb, t_single, t_qubit,
                                 t_single_star, t_qubit_star, grid, ca,
                                 capacity(CA40["d_eff"], CA40["n_aux"]),
                                 qd, qb_raw)
    (ROOT / "results" / "cluster_bound.md").write_text(
        "\n".join(CM) + "\n", encoding="utf-8")
    CJ["commit"] = commit()
    (ROOT / "results" / "cluster_bound.json").write_text(
        json.dumps(CJ, indent=2) + "\n", encoding="utf-8")

    ok = chk1 and chk2 and chk3 and chk4

    # ---- markdown ---------------------------------------------------------
    M = []
    M.append("# Multi-ion split of the qudit register")
    M.append("")
    M.append(f"Generated by `scripts/split_benchmark.py` at commit {commit()}.")
    M.append("Rows come from the compiled qudit tape through the Table D1 "
             "classifier (`scripts/appendix_d_tables_tex.py`); class means and "
             "closed-form counts from `scripts/runtime_final_regen.py`.")
    M.append(f"Qubit reference total {f_qb:.4f} us "
             "(`results/appendix_d_tables.txt`), unchanged by the split.")
    M.append("")
    M.append("Crossing gates are priced in MS windows of 200 us (rule R7) "
             "with the qubit reference's convention: a crossing X_E is an "
             "exchange, two windows, like J; a crossing C_{Pi_x}D is the "
             "link-modulated exchange, one window, like J_l. A crossing edge "
             "therefore costs 5 windows.")
    M.append("")
    M.append("Capacity per ion is d_eff - 1 - n_aux excited states: one level "
             "is the reference state G, and n_aux are auxiliary levels "
             "reserved for the level-selective mode operations. Generic "
             "platform n_aux = 1; 40Ca+ d_eff = 8, n_aux = 2, capacity 5.")
    M.append("")
    M.append("## Part A. N_c = 3 at d_eff = 4, n_aux = 1, "
             "ion 1 = {G, A, B}, ion 2 = {C}")
    M.append("")
    M.append("| # | gate | levels | angle | rule | T (us) | crossing |")
    M.append("|---|---|---|---|---|---|---|")
    for r in rows:
        a = "--" if r["angle"] is None else f"{r['angle']:+.4f}"
        M.append(f"| {r['n']} | {r['gate']} | {r['levels']} | {a} | "
                 f"{r['rule']} | {r['T']:.3f} | "
                 f"{'yes' if r['crossing'] else ''} |")
    M.append("")
    M.append(f"Crossing gates: {len(cross_rows)} "
             f"(rows {', '.join(str(r['n']) for r in cross_rows)}), "
             f"{sum(r['windows'] for r in cross_rows)} windows.")
    M.append(f"Their single-ion price was "
             f"{sum(r['T_single'] for r in cross_rows):.4f} us; "
             f"as two-ion gates they cost "
             f"{sum(r['T'] for r in cross_rows):.1f} us.")
    M.append("")
    M.append(f"- unsplit total (reproduces Table D1): {T_unsplit:.4f} us")
    M.append(f"- split total: {T_A:.4f} us")
    M.append(f"- ratio T_qubit / T_qudit: {f_qb:.4f} / {T_A:.4f} = "
             f"{f_qb / T_A:.4f}")
    M.append("")
    M.append("## Part B. Generic platform, n_aux = 1, star coupling from A")
    M.append("")
    M.append("Capacity = d_eff - 2; N_ion = ceil(N_c/capacity); crossing "
             "edges = max(0, N_c - capacity); each crossing edge is 2 X_E at "
             "two windows plus 1 C_{Pi_x}D at one window, 5 windows.")
    M.append("")
    M.append("| N_c | d_eff | capacity | N_ion | crossing edges | windows | "
             "T_qudit (us) | T_qubit (us) | ratio |")
    M.append("|---|---|---|---|---|---|---|---|---|")
    for r in grid:
        M.append(f"| {r['N_c']} | {r['d_eff']} | {r['capacity']} | "
                 f"{r['N_ion']} | {r['crossing_edges']} of {r['edges']} | "
                 f"{r['windows']} | {r['T_qudit']:.1f} | "
                 f"{r['T_qubit']:.1f} | {r['ratio']:.2f} |")
    M.append("")
    M.append("## Part B. 40Ca+, d_eff = 8, n_aux = 2, capacity 5")
    M.append("")
    M.append("| N_c | capacity | N_ion | crossing edges | windows | "
             "T_qudit (us) | T_qubit (us) | ratio |")
    M.append("|---|---|---|---|---|---|---|---|")
    for r in ca:
        M.append(f"| {r['N_c']} | {r['capacity']} | {r['N_ion']} | "
                 f"{r['crossing_edges']} of {r['edges']} | {r['windows']} | "
                 f"{r['T_qudit']:.1f} | {r['T_qubit']:.1f} | "
                 f"{r['ratio']:.2f} |")
    M.append("")
    M.append("The qubit reference carries 5(N_c-1) windows per step and a "
             "split qudit 5 max(0, N_c - capacity). Once the window time "
             "dominates both, the ratio tends to (N_c-1)/(N_c-capacity), "
             "which approaches 1 as N_c grows at fixed capacity.")
    M.append("")
    M.append("The same quantity is re-derived from the edge costs alone, "
             "with the cluster size K free, in `results/cluster_bound.md`, "
             "together with the bound and the (N_c-1)/(N_c-K) limit.")
    M.append("")
    M.append("## Checks")
    M.append("")
    M.append(f"- generic N_c=3, d_eff=8 gives {c1['T_qudit']:.4f} us and ratio "
             f"{c1['ratio']:.4f}; one decimal {c1['T_qudit']:.1f} us and "
             f"{c1['ratio']:.1f}. Target 210.5 and 10.5: "
             f"{'PASS' if chk1 else 'FAIL'}.")
    M.append(f"- generic N_c=3, d_eff=4 closed form {c2['T_qudit']:.4f} us "
             f"against Part A {T_A:.4f} us, difference {chk2_diff:+.4f} us: "
             f"{'PASS' if chk2 else 'FAIL'} at 0.05 us. The residual is "
             f"the closed form removing the crossing X_E at the class mean, "
             f"2 x {mXE:.4f} = {2 * mXE:.4f} us, where Part A removes their "
             f"own A-C price, {xe_removed:.4f} us.")
    M.append(f"- 40Ca+ N_c=3 gives {ca3['T_qudit']:.4f} us and ratio "
             f"{ca3['ratio']:.4f}, the single-ion values. Target 210.5 and "
             f"10.5: {'PASS' if chk3 else 'FAIL'}.")
    M.append(f"- closed-form qubit total at N_c=3 {t_qubit(3):.4f} us against "
             f"{f_qb:.4f} us in `results/appendix_d_tables.txt`.")
    M.append(f"- cluster speedup check, see `results/cluster_bound.md`: "
             f"{'PASS' if chk4 else 'FAIL'}.")
    M.append("")
    M.append("## Side note: crossing X_E at one window")
    M.append("")
    M.append("Not the convention used above. Shown only for comparison with "
             "the first draft of this benchmark, which priced a crossing X_E "
             "at one window; that under-prices it relative to the qubit "
             "reference's own treatment of exchange. Generic platform only.")
    M.append("")
    M.append(f"Part A: {T_A_side:.4f} us, ratio {f_qb / T_A_side:.4f}.")
    M.append("")
    M.append("| N_c | d_eff | T_qudit (us) | ratio |")
    M.append("|---|---|---|---|")
    for r in grid:
        if r["crossing_edges"]:
            M.append(f"| {r['N_c']} | {r['d_eff']} | "
                     f"{r['T_qudit_one_window']:.1f} | "
                     f"{r['ratio_one_window']:.2f} |")
    M.append("")

    (ROOT / "results" / "split_benchmark.md").write_text(
        "\n".join(M) + "\n", encoding="utf-8")

    J = dict(commit=commit(), T_MS_us=A.T_MS, qubit_total_Nc3_us=f_qb,
             windows_per_crossing_instance=WIN,
             capacity_rule="d_eff - 1 - n_aux",
             part_A=dict(layout={"ion1": ["G", "A", "B"], "ion2": ["C"]},
                         d_eff=4, n_aux=1,
                         rows=rows, unsplit_total_us=T_unsplit,
                         split_total_us=T_A, ratio=f_qb / T_A,
                         crossing_rows=[r["n"] for r in cross_rows]),
             part_B=dict(mean_XE_us=mXE, mean_CPixD_us=mCX,
                         windows_per_crossing_edge=w_edge,
                         generic=dict(n_aux=GENERIC["n_aux"], grid=grid),
                         ca40=dict(d_eff=CA40["d_eff"], n_aux=CA40["n_aux"],
                                   capacity=capacity(CA40["d_eff"],
                                                     CA40["n_aux"]),
                                   grid=ca)),
             checks=dict(generic_Nc3_d8=dict(T=c1["T_qudit"],
                                             ratio=c1["ratio"], passed=chk1),
                         generic_Nc3_d4_vs_partA=dict(closed=c2["T_qudit"],
                                                      partA=T_A,
                                                      diff=chk2_diff,
                                                      passed=chk2),
                         ca40_Nc3=dict(T=ca3["T_qudit"], ratio=ca3["ratio"],
                                       passed=chk3)),
             side_note_one_window_XE=dict(
                 note="not the adopted convention; generic platform only",
                 part_A_total_us=T_A_side, part_A_ratio=f_qb / T_A_side))
    (ROOT / "results" / "split_benchmark.json").write_text(
        json.dumps(J, indent=2) + "\n", encoding="utf-8")
    print("\n".join(M))
    print()
    print("\n".join(CM))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
