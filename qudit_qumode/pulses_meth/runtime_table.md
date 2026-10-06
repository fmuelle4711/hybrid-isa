# Runtime table (final) — Meth operating point

All gate times read from saved files.  Never retyped.

## Inputs

| gate | value | source | derivation | flag |
|---|---:|---|---|---|
| QCD (α=0.008) | 2.566 µs | `pulses_meth\hypulse_validation\final_v3.json` | T_QCD(alpha=0.008) = T_QCD(alpha=0.5)_final_v3 * (0.008/0.5) = 160.4 * 0.016000 = 2.5664 us | overhead-limited (FINDING 1: bound never binds at small alpha) |
| QCR (T_paper) | 55.93 µs | `pulses_meth\appendix_table.md` | T_paper = 2*T + t_pi with t_pi = 5 us ASSUMED |  |
| QT-SB (θ=0.0015) | 0.6494 µs | `results\qtsb_duration_sweep.txt` | 0.5 motional periods = 0.5 * 2pi/nu = 0.649351 us; flat pulse at theta/(eta*T) = 9.2% of Omega_c; 1-F_indep_vz = 3.438e-09, err/signal = 0.00131 | ADOPTED, supersedes T_analytic = 0.0597 us, which is 0.046 motional periods and does not function (err/signal = 0.9886) |
| Rotation | 5 µs | `pulses_meth\appendix_table.md` | 5 us ASSUMED entirely; no anchor | ASSUMED |
| MS | 200 µs | `pulses_meth\appendix_table.md` | Meth SI Note II p.3: 0.2 ms per gate (angle-independent) |  |
| QT bounds | [0.105, 11.0] µs | `pulses_meth\appendix_table.md` | qutip-qtrl GRAPE, 200 slots, aggregate field cap sqrt(sum|Omega_k|^2) <= Omega_max; high = calibrated theta=pi/4, low = angle-scaled to circuit angle 0.0075 rad (overhead-limited, not_in_meth) | Omega_max/2pi = 100 kHz ASSUMED |
| QPS | 0.0 µs | virtual-Z | frame update (McKay 2017); T = 0 | none |
| τ_motional | 27.4 ms | `hypulse/calibrations/innsbruck_meth_2025.py` | Meth Methods p.8 motional coherence | anchor |

## Op counts per Trotter step

Naming: ISA names are used throughout. The simulation-side gate labels in `simulations/` differ and map as: `chi` = **CR** (σ_z ⊗ n̂, dispersive conditional rotation), `cd` = **CD** (σ_z ⊗ x̂, conditional displacement), `J` / `Jl` = **MS-type** (2-qubit XY carrier / 3-qubit sideband), `rz` = virtual-Z (free). On the qudit side the same two primitives are `QCR` (Π_l ⊗ n̂) and `QCD`; QCR and CR differ by an unconditional mode rotation absorbed into the free frame (FINDING 5), so both are priced identically.

### Per-primitive breakdown

| primitive | qudit count | qubit count | unit time (µs) |
|---|---|---|---:|
| CR / QCR  (σ_z⊗n̂) | 2Nc = 6 | 2Nc = 6 | 55.93 |
| CD / QCD  (σ_z⊗x̂) | 2(Nc+1) = 8 | 2(Nc+1) = 8 | 2.5664 |
| Rotation  (mode free) | 2(Nc+1) = 8 | 2(Nc+1) = 8 | 5.0 |
| QPS / rz  (virtual-Z) | 2 = 6 | 2(Nc-1) = 4 | 0 |
| QT (carrier) | 2 = 4 | — | [0.105, 11.0] |
| QT-SB (sideband) | Nc-1 = 2 | — | 0.6494 |
| MS-type J  (2-qubit XY) | — | 2(Nc-1) = 4 | 200 |
| MS-type Jl (sideband) | — | Nc-1 = 2 | 200 |

Counts in the `= n` columns are at **Nc = 3**, read from the compiled circuits ([build_generalized_qudit_circuit](../simulations/generalized_28_validation/scaling.py), [build_one_hot_trotter_step](../simulations/vibronic_scaling.py)), not from the formulas.

### Totals

| Nc | qudit total (7Nc+7) | qubit total (11Nc-1) | qubit MS-type (3(Nc-1)) |
|---:|---:|---:|---:|
| 3 | 34 | 32 | 6 |
| 6 | 67 | 65 | 15 |
| 8 | 89 | 87 | 21 |
| 12 | 133 | 131 | 33 |

## Per-step runtime and qubit/qudit ratio

T_QT bounds: **low = 0.105 µs** (optimistic, angle-scaled), **high = 11.0 µs** (pessimistic, calibrated π/4).

| Nc | T_qudit @ T_QT=0.105 µs | T_qudit @ T_QT=11.0 µs | T_qubit seq (µs) | T_qubit MS×3 (µs) | T_qubit MS×1 (µs) | ratio seq | **ratio MS×3 (paper)** | ratio MS×1 (bound) |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 3 | 397.8 | 441.4 | 2396.1 | 1396.1 | 996.1 | [5.43×, 6.02×] | **[3.16×, 3.51×]** | [2.26×, 2.50×] |
| 6 | 781.4 | 890.3 | 5777.1 | 1777.1 | 1377.1 | [6.49×, 7.39×] | **[2.00×, 2.27×]** | [1.55×, 1.76×] |
| 8 | 1037.1 | 1189.6 | 8031.1 | 2031.1 | 1631.1 | [6.75×, 7.74×] | **[1.71×, 1.96×]** | [1.37×, 1.57×] |
| 12 | 1548.5 | 1788.2 | 12539.0 | 2539.0 | 2139.0 | [7.01×, 8.10×] | **[1.42×, 1.64×]** | [1.20×, 1.38×] |


### MS-window accounting (J = 2 windows, ADOPTED)

`H_J = g(X_mX_n + Y_mY_n)/2` per pair. One MS window generates only `sigma_phi (x) sigma_phi = cos^2(phi) XX + sin^2(phi) YY + cos(phi)sin(phi)(XY+YX)`; killing the cross terms forces phi = 0 (XX alone) or phi = pi/2 (YY alone), so **no single window gives XX+YY**. Two windows are *exact*, not approximate, because `[X(x)X, Y(x)Y] = 0` hence `exp(-i t (XX+YY)) = exp(-i t XX) exp(-i t YY)`. Derivation and source quotes: [results/j_ms_convention.txt](../results/j_ms_convention.txt).

At N_c = 3, arithmetic in full:

```
  non-MS part : CR 6x55.93 + CD 8x2.5664000000000002 + Rot 8x5.0 + rz 4x0
              = 335.58 + 20.53 + 40.00 + 0 = 396.11 us
  MS windows  : J 4x2 + Jl 2x1 = 10   ->  10x200 = 2000 us
  T_qubit     = 396.11 + 2000 = 2396.11 us
```

**SUPERSEDED** -- the previous table priced J at **1** window (6 windows total, T_qubit = 1596.11 us, sequential ratio **[3.82x, 4.03x]**, parallel-MS **[2.38x, 2.51x]**). Reason: a single Molmer-Sorensen window cannot produce XX+YY (algebra above), and the repo's own circuit code ([circuit_fig89_10_hqml_qnode_b_qubit.py:130,180](../simulations/circuit_fig89_10_hqml_qnode_b_qubit.py#L130)) already emits `IsingXX` and `IsingYY` as separate gates. The revision moves both brackets **in the qudit encoding's favour**, so the earlier published numbers were the conservative ones.


### H_Jl pricing from its decomposition (PRINT ONLY -- NOT ADOPTED)

`H_Jl = g_abl*(sxa*sxb + sya*syb)*xl/2 + g_acl*(...)*xl/2` is a 2-qubit exchange **times** `x_l`, so it is a 3-body term. It is realised not by bare MS windows but by a CD sandwich. Gate list per coupling, quoted from [circuit_fig89_10_hqml_qnode_b_qubit.py:137-147](../simulations/circuit_fig89_10_hqml_qnode_b_qubit.py#L137):

```
  qml.Hadamard(QB[0]); qml.Hadamard(QB[i])
  qml.CNOT([QB[i], QB[0]])
  qml.SWAP([QB[0], ANC])
  hqml.ConditionalDisplacement(r, phi, [ANC, MODES[3]])
  qml.SWAP([QB[0], ANC])
  qml.CNOT([QB[i], QB[0]])
  qml.Hadamard(QB[0]); qml.Hadamard(QB[i])
```

i.e. **4 H + 2 CNOT + 2 SWAP + 1 CD** per coupling. Costing it:

| part | count | unit (us) | subtotal (us) | status |
|---|---:|---:|---:|---|
| H (carrier pi/2) | 4 | 2.5 | 10.00 | **ASSUMED** t_pi = 5.0 us at Omega/2pi = 100 kHz |
| CNOT | 2 | 200 | 400.00 | **ASSUMED** -- *no CNOT time is anchored anywhere in this repo*; priced at T_MS since a TI CNOT is one MS plus locals |
| CD (link) | 1 | 2.5664 | 2.5664 | derived (T_QCD) |
| SWAP | 2 | -- | 0 | convention-only: the source calls the SWAPs a Vu hardware-mapping convention, *"equivalent under PennyLane"*, so they are not charged |
| **total per Jl** | | | **412.57** | vs the **200 us** placeholder |

Decomposed Jl costs **412.57 us**, i.e. **2.06x** the 200 us placeholder -- it is *more* expensive, not less, and the excess is almost entirely the two CNOTs. Step time would move 2396.11 -> 2821.24 us at N_c = 3.

| convention | T_qubit (us) | sequential ratio | parallel-MS ratio |
|---|---:|---|---|
| Jl = 200 us placeholder (**adopted**) | 2396.11 | [5.43x, 6.02x] | [3.16x, 3.51x] |
| Jl decomposed (**not adopted**) | 2821.24 | [6.39x, 7.09x] | [3.64x, 4.04x] |

**Not adopted** because the dominant term is an **ASSUMED** CNOT time that no anchor in this repo supports. Adopting it would move the headline further in the qudit's favour on the strength of an unanchored number, which is the wrong direction to be sloppy in. Anchoring a TI CNOT/CZ duration would settle it.

**Parallel-MS (Figgatt) variants.** `MS×3` is the **paper-facing** number: the 3(N_c−1) MS-type gates of a step split as (N_c−1) two-qubit J forward, (N_c−1) three-qubit Jl in the middle at full τ, (N_c−1) J reverse; merging across the middle block would require the commutation Trotterisation assumes absent, so **3 windows is the defensible floor**. `MS×1` is the **most qubit-favourable bound and is likely unphysical** — besides the forward/middle/reverse split, the J gates are A↔R for every R and therefore all share qubit A (a shared-control fan-out, not the disjoint pairs Figgatt et al. parallelise). It is retained only as a lower bound on T_qubit.

## Trotter steps until motional coherence limit τ_mot = 27.4 ms

| Nc | steps_qudit (@ T_QT low → high) | steps_qubit seq | steps_qubit MS×3 (paper) | steps_qubit MS×1 (bound) |
|---:|---|---:|---:|---:|
| 3 | [62, 68] | 11 | 19 | 27 |
| 6 | [30, 35] | 4 | 15 | 19 |
| 8 | [23, 26] | 3 | 13 | 16 |
| 12 | [15, 17] | 2 | 10 | 12 |

## Notes

- Every input gate time is read (not retyped) from the sources cited in the Inputs table; the runtime is computed as `Σ op_count × T_op` per Trotter step.
- QCD α=0.008 T is linear-scaled from the α=0.5 GRAPE headline (final_v3.json) using the Liu et al. PRX Quantum 7, 010201 (2026) linear-in-α scaling; flagged **overhead-limited** because the amp bound never binds at α=0.008 (FINDING 1).
- QT-SB θ=0.0015 T is the analytic minimum `θ/SIDEBAND_BOUND`; flagged **overhead-limited** for the same reason.
- QT-SB uses the HyPulse GRAPE `T_analytic` single value (θ/SIDEBAND_BOUND); the archived `[0.263, 87.67]` bounds came from a different per-pulse-overhead treatment and are superseded.
- QT is bounded [low, high] = [0.105, 11.0] µs, both from the aggregate-field-capped qutip-qtrl GRAPE run (`step_qt_meth_aggcap.py`): high = calibrated θ=π/4, low = angle-scaled to the circuit angle 0.0075 rad.  T_QT scales as 1/Ω and Ω_max/2π = 100 kHz is ASSUMED, so both bounds rescale with any revision of that assumption.
- Figgatt variant: all MS-type gates within a Trotter step are batched into a single MS window (`T_MS = 200 µs`), removing the `3(Nc-1)` multiplicity in the qubit encoding.  Applied only to the qubit side.
- QPS and rz are virtual (frame-update; T = 0).
- MS-angle-independent per Meth SI Note II p.3: the entire runtime asymmetry rests on this Meth-quoted simplification.
