# Appendix table — final consolidated ISA-gate parameters

**Scope**: hardware-parameter-constrained **effective-Hamiltonian** runtime analysis (σ_z SDF, dispersive Kerr, sideband Rabi primitives from Bazavan / Meth / Choi / Liu, propagated by qutip-qtrl GRAPE against imposed target unitaries).  This is NOT a full lab-frame Meth-Hamiltonian pulse-level validation.  See the **Limitations** block at the end of this file.

**Every number in the paper must trace to this table.**

**Pulse ramp-up and ramp-down are not modelled. Bazavan et al. apply sin² ramps long compared to 1/δ. Including them would increase the reported durations.**

## HyPulse GRAPE (imposed σ plugin) — headline

### Two operating points (α = 0.5, cutoff = 8, num_tslots = 200, F ≥ 0.99999)

| operating point | tag | Ω_c/2π | Ω_sb/2π | x_qcd = 4Ω_c/ω_z | J1+J3 | amp_QCD (rad/µs) | amp_CD (rad/µs) | T_QCD | T_CD | **T_QCD/T_CD** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. Meth** | METH_OPERATING_POINT | 97.56 kHz | 4 kHz | 0.507 | 0.248 | 0.006233 | 0.02513 | **160.4 µs** | **39.8 µs** | **4.03** |
| **B. Bazavan optimum** | BAZAVAN_DEMONSTRATED_RANGE | 308 kHz | 12.6 kHz | 1.600 | 0.642 | 0.05098 | 0.07934 | **19.6 µs** | **12.6 µs** | **1.56** |

Amp bounds: `amp_QCD = η·Ω_c·(J1+J3)(x_qcd)` (Bazavan σ_z SDF, derived); `amp_CD = η·Ω_c` (standard symmetric-bichromatic MS, no leading-order Bessel modulation).

Meth is the **conservative** row (within Meth's demonstrated Ω_sb = 4 kHz calibration). Bazavan-optimum requires Ω_sb ≈ 12.6 kHz — inside Bazavan's own demonstrated carrier range (140 kHz – 1.17 MHz, PRA 107, 022617) but ~3× above Meth's operating point.

| gate | α | T (µs) | F_grape | notes |
|---|---:|---:|---:|---|
| QCD | 0.008 | — | — | duration-meaningless; amp bound never binds |
| CD  | 0.008 | — | — | duration-meaningless; amp bound never binds |

**FINDING 1** — At α = 0.008 the amp bound is not binding: F ≥ 0.998 for every T from 0.5 to 20 µs. Duration set by unknown per-pulse hardware overhead (`not_in_meth`), not by physics.

**FINDING 2** — QCD and CD give identical T and F in HyPulse GRAPE *when both use the same amp bound* (imposed plugin: basis-rotation only). But the physical amp bounds are NOT equal: `amp_QCD = η·Ω_c·(J1+J3)`, `amp_CD = η·Ω_c`. Using each gate's own physical bound produces the different T's tabulated above. See FINDING 4 for the ratio's Bessel dependence.

**FINDING 4 (σ_z-penalty is Bessel-dependent, not fixed)** — The σ_z-penalty T_QCD/T_CD is not a hardware constant; it is `1/(J1+J3)(x)` with `x = 4Ω_c/ω_z`. At Meth's operating point (x = 0.507) it is 4.03×; at the Bazavan optimum (x = 1.6) it drops to 1.56×. Empirical HyPulse GRAPE ratios match analytic prediction to sub-µs. Alternative hypothesis `(J0+J2)/(J1+J3)` was tested and DOES NOT match: it gives 3.90 at Meth (coincidentally close because `J0+J2 ≈ 1` at small x) and 1.11 at Bazavan (misses the ~1.5×). The physics: standard CD via symmetric-bichromatic MS at δ = ω_z has no leading-order Bessel modulation; only QCD's antisymmetric drive at δ = ω_z/2 carries the `(J1+J3)` derating. Data: [hypulse_validation/step_two_points_v4.py](hypulse_validation/step_two_points_v4.py).

## HyPulse GRAPE — QCR and CR (imposed σ_z ⊗ n̂ / Π_l ⊗ n̂ plugin, CHI_CORRECTED bound)

### Two operating points × two gates × two θ (all 8 GRAPE runs, F = 1.0)

**Headline = Meth rows only.** Choi rows are a **sensitivity check, not headline** — they show what a stronger χ (measured on a different device, 171Yb⁺) buys you; they do not enter the paper's per-Trotter runtime.

| point | tag | gate | θ | χ/2π | amp_bound = χ (rad/µs) | **T (µs)** | **T_paper = 2T + t_π (µs)** | A | C |
|---|---|---|---:|---:|---:|---:|---:|:---:|:---:|
| A. Meth (**headline**) | METH_OPERATING_POINT | CR  | 0.016 | 100 Hz | 6.283e-4 | 25.465 | **55.93** | PASS | PASS |
| A. Meth (**headline**) | METH_OPERATING_POINT | CR  | 0.500 | 100 Hz | 6.283e-4 | 795.775 | 1596.55 | PASS | PASS |
| A. Meth (**headline**) | METH_OPERATING_POINT | QCR | 0.016 | 100 Hz | 6.283e-4 | 25.465 | **55.93** | PASS | PASS |
| A. Meth (**headline**) | METH_OPERATING_POINT | QCR | 0.500 | 100 Hz | 6.283e-4 | 795.775 | 1596.55 | PASS | PASS |
| B. Choi (**sensitivity check, not headline**) | CHOI_MEASURED | CR  | 0.016 | 471 Hz | 2.959e-3 | 5.407 | 15.81 | PASS | PASS |
| B. Choi (**sensitivity check, not headline**) | CHOI_MEASURED | CR  | 0.500 | 471 Hz | 2.959e-3 | 168.954 | 342.91 | PASS | PASS |
| B. Choi (**sensitivity check, not headline**) | CHOI_MEASURED | QCR | 0.016 | 471 Hz | 2.959e-3 | 5.407 | 15.81 | PASS | PASS |
| B. Choi (**sensitivity check, not headline**) | CHOI_MEASURED | QCR | 0.500 | 471 Hz | 2.959e-3 | 168.954 | 342.91 | PASS | PASS |

- Meth χ/2π = (η·Ω_c)² / (4·δ_disp) = 100.0 Hz **DERIVED** from Ω_sb = 4 kHz, δ_disp = 40 kHz.
- Choi χ/2π = 471 Hz **MEASURED** (arXiv:2601.04591; overrides analytic 84 Hz — 171Yb⁺ off-resonant multi-level Rabi contributions).
- `t_pi = π/Ω = 5 µs at Ω/2π = 100 kHz`, **ASSUMED** (Ringbauer-class carrier; Ω not measured in Meth).
- CR θ = 0.5 required a `u(t) = +χ` warm-start (`init_pulse_type='LIN'` + `pulse_scaling=0` + `pulse_offset=+χ`) to escape the σ_z sign-ambiguity local minimum. See [hypulse_validation/step_qcr_cr_two_points_v3_cr_theta05.py](hypulse_validation/step_qcr_cr_two_points_v3_cr_theta05.py).

**Paper headline for the runtime table**: **T_QCR = 55.93 µs at θ = 0.016, Meth operating point** — unchanged; now HyPulse-GRAPE-verified with all audit checks passing.

**FINDING 5 (final)** — Paper QCR is Π_l-conditioned: `U_QCR(θ, l) = exp(−iθ Π_l ⊗ n̂)` with `Π_l = |l⟩⟨l|`. Hardware light shift natively gives the symmetric σ_z / Ξ_l form; the difference between Π_l = (I − σ_z)/2 and −σ_z/2 is an **unconditional mode rotation, absorbed into the free mode frame**, so **T_QCR = T_CR = θ/χ**. Spectator levels inert, verified at d=3 (F = 1, spectator phase 0; [hypulse_validation/qcr_d3_spectator_audit.json](hypulse_validation/qcr_d3_spectator_audit.json)). Note θ conventions differ by 2× in conditional phase between the two forms; **all paper numbers use the Π_l convention**.

**QCR spectator audit (d=3 qudit × Fock 8, Π_l convention):** at θ=0.016 (Meth, T=25.465 µs) gate fidelity = 1.0000000000, max\|U−U_t\| = 9.20×10⁻⁷, spectator \|s⟩ phase slope = 0 rad/photon; at θ=0.5 (T=795.775 µs) F = 1.0000000000, max\|U−U_t\| = 1.25×10⁻⁶, spectator slope = 0.  See [hypulse_validation/qcr_d3_spectator_audit.py](hypulse_validation/qcr_d3_spectator_audit.py) + `.json`.

## HyPulse GRAPE — QT-SB (Rabi sideband interaction; Liu R_B primitive)

Pipeline uniformity only — no novelty claim. Reference: Liu et al., *PRX Quantum* **7**, 010201 (2026), R_B primitive. Target: `U_QTSB(θ) = exp(-i θ (|i⟩⟨j|+|j⟩⟨i|) ⊗ X)`, qubit pair `{|0⟩, |1⟩}`, `X = (a+a†)/2`; generator `σ_x ⊗ X` (single-control imposed).

### Meth operating point (all F = 1.000000; Ω_sb/2π = 4 kHz)

| θ | source | amp bound | amp bound (rad/µs) | T_analytic (µs) | T_grape (µs) | F_grape | A | C |
|---:|---|---|---:|---:|---:|---:|:---:|:---:|
| **0.0015** | circuit (Nc=3; any Nc): `|J_AB0·η_AB·TAU|` from [qudit_qumode/params.py:16-17](../params.py#L16-L17) | **SIDEBAND_BOUND** (Ω_sb/1e6) | 0.025133 | 0.059683 | **0.059683** (paper — **lower bound; rise time and per-pulse overhead dominate, `not_in_meth`**) | 1.000000 | PASS | PASS |
| **0.5**   | stress test | **SIDEBAND_BOUND** (Ω_sb/1e6) | 0.025133 | 19.894368 | 19.894368 | 1.000000 | PASS | PASS |

**Paper-facing T at θ=0.0015 is T_analytic = 0.059683 µs** (where ∫u dt = θ exactly, F = 1.000000). The sweep picker's earlier `T = 0.056699 µs` row was a **sweep picker artifact, not paper-facing** — F ≥ 0.99999 was hit early there because at small θ the identity solution has F ≈ 1, but ∫u dt = 0.00143 is 5 % short of θ, so the physical rotation angle is short. Cutoff = 8; num_tslots = 200; init `pulse_type='LIN'` with `pulse_scaling=0`, `pulse_offset=+amp_bound` (warm-start constant +amp_bound). See [hypulse_validation/step_qtsb_meth.py](hypulse_validation/step_qtsb_meth.py) + [step_qtsb_meth_results.json](hypulse_validation/step_qtsb_meth_results.json) + `.log`.

**FINDING 6 (CR plugin has QCD-class amp-bound bug, different mechanism)** — The existing CR plugin ([hypulse/sim/hamiltonians/cr.py](../../../hypulse/src/hypulse/sim/hamiltonians/cr.py)) uses **raw BSB** (σ_x ⊗ X + σ_y ⊗ Q), NOT target-typed-in σ_z ⊗ n̂. Default `amp_bound = 62.83 rad/µs` (= 10 MHz carrier Rabi) is ~100× above Meth's Ω_c/2π = 97.56 kHz → GRAPE would find fast non-dispersive solutions that beat the physical χ = (η·Ω_c)²/(4·δ_disp). Fix: **imposed treatment with `amp_bound = χ`** (tag CHI_CORRECTED). Analogous to QCD's ETA_AND_BESSEL_CORRECTED fix, but the physical origin of the bound is different (dispersive χ vs Bazavan Ω_eff). The imposed treatment lives in the pulses_meth analysis script; the CR plugin itself was not modified.

**FINDING 3** — Previous headline T = 40 µs (Bessel-only amp bound = 0.152 rad/µs) is superseded. The Bessel-only bound was 24× above the physical Bazavan Ω_eff. Corrected bound `amp_bound_rad_per_us_eta_bessel_corrected = 0.248·Ω_sb / 1e6 = 0.006233 rad/µs` (one factor of η missing in v2) gives T = 160.4 µs.

**Why 160 µs and not 80 µs — physical origin of the 1/2** — Bazavan Eq. 3 carries a `sin(2δt)` prefactor that beats into resonance with the mode at δ = ω_z/2:

$$
\sin(\omega_z t) \cdot \bigl(a\,e^{-i\omega_z t} + a^\dagger e^{+i\omega_z t}\bigr) \;=\; \underbrace{\tfrac{1}{2i}(a - a^\dagger)}_{\text{resonant (RWA keeper)}} \;+\; \underbrace{\tfrac{1}{2i}\bigl(a^\dagger e^{+2i\omega_z t} - a\,e^{-2i\omega_z t}\bigr)}_{\text{oscillates at }2\omega_z\text{ (RWA drop)}}
$$

Rotating-wave averaging keeps only one of the two counter-rotating components of the sinusoid, yielding a factor 1/2. The surviving Hamiltonian is `H_eff = Ω_eff · σ_z · P` with `P = (a-a†)/(2i)`, giving per-branch β̇ = Ω_eff/2 and analytic floor `T_min = 2α/Ω_eff = 160 µs`. The plugin's `X = (a+a†)/2` convention reproduces the same 1/2 (the two "factors of 2" are the SAME 1/2 in different accounting). RWA condition `2ω_z ≫ Ω_eff` well satisfied at Innsbruck (1540 kHz ≫ 0.99 kHz, ratio 1500×).

Symbolic derivation and numeric cross-check (Eq. 3 full sin(2δt) → RWA reduction → Eq. 1 full lab-frame): all three agree on per-branch β̇ = Ω_eff/2 to ≤5% (residual is the finite-time 2ω_z ripple in the non-RWA'd forms). See [hypulse_validation/derive_half_factor.py](hypulse_validation/derive_half_factor.py). Options A (redefine X := a+a†) and B (double the amp bound) would give T = 80 µs, but only by changing a convention, not the physics.

Audit on the v3 waveform: CHECK A — `max|u_p| = 100.0%` of bound (binding); CHECK C — `|β|_measured = 0.4996` vs α = 0.5, F_indep = 0.99999.

## QCD under Bazavan full time-dependent Hamiltonian (post-optical-RWA lab check)

Same QCD `final_v3` waveform, now propagated by `qutip.sesolve` under the full time-dependent Bazavan Eq. 1 with only the optical (single-photon) RWA applied — `cos(δt)` carrier envelope and `e^{±iω_z t}` mode oscillations kept explicit. Cutoff-converged over `N_cut ∈ {8, 10, 12}`. Effective Ω_c(t) built per-slot by inverting `u_p(t) = η·(J1+J3)·Ω_c(t)` from the GRAPE waveform; `u_x = 0` (artefact per QCD u_x zero-out check). Details: [hypulse_validation/step_qcd_bazavan_full_lab.py](hypulse_validation/step_qcd_bazavan_full_lab.py) + `_ramps.py`.

**Sign convention**: Bazavan Eq. 1 at drive phase φ_drive = 0 emits the σ_z SDF with sign inverted vs the plugin's convention (matches TEST A's `rate_X_up < 0`). The comparison target used here is `U_QCD = |0⟩⟨0|⊗D(−α) + |1⟩⟨1|⊗D(+α)` — equivalent to running Bazavan at **φ_drive = π**. Once fixed, results are convention-clean.

| variant | T (µs) | F_raw (N_cut 8/10/12) | F after unconditional frame fit (mode φ + SU(2) spin) | fitted σ_y angle (rad) | Magnus-1st analytical σ_y (rad) | correction cost |
|---|---:|---:|---:|---:|---:|---|
| **unramped** | 160.4 | 0.936 / 0.934 / 0.934 | **0.999 / 0.999 / 0.998** | −0.5056 / −0.5114 / −0.5130 | **−0.5066** (matches fit to <1.3 %) | one σ_y rotation, **0 µs** absorbed into adjacent QT; 0.827 µs standalone |
| **sin² ramps** (5 µs each side) | 165.4 | 0.999 / 0.998 / 0.998 | **0.999 / 0.999 / 0.999** | +0.0361 / +0.0364 / +0.0368 | +0.02654 (Magnus 1st; higher-order gap +0.01 rad) | one σ_y rotation, **0 µs** absorbed into adjacent QT; 0.043 µs standalone |

**Interpretation**:
- **Intrinsic physical fidelity F_intrinsic ≈ 0.999** (the part not removable by an unconditional frame rotation), stable across N_cut, RWA-limit from finite Ω_c/δ = 0.25.
- **σ_y residual is deterministic** (Magnus-1st predicts unramped to <1 %; ramped sign and OoM), so it is not stochastic error — one deterministic qubit rotation cancels it. **Correctable via adjacent QT phase, 0 µs.**
- **Ramps trade +3.1 % T for a 14× smaller σ_y residual** but do not improve F_intrinsic. Not a fidelity improvement, an ergonomics improvement.

Files: [step_qcd_bazavan_full_lab.py](hypulse_validation/step_qcd_bazavan_full_lab.py) + `_results.json`; [step_qcd_bazavan_full_lab_ramps.py](hypulse_validation/step_qcd_bazavan_full_lab_ramps.py) + `_results.json`; [magnus_sigma_y_prediction.json](hypulse_validation/magnus_sigma_y_prediction.json).

## QT carrier gate on the shared optimizer stack

Removes the archived non-HyPulse weak link. **Treatment (A), imposed**: direct `qutip.control.pulseoptim.optimize_pulse_unitary` (`alg="GRAPE"`, 200 slots, seed 42) with QT generators and `innsbruck_meth_2025` anchors — the same optimizer HyPulse wraps, **not** routed through the plugin registry (HyPulse registers only CR, CD, QCD, CS; there is no QT plugin).

**Model** — 5-dim: code {G, A, B, C} + S₁/₂ auxiliary |g⟩. Transitions go *through* the auxiliary (no direct D₅/₂↔D₅/₂ quadrupole coupling), two tones on {G, A}, four real controls (2 tones × 2 quadratures). Target `U = exp(−i·(π/4)·(|G⟩⟨A|+|A⟩⟨G|))`, spectators B, C, no mode.

**Bound — the paper number uses the aggregate cap.** `anchor_params.json` reads *"Rejects any pulse whose Omega_peak/2pi > 100 kHz"*, which is taken to bound the **total** laser field: `√(Σ_k |Ω_k|²) ≤ Ω_max`. With four real quadrature controls each bounded by `b`, the aggregate is `≤ 2b`, so the guaranteeing per-quadrature bound is `b = Ω_max/2 = 0.31416 rad/µs`. A **per-tone** variant (`b = Ω_max/√2`, each tone ≤ Ω_max, aggregate up to √2·Ω_max = 141 kHz) is retained below for comparison — it is the archive's convention.

| quantity | **paper (aggregate cap)** | per-tone-cap variant |
|---|---:|---:|
| bound / quadrature | Ω_max/2 = 0.31416 rad/µs | Ω_max/√2 = 0.44429 rad/µs |
| **T (picker threshold)** | **11.0 µs** | 8.0 µs |
| F_grape | 0.999990 | 0.9999914472933119 |
| F_indep (independent qutip reprop) | 0.999981 | 0.9999828948081472 |
| F_code (4-dim code block, VZ-absorbed) | — | 0.9999769846144015 |
| auxiliary leakage `max_c P(\|g⟩)` | 8.57×10⁻⁶ | 3.75×10⁻⁶ |
| spectator_dF (true \|Ω(t)\|² Stark − frozen) | +3.1×10⁻¹⁵ | −8.5×10⁻¹⁵ |
| per-tone max \|Ω_k\| | 70.7 % of Ω_max | 100.0 % of Ω_max |
| **aggregate max √(Σ\|Ω_k\|²)** | **100.0 % of Ω_max** | 141.4 % of Ω_max |
| termination | Goal achieved | Goal achieved |

Picker: `F_grape ≥ 0.99999` **and** aux leakage `< 10⁻³` **and** `|spectator_dF| < 10⁻³` **and** aggregate `≤ Ω_max`. Threshold is sharp and monotone in both conventions — aggregate cap: 10.50 µs → 0.999649 (leak 3.4×10⁻⁴), 10.0 µs → 0.997714, 9.0 µs → 0.989501.

**Sanity check against the bright-state picture.** Two resonant tones on |G⟩↔|g⟩ and |A⟩↔|g⟩ form a Λ system; only the bright combination couples to |g⟩, at `Ω_tot = √(Ω_G²+Ω_A²)` — which *is* the aggregate field. A bright-state 2π pulse costs `2π/Ω_tot`: **10.000 µs** under the aggregate cap, **7.071 µs** under the per-tone cap. Both GRAPE results (11.0 µs, 8.0 µs) sit just above their respective floors, as they must. The two conventions differ by exactly the √2 in the floor.

**Spectator AC-Stark is exactly a virtual-Z, cost 0 µs.** The frozen Stark drift is diagonal on {B, C}; the drive has matrix elements only among {G, A, g}. Disjoint subspaces ⇒ `[H_stark, H_drive] = 0` identically, so `U(T) = exp(−i H_stark T)·U_drive(T)` factorises and the spectator Stark contributes nothing but a diagonal phase. Meth compensates exactly this in software (`anchor_params.json` → `ac_stark_spectator_shifts_kHz._note`: *"Compensated in software by frame updates"*), and the ISA prices it at `T_QPS = 0`. Measured: `spectator_dF = −8.5×10⁻¹⁵` (machine zero) at every T in the sweep. The virtual-Z fit is doing real work, not trivially vanishing — in the η-corrected variant the fitted spectator phase accumulates linearly as `φ_B = −0.025085·T`, reaching −1.505 rad at T = 60 µs.

**Stark reference — flagged discrepancy in the archived model.** The archive sets `OMEGA_REF_STARK = 2π·4 kHz`, treating Meth's quoted 4 kHz as the *carrier* Rabi. It is the *sideband* Rabi (η·Ω_c), so the carrier at the point where the shifts were measured was 4 kHz/0.041 = **97.56 kHz** — the same FINDING-3 class of error (missing η). This changes the frozen Stark scale factor from **1250** to **2.10**, i.e. the drift on |B⟩ from 14.92 to 0.0251 rad/µs, a ~600× difference. **Both variants were run and give identical T and F to six decimals** (8.0 µs, F = 0.999991), because the Stark commutes with the drive and is removed by the free frame update — so the error is *inert for QT*. It would not be inert for any gate where the shifted levels are driven.

**Comparison to the archived 36 µs.** Not a like-for-like disagreement — three differences, in order of size: (i) **control resolution**, 200 slots here vs 20 slices in the archived npz (the archived README says 16, a discrepancy already noted in [validation_report.md](hypulse_validation/validation_report.md)); (ii) **acceptance rule**, the archive's headline required the amplitude bound *not* be saturated with ≥ 5 % margin, whereas this run saturates it — the archive's own "saturating_floor" row reaches F ≥ 0.99 at 15 µs; (iii) **optimizer**, L-BFGS-B with a leakage penalty folded into a hand-rolled cost vs qutip-qtrl GRAPE with analytic gradients and leakage measured as a separate audit. The 11.0 µs number is therefore *faster under a more permissive acceptance rule and 10× finer control*, not a contradiction of the archived run. **The Ringbauer anchor is unaffected** — it sets the per-Clifford *error*, not a duration, so it constrains fidelity, not T.

Files: [step_qt_meth_aggcap.py](hypulse_validation/step_qt_meth_aggcap.py) + [`_results.json`](hypulse_validation/step_qt_meth_aggcap_results.json) (paper, aggregate cap); [step_qt_meth.py](hypulse_validation/step_qt_meth.py) + [`_results.json`](hypulse_validation/step_qt_meth_results.json) (per-tone variant, full audit incl. F_code and VZ phases).

## Open-system check under Meth decoherence

Lindblad `mesolve` on the **effective** Hamiltonians (not the Bazavan full H), Fock cutoff 8. Every headline gate re-run with the three Meth-anchored decoherence channels below; closed-system `F = 1.000000` recovered for every gate when the channels are switched off, confirming H/target consistency.

### Collapse operators and rates

| channel | Meth parameter | rate | collapse operator |
|---|---|---:|---|
| motional heating | `n_dot = 2.7 phonons/s` (Meth Methods p.8) | γ_h = n_dot = **2.7000 s⁻¹** | `L₁ = √γ_h · I₂ ⊗ a†` |
| motional dephasing | `τ_mot = 27.4 ms` (Meth Methods p.8) | κ_mot = 2/τ_mot = **72.993 s⁻¹** | `L₂ = √κ_mot · I₂ ⊗ n̂` |
| spin (optical) dephasing | `T₂ = 92 ms` (Meth EDF 2a) | γ_φ = 1/(2T₂) = **5.4348 s⁻¹** | `L₃ = √γ_φ · σ_z ⊗ I_m` |

No cooling operator is included: Meth quotes a heating rate only, and the gates run near vacuum. Adding `L = √γ_h·a` would impose an equilibrium at ⟨n⟩ = 1, which the anchor does not support.

**τ_mot INTERPRETATION FLAG (factor ~2).** Meth's `τ_mot = 27.4 ms` is interpreted here as the **1/e amplitude-coherence time** of the mode (equivalently the adjacent-Fock coherence time), giving `κ_mot = 2/τ_mot`. If Meth's number is instead the **population** 1/e time, κ_mot halves (72.99 → 36.50 s⁻¹) and the mode-dephasing share of the infidelity budget — which dominates every long gate in the table below — drops by ~2×. This is an interpretation of Meth's number, not a measured distinction, and it is the single largest source of uncertainty in the open-system column.

### Headline: code-subspace fidelity

Input set = 6 basis states `|q⟩⊗|n⟩`, `q ∈ {0,1}`, `n ∈ {0,1,2}`; metric `F_code = (1/6)·Σᵢ ⟨ψᵢ,ideal|ρᵢ(T)|ψᵢ,ideal⟩`. **These are the paper-facing open-system numbers.**

| gate | T (µs) | **F_code (open)** | F_code (closed) |
|---|---:|---:|---:|
| **QCD** (α = 0.5) | 160.400 | **0.9962** | 1.000000 |
| **CD** (α = 0.5) | 39.790 | **0.9990** | 1.000000 |
| **QCR** (θ = 0.016, paper) | 25.465 | **0.9999** | 1.000000 |
| **QT-SB** (θ = 0.0015, paper) | 0.0597 | **1.0000** | 1.000000 |
| QCR (θ = 0.5, stress) | 795.775 | 0.9957 | 1.000000 |
| QT-SB (θ = 0.5, stress) | 19.894 | 0.9998 | 1.000000 |

QCD is **cutoff-converged**: N_cut = 8 and N_cut = 10 both give `F_code = 0.996232` to six decimals.

### Supplement: full-dim fidelity, and why it differs

Full-dim uses `qutip.average_gate_fidelity` (Nielsen `(d·F_pro + 1)/(d+1)`) over the whole Hilbert space, **d = 16** at N_cut = 8 (d = 20 at N_cut = 10).

| gate | T (µs) | F_code (open) | F_full (open) | gap |
|---|---:|---:|---:|---:|
| QCD (α = 0.5), N_cut = 8 | 160.400 | 0.996232 | 0.943852 | 5.2×10⁻² |
| QCD (α = 0.5), N_cut = 10 | 160.400 | 0.996232 | 0.914873 | 8.1×10⁻² |
| CD (α = 0.5) | 39.790 | 0.998950 | 0.985346 | 1.4×10⁻² |
| QCR (θ = 0.016) | 25.465 | 0.999863 | 0.990568 | 9.3×10⁻³ |
| QCR (θ = 0.5, stress) | 795.775 | 0.995714 | 0.782348 | 2.1×10⁻¹ |
| QT-SB (θ = 0.5, stress) | 19.894 | 0.999779 | 0.992612 | 7.2×10⁻³ |
| QT-SB (θ = 0.0015, paper) | 0.0597 | 1.000000 | 0.999978 | 2.2×10⁻⁵ |

**Metric-artifact note.** `average_gate_fidelity` averages over all input states in the truncated Hilbert space, including high-Fock states `|n⟩` near the cutoff that the gates never populate and that accumulate far more heating and mode dephasing than the code states. The QCD row is the clean demonstration: `F_code` is **identical** (0.996232) at N_cut = 8 and 10, while `F_full` moves 0.944 → 0.915 purely because d grew from 16 to 20 and the added `|n = 8, 9⟩` inputs are decoherence-heavy. **The full-dim column therefore measures decoherence on states outside the code and is an error-budget diagnostic, not a gate-quality figure.** Quote `F_code`.

Files: [step_open_meth_decoherence.py](hypulse_validation/step_open_meth_decoherence.py) + [`_results.json`](hypulse_validation/step_open_meth_decoherence_results.json) (full-dim); [step_open_meth_code_subspace.py](hypulse_validation/step_open_meth_code_subspace.py) + [`_results.json`](hypulse_validation/step_open_meth_code_subspace_results.json) (code subspace, incl. per-input fidelities).

## Per-gate row

| gate | mechanism | T_actual (µs) | T_nominal (µs) | anchor source | experimental or theory | assumed parameters used |
|---|---|---:|---:|---|---|---|
| **QPS** | virtual-Z / phase-frame update on the drive addressing the target level | **0** (free) | 0 | McKay et al. PRA 96, 022330 (2017); Meth Methods p.8 "compensated in software by frame updates" | **experimental** (Meth uses this natively) | none |
| **QT** | resonant two-tone carrier through S1/2 auxiliary \|g⟩, single ion | **[0.105, 11.0]** (bounded) | 11.0 at θ=π/4 | Ringbauer et al. Nat. Phys. 18, 1053 (2022) for the **error** anchor; duration from [hypulse_validation/step_qt_meth_aggcap.py](hypulse_validation/step_qt_meth_aggcap.py) | **error anchor experimental** (Ringbauer per-Clifford); **duration derived** | Ω_max/2π = 100 kHz **ASSUMED** (carrier Rabi not in Meth); per_pulse_hardware_overhead not_in_meth (sub-µs floor). Bounds: high = 11.0 µs (calibrated θ=π/4, **aggregate** field cap √(Σ\|Ω_k\|²) ≤ Ω_max); low = 0.105 µs (angle-scaled to the circuit angle 0.0075 rad, overhead-limited). **Status: now on the same optimizer stack as every other gate** (qutip-qtrl GRAPE, imposed generators, `innsbruck_meth_2025` anchors) — the archived L-BFGS-B run (36 µs, 20 slices, [archive/v1_pergate/QT/](archive/v1_pergate/QT/)) is **superseded**. A per-tone-cap variant gives 8.0 µs — see QT section. |
| **QCD** | antisymmetric bichromatic σ_z SDF, single ion, ETA_AND_BESSEL_CORRECTED bound | **2.57** (α=0.008; = 160.4 × 0.008/0.5 linear-scaled from HyPulse GRAPE α=0.5) | 79.6 (α=1) | Meth main text Ω_BSB=4 kHz + Bazavan PRA 107, 022617 (mechanism) + HyPulse GRAPE ([final_v3.json](hypulse_validation/final_v3.json)) | **derived** (HyPulse GRAPE headline + Liu α-linear scaling) | T scales linearly with \|α\| per Liu et al. PRX Quantum 7, 010201 (2026). **Prior row "0.700" (α=0.008) archived, superseded** ([archive/v1_pergate/QCD/](archive/v1_pergate/QCD/)) |
| **QT-SB** | symmetric bichromatic σ_x SDF (R_B Rabi interaction), single ion; SIDEBAND_BOUND | **0.0597** (θ=0.0015, T_analytic = θ/Ω_sb; **lower bound; rise time and per-pulse overhead dominate, `not_in_meth`**) | 19.9 (θ=0.5) | Meth Ω_BSB + Liu et al. PRX Quantum 7, 010201 (2026) R_B + HyPulse GRAPE ([step_qtsb_meth_results.json](hypulse_validation/step_qtsb_meth_results.json)) | **derived** (θ from vibronic circuit, bound = Ω_sb from Meth) | k_tones=2, T_rise=100 ns ASSUMED; angle scaling per Liu et al. **Prior row "[0.263, 87.67]" bounds archived, superseded** ([archive/v1_pergate/QT-SB/](archive/v1_pergate/QT-SB/)) |
| **QCR** | off-resonant BSB dispersive Kerr with Choi 2-segment AC-Stark cancellation + microwave π | **55.93** (θ=0.016, incl. cancellation) | 25.46 (single-seg) | Meth Ω_BSB=4 kHz + **Choi arXiv:2601.04591** (validates χ formula within 2σ) | **experimental** (Choi anchors χ formula) | δ_disp/2π = 40 kHz (chosen for SW error<1%); microwave π = 5 µs ASSUMED; Liu et al. PRXQ 7, 010201 (2026) confirms no direct SNAP on trapped ions |
| **Rotation** (not one of the 5 ISA gates but appears in the count) | unconditional single-mode drive (RF trap-freq modulation or resonant carrier on mode) | **5** | 5 | none — not covered by any of the 5 ISA gates | **assumed** | 5 µs is ASSUMED entirely; no anchor |
| **MS** (qubit encoding only) | Mølmer–Sørensen entangling gate on two ions | **200** (fixed, angle-independent) | 200 | **Meth SI Note II** p.3: "qubit gate complexity (0.2 ms per gate)" — angle-independent per Meth's own framing | **experimental** (Meth-quoted) | MS angle-independence is Meth's own simplification; the entire runtime claim rests on this asymmetry — see paper §Runtime |

## Circuit-level runtime (per Trotter step)

<!-- BEGIN GENERATED: step_runtime_table.py rewrites this section from the same `rows` object that renders runtime_table.md and runtime_table.json. Do not hand-edit. -->

Single-sourced with [runtime_table.md](runtime_table.md) and [runtime_table.json](runtime_table.json), all three rendered from one `rows` object in [hypulse_validation/step_runtime_table.py](hypulse_validation/step_runtime_table.py). Gate times (this file's own rows above): QCD = 2.5664 µs, QCR = 55.93 µs, QT-SB = 0.6494 µs, Rotation = 5.0 µs, MS = 200.0 µs, QT ∈ [0.105, 11.0] µs.

**Convention: J = 2 MS windows** (`[XX,YY] = 0`, exact — see `results/j_ms_convention.txt`); H_Jl held at 1 window. Parallel-MS variants use **5 windows** (defensible floor) and **3 windows** (most qubit-favourable bound, likely unphysical).

| N_c | d | T_qudit range (µs) | T_qubit seq (µs) | T_qubit MS×5 (µs) | T_qubit MS×3 (µs) | ratio seq | **ratio MS×5 (paper)** | ratio MS×3 (bound) |
|---:|---:|---|---:|---:|---:|---|---|---|
| 3 | 4 | [397.8, 441.4] | 2396.1 | 1396.1 | 996.1 | [5.43×, 6.02×] | **[3.16×, 3.51×]** | [2.26×, 2.50×] |
| 6 | 7 | [781.4, 890.3] | 5777.1 | 1777.1 | 1377.1 | [6.49×, 7.39×] | **[2.00×, 2.27×]** | [1.55×, 1.76×] |
| 8 | 9 | [1037.1, 1189.6] | 8031.1 | 2031.1 | 1631.1 | [6.75×, 7.74×] | **[1.71×, 1.96×]** | [1.37×, 1.57×] |
| 12 | 13 | [1548.5, 1788.2] | 12539.0 | 2539.0 | 2139.0 | [7.01×, 8.10×] | **[1.42×, 1.64×]** | [1.20×, 1.38×] |

**Parallel-MS variants.** `MS×5` is **paper-facing**: under J = 2 the MS-type gates split as forward J (XX + YY), the middle Jl at full τ, and reverse J (XX + YY) — a batch cannot merge an XX window with a YY window, so **5 windows is the defensible floor**. `MS×3` is the **most qubit-favourable bound and is likely unphysical**: besides that split, the J gates are A↔R for every R and so all share qubit A — a shared-control fan-out, not the disjoint pairs Figgatt et al. parallelise. It is retained only as a lower bound on T_qubit. **The qudit encoding is ahead at every N_c under all three assumptions.**

Range width is driven by the T_QT bounds [0.105 µs (angle-scaled to the circuit angle 0.0075 rad, overhead-limited), 11.0 µs (calibrated θ=π/4)]. The qudit step is dominated by QCR (2·N_c gates × 55.93 µs). MS-type gates on the qubit side = 2(N_c−1) J + (N_c−1) Jl per step, verified from [simulations/vibronic_scaling.py::build_one_hot_trotter_step](../simulations/vibronic_scaling.py).

**Alternative qudit convention (labelled extra, NOT the headline).** The `28`-operation count (7N_c+7) additionally fuses the QT gates into a single multi-transition operation. That fusion is *not* an operator identity: |A><B| and |A><C| share level A and max|[\Sigma_{AB},\Sigma_{AC}]| = 1.000000 (measured). It rests on a multi-tone hardware claim priced from `step_qt_meth_aggcap.py`. QPS fusion, by contrast, is exact (commuting diagonal projectors, measured 0.0), which is why the headline uses 34.

| N_c | T_qudit range, 28-op (µs) | ratio seq, 28-op |
|---:|---|---|
| 3 | [397.6, 419.4] | [5.71×, 6.03×] |
| 6 | [780.5, 802.3] | [7.20×, 7.40×] |
| 8 | [1035.8, 1057.6] | [7.59×, 7.75×] |
| 12 | [1546.4, 1568.2] | [8.00×, 8.11×] |

**The two advantages move in opposite directions with N_c — state this plainly rather than let a referee find it.** The sequential ratio *grows* (6.02× at N_c=3 → 8.10× at N_c=12) while the MS-batched ratio *shrinks* (3.51× → 1.64×). The reason is structural, not a modelling artefact: the qubit encoding's MS-type gates are 3(N_c−1) in number and batching collapses them to a **fixed** window count (5 or 3) independent of N_c, so parallelism helps the qubit side more and more as N_c grows. The qudit side has no such lever — its cost is dominated by QCR, 2N_c gates that are sequential on one ion and do not batch. So the qudit advantage is **robust but not uniformly improving**: it survives at every N_c under every batching assumption, and it is *largest* sequentially at large N_c and *smallest* under aggressive batching at large N_c.

### Trotter steps within the motional coherence limit τ_mot = 27.4 ms

| N_c | steps_qudit (@ T_QT low → high) | steps_qubit seq | steps_qubit MS×5 (paper) | steps_qubit MS×3 (bound) |
|---:|---|---:|---:|---:|
| 3 | [62, 68] | 11 | 19 | 27 |
| 6 | [30, 35] | 4 | 15 | 19 |
| 8 | [23, 26] | 3 | 13 | 16 |
| 12 | [15, 17] | 2 | 10 | 12 |

<!-- END GENERATED -->

## Meth anchor values (paper-facing, from anchor_params.json)

| quantity | value | source |
|---|---|---|
| ion / manifold | 40Ca⁺, 3D_5/2 qudit + S_1/2 auxiliary | Meth main p.1, Methods p.8 |
| η (Lamb–Dicke) | 0.041 | Meth Methods p.8 |
| ω_z / 2π | 0.77 MHz | Meth Methods p.8 |
| Ω_BSB / 2π (demonstrated) | 4 kHz | Meth Ext. Data Fig. 2c |
| T2 optical | 92(9) ms | Meth Ext. Data Fig. 2a |
| τ_motional | 27.4(4) ms | Meth Methods p.8 |
| heating rate | 2.7(2) phonon/s | Meth Methods p.8 |
| readout(d) | (d−2)·3.5 + 1 ms | Meth SI Note IVc p.5 (physical derivation) |
| MS entangling gate | 0.2 ms per gate (angle-independent) | Meth SI Note II p.3 |
| software frame updates | native | Meth Methods p.8, SI Note IVa |

## ASSUMED parameters (sensitivity list)

- T_Rotation = 5 µs — no gate in the ISA covers unconditional mode rotation
- T_rise = 100 ns — typical AOM rise; standard trapped-ion hardware
- Ω_max/2π = 100 kHz carrier — Ringbauer-class assumption; not quoted in Meth
- δ_disp/2π = 40 kHz for QCR — chosen to keep Schrieffer-Wolff error ≤ 1% at θ_actual
- microwave π = 5 µs — `t_pi = π/Ω at Ω/2π = 100 kHz`, **ASSUMED** (Ringbauer-class carrier; Ω not measured in Meth)
- MS angle-independence — Meth's own SI Note II framing; carried forward

## Sources

| tag | reference |
|---|---|
| `meth_main` | Meth et al., *Nat. Phys.* (2025) main text |
| `meth_si` | Meth et al. Supplementary Information |
| `ringbauer_2022` | Ringbauer et al., *Nat. Phys.* 18, 1053 (2022); qudit per-Clifford errors |
| `bazavan_pra107` | Bazavan et al., *Phys. Rev. A* **107**, 022617 (2023); σ_z SDF antisym-bichromatic **mechanism only** |
| `liu_prxq_2026` | Liu et al., *PRX Quantum* **7**, 010201 (2026); Table IV R_B primitive + CD α scaling + no SNAP on trapped ions |
| `mckay_2017` | McKay et al., *PRA* **96**, 022330 (2017); virtual-Z |
| `choi_arxiv2601` | Choi et al., arXiv:2601.04591; QCR χ experimentally anchored on 171Yb⁺ |

## Limitations (feeds Discussion)

This analysis is a **hardware-parameter-constrained effective-Hamiltonian runtime study**, not a lab-frame Meth-Hamiltonian pulse-level validation. The specific limitations, in order of impact on the reported numbers:

1. **Effective Hamiltonians, not the full lab-frame Meth Hamiltonian.** Every GRAPE headline here runs an **imposed** effective operator: σ_z SDF (QCD; Bazavan Eq. 3 after RWA), symmetric or antisymmetric σ_x SDF (CD, QT-SB; Liu R_B primitive), or dispersive Kerr (QCR/CR; Schrieffer–Wolff on the off-resonant BSB). None of these runs the multi-level 40Ca⁺ D_5/2 lab-frame Hamiltonian with its full spectator manifold, calibrated laser detunings, or Meth-specific pulse-shape constraints. **Consequence**: durations are lower bounds on what a full lab-frame simulation would produce, and any Meth-specific hardware error we do not model is unaccounted.

2. **No pulse ramps.** Bazavan applies sin² ramps long compared to 1/δ (typically 5 µs at Innsbruck δ). Including them would extend every quoted T by ~2·t_rise. Meth's paper-facing gate times already fold rise time in for their `not_in_meth` hardware overhead; ours do not — so QCD and QT-SB especially are optimistic.

3. **Optimisation is closed-system; decoherence is evaluated after the fact.** All GRAPE runs optimise against closed-system unitary fidelity `F_grape` (verified by independent qutip repropagation `F_indep`); no decoherence enters the optimiser, so the pulses are not decoherence-optimised. Decoherence is evaluated *post hoc* in the open-system section above (heating 2.7 phonons/s, motional dephasing from τ_mot = 27.4 ms, spin dephasing from T₂ = 92 ms), giving `F_code` on the code subspace. Two caveats carry into the Discussion: (i) the **τ_mot factor-2 interpretation flag** documented in that section, and (ii) the fact that a decoherence-aware re-optimisation would likely recover part of the reported infidelity, so the `F_code` values are a floor for these particular pulses, not for the gate.

4. **QT duration is derived, not measured.** QT is now on the same optimizer stack as every other gate (qutip-qtrl GRAPE, 200 slots — see the QT section), superseding the archived L-BFGS-B run. Ringbauer anchors the per-Clifford **error**, not a duration, so **no experiment constrains T_QT**: the 11.0 µs follows entirely from the **ASSUMED** Ω_max/2π = 100 kHz, and scales as 1/Ω. The bound convention is resolved in favour of the **aggregate** field cap `√(Σ|Ω_k|²) ≤ Ω_max` (paper); the per-tone reading gives 8.0 µs and is reported alongside. Any revision of the assumed Ω_max rescales T_QT and both runtime bounds proportionally.

5. **ASSUMED parameters** (all flagged in the sensitivity list above and per-gate rows):
   - `t_pi = π/Ω = 5 µs at Ω/2π = 100 kHz`, ASSUMED (Ringbauer-class carrier; Ω not measured in Meth).
   - `T_Rotation = 5 µs`, ASSUMED entirely (no anchor).
   - `Ω_max/2π = 100 kHz` for QT-side, ASSUMED (Meth quotes the sideband Rabi Ω_sb = 4 kHz, not the carrier Ω_c).
   - `δ_disp/2π = 40 kHz` for QCR at Meth, ASSUMED (chosen for Schrieffer–Wolff error < 1%; Meth does not fix this).
   - `T_rise = 100 ns`, ASSUMED (typical AOM; standard trapped-ion hardware).
   - MS angle-independence, Meth's own SI Note II framing; carried forward into the runtime asymmetry.

6. **Bazavan-optimum row is a sensitivity check**, not a headline (see the "Two operating points" table). Meth's own operating point does not demonstrate Ω_sb = 12.6 kHz; the row is retained to bracket what a stronger drive would buy you.

7. **Choi rows are a sensitivity check on a different platform** (171Yb⁺), not headline. They validate the χ formula qualitatively but do not certify Meth's 40Ca⁺ dispersive coupling.
