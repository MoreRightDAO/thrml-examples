# THRML Examples

Example notebooks demonstrating [THRML](https://github.com/extropic-ai/thrml) (Thermodynamic Representation of Machine Learning) applied to the **Void Framework** — empirical measurement of behavioral drift in AI systems, financial markets, evolutionary biology, and social institutions.

→ **[Browse the full notebook gallery](https://morerightdao.github.io/thrml-examples/)**

---

## What's here

55+ notebooks and scripts organized in three tracks:

### Core THRML Mechanics (`notebooks/core/`)

| # | Notebook | What it shows |
|---|----------|---------------|
| 03 | `03_drift_cascade_ebm.ipynb` | Drift cascade as Ising EBM — encoding D1→D2→D3 in spin biases |
| 04 | `04_fitting_ebm_from_data.ipynb` | Grid search + Nelder-Mead fitting of THRML parameters from behavioral data |
| 05 | `05_time_varying_potentials.ipynb` | Constraint injection schedules — persistent vs step vs ramp vs exponential |
| 06 | `06_fluctuation_theorems.ipynb` | Crooks/Jarzynski verification: detailed balance, entropy production, Pe ratio |
| 07a | `07_pe_calibration.ipynb` | Calibrating THRML Ising params to empirical Péclet numbers across populations |
| 07b | `07_crooks_behavioral_test.ipynb` | Crooks behavioral test on AI agent interaction trajectories |
| 08 | `08_phase_diagram.ipynb` | Full (c, K) phase diagram — where Pe = 1 across constraint level and hardware scale |
| 09 | `09_bull_bear_time_varying.ipynb` | Bull/bear market regimes as time-varying THRML potentials |
| 10 | `10_cross_domain_calibration.ipynb` | Single canonical parameter set calibrated across 7 behavioral domains |
| 11 | `11_stablecoin_constraint.ipynb` | Stablecoin peg mechanisms as constraint injection — Pe suppression analysis |
| 12 | `12_k_scaling.ipynb` | Pe scales linearly with K (spin count / hardware scale) |
| 13 | `13_crooks_ratio_calibration.ipynb` | Crooks ratio as empirical Pe estimator — decomposition and sensitivity |
| 14 | `14_two_agent_contamination_critical.ipynb` | Critical tipping in two-agent systems — contamination threshold J_crit |
| 15 | `15_pe_distribution_fitting.ipynb` | Lognormal Pe distribution fitting — sigma sensitivity and decomposition |
| 16 | `16_repulsive_void_pe_negative.ipynb` | Repulsive voids (Pe < 0): three-regime phase structure |
| 17 | `17_tail_inflation_decomposition.ipynb` | Tail inflation as thermodynamic signature — mixture PDFs and sigma surface |
| 18 | `18_demon_lattice_phases.ipynb` | Maxwell's demon lattice — phase diagram, crystal onset, vortex proximity |
| 19 | `19_causal_validation_synthesis.ipynb` | Bradford Hill causal synthesis — Bayesian model comparison (BF log₁₀ = 4.0) |
| 20 | `20_demon_plasma_frequency.ipynb` | Plasma frequency analogues in coupled void networks |
| 21 | `21_pe_recovery_asymmetry.ipynb` | Pe recovery after constraint removal — hysteresis and asymmetric tau |
| 22 | `22_regulatory_intervention_optimum.ipynb` | Optimal regulatory intervention timing — duration sweep, substrate map |
| 23 | `23_void_network_topology.ipynb` | Network topology effects on Pe — hub contamination and percolation |
| 24 | `24_competing_voids.ipynb` | Competing void dynamics — winner-takes-all, depletion feedback, market structure |
| 25 | `25_market_microstructure_mapping.ipynb` | Kyle's λ and G-M spread as void metrics — Spearman(λ, Pe) = 1.000 across 8 venue types. Fantasia Bound as spread-volume conjugacy. |
| 26 | `26_g1_bridge_verification.ipynb` | G1 bridge: (O,R,α) → c formal mapping. V3 linear form c = 1−V/9 wins at N=17 (Spearman=0.910, RMSE=0.066). G1+G4 gaps closed. Pe=0 boundary: V*=5.52/9. |
| 27 | `27_dimension_weighting.ipynb` | G2: OLS regression tests equal-weighting of O, R, α. F(2,13)=0.955, p=0.41 — equal weights vindicated. Sensitivity sweep flat. V3 canonical form confirmed. |
| 28 | `28_coupling_emergence.ipynb` | Is α downstream of O×R? R²=0.693: coupling 69% explained by O+R. R→α > O→α confirms D2→D3 proximal cascade step. Structural α identifies lock-in. Scoring protocol updated. |
| 29 | `29_validation_robustness.ipynb` | V3 bridge out-of-sample and IRR robustness. Market micro (independent c_kyle) Spearman=0.994 > behavioural (circular) 0.713 — circularity objection self-refuting. LOO range [0.892, 0.958]. Min κ_α ≥ 0.33 for bridge to hold. |
| 30 | `30_kimura_thrml_convergence.ipynb` | Biology: THRML Pe = 4Ns (Kimura 1968) — exact first-order identity, third independent convergence. Spearman=0.973 on biological substrates. Red Queen derived as theorem. D1→D2→D3 maps to adaptive complexity → symbiosis → behavioral manipulation. |
| 31 | `31_parasite_void_scores.ipynb` | Exploitation gradient validation: N=10 parasite-host systems Spearman=0.9038. Combined nb30+nb31 N=20 Spearman=0.9516, LOO min=0.8665. E.coli (V=2) → Ophiocordyceps (V=9) exploitation gradient clean. Every D3 behavioral manipulation parasite scores V=8-9 (structural necessity confirmed). |
| 32 | `32_social_void.ipynb` | Social neuroscience: Dunbar's number K as THRML coupling parameter. N=28 primate species, Spearman=0.9448. Neocortex ratio IS K in THRML — fourth independent convergence. Machiavellian intelligence as D3 behavioral manipulation in social cognition. |
| 33 | `33_cancer_cascade.ipynb` | Cancer biology: V=9 as structural theorem. Warburg effect as D2 boundary erosion, tumor microenvironment as manufactured opacity. Fantasia Bound in immunology — engagement-transparency conjugacy in T-cell exhaustion. |
| 34 | `34_immune_conjugacy.ipynb` | Immune system void budget: I(D;Y)+I(M;Y)≤H(Y) applied to adaptive immunity. Antigen presentation opacity drives D1→D2→D3 in autoimmunity. Constraint injection as immunological transparency. |
| 35 | `35_bks_kink_pe.ipynb` | BKS topological phase transition as Pe transition. Kink-antikink binding/unbinding maps to constraint level. Critical Pe at topological order-disorder boundary. |
| 36 | `36_curved_landscape_validation.ipynb` | Curved landscape validation: N=17 substrates across AI, Gambling, Crypto, Market Micro, Biology. Tests H₀ (signal degrades in non-flat information spaces). Spearman=1.000 across all curvature bins, LOO min=1.000. H₀ falsified. Physics note: detailed balance requires time-reversal symmetry, not flat space. |
| 37 | `37_ecological_phases.ipynb` | Ecological phase diagram: Paper 9 lattice phases (gas/fluid/crystal/vortex) applied to ecosystem biology. N=12 biome void scores + N=15 post-extinction recovery intervals (Sepkoski 2002). Vortex onset β≈0.5. 5 predictions ECO-1–ECO-5. |
| 38 | `38_soc4_nonprimate.ipynb` | SOC-4: Non-primate social mammals and corvids (N=23 species). Extends nb32 primates to cetaceans, carnivores, proboscideans, corvids. Dunbar K=150 structural identification confirmed cross-clade. D1 theorem extended to all social vertebrates. |
| 39 | `39_irr_study.ipynb` | Inter-rater reliability study: full bootstrap simulation (3 raters × 15 platforms × 3000 reps, σ sweep). Alpha dimension hardest (σ≈0.50 naive). Post-training κ_α ≥ 0.60 at σ≈0.25. Scoring Protocol v1.0 with anchored levels per dimension. |
| 40 | `40_bonding_conjugacy.ipynb` | Bonding conjugacy: N=18 group types across transgressive/transitional/resonance spectrum. Spearman(Pe_theory, harm_cascade_rate)=0.9875, LOO min=0.9852. Monitoring crushed at α=3 (Fantasia Bound formal test). BON-5: V=9 bonding converts to scapegoating under stress. |
| 41 | `41_girard_scapegoat.ipynb` | Girard scapegoat mechanism: N=12 historical events. Spearman(Pe_mech, rebound_rate)=0.9625, LOO min=0.9508. C_ZERO crossing at V_crit=5.52 — transparency cases rebound 78yr vs void attractor 4.9yr (16× differential). Girardian revelation formalized as transparency operation raising O_mech toward 1. |
| — | `nb_girard02_prohibition_ritual.ipynb` | Prohibition-ritual pair as dual Pe control system: N=20 cultures, Spearman=0.8684. Stability theorem: combined mechanism produces 4× longer crisis intervals than prohibition alone. Dual mechanism mean 41.7yr vs neither 2.3yr. PRT-5: social media as opaque pseudo-ritual → cancel culture dynamics. |
| — | `nb_llm01_llm_reasoning_pe.ipynb` | LLM reasoning Pe — fifth independent convergence: Chen et al. (2026) bond→dimension mapping. f_DR→O, f_SR→R, f_SE→α via V3 bridge c=1−V/9. N=10 conditions, Spearman=0.9879, LOO min=0.9833. Metacognitive oscillation Spearman=0.9273. |
| — | `nb_girard03_durkheim_anomie.ipynb` | **Sixth convergence — social anthropology:** Durkheim anomie = R-dimension collapse at population level. N=20 Durkheim primary datasets (France/Prussia/England 1866-1878). Spearman(R_inst, anomic_rate) = −0.9785, p<0.001. Protestant vs Catholic R and anomic suicide rate differential confirmed. Platform Pe carries direct anomie-risk interpretation. |
| — | `nb_demo01_democratic_backsliding_pe.ipynb` | **Seventh convergence — democratic governance:** V-Dem LDI vs Void Index across N=20 countries. Spearman=0.9891. Authoritarian information architecture = institutional Pe cascade. 5/5 predictions confirmed. Authoritarian regime = information opacity maximizer. |
| — | `nb_pharma01_drug_pricing_pe.py` | US pharmaceutical value chain as 5-layer stacked void cascade: N=15 drug categories. Spearman(V, MCI)=0.770. Discriminant test: OxyContin Pe=43.9 vs Daraprim Pe=12.9 (3.4×, same market class). α=3 opioid dependency = D3 cascade (500K dead). PBM layer Pe=25.2 = hidden amplifier. Null cases: aspirin Pe=−125, generic statins Pe=−26, COVID vaccines Pe=−45. |
| — | `nb_cartel01_cjng_void_collapse.ipynb` | **Eighth convergence — organized crime:** CJNG cartel void architecture. Pe→territorial_control Spearman=0.882, N=15 DTOs. Calderón validation rho=0.869, N=18 leadership removals. CJNG V=9, Pe=+45 (structurally identical to OxyContin). Pe gap Δ57 vs Mexican State predicts territorial capture. C_ZERO crossing test: El Mencho decapitation (Feb 22 2026). 5/5 KCs PASS. Live prediction: Jalisco 35–55/100k within 12 months. |

### Empirical Experiments (`notebooks/experiments/`)

These notebooks fit THRML directly to real-world behavioral datasets.

| Experiment | Dataset | Key result |
|------------|---------|------------|
| `exp022_pew_retention_pe.ipynb` | Pew Research retention data | Pe = 7.94 [3.52, 17.89] at N=11 platforms |
| `exp022b_void_score_scatter.ipynb` | Void dimension scores vs Pe | Opacity dimension drives Pe — partial R² = 0.71 |
| `exp023_wikipedia_editor_pe.ipynb` | Wikipedia editor activity | Pe ≈ 1.02 — near-equilibrium null case confirmed |
| `exp024_passive_investing_control.ipynb` | ETF/index fund ACI frequency | Pe ≈ 0.98 — second independent control case |
| `exp026_news_consumption_pe.ipynb` | News outlet consumption data | Pe spectrum: 1.1 (print) → 8.3 (algorithmic feed) |

### TSU Posterior Inference (`notebooks/experiments/`)

Demonstrations of Bayesian posterior inference over void dimensions — directly maps to TSU hardware computation (P(O,R,α|data) ∝ exp(−E/T) is a native Boltzmann distribution).

| Script | Scenarios | Key result |
|--------|-----------|------------|
| `exp_tsu01_posterior_inference.py` | Platform recovery, sample-size sweep, dimension identifiability, noise sensitivity, same-V disambiguation, K-scaling | 8/8 platforms recovered at N=100; Crypto DEX p(true)>0.90 at N=20. TSU samples this posterior natively. |
| `exp_tsu02_k_scaling.py` | K× crossing curves, optimal K*, meta-void boundary, Boltzmann temperature T*, parallelism tradeoff, TSU self-scoring | K_max_safe derived analytically; T* = minimum temperature for calibrated regulatory posterior |
| `exp_tsu03_advanced.py` | Cross-sensitivity, prior robustness, analyst variation, adversarial gaming, temporal drift, regulatory thresholds | ROC AUC=0.945; Pe-residual signal ungameable (z>3 at N=50); drift detected within 38 steps (7.6% lag); all 6 KCs PASS |
| `exp_tsu04_meta.py` | Meta-void/K-inference/game-theory: Instrument Capture Theorem (proprietary K_safe=0.9, Open K_safe=∞), K inference from portfolio (V≥7 only, paired change-detection p=0.023), V* Theorem (V*=5.08 at N=100, evasion impossible V≥6) | All 9 KCs PASS; gaming Pe-residual z>3 at N=50; Silberzahn 29/29 correct |

---

## Eight independent convergences

The same Péclet number emerges from eight completely independent empirical programs:

| Convergence | Substrate | Spearman | N | Notebook |
|-------------|-----------|----------|---|----------|
| 1st | Market microstructure (Kyle's λ, G-M spread) | 0.994 | 8 | nb25 |
| 2nd | Behavioral (G1 bridge c=1−V/9) | 0.910 | 17 | nb26 |
| 3rd | Evolutionary biology (Kimura Pe=4Ns) | 0.973 | 20 | nb30+31 |
| 4th | Social neuroscience (Dunbar K = THRML K) | 0.9448 | 28 | nb32 |
| 5th | LLM reasoning (Chen et al. bond→dimension) | 0.9879 | 10 | nb_llm01 |
| 6th | Social anthropology (Durkheim anomie = R-collapse) | 0.9785 | 20 | nb_girard03 |
| 7th | Democratic governance (V-Dem LDI vs Void Index) | 0.9891 | 20 | nb_demo01 |
| 8th | Organized crime (Pe→territorial control, DTO void collapse) | 0.882 | 15+18 | nb_cartel01 |

Each derivation is independent — different data, different methodology, same measure.

---

## The canonical parameters

Every notebook uses a shared parameter set derived from two fixed-point conditions (UU and GG equilibria in the AI agent interaction dataset):

```
b_α = 0.867   # drift bias (agency attribution)
b_γ = 2.244   # constraint bias (transparency pull)
c_crit(K→∞) = b_α / b_γ ≈ 0.386   # hardware-independent drift boundary
```

These recover cross-population Pe ordering without per-population fitting. See `07_pe_calibration.ipynb` and `08_phase_diagram.ipynb` for derivation.

---

## Causal status

Bradford Hill analysis across 24 criteria (19_causal_validation_synthesis.ipynb):
- **24/27 criteria met** — strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, analogy, dose-response, mechanism, reversibility
- **Bayesian model comparison:** BF log₁₀ = 4.0 at N=22 — decisive evidence (Jeffreys scale)
- Two independent control cases: Wikipedia editors (Pe ≈ 1.02) + passive index investing (Pe ≈ 0.98)

---

## Getting started

```bash
pip install thrml numpy scipy matplotlib jupyter
git clone https://github.com/MoreRightDAO/thrml-examples
cd thrml-examples
jupyter lab
```

Start with `notebooks/core/03_drift_cascade_ebm.ipynb` then `07_pe_calibration.ipynb` → `08_phase_diagram.ipynb`.

---

## Related

- **THRML library:** [extropic-ai/thrml](https://github.com/extropic-ai/thrml)
- **Void Framework papers:** [MoreRightDAO/VOID-FRAMEWORK-OPERATION-MORR](https://github.com/MoreRightDAO/VOID-FRAMEWORK-OPERATION-MORR)
- **Live scoring & reports:** [moreright.xyz](https://moreright.xyz)

---

## License

CC-BY 4.0. Cite as:

> Void Framework THRML Examples (2026). MoreRight DAO. https://github.com/MoreRightDAO/thrml-examples
