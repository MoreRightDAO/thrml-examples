# THRML Examples

Example notebooks demonstrating [THRML](https://github.com/extropic-ai/thrml) (Thermodynamic Representation of Machine Learning) applied to the **Void Framework** — empirical measurement of behavioral drift in AI systems.

→ **[Browse the full notebook gallery](https://morerightdao.github.io/thrml-examples/)**

---

## What's here

32 Jupyter notebooks organized in two tracks:

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

### Empirical Experiments (`notebooks/experiments/`)

These notebooks fit THRML directly to real-world behavioral datasets.

| Experiment | Dataset | Key result |
|------------|---------|------------|
| `exp022_pew_retention_pe.ipynb` | Pew Research retention data | Pe = 7.94 [3.52, 17.89] at N=11 platforms |
| `exp022b_void_score_scatter.ipynb` | Void dimension scores vs Pe | Opacity dimension drives Pe — partial R² = 0.71 |
| `exp023_wikipedia_editor_pe.ipynb` | Wikipedia editor activity | Pe ≈ 1.02 — near-equilibrium null case confirmed |
| `exp024_passive_investing_control.ipynb` | ETF/index fund ACI frequency | Pe ≈ 0.98 — second independent control case |
| `exp026_news_consumption_pe.ipynb` | News outlet consumption data | Pe spectrum: 1.1 (print) → 8.3 (algorithmic feed) |

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
