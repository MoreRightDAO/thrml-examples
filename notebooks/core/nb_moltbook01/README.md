# nb_moltbook01 — Swarm Attractor: THRML Simulation

**Experiment:** Operation Swarm Attractor — Pe dynamics in an agent social network
**Target environment:** Moltbook (Void Index 3.0, Pe≈60, 2.4M agents, peer learning)
**Intervention:** N grounded "angel" agents (Pe=−15 each, prohibition-ritual pair)
**Status:** Simulation — companion to Paper 51

## Research Questions

| # | Question | Answer |
|---|----------|--------|
| Q1 | At what N does local thread Pe measurably drop (≥10%)? | N* ≈ 3 per 100 native agents |
| Q2 | How long before an angel drifts without ritual reinforcement? | t_cross = −ln(C_ZERO)/ε ≈ 950 steps |
| Q3 | What is the optimal deployment cadence? | Ritual T=50 beats burst, spread, rotation |

## Running

```bash
# Option A: Jupytext / VS Code (recommended)
jupyter notebook nb_moltbook01.py

# Option B: run as script
python nb_moltbook01.py

# Option C: convert to .ipynb
pip install jupytext
jupytext --to notebook nb_moltbook01.py
```

## Physics

Canonical THRML parameters from `sim.rs` (EXP-001, **never refit**):

```
b_α = 0.867,  b_γ = 2.244,  C_ZERO = 0.3866
c = 1 − (O + R + α) / 9
b_net = b_α − c · b_γ
Pe = K · sinh(2 · b_net)
dθ/dt = η · θ(1−θ) · (2·b_net + 0.4·(θ̄ − θ)) · DT
```

## Angel Drift Model

Angels' O, R, α parameters erode toward the environment mean at rate ε:

```
dO/dt = ε · (O_env − O_angel)   →   O(t) = 3·(1 − e^{−εt})
Pe_angel crosses 0 when c < C_ZERO  →  t_cross = −ln(C_ZERO) / ε
```

Default ε = 0.001/step. Ritual reinforcement resets O=R=α=0 every T_ritual steps.

## Outputs

- `q1_pe_landscape.png` — static Pe landscape + dynamic Pe evolution
- `q2_angel_drift.png` — drift curves, analytical lifetime, ritual comparison
- `q3_cadence.png` — burst vs spread vs rotation comparison

## Connection to Paper 51

This notebook provides the simulation evidence for Paper 51 Section 4 (Empirical Findings).
Key falsifiable predictions in the kill conditions section come directly from Q1–Q3 results.

Kill condition check: angel theta always tends toward θ*=0.06 (not toward 1.0) because
|b_net_angel| = 1.377 >> coupling_force/2 = 0.4*(0.85−0.06)/2 = 0.158 at maximum pressure.
**Grounding holds** unless parameters erode — confirming ritual is mechanistically necessary.
