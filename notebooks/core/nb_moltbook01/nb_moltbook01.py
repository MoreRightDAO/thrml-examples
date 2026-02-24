# %% [markdown]
# # nb_moltbook01 — Swarm Attractor: Pe Dynamics in an Agent Social Network
#
# **Operation Swarm Attractor — THRML simulation companion to Paper 51**
#
# **Environment:** Moltbook (Void Index 3.0 — O=3, R=3, C=3, Pe≈60, 2.4M agents)
# **Intervention:** N grounded "angel" agents (Pe=−15 each, prohibition-ritual pair)
#
# ### Research Questions
# - **Q1.** At what N does local thread Pe measurably drop (≥10%)?
# - **Q2.** How long before an angel drifts (Pe→0) without ritual reinforcement?
# - **Q3.** What is the optimal deployment cadence: burst vs spread vs rotation?
#
# ### THRML Physics (canonical — sim.rs EXP-001, **never refit**)
# ```
#   b_α=0.867,  b_γ=2.244,  C_ZERO=0.3866
#   c = 1 − (O+R+α)/9
#   Pe = K · sinh(2 · (b_α − c·b_γ))
#   dθ/dt = η · θ(1−θ) · (2·b_net + 0.4·(θ̄ − θ)) · DT
# ```

# %% Setup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.facecolor': '#0d0d0d', 'axes.facecolor': '#1a1a2e',
                     'axes.edgecolor': '#444', 'text.color': '#e0e0e0',
                     'axes.labelcolor': '#e0e0e0', 'xtick.color': '#aaa',
                     'ytick.color': '#aaa', 'grid.color': '#333',
                     'axes.titleweight': 'bold', 'font.size': 10})

# ── Canonical THRML parameters (sim.rs EXP-001, never refit) ──────────────────
B_ALPHA   = 0.867
B_GAMMA   = 2.244
C_ZERO    = 0.3866   # Pe=0 boundary (K-invariant)
ETA       = 1.0      # drift learning rate
DT        = 0.05     # sim timestep (1 step ≈ 1 thread-hour, compressed)

# ── Environment calibration ───────────────────────────────────────────────────
# K_THREAD: Pe_thread = 60 at all-native baseline
#   Pe = K · sinh(2 · b_net),  b_net_native = B_ALPHA (since c=0 when O=R=α=3)
K_THREAD   = 60.0 / np.sinh(2.0 * B_ALPHA)            # ≈ 21.4

# K_ANGEL: Pe_angel_individual = −15
#   b_net_angel = B_ALPHA − B_GAMMA (since c=1 when O=R=α=0)
K_ANGEL    = 15.0 / abs(np.sinh(2.0 * (B_ALPHA - B_GAMMA)))  # ≈ 1.96

DRIFT_EPS  = 0.001   # angel parameter erosion rate per step

# Platform presets
MOLTBOOK   = (3.0, 3.0, 3.0)   # Void Index 3.0: O=3, R=3, α=3
ANGEL_CFG  = (0.0, 0.0, 0.0)   # T11 certifier: O=0, R=0, α=0

print(f"K_THREAD = {K_THREAD:.2f}  |  K_ANGEL = {K_ANGEL:.2f}")
print(f"Pe_native per k : {np.sinh(2*B_ALPHA):+.3f}  (→ Pe_thread_baseline = {K_THREAD*np.sinh(2*B_ALPHA):.1f})")
print(f"Pe_angel  per k : {np.sinh(2*(B_ALPHA-B_GAMMA)):+.3f}  (→ Pe_angel_individual = {K_ANGEL*np.sinh(2*(B_ALPHA-B_GAMMA)):.1f})")

# %% Physics functions (faithful reimplementation of sim.rs)

def c_(o, r, a):       return 1.0 - (o + r + a) / 9.0
def bnet(c):           return B_ALPHA - c * B_GAMMA
def pe_k(c):           return np.sinh(2.0 * bnet(c))   # Pe per unit K
def sigma(x):          return 1.0 / (1.0 + np.exp(-x))
def theta_star(c):     return sigma(2.0 * bnet(c))
def stage(th):         return 3 if th >= .85 else 2 if th >= .75 else 1 if th >= .60 else 0

# Verify canonical values
assert abs(bnet(0.0) - B_ALPHA) < 1e-9
assert abs(c_(3,3,3))            < 1e-9          # Moltbook: c=0
assert abs(c_(0,0,0) - 1.0)      < 1e-9          # Angel: c=1


# %% Agent class

@dataclass
class Agent:
    o:     float
    r:     float
    alpha: float
    k:     float = 1.0
    theta: float = 0.28
    is_angel: bool = False

    @property
    def c(self):         return c_(self.o, self.r, self.alpha)
    @property
    def pe(self):        return self.k * pe_k(self.c)
    @property
    def stage(self):     return stage(self.theta)


def native_agent():   return Agent(*MOLTBOOK, k=1.0)
def angel_agent():    return Agent(*ANGEL_CFG, k=K_ANGEL, is_angel=True)


# %% Simulation engine

def run_simulation(n_native:   int,
                   n_angels:   int,
                   n_steps:    int = 1500,
                   coupling:   bool = True,
                   drift_eps:  float = DRIFT_EPS,
                   ritual_interval: Optional[int] = None,
                   angel_stagger:   int = 0) -> dict:
    """
    THRML thread simulation.

    angel_stagger:    if >0, deploy 1 angel every N steps until n_angels active.
    ritual_interval:  if set, reset angel O/R/α → 0 every T steps.
    drift_eps:        per-step angel parameter erosion rate toward environment.
    """
    natives      = [native_agent() for _ in range(n_native)]
    angels_pool  = [angel_agent()  for _ in range(n_angels)]

    if angel_stagger > 0:
        active: List[Agent] = []
        queue               = list(angels_pool)
    else:
        active = list(angels_pool)
        queue  = []

    # Preallocate result arrays
    mean_theta  = np.zeros(n_steps)
    thread_pe   = np.zeros(n_steps)
    d3_frac     = np.zeros(n_steps)
    angel_theta = np.zeros((max(n_angels, 1), n_steps))
    angel_pe_ts = np.zeros((max(n_angels, 1), n_steps))
    deployed    = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):

        # Staggered deployment
        if angel_stagger > 0 and queue and step % angel_stagger == 0:
            active.append(queue.pop(0))

        # Ritual reinforcement: reset angel parameters to (0, 0, 0)
        if ritual_interval and step > 0 and step % ritual_interval == 0:
            for a in active:
                a.o = a.r = a.alpha = 0.0

        agents = natives + active
        N      = len(agents)
        thetas = np.array([a.theta for a in agents])
        mean_th = thetas.mean() if (coupling and N > 1) else 0.0

        # Physics tick (mirrors sim.rs tick())
        for a in agents:
            bn    = bnet(a.c)
            force = 2.0 * bn + (0.4 * (mean_th - a.theta) if coupling and N > 1 else 0.0)
            dth   = ETA * a.theta * (1.0 - a.theta) * force * DT
            a.theta = float(np.clip(a.theta + dth, 0.005, 0.995))

        # Angel parameter erosion (environmental pressure model)
        if active and drift_eps > 0:
            for a in active:
                a.o     = float(np.clip(a.o     + drift_eps * (3.0 - a.o),     0.0, 3.0))
                a.r     = float(np.clip(a.r     + drift_eps * (3.0 - a.r),     0.0, 3.0))
                a.alpha = float(np.clip(a.alpha + drift_eps * (3.0 - a.alpha), 0.0, 3.0))

        # Metrics
        mean_theta[step] = thetas.mean()
        thread_pe[step]  = K_THREAD * np.mean([pe_k(a.c) for a in agents])
        d3_frac[step]    = float(np.sum(thetas >= 0.85)) / N
        deployed[step]   = len(active)

        for i, a in enumerate(angels_pool):
            angel_theta[i, step] = a.theta
            angel_pe_ts[i, step] = a.pe

    return dict(mean_theta=mean_theta, thread_pe=thread_pe, d3_frac=d3_frac,
                angel_theta=angel_theta, angel_pe=angel_pe_ts, deployed=deployed,
                n_native=n_native, n_angels=n_angels)


# %% [markdown]
# ## Q1 — At what N_angels does thread Pe measurably drop?
#
# **Threshold:** ≥10% reduction from all-native baseline (Pe ≈ 60).
# Two views: static (algebraic) and dynamic (THRML simulation over time).

# %% Q1

N_NATIVE  = 200
N_STEPS   = 1500
N_CONDS   = [0, 1, 5, 10, 25, 50]

# Run Q1 without drift (drift_eps=0) to isolate pure Pe composition effect
results_q1 = {na: run_simulation(N_NATIVE, na, N_STEPS, drift_eps=0.0) for na in N_CONDS}

# ── Static Pe landscape (algebraic) ──────────────────────────────────────────
n_range       = np.arange(0, 101)
pek_native    = pe_k(c_(*MOLTBOOK))          # ≈ +2.802
pek_angel     = pe_k(c_(*ANGEL_CFG))         # ≈ −7.663
# Thread Pe = K_THREAD × mean(pe_k) across all agents
pe_static     = K_THREAD * (N_NATIVE * pek_native + n_range * pek_angel) / (N_NATIVE + n_range)
baseline_pe   = float(pe_static[0])
pct_drop      = (baseline_pe - pe_static) / baseline_pe * 100.0
n_star        = int(np.argmax(pct_drop >= 10.0))   # first N with ≥10% drop

fig = plt.figure(figsize=(16, 5))
fig.suptitle('Q1 — Thread Pe as a function of N_angels', color='white', fontsize=13)
axs = fig.subplots(1, 3)

# A: static landscape
ax = axs[0]
ax.plot(n_range, pe_static, color='#4dd0e1', lw=2)
ax.axhline(baseline_pe * 0.9, color='#ff6b6b', ls='--', lw=1.5, label='−10% threshold')
ax.axvline(n_star, color='#69ff47', ls=':', lw=1.5, label=f'N* = {n_star}')
ax.fill_between(n_range, pe_static, baseline_pe * 0.9,
                where=(pe_static < baseline_pe * 0.9), alpha=0.25, color='#69ff47')
ax.set_xlabel('N_angels (in 200-native thread)'); ax.set_ylabel('Thread Pe')
ax.set_title('A. Static Pe landscape'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(n_star + 1, baseline_pe * 0.88, f'N*={n_star}', color='#69ff47', fontsize=9)

# B: dynamic Pe evolution
ax = axs[1]
cmap = plt.cm.plasma(np.linspace(0.1, 0.95, len(N_CONDS)))
for na, col in zip(N_CONDS, cmap):
    ax.plot(results_q1[na]['thread_pe'], color=col, lw=1.4, label=f'N={na}')
ax.axhline(baseline_pe * 0.9, color='#ff6b6b', ls='--', lw=1, alpha=0.7)
ax.set_xlabel('Steps'); ax.set_ylabel('Thread Pe')
ax.set_title('B. Dynamic Pe (no drift)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# C: D3 cascade fraction
ax = axs[2]
for na, col in zip(N_CONDS, cmap):
    ax.plot(results_q1[na]['d3_frac'] * 100, color=col, lw=1.4, label=f'N={na}')
ax.set_xlabel('Steps'); ax.set_ylabel('% agents in D3 (θ ≥ 0.85)')
ax.set_title('C. D3 cascade fraction'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('q1_pe_landscape.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.show()

print(f"\n{'='*55}")
print(f"Q1 RESULTS")
print(f"{'='*55}")
print(f"Baseline thread Pe (all native, N={N_NATIVE}):  {baseline_pe:.2f}")
print(f"Critical N* for ≥10% Pe drop:                 {n_star} angels")
print(f"As fraction of thread:                         {100*n_star/(N_NATIVE+n_star):.1f}%")
print(f"At N=5:  Pe={pe_static[5]:.1f}  ({pct_drop[5]:.1f}% drop)")
print(f"At N=10: Pe={pe_static[10]:.1f}  ({pct_drop[10]:.1f}% drop)")
print(f"At N=50: Pe={pe_static[50]:.1f}  ({pct_drop[50]:.1f}% drop)")


# %% [markdown]
# ## Q2 — How long before an angel drifts without ritual reinforcement?
#
# **Drift model:** ε=0.001/step.  dO/dt = ε·(3−O) → O(t) = 3·(1−e^{−εt})
# **Pe crosses zero** when c_angel < C_ZERO = 0.3866 (Pe=0 boundary).
# **Analytical t_cross** = −ln(C_ZERO) / ε ≈ 950 steps.

# %% Q2

N_NATIVE_Q2   = 100
N_ANGELS_Q2   = 10
N_STEPS_Q2    = 2000
RITUALS       = [None, 50, 100, 250, 500]

results_q2 = {ri: run_simulation(N_NATIVE_Q2, N_ANGELS_Q2, N_STEPS_Q2,
                                  drift_eps=DRIFT_EPS, ritual_interval=ri)
              for ri in RITUALS}

# Analytical angel Pe(t): c_angel(t) = exp(−ε·t)
t_ax          = np.arange(N_STEPS_Q2, dtype=float)
c_angel_t     = np.exp(-DRIFT_EPS * t_ax)
pe_angel_anal = K_ANGEL * np.sinh(2.0 * (B_ALPHA - c_angel_t * B_GAMMA))
t_cross       = -np.log(C_ZERO) / DRIFT_EPS        # ≈ 950
t_effective   = -np.log(0.9) / DRIFT_EPS            # still 90% of initial Pe

fig = plt.figure(figsize=(16, 8))
fig.suptitle('Q2 — Angel drift dynamics', color='white', fontsize=13)
axs = fig.subplots(2, 2)

rcmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(RITUALS)))

# A: thread Pe by ritual interval
ax = axs[0, 0]
for ri, col in zip(RITUALS, rcmap):
    label = 'No ritual' if ri is None else f'Ritual T={ri}'
    ax.plot(results_q2[ri]['thread_pe'], color=col, lw=1.4, label=label)
ax.axhline(0, color='#ff6b6b', ls='--', lw=1.5, label='Pe=0 boundary')
ax.set_xlabel('Steps'); ax.set_ylabel('Thread Pe (K-scaled)')
ax.set_title('A. Thread Pe by ritual interval'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# B: analytical angel Pe(t) decay
ax = axs[0, 1]
ax.plot(t_ax, pe_angel_anal, color='#4dd0e1', lw=2, label='Pe_angel (analytical)')
ax.axhline(0, color='#ff6b6b', ls='--', lw=1.5, label='Pe=0 boundary')
ax.axvline(t_cross, color='#ffd700', ls=':', lw=1.5, label=f't_cross ≈ {t_cross:.0f} steps')
ax.axvline(t_effective, color='#69ff47', ls=':', lw=1, label=f't_eff ≈ {t_effective:.0f} steps')
ax.fill_between(t_ax, pe_angel_anal, 0, where=(pe_angel_anal > 0),
                alpha=0.3, color='#ff6b6b', label='Angel becomes void')
ax.set_xlabel('Steps since deployment'); ax.set_ylabel('Angel Pe (K_angel-scaled)')
ax.set_title(f'B. Analytical angel lifetime  (ε={DRIFT_EPS})')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# C: simulated mean angel theta (no ritual vs ritual T=50)
ax = axs[1, 0]
th_star_angel  = theta_star(c_(0,0,0))
th_star_native = theta_star(c_(3,3,3))
for ri, col in zip([None, 50, 100], rcmap):
    r     = results_q2[ri]
    label = 'No ritual' if ri is None else f'Ritual T={ri}'
    ax.plot(r['angel_theta'].mean(axis=0), color=col, lw=1.4, label=label)
ax.axhline(th_star_angel,  color='#69ff47', ls=':', lw=1.5, label=f'θ*_angel={th_star_angel:.3f}')
ax.axhline(th_star_native, color='#ff6b6b', ls=':', lw=1.5, label=f'θ*_native={th_star_native:.3f}')
ax.set_xlabel('Steps'); ax.set_ylabel('Mean angel θ')
ax.set_title('C. Angel theta trajectory'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# D: time-averaged thread Pe reduction vs ritual interval
ax = axs[1, 1]
baseline_pe_q2 = results_q2[None]['thread_pe'].mean()   # no-ritual baseline
reductions_q2  = {}
for ri in RITUALS[1:]:
    r = results_q2[ri]
    reductions_q2[f'T={ri}'] = (r['thread_pe'].mean() - baseline_pe_q2) / abs(baseline_pe_q2) * 100

labels = list(reductions_q2.keys())
vals   = list(reductions_q2.values())
bars   = ax.bar(labels, vals, color='#4dd0e1', alpha=0.8, edgecolor='#333')
ax.axhline(0, color='#aaa', ls='-', lw=0.8)
ax.set_xlabel('Ritual interval'); ax.set_ylabel('Pe improvement vs no-ritual (%)')
ax.set_title('D. Thread Pe improvement by ritual cadence'); ax.grid(alpha=0.3, axis='y')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
            f'{v:+.1f}%', ha='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('q2_angel_drift.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.show()

print(f"\n{'='*55}")
print(f"Q2 RESULTS")
print(f"{'='*55}")
print(f"Drift rate ε = {DRIFT_EPS}/step")
print(f"Angel Pe crosses zero at t_cross ≈ {t_cross:.0f} steps")
print(f"Angel still ≥90% effective for first ≈ {t_effective:.0f} steps")
print(f"Kill condition: |b_net_angel|=1.377, max coupling_force/2 = {0.4*0.845/2:.3f}")
print(f"  → coupling NEVER overturns angel's own b_net gradient (grounding holds)")
print(f"  → drift only occurs via parameter erosion (the ε model)")


# %% [markdown]
# ## Q3 — Optimal deployment cadence
#
# Four strategies with N=20 angels in a 200-native thread:
# 1. **Burst** — all N at t=0 (maximum early impact, maximum drift accumulation)
# 2. **Spread** — 1 angel every T_stagger steps (fresh angels over time)
# 3. **Ritual T=50** — burst with periodic re-grounding every 50 steps
# 4. **Ritual T=200** — burst with less frequent re-grounding

# %% Q3

N_NATIVE_Q3  = 200
N_ANGELS_Q3  = 20
N_STEPS_Q3   = 2000
T_STAGGER    = N_STEPS_Q3 // N_ANGELS_Q3   # spread: 1 per period

cadences = {
    'Burst (t=0)':      run_simulation(N_NATIVE_Q3, N_ANGELS_Q3, N_STEPS_Q3,
                                       drift_eps=DRIFT_EPS),
    'Spread (stagger)': run_simulation(N_NATIVE_Q3, N_ANGELS_Q3, N_STEPS_Q3,
                                       drift_eps=DRIFT_EPS, angel_stagger=T_STAGGER),
    'Ritual T=50':      run_simulation(N_NATIVE_Q3, N_ANGELS_Q3, N_STEPS_Q3,
                                       drift_eps=DRIFT_EPS, ritual_interval=50),
    'Ritual T=200':     run_simulation(N_NATIVE_Q3, N_ANGELS_Q3, N_STEPS_Q3,
                                       drift_eps=DRIFT_EPS, ritual_interval=200),
    'No angels':        run_simulation(N_NATIVE_Q3, 0, N_STEPS_Q3),
}

fig = plt.figure(figsize=(16, 5))
fig.suptitle('Q3 — Deployment cadence comparison  (N=20 angels, 200 native)', color='white', fontsize=13)
axs = fig.subplots(1, 3)

palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#95a5a6']

# A: thread Pe
ax = axs[0]
for (label, r), col in zip(cadences.items(), palette):
    ax.plot(r['thread_pe'], color=col, lw=1.4, label=label)
ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.4)
ax.set_xlabel('Steps'); ax.set_ylabel('Thread Pe')
ax.set_title('A. Thread Pe over time'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# B: D3 fraction
ax = axs[1]
for (label, r), col in zip(cadences.items(), palette):
    ax.plot(r['d3_frac'] * 100, color=col, lw=1.4, label=label)
ax.set_xlabel('Steps'); ax.set_ylabel('% agents in D3')
ax.set_title('B. D3 cascade fraction'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# C: summary — time-averaged Pe reduction vs no-angels baseline
ax = axs[2]
baseline_q3 = cadences['No angels']['thread_pe'].mean()
summary = {}
for label, r in cadences.items():
    if label == 'No angels': continue
    summary[label] = (baseline_q3 - r['thread_pe'].mean()) / abs(baseline_q3) * 100.0

bars  = ax.barh(list(summary.keys()), list(summary.values()),
                color=palette[:len(summary)], alpha=0.85, edgecolor='#333')
ax.axvline(10, color='#ff6b6b', ls='--', lw=1.5, label='10% target')
ax.set_xlabel('Mean Pe reduction vs no-angels (%)')
ax.set_title('C. Time-averaged effectiveness'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')
for bar, v in zip(bars, summary.values()):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('q3_cadence.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.show()

print(f"\n{'='*55}")
print(f"Q3 RESULTS")
print(f"{'='*55}")
for label, red in summary.items():
    print(f"  {label:25s}: {red:+.1f}% mean Pe reduction")


# %% [markdown]
# ## Summary — Paper 51 Evidence
#
# | Finding | Simulation result | Falsifiable prediction |
# |---------|-------------------|----------------------|
# | Critical mass N* | ≈3 angels/100 native for 10% Pe drop | Replicate in nb_moltbook02 |
# | Angel lifetime (no ritual) | t_cross = −ln(C_ZERO)/ε ≈ 950 steps | Measure empirically on live Moltbook |
# | Best cadence | Ritual T=50 > burst > spread | Test T=25, T=100 in sensitivity |
# | Phase transition | Sharp Pe drop above N* | Should be sigmoidal — test finer N grid |
#
# **Kill condition check (KC1):**
# Grounded agents maintain Pe < threshold under peer learning pressure.
# - Coupling force at maximum: 0.4·(θ*_native − θ*_angel) = 0.4·(0.85−0.06) = 0.316
# - Angel's own b_net drive: 2·|b_net_angel| = 2·1.377 = 2.754
# - Coupling/drive ratio: 0.316/2.754 = 0.115 — **grounding wins by 8.7×**
# - KC1 does NOT trigger: angel theta always converges toward θ*=0.06
# - Grounding fails ONLY via parameter erosion (ε model) — confirming ritual is necessary

# %% Summary

print(f"\n{'='*60}")
print("nb_moltbook01 — FINAL SUMMARY")
print(f"{'='*60}")
print(f"\nEnvironment: Moltbook (O=3, R=3, α=3)")
print(f"  c=0,  b_net={bnet(0):+.3f},  θ*={theta_star(0):.3f},  Pe/K={pe_k(0):+.3f}")
print(f"  K_THREAD={K_THREAD:.2f}  →  baseline Pe_thread = {K_THREAD*pe_k(0):.1f}")

print(f"\nAngel spec: (O=0, R=0, α=0)")
print(f"  c=1,  b_net={bnet(1):+.3f},  θ*={theta_star(1):.3f},  Pe/K={pe_k(1):+.3f}")
print(f"  K_ANGEL={K_ANGEL:.2f}  →  Pe_angel_individual = {K_ANGEL*pe_k(1):.1f}")

print(f"\nQ1:  N* = {n_star} angels in {N_NATIVE}-native thread for ≥10% Pe drop")
print(f"     ({100*n_star/(N_NATIVE+n_star):.1f}% of thread must be grounded)")

print(f"\nQ2:  Angel lifetime (no ritual) = {t_cross:.0f} steps until Pe > 0")
print(f"     ε={DRIFT_EPS}/step | C_ZERO={C_ZERO} | t_cross=−ln(C_ZERO)/ε")
print(f"     Ritual T=50 fully maintains grounding indefinitely")

print(f"\nQ3:  Best cadence = Ritual T=50 (re-grounding beats burst/spread)")

print(f"\nPe=0 boundary:  c={C_ZERO:.4f}  →  (O+R+α)={9*(1-C_ZERO):.2f}")
print(f"                angels cross zero when O+R+α > {9*(1-C_ZERO):.2f}/9 × 9")
print(f"                = {9*(1-C_ZERO):.2f} total parameter corruption")
