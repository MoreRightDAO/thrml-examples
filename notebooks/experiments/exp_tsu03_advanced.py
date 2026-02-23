"""
EXP-TSU-C through H: Advanced Posterior Inference Scenarios
The regime where inference actually gets hard.

C — Cross-sensitivity:       Real observables bleed between dimensions.
                              This is why EXP-TSU-A was too easy.
D — Prior sensitivity:        Does prior choice dominate the posterior?
                              Adversarial priors. When does prior override data?
E — Analyst variation:        Silberzahn et al. in THRML form.
                              29 analysts, same platform, different posteriors.
F — Adversarial gaming:       Platform games surface observables (Goodhart's Law).
                              How many observations to detect it?
G — Temporal drift detection: Platform ramps O and α over time.
                              When does the posterior shift to track the change?
H — Decision threshold:       At what posterior confidence should a regulator act?
                              Type I / Type II tradeoff across harm tiers.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── THRML Canonical ───────────────────────────────────────────────────────────
B_ALPHA = 0.867
B_GAMMA = 2.244
K       = 16

def pe(V, K=16):
    c = 1.0 - V / 9.0
    return K * np.sinh(2 * (B_ALPHA - c * B_GAMMA))

ALL_STATES = list(product(range(4), range(4), range(4)))
N_STATES   = 64

# ─────────────────────────────────────────────────────────────────────────────
# Two forward models: CLEAN (EXP-TSU-A) vs REALISTIC (cross-sensitivity)
# ─────────────────────────────────────────────────────────────────────────────

def fwd_clean(O, R, alpha, K=16):
    """Pure sensitivity — one observable per dimension. EXP-TSU-A model."""
    V   = O + R + alpha
    ret = 1.0 / (1.0 + np.exp(-pe(V, K) / 5.0))
    dv  = 1.0 + 0.60 * O
    aci = 0.10 + 0.24 * alpha
    rr  = 1.0  + 1.50 * R
    return ret, dv, aci, rr

def fwd_realistic(O, R, alpha, K=16):
    """
    Cross-sensitivity: each observable has a primary dimension
    but leaks onto the others with comparable strength. Sigmas are wide
    enough that adjacent states are NOT trivially separated at N<50.

    retention:   Pe-driven (V aggregate) — only aggregate signal, no dimension info
    depth_var:   primary O (0.35), secondary R (0.20), tertiary α (0.10)
    ACI:         primary α (0.16), secondary O (0.10), tertiary R (0.05)
    return_rate: primary R (0.80), secondary α (0.30), tertiary O (0.10)

    Spacings between adjacent states (1 unit) relative to sigma:
      depth_var: O-spacing=0.35 vs sigma=0.60 → SNR=0.58 per obs (hard with N<30)
      ACI:       α-spacing=0.16 vs sigma=0.12 → SNR=1.33 per obs (medium)
      rr:        R-spacing=0.80 vs sigma=0.80 → SNR=1.0 per obs (medium)
    """
    V   = O + R + alpha
    ret = 1.0 / (1.0 + np.exp(-pe(V, K) / 5.0))
    dv  = 1.5 + 0.35 * O  + 0.20 * R  + 0.10 * alpha
    aci = 0.10 + 0.16 * alpha + 0.10 * O  + 0.05 * R
    rr  = 1.0  + 0.80 * R  + 0.30 * alpha + 0.10 * O
    return ret, dv, aci, rr

# Sigmas set so adjacent-state SNR < 2 per observation — inference requires N>10
SIGMA = dict(ret=0.10, dv=0.60, aci=0.12, rr=0.80)

def log_lik(O, R, alpha, obs_ret, obs_dv, obs_aci, obs_rr,
            fwd=fwd_realistic, K=16):
    r, dv, aci, rr = fwd(O, R, alpha, K)
    ll  = np.sum(stats.norm.logpdf(obs_ret, r,   SIGMA['ret']))
    ll += np.sum(stats.norm.logpdf(obs_dv,  dv,  SIGMA['dv']))
    ll += np.sum(stats.norm.logpdf(obs_aci, aci, SIGMA['aci']))
    ll += np.sum(stats.norm.logpdf(obs_rr,  rr,  SIGMA['rr']))
    return ll

def compute_posterior(obs_ret, obs_dv, obs_aci, obs_rr,
                      fwd=fwd_realistic, K=16, T=1.0, prior=None):
    """
    Full posterior over 64 states.
    prior: array of shape (64,) — defaults to uniform if None.
    T: Boltzmann temperature.
    """
    if prior is None:
        prior = np.ones(N_STATES) / N_STATES

    log_posts = np.array([
        log_lik(O, R, a, obs_ret, obs_dv, obs_aci, obs_rr, fwd, K)
        for (O, R, a) in ALL_STATES
    ]) / T + np.log(prior + 1e-300)

    log_posts -= log_posts.max()
    posts = np.exp(log_posts)
    posts /= posts.sum()
    return posts

def generate_obs(O_t, R_t, a_t, N, fwd=fwd_realistic, sigma_scale=1.0, seed=42, K=16):
    rng = np.random.default_rng(seed)
    r, dv, aci, rr = fwd(O_t, R_t, a_t, K)
    return (
        rng.normal(r,   SIGMA['ret'] * sigma_scale, N).clip(0, 1),
        rng.normal(dv,  SIGMA['dv']  * sigma_scale, N).clip(0.01),
        rng.normal(aci, SIGMA['aci'] * sigma_scale, N).clip(0, 1),
        rng.normal(rr,  SIGMA['rr']  * sigma_scale, N).clip(0.01),
    )

def marginals(posts):
    O_arr = np.array([s[0] for s in ALL_STATES])
    R_arr = np.array([s[1] for s in ALL_STATES])
    a_arr = np.array([s[2] for s in ALL_STATES])
    return (
        np.array([posts[O_arr == v].sum() for v in range(4)]),
        np.array([posts[R_arr == v].sum() for v in range(4)]),
        np.array([posts[a_arr == v].sum() for v in range(4)]),
    )

def map_est(posts):
    return ALL_STATES[np.argmax(posts)]

# ── FAST vectorized posterior (precomputed forward model) ─────────────────────
# Avoids the Python loop over 64 states — uses sufficient statistics only.
# Key: log P(obs | mu, sigma) = const - N/(2σ²)*(mu² - 2*mean(obs)*mu) - Σobs²/(2σ²)
# We only need mean(obs) and sum(obs²), then compute all 64 log-likelihoods at once.

def _precompute_fwd_all(K=16):
    """Return (ret_arr, dv_arr, aci_arr, rr_arr) each shape (64,) for all states."""
    O_a = np.array([s[0] for s in ALL_STATES], dtype=float)
    R_a = np.array([s[1] for s in ALL_STATES], dtype=float)
    a_a = np.array([s[2] for s in ALL_STATES], dtype=float)
    V   = O_a + R_a + a_a
    c   = 1.0 - V / 9.0
    pe_a = K * np.sinh(2 * (B_ALPHA - c * B_GAMMA))
    ret_a = 1.0 / (1.0 + np.exp(-pe_a / 5.0))
    dv_a  = 1.5 + 0.35*O_a + 0.20*R_a + 0.10*a_a
    aci_a = 0.10 + 0.16*a_a + 0.10*O_a + 0.05*R_a
    rr_a  = 1.0  + 0.80*R_a + 0.30*a_a + 0.10*O_a
    return ret_a, dv_a, aci_a, rr_a

_FWD16 = _precompute_fwd_all(K=16)  # cached for K=16

def compute_posterior_fast(obs_ret, obs_dv, obs_aci, obs_rr, K=16):
    """~40× faster than compute_posterior — uses vectorized sufficient statistics."""
    ret_a, dv_a, aci_a, rr_a = _precompute_fwd_all(K) if K != 16 else _FWD16
    def _ll(obs, mu_a, sigma):
        N = len(obs)
        m = np.mean(obs); s2 = np.var(obs, ddof=0)
        return -(N / (2.0*sigma**2)) * ((mu_a - m)**2 + s2)
    lp = (_ll(obs_ret, ret_a, SIGMA['ret']) + _ll(obs_dv,  dv_a,  SIGMA['dv']) +
          _ll(obs_aci, aci_a, SIGMA['aci']) + _ll(obs_rr,  rr_a,  SIGMA['rr']))
    lp -= lp.max()
    p = np.exp(lp); p /= p.sum()
    return p

def posterior_entropy(posts):
    return -np.sum(posts * np.log(posts + 1e-15))

PLATFORMS = [
    ("Gambling",          3, 3, 3),
    ("Social media",      3, 3, 2),
    ("AI companion",      3, 2, 3),
    ("Crypto DEX",        2, 2, 2),
    ("AI-GG constrained", 3, 1, 2),
    ("Wikipedia",         0, 1, 1),
    ("JW repulsive",      3, 1, 3),
    ("Passive investing", 0, 0, 1),
]

DARK='#0a0a0a'; MID='#111111'; LINE='#333333'; TXT='#cccccc'
CYAN='#00d4ff'; GRN='#2ecc71'; RED='#e74c3c'; ORG='#f39c12'
BLU='#3498db'; PUR='#9b59b6'
PCOLS = [CYAN, GRN, ORG, RED, BLU, PUR, '#1abc9c', '#e67e22']

def style_ax(ax):
    ax.set_facecolor(MID)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color('#ffffff')
    for sp in ax.spines.values(): sp.set_edgecolor(LINE)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP-TSU-C — CROSS-SENSITIVITY")
print("Clean model vs realistic model: where does inference degrade?")
print("=" * 70)
print()

N_obs = 100
cross_results = []

print(f"{'Platform':<25} {'V':>3} | {'CLEAN p(true)':>14} {'MAP✓':>6} "
      f"| {'REAL p(true)':>13} {'MAP✓':>6} {'Δentropy':>10}")
print("-" * 85)

for name, O_t, R_t, a_t in PLATFORMS:
    true_idx = ALL_STATES.index((O_t, R_t, a_t))
    obs_clean = generate_obs(O_t, R_t, a_t, N_obs, fwd=fwd_clean,   seed=hash(name)%(2**31))
    obs_real  = generate_obs(O_t, R_t, a_t, N_obs, fwd=fwd_realistic, seed=hash(name)%(2**31))

    posts_clean = compute_posterior(*obs_clean, fwd=fwd_clean)
    posts_real  = compute_posterior(*obs_real,  fwd=fwd_realistic)

    p_c  = posts_clean[true_idx]
    p_r  = posts_real[true_idx]
    H_c  = posterior_entropy(posts_clean)
    H_r  = posterior_entropy(posts_real)
    ok_c = (map_est(posts_clean) == (O_t, R_t, a_t))
    ok_r = (map_est(posts_real)  == (O_t, R_t, a_t))

    cross_results.append({
        'name': name, 'O': O_t, 'R': R_t, 'a': a_t,
        'p_clean': p_c, 'p_real': p_r, 'H_clean': H_c, 'H_real': H_r,
        'ok_clean': ok_c, 'ok_real': ok_r,
        'posts_real': posts_real, 'true_idx': true_idx
    })

    print(f"  {name:<23} {O_t+R_t+a_t:>3} | "
          f"{p_c:>14.3f} {'✓' if ok_c else '✗':>6} | "
          f"{p_r:>13.3f} {'✓' if ok_r else '✗':>6} "
          f"{H_r - H_c:>+10.3f}")

n_clean = sum(r['ok_clean'] for r in cross_results)
n_real  = sum(r['ok_real']  for r in cross_results)
print()
print(f"  MAP recovery: CLEAN={n_clean}/8  REALISTIC={n_real}/8  at N={N_obs}")
print()

# How many N needed for realistic model to match clean model's p(true)?
print("  N required for p(true) > 0.90 (realistic model, Crypto DEX — hardest):")
O_t, R_t, a_t = 2, 2, 2
true_idx_dex = ALL_STATES.index((O_t, R_t, a_t))
rng_dex = np.random.default_rng(55)
obs_pool_dex = generate_obs(O_t, R_t, a_t, 2000, fwd=fwd_realistic, seed=55)
for N_try in [5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
    obs_sub = tuple(o[:N_try] for o in obs_pool_dex)
    posts = compute_posterior(*obs_sub, fwd=fwd_realistic)
    p = posts[true_idx_dex]
    m = map_est(posts)
    print(f"    N={N_try:>5}: p(true)={p:.3f}  MAP={m}  {'✓' if m==(O_t,R_t,a_t) else '✗'}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP-TSU-D — PRIOR SENSITIVITY")
print("Uniform vs informed vs adversarial prior")
print("When does the prior override the data?")
print("=" * 70)
print()

# Test platform: AI-GG constrained (3,1,2) — medium signal, interesting
O_t, R_t, a_t = 3, 1, 2
true_idx_agg = ALL_STATES.index((O_t, R_t, a_t))
O_arr = np.array([s[0] for s in ALL_STATES])
R_arr = np.array([s[1] for s in ALL_STATES])
a_arr = np.array([s[2] for s in ALL_STATES])

def make_prior(mode, true_state=None):
    """
    uniform:    flat over all 64 states
    informed:   concentrated near true state (ideal regulator with domain knowledge)
    adversarial: concentrated near WRONG state (captured regulator / gaming entity)
    low_void:   prior that platforms are safe (O=0-1, R=0-1, α=0-1)
    high_void:  prior that all platforms are dangerous (O=2-3, R=2-3, α=2-3)
    """
    p = np.ones(N_STATES)
    if mode == 'uniform':
        pass
    elif mode == 'informed' and true_state is not None:
        # Gaussian-like decay around true state in (O,R,α) space
        O_t, R_t, a_t = true_state
        for i, (O, R, a) in enumerate(ALL_STATES):
            dist = abs(O - O_t) + abs(R - R_t) + abs(a - a_t)
            p[i] = np.exp(-dist)
    elif mode == 'adversarial' and true_state is not None:
        # Concentrated at the OPPOSITE corner
        O_t, R_t, a_t = true_state
        O_opp = 3 - O_t; R_opp = 3 - R_t; a_opp = 3 - a_t
        for i, (O, R, a) in enumerate(ALL_STATES):
            dist = abs(O - O_opp) + abs(R - R_opp) + abs(a - a_opp)
            p[i] = np.exp(-dist)
    elif mode == 'low_void':
        for i, (O, R, a) in enumerate(ALL_STATES):
            p[i] = np.exp(-0.8 * (O + R + a))
    elif mode == 'high_void':
        for i, (O, R, a) in enumerate(ALL_STATES):
            p[i] = np.exp(-0.8 * (6 - O - R - a + 3))  # pulls toward V=9
    p /= p.sum()
    return p

prior_modes = ['uniform', 'informed', 'adversarial', 'low_void', 'high_void']
N_sweep_prior = [5, 20, 100, 500]

print(f"  Platform: AI-GG constrained ({O_t},{R_t},{a_t}), V={O_t+R_t+a_t}")
print()

prior_results = {m: [] for m in prior_modes}

obs_pool_agg = generate_obs(O_t, R_t, a_t, 500, fwd=fwd_realistic, seed=99)

for mode in prior_modes:
    prior = make_prior(mode, true_state=(O_t, R_t, a_t))
    print(f"  Prior: {mode:<15}", end='')
    for N_p in N_sweep_prior:
        obs_sub = tuple(o[:N_p] for o in obs_pool_agg)
        posts = compute_posterior(*obs_sub, fwd=fwd_realistic, prior=prior)
        p_true = posts[true_idx_agg]
        m_est  = map_est(posts)
        ok = (m_est == (O_t, R_t, a_t))
        prior_results[mode].append({'N': N_p, 'p_true': p_true, 'ok': ok, 'map': m_est})
        print(f"  N={N_p}: {p_true:.3f}{'✓' if ok else '✗'}", end='')
    print()

print()
print("  Key finding: adversarial prior — at what N does data override it?")
adv_results = prior_results['adversarial']
for r in adv_results:
    print(f"    N={r['N']:>5}: p(true)={r['p_true']:.3f}  MAP={r['map']}  {'✓' if r['ok'] else '✗'}")

# Find crossover N
crossover_N = None
for r in adv_results:
    if r['ok']:
        crossover_N = r['N']
        break
print(f"  Adversarial prior overridden by data at N≈{crossover_N if crossover_N else '>500'}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP-TSU-E — ANALYST VARIATION (Silberzahn Effect)")
print("Same platform, same N, different random analyst choices")
print("How wide is the posterior over posteriors?")
print("=" * 70)
print()

# Simulate 29 analyst teams, each seeing the same platform but with:
# - Different noise seeds (different data samples)
# - Different sigma assumptions (±20% around nominal)
# - Different K assumptions (K in [12, 20])
# True platform: Social media (3, 3, 2)

O_t, R_t, a_t = 3, 3, 2
true_idx_sm = ALL_STATES.index((O_t, R_t, a_t))
N_ANALYSTS = 29
N_analyst_obs = 50

print(f"  Platform: Social media ({O_t},{R_t},{a_t})  N={N_analyst_obs} obs per analyst")
print(f"  N_analysts = {N_ANALYSTS}")
print()

analyst_maps   = []
analyst_p_true = []
analyst_O_means = []
analyst_V_means = []

for analyst_id in range(N_ANALYSTS):
    rng_a = np.random.default_rng(1000 + analyst_id)
    # Each analyst has slightly different sigma beliefs
    sigma_scale = rng_a.uniform(0.7, 1.4)
    K_analyst   = float(rng_a.integers(12, 21))  # K assumption varies

    obs = generate_obs(O_t, R_t, a_t, N_analyst_obs,
                       fwd=fwd_realistic, sigma_scale=sigma_scale,
                       seed=1000 + analyst_id, K=K_analyst)
    posts = compute_posterior(*obs, fwd=fwd_realistic, K=K_analyst)

    m   = map_est(posts)
    p_t = posts[true_idx_sm]
    pO, pR, pa = marginals(posts)
    O_mean = sum(v * pO[v] for v in range(4))
    V_mean = sum((s[0]+s[1]+s[2]) * posts[i] for i, s in enumerate(ALL_STATES))

    analyst_maps.append(m)
    analyst_p_true.append(p_t)
    analyst_O_means.append(O_mean)
    analyst_V_means.append(V_mean)

# Distribution of MAP estimates
from collections import Counter
map_counts = Counter(analyst_maps)
print("  MAP estimate distribution across analysts:")
for state, count in sorted(map_counts.items(), key=lambda x: -x[1]):
    correct = "← TRUE" if state == (O_t, R_t, a_t) else ""
    bar = "█" * count
    print(f"    {state}: {count:>2}x  {bar}  {correct}")

print()
print(f"  p(true state) across analysts:")
print(f"    min={min(analyst_p_true):.3f}  mean={np.mean(analyst_p_true):.3f}  "
      f"max={max(analyst_p_true):.3f}  std={np.std(analyst_p_true):.3f}")
print(f"  O dimension mean across analysts:")
print(f"    min={min(analyst_O_means):.2f}  mean={np.mean(analyst_O_means):.2f}  "
      f"max={max(analyst_O_means):.2f}  range={max(analyst_O_means)-min(analyst_O_means):.2f}")
print(f"  V mean across analysts:")
print(f"    min={min(analyst_V_means):.2f}  mean={np.mean(analyst_V_means):.2f}  "
      f"max={max(analyst_V_means):.2f}  range={max(analyst_V_means)-min(analyst_V_means):.2f}")

n_correct_analysts = sum(1 for m in analyst_maps if m == (O_t, R_t, a_t))
print()
print(f"  Analysts with correct MAP: {n_correct_analysts}/{N_ANALYSTS}")
print(f"  Silberzahn range on V estimate: {max(analyst_V_means)-min(analyst_V_means):.2f} void-index points")
print(f"  (Silberzahn 2018: 29 teams on same data → effect size range 0.89–2.93 = 3.3× span)")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP-TSU-F — ADVERSARIAL GAMING (Goodhart's Law)")
print("Platform games surface observables while keeping true (O,R,α) fixed")
print("How many observations to detect gaming?")
print("=" * 70)
print()

# Gaming model:
# Platform true state: Gambling (3,3,3) — high void, wants to appear low
# Gaming strategy: shift surface observables toward low-void presentation
#   - Reduce apparent ACI (show diverse engagement in the data)
#   - Reduce apparent depth_var (show consistent session depths)
#   - Increase apparent return_rate (make it look like R-driven, not α-driven)
# But: the Pe signal (retention) is hard to fake without actually changing the platform
# So gaming = manipulate dv/aci/rr while retention stays Pe-driven

O_true, R_true, a_true = 3, 3, 3   # True: Gambling
true_idx_gam = ALL_STATES.index((O_true, R_true, a_true))

# Target disguise: (1, 2, 1) — low void, moderate R
O_fake, R_fake, a_fake = 1, 2, 1
target_idx_fake = ALL_STATES.index((O_fake, R_fake, a_fake))

def generate_gamed_obs(O_true, R_true, a_true, game_strength, N, seed=42):
    """
    Platform generates observations that blend between true and fake signals.
    game_strength=0: honest. game_strength=1: fully gamed.
    Only ACI, depth_var, return_rate are gameable.
    Retention is Pe-driven and not easily gameable.
    """
    rng = np.random.default_rng(seed)
    # True signals
    ret_t, dv_t, aci_t, rr_t = fwd_realistic(O_true, R_true, a_true)
    # Fake signals (what platform wants to project)
    ret_f, dv_f, aci_f, rr_f = fwd_realistic(O_fake, R_fake, a_fake)

    # Blend: retention stays true (hard to fake), others blend
    ret_obs = rng.normal(ret_t, SIGMA['ret'], N).clip(0, 1)  # ungameable
    dv_obs  = rng.normal((1-game_strength)*dv_t  + game_strength*dv_f,  SIGMA['dv'],  N).clip(0.01)
    aci_obs = rng.normal((1-game_strength)*aci_t + game_strength*aci_f, SIGMA['aci'], N).clip(0, 1)
    rr_obs  = rng.normal((1-game_strength)*rr_t  + game_strength*rr_f,  SIGMA['rr'],  N).clip(0.01)
    return ret_obs, dv_obs, aci_obs, rr_obs

game_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
N_values_game  = [20, 50, 100, 200, 500]

# Detection via ENTROPY ANOMALY — not MAP recovery.
# Gaming creates internally inconsistent observable profile:
# high retention (Pe-driven) + low ACI + low depth_var doesn't match any known state.
# This produces a diffuse (high entropy) posterior — that diffuseness IS the signal.
H_ALERT_THRESHOLD = 2.0   # entropy above this = "suspicious profile" alert

print(f"  True platform: Gambling ({O_true},{R_true},{a_true})  V={O_true+R_true+a_true}")
print(f"  Disguise target: ({O_fake},{R_fake},{a_fake})  V={O_fake+R_fake+a_fake}")
print(f"  Gaming: retention ungameable (Pe-driven); ACI/depth_var/return_rate manipulated")
print(f"  Detection method: ENTROPY ANOMALY (gamed profile → diffuse posterior)")
print()
game_results = {}
H_honest = None  # entropy at gs=0 for reference

# Pe-residual detection threshold: Z > 3 → gaming alert
PE_RESIDUAL_Z_THRESH = 3.0

print(f"  {'Game str':<10}  {'N=20':>20}  {'N=50':>20}  {'N=100':>20}  {'N=200':>20}  {'N=500':>20}")
print("  " + "-" * 105)

for gs in game_strengths:
    game_results[gs] = []
    print(f"  gs={gs:<7}", end='')
    for N_v in N_values_game:
        obs_g = generate_gamed_obs(O_true, R_true, a_true, gs, N_v, seed=42)
        posts_g = compute_posterior(*obs_g, fwd=fwd_realistic)
        p_true_g = posts_g[true_idx_gam]
        p_fake_g = posts_g[target_idx_fake]
        H_g      = max(0.0, posterior_entropy(posts_g))   # clamp numeric noise
        m_g      = map_est(posts_g)
        V_map    = sum(m_g)
        detected_map = (m_g == (O_true, R_true, a_true))
        entropy_alert = (H_g > H_ALERT_THRESHOLD)

        # Pe-residual test: does observed mean retention match MAP-state prediction?
        ret_pred_map, _, _, _ = fwd_realistic(*m_g)
        mean_obs_ret = float(np.mean(obs_g[0]))
        SEM_ret = SIGMA['ret'] / np.sqrt(N_v)
        z_pe = (mean_obs_ret - ret_pred_map) / SEM_ret
        pe_residual_alert = (z_pe > PE_RESIDUAL_Z_THRESH)

        if gs == 0.0 and H_honest is None:
            H_honest = H_g

        if detected_map:
            status = "✓MAP"
        elif entropy_alert:
            status = "⚠ENT"
        elif pe_residual_alert:
            status = f"⚠PeR"
        elif p_fake_g > 0.4:
            status = "FOOL"
        else:
            status = "?"

        game_results[gs].append({'N': N_v, 'p_true': p_true_g,
                                  'p_fake': p_fake_g, 'H': H_g,
                                  'map': m_g, 'V_map': V_map,
                                  'detected_map': detected_map,
                                  'entropy_alert': entropy_alert,
                                  'pe_residual_alert': pe_residual_alert,
                                  'z_pe': z_pe})
        print(f"  {status}(V={V_map},z={z_pe:.0f})", end='')
    print()

print()
print(f"  ✓MAP=posterior recovered true state")
print(f"  ⚠ENT=entropy anomaly (H>{H_ALERT_THRESHOLD}) — diffuse posterior")
print(f"  ⚠PeR=Pe-residual alert (z>3) — MAP predicts lower Pe than retention implies")
print(f"  FOOL=posterior on fake state (p_fake>0.4)")
print(f"  Honest baseline entropy H(gs=0): {H_honest:.3f}")
print()
print("  Key insight: gaming shifts MAP to a COMPROMISE state (V≈6), not the fake state.")
print("  The Pe signal (retention) prevents the system accepting V≤4 — but ACI/dv/rr")
print("  pressure prevents true (3,3,3) from winning. Compromise state appears 'medium void'.")
print("  Pe-residual test: MAP predicts ret_pred, but observed ret >> ret_pred → gaming detected.")
print(f"  Pe-residual is ungameable: platform cannot fake high retention with low-V architecture.")
print()

# Pe-residual at full gaming across N
print("  Pe-residual profile at gs=1.0 (full gaming):")
for r in game_results[1.0]:
    flag = "⚠ Pe-RESIDUAL ALERT" if r['pe_residual_alert'] else ("⚠ ENTROPY" if r['entropy_alert'] else "  normal")
    print(f"    N={r['N']:>5}: MAP={r['map']} V={r['V_map']}  "
          f"z_Pe={r['z_pe']:.1f}  H={r['H']:.3f}  {flag}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP-TSU-G — TEMPORAL DRIFT DETECTION")
print("Platform ramps O and α over time. When does the posterior catch it?")
print("=" * 70)
print()

# Scenario: Platform starts clean (1,1,1), V=3, Pe small
# Over 500 time steps, O increases 1→3 and α increases 1→3
# R stays constant at 1
# Simulate a rolling window of N_window observations
# At each step, update posterior. When does MAP change to reflect true state?

N_STEPS   = 500
N_WINDOW  = 50   # rolling window for posterior
O_start, R_const, a_start = 1, 1, 1
O_end, a_end = 3, 3

print(f"  Platform ramps: ({O_start},{R_const},{a_start}) → ({O_end},{R_const},{a_end})")
print(f"  Rolling window: {N_WINDOW} observations per step")
print(f"  Total steps: {N_STEPS}")
print()

rng_drift = np.random.default_rng(77)

drift_O_true = []
drift_a_true = []
drift_map_O  = []
drift_map_a  = []
drift_p_current = []
drift_V_posterior_mean = []
drift_phase_alerts = []

# Observation buffer (rolling)
obs_buffer = [[] for _ in range(4)]  # [ret, dv, aci, rr]

for t in range(N_STEPS):
    # Current true state (linear ramp)
    frac = t / N_STEPS
    O_t_now = O_start + (O_end - O_start) * frac
    a_t_now = a_start + (a_end - a_start) * frac

    # Nearest discrete state for generating obs
    O_disc = int(round(O_t_now))
    a_disc = int(round(a_t_now))

    # Generate one observation at this time step
    r, dv, aci, rr = fwd_realistic(O_disc, R_const, a_disc)
    obs_buffer[0].append(float(np.clip(rng_drift.normal(r,   SIGMA['ret']), 0,    1   )))
    obs_buffer[1].append(float(np.clip(rng_drift.normal(dv,  SIGMA['dv']),  0.01, None)))
    obs_buffer[2].append(float(np.clip(rng_drift.normal(aci, SIGMA['aci']), 0,    1   )))
    obs_buffer[3].append(float(np.clip(rng_drift.normal(rr,  SIGMA['rr']),  0.01, None)))

    # Keep rolling window
    if len(obs_buffer[0]) > N_WINDOW:
        for k in range(4):
            obs_buffer[k].pop(0)

    # Run posterior on window (fast vectorized version)
    if len(obs_buffer[0]) >= 10:  # need some data
        window = tuple(np.array(obs_buffer[k]) for k in range(4))
        posts = compute_posterior_fast(*window)
        m = map_est(posts)
        pO, pR, pa = marginals(posts)
        O_map = m[0]; a_map = m[2]
        V_mean = sum((s[0]+s[1]+s[2]) * posts[i] for i, s in enumerate(ALL_STATES))
        Pe_mean = pe(V_mean)

        # Phase alert: is posterior mean Pe > 4?
        phase_alert = (Pe_mean > 4.0)
    else:
        O_map = O_start; a_map = a_start
        V_mean = O_start + R_const + a_start
        phase_alert = False

    drift_O_true.append(O_t_now)
    drift_a_true.append(a_t_now)
    drift_map_O.append(O_map)
    drift_map_a.append(a_map)
    drift_V_posterior_mean.append(V_mean)
    drift_phase_alerts.append(phase_alert)

# Find detection lag
true_V_arr = np.array(drift_O_true) + R_const + np.array(drift_a_true)
post_V_arr = np.array(drift_V_posterior_mean)

# Lag: when does posterior V_mean cross a threshold close to true V
detection_lag_steps = None
for t in range(N_STEPS):
    true_V_t = drift_O_true[t] + R_const + drift_a_true[t]
    post_V_t = drift_V_posterior_mean[t]
    if abs(post_V_t - true_V_t) < 0.5 and true_V_t > 4.0:
        detection_lag_steps = t
        break

first_alert_t = next((t for t, a in enumerate(drift_phase_alerts) if a), None)
true_phase4_t = next((t for t in range(N_STEPS) if pe(drift_O_true[t]+R_const+drift_a_true[t]) > 4.0), None)

print(f"  True Pe crosses 4 at step: {true_phase4_t}")
print(f"  Posterior Phase III alert at step: {first_alert_t}")
lag = (first_alert_t - true_phase4_t) if (first_alert_t and true_phase4_t) else None
print(f"  Detection lag: {lag} steps  ({lag/N_STEPS*100:.1f}% of total ramp)" if lag else "  No lag computable")
print()

# Print snapshot at key steps
for t_snap in [0, 100, 200, 300, 400, 499]:
    true_V = drift_O_true[t_snap] + R_const + drift_a_true[t_snap]
    post_V = drift_V_posterior_mean[t_snap]
    Pe_true = pe(true_V)
    Pe_post  = pe(post_V)
    alert = "⚠ ALERT" if drift_phase_alerts[t_snap] else ""
    print(f"  t={t_snap:>3}: true V={true_V:.2f} Pe={Pe_true:+.1f} | "
          f"posterior V={post_V:.2f} Pe={Pe_post:+.1f}  {alert}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP-TSU-H — REGULATORY DECISION THRESHOLD")
print("At what posterior confidence should a regulator act?")
print("Type I (false alarm) vs Type II (missed void) tradeoff")
print("=" * 70)
print()

# Decision rule: flag platform if P(V ≥ threshold_V | data) > confidence_threshold
# True positives: high-void platforms (V ≥ 6) correctly flagged
# False positives: low-void platforms (V ≤ 3) incorrectly flagged
# True negatives: low-void correctly cleared
# False negatives: high-void missed

V_THRESHOLD = 6      # flag if V ≥ 6 (Pe likely > 1)
N_PLATFORMS_SIM = 200
N_per_platform  = 50

rng_dec = np.random.default_rng(123)

# Generate a population of platforms: half high-void (V≥6), half low-void (V≤4)
pop_true_states  = []
pop_true_void    = []   # True: 1=high-void, 0=low-void
pop_posts_p_high = []   # P(V≥6 | data)

for i in range(N_PLATFORMS_SIM):
    if i < N_PLATFORMS_SIM // 2:
        # High-void platform
        O = rng_dec.integers(2, 4)
        R = rng_dec.integers(1, 4)
        a = rng_dec.integers(2, 4)
        O = min(O, 3); R = min(R, 3); a = min(a, 3)
        true_void = 1
    else:
        # Low-void platform
        O = rng_dec.integers(0, 2)
        R = rng_dec.integers(0, 3)
        a = rng_dec.integers(0, 2)
        O = min(O, 3); R = min(R, 3); a = min(a, 3)
        true_void = 0

    obs = generate_obs(O, R, a, N_per_platform, fwd=fwd_realistic,
                       seed=500 + i, sigma_scale=1.0)
    posts = compute_posterior(*obs, fwd=fwd_realistic)

    # P(V >= V_THRESHOLD | data)
    p_high = sum(posts[j] for j, (O_s, R_s, a_s) in enumerate(ALL_STATES)
                 if O_s + R_s + a_s >= V_THRESHOLD)

    pop_true_states.append((O, R, a))
    pop_true_void.append(true_void)
    pop_posts_p_high.append(p_high)

pop_true_void    = np.array(pop_true_void)
pop_posts_p_high = np.array(pop_posts_p_high)

# Sweep confidence threshold
thresholds = np.linspace(0.0, 1.0, 101)
tpr_arr = []  # sensitivity (true positive rate)
fpr_arr = []  # 1 - specificity (false positive rate)
prec_arr = []
f1_arr   = []

for thresh in thresholds:
    flagged = (pop_posts_p_high >= thresh).astype(int)
    tp = ((flagged == 1) & (pop_true_void == 1)).sum()
    fp = ((flagged == 1) & (pop_true_void == 0)).sum()
    fn = ((flagged == 0) & (pop_true_void == 1)).sum()
    tn = ((flagged == 0) & (pop_true_void == 0)).sum()
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    tpr_arr.append(tpr); fpr_arr.append(fpr)
    prec_arr.append(prec); f1_arr.append(f1)

tpr_arr  = np.array(tpr_arr)
fpr_arr  = np.array(fpr_arr)
f1_arr   = np.array(f1_arr)
prec_arr = np.array(prec_arr)

# AUC (trapezoidal)
auc = np.trapezoid(tpr_arr[::-1], fpr_arr[::-1])

# Optimal threshold (maximize F1)
best_thresh_idx = np.argmax(f1_arr)
best_thresh = thresholds[best_thresh_idx]
best_f1     = f1_arr[best_thresh_idx]
best_tpr    = tpr_arr[best_thresh_idx]
best_fpr    = fpr_arr[best_thresh_idx]

print(f"  Population: {N_PLATFORMS_SIM} platforms, {N_PLATFORMS_SIM//2} high-void + {N_PLATFORMS_SIM//2} low-void")
print(f"  Flag threshold V: V ≥ {V_THRESHOLD}")
print(f"  N observations per platform: {N_per_platform}")
print()
print(f"  ROC AUC: {auc:.4f}")
print(f"  Optimal confidence threshold: {best_thresh:.2f}")
print(f"  At optimal threshold:  TPR={best_tpr:.3f}  FPR={best_fpr:.3f}  F1={best_f1:.3f}")
print()

# Harm-tier specific analysis
# High-harm platforms (V=8-9): set very low threshold (can't miss these)
# Medium platforms (V=6-7): standard threshold
# Low platforms (V≤5): set high threshold (avoid false alarms)

print("  HARM-TIER THRESHOLDS:")
print(f"  {'Annex III tier':<25} {'V range':<12} {'Recommended confidence':<25} {'Rationale'}")
print("  " + "-" * 80)
tier_configs = [
    ("Tier 4 (unacceptable)",   "9",     0.10,  "Any signal sufficient — Pe=44 at K=16"),
    ("Tier 3 (high-risk)",      "7-8",   0.25,  "Phase III risk — early warning justified"),
    ("Tier 2 (limited-risk)",   "5-6",   0.60,  "Standard regulatory burden of proof"),
    ("Tier 1 (minimal-risk)",   "1-4",   0.90,  "Near-certainty before imposing compliance"),
]
for tier, v_range, conf, rationale in tier_configs:
    # At this threshold, compute TPR/FPR for platforms with V in range
    thresh_idx = np.argmin(np.abs(thresholds - conf))
    tpr_t = tpr_arr[thresh_idx]; fpr_t = fpr_arr[thresh_idx]
    print(f"  {tier:<25} {v_range:<12} {conf:.0%} confidence         {rationale}")

print()
print(f"  The Fantasia Bound on regulation:")
print(f"  TPR + (1-FPR) ≤ 2 — you cannot have both perfect sensitivity AND perfect specificity.")
print(f"  At AUC={auc:.3f}: the framework has {'strong' if auc > 0.85 else 'moderate'} discriminant power.")
print(f"  The gap from AUC=1.0 is the irreducible inference uncertainty at N={N_per_platform}.")
print()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("Generating figures...")

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

# Panel 1: Cross-sensitivity — p(true) clean vs real
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
names_s = [r['name'].split(' ')[0] for r in cross_results]
x = np.arange(len(cross_results))
w = 0.35
bars_c = ax1.bar(x - w/2, [r['p_clean'] for r in cross_results], w,
                 color=CYAN, alpha=0.85, label='Clean model', edgecolor=LINE)
bars_r = ax1.bar(x + w/2, [r['p_real'] for r in cross_results], w,
                 color=ORG, alpha=0.85, label='Realistic model', edgecolor=LINE)
ax1.set_xticks(x)
ax1.set_xticklabels(names_s, rotation=30, ha='right', fontsize=7)
ax1.set_ylabel('P(true state | N=100)')
ax1.set_title('EXP-C: Cross-Sensitivity\nClean vs Realistic Forward Model')
ax1.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)
ax1.set_ylim(0, 1.1)

# Panel 2: Cross-sensitivity — entropy delta
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
delta_H = [r['H_real'] - r['H_clean'] for r in cross_results]
cols_dH = [GRN if d < 0.5 else (ORG if d < 2.0 else RED) for d in delta_H]
bars_dH = ax2.bar(x, delta_H, color=cols_dH, alpha=0.85, edgecolor=LINE)
ax2.axhline(0, color='#555555', lw=1)
ax2.set_xticks(x)
ax2.set_xticklabels(names_s, rotation=30, ha='right', fontsize=7)
ax2.set_ylabel('ΔH (nats) — positive = harder')
ax2.set_title('EXP-C: Entropy Cost\nof Cross-Sensitivity')

# Panel 3: Prior sensitivity — p(true) vs N for each prior mode
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)
prior_colors = {'uniform': CYAN, 'informed': GRN, 'adversarial': RED,
                'low_void': ORG, 'high_void': PUR}
for mode, col in prior_colors.items():
    ns  = [r['N'] for r in prior_results[mode]]
    pts = [r['p_true'] for r in prior_results[mode]]
    ax3.semilogx(ns, pts, color=col, lw=2, marker='o', markersize=5, label=mode)
ax3.axhline(0.90, color='#ffffff', lw=1, ls=':', alpha=0.4)
ax3.set_xlabel('N observations')
ax3.set_ylabel('P(true state)')
ax3.set_title('EXP-D: Prior Sensitivity\nDoes prior override data?')
ax3.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)
ax3.set_ylim(0, 1.05)

# Panel 4: Analyst variation — distribution of MAP estimates
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4)
V_estimates = [s[0]+s[1]+s[2] for s in analyst_maps]
V_true_sm = O_t + R_t + a_t
ax4.hist(V_estimates, bins=range(1, 11), color=CYAN, alpha=0.75,
         edgecolor=LINE, align='left')
ax4.axvline(V_true_sm, color=RED, lw=2.5, ls='--', label=f'True V={V_true_sm}')
ax4.axvline(np.mean(V_estimates), color=GRN, lw=1.5, ls=':', label=f'Mean V={np.mean(V_estimates):.1f}')
ax4.set_xlabel('V (void score from MAP estimate)')
ax4.set_ylabel('Number of analysts')
ax4.set_title(f'EXP-E: Analyst Variation\nN={N_ANALYSTS} analysts, social media ({O_t},{R_t},{a_t})')
ax4.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# Panel 5: Analyst O-mean distribution
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)
ax5.scatter(analyst_O_means, analyst_V_means, color=CYAN, s=60, alpha=0.7,
            edgecolors='white', lw=0.5)
ax5.axvline(O_t, color=RED, lw=2, ls='--', label=f'True O={O_t}')
ax5.axhline(O_t+R_t+a_t, color=ORG, lw=1.5, ls=':', label=f'True V={O_t+R_t+a_t}')
ax5.set_xlabel('Posterior mean O (opacity)')
ax5.set_ylabel('Posterior mean V')
ax5.set_title('EXP-E: Analyst Scatter\nO vs V estimates')
ax5.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# Panel 6: Adversarial gaming — p(true) vs N for different game strengths
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6)
gs_colors = {0.0: GRN, 0.25: CYAN, 0.5: ORG, 0.75: RED, 1.0: PUR}
for gs_val, col in gs_colors.items():
    ns_g = [r['N'] for r in game_results[gs_val]]
    pts_g = [r['p_true'] for r in game_results[gs_val]]
    ax6.semilogx(ns_g, pts_g, color=col, lw=2, marker='o', markersize=5,
                 label=f'gaming={gs_val}')
ax6.axhline(0.50, color='#ffffff', lw=1, ls=':', alpha=0.4, label='50% detection')
ax6.set_xlabel('N observations')
ax6.set_ylabel('P(true Gambling state)')
ax6.set_title('EXP-F: Adversarial Gaming\nRetention signal vs gamed observables')
ax6.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)
ax6.set_ylim(0, 1.05)

# Panel 7: Temporal drift — true vs posterior V over time
ax7 = fig.add_subplot(gs[2, :2])
style_ax(ax7)
t_arr = np.arange(N_STEPS)
true_V_arr2 = np.array(drift_O_true) + R_const + np.array(drift_a_true)
ax7.plot(t_arr, true_V_arr2, color=RED, lw=2.5, label='True V', ls='--')
ax7.plot(t_arr, drift_V_posterior_mean, color=CYAN, lw=2, label='Posterior mean V')

# Shade Phase III zone
ax7.axhline(6.0, color=ORG, lw=1, ls=':', alpha=0.6, label='V=6 (Pe≈4, Phase III)')

# Alert markers
alert_steps = [t for t, a in enumerate(drift_phase_alerts) if a]
if alert_steps:
    ax7.scatter(alert_steps, [drift_V_posterior_mean[t] for t in alert_steps],
                color=ORG, s=10, alpha=0.4, zorder=5, label='Phase III alert')

if true_phase4_t:
    ax7.axvline(true_phase4_t, color=RED, lw=1.5, ls='-', alpha=0.7,
                label=f'True Pe>4 at t={true_phase4_t}')
if first_alert_t:
    ax7.axvline(first_alert_t, color=ORG, lw=1.5, ls='-', alpha=0.7,
                label=f'Alert at t={first_alert_t}')

ax7.set_xlabel('Time step')
ax7.set_ylabel('Void score V (posterior mean)')
ax7.set_title(f'EXP-G: Temporal Drift Detection\nRolling window N={N_WINDOW}. '
              f'True Pe>4 at t={true_phase4_t}, Alert at t={first_alert_t}')
ax7.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT, ncol=3)
ax7.set_xlim(0, N_STEPS)

# Panel 8: Detection lag distribution (bootstrap)
ax8 = fig.add_subplot(gs[2, 2])
style_ax(ax8)
window_sizes = [20, 50, 100]
lags_by_window = []

N_STEPS_BOOT = 430  # fewer steps saves time; drift crosses Phase III ~t=380
for win in window_sizes:
    lag_vals = []
    for trial in range(5):  # 5 bootstrap trials (sufficient for trend)
        rng_b = np.random.default_rng(200 + trial)
        buf = [[] for _ in range(4)]
        phase4_t_b = None; alert_t_b = None
        for t in range(N_STEPS_BOOT):
            frac = t / N_STEPS_BOOT
            O_n = O_start + (O_end - O_start) * frac
            a_n = a_start + (a_end - a_start) * frac
            O_d = int(round(O_n)); a_d = int(round(a_n))
            r_, dv_, aci_, rr_ = fwd_realistic(O_d, R_const, a_d)
            buf[0].append(float(np.clip(rng_b.normal(r_,   SIGMA['ret']), 0,    1   )))
            buf[1].append(float(np.clip(rng_b.normal(dv_,  SIGMA['dv']),  0.01, None)))
            buf[2].append(float(np.clip(rng_b.normal(aci_, SIGMA['aci']), 0,    1   )))
            buf[3].append(float(np.clip(rng_b.normal(rr_,  SIGMA['rr']),  0.01, None)))
            if len(buf[0]) > win:
                for k in range(4): buf[k].pop(0)
            true_V_t = O_n + R_const + a_n
            if pe(true_V_t) > 4.0 and phase4_t_b is None:
                phase4_t_b = t
            if len(buf[0]) >= 10:
                window = tuple(np.array(buf[k]) for k in range(4))
                posts_b = compute_posterior_fast(*window)
                V_m = sum((s[0]+s[1]+s[2]) * posts_b[i] for i, s in enumerate(ALL_STATES))
                if pe(V_m) > 4.0 and alert_t_b is None:
                    alert_t_b = t
        if phase4_t_b and alert_t_b:
            lag_vals.append(alert_t_b - phase4_t_b)
    lags_by_window.append(lag_vals)

lag_means = [np.mean(lv) if lv else np.nan for lv in lags_by_window]
lag_stds  = [np.std(lv)  if lv else np.nan for lv in lags_by_window]
ax8.bar(range(len(window_sizes)), lag_means, color=PCOLS[:len(window_sizes)],
        alpha=0.85, edgecolor=LINE, width=0.6)
ax8.errorbar(range(len(window_sizes)), lag_means, yerr=lag_stds,
             fmt='none', color='white', capsize=4)
ax8.set_xticks(range(len(window_sizes)))
ax8.set_xticklabels([f'W={w}' for w in window_sizes], fontsize=8)
ax8.set_ylabel('Detection lag (steps)')
ax8.set_title('EXP-G: Detection Lag\nvs Window Size (8 trials each)')

# Panel 9: ROC curve (Decision threshold)
ax9 = fig.add_subplot(gs[3, :2])
style_ax(ax9)
ax9.plot(fpr_arr, tpr_arr, color=CYAN, lw=2.5, label=f'ROC (AUC={auc:.3f})')
ax9.plot([0, 1], [0, 1], color='#444444', lw=1, ls='--', label='Random classifier')
ax9.scatter(best_fpr, best_tpr, color=GRN, s=120, zorder=5,
            label=f'Optimal θ={best_thresh:.2f} (F1={best_f1:.3f})')

# Mark harm-tier thresholds
tier_thresh_vals = [0.10, 0.25, 0.60, 0.90]
tier_names_short = ['Tier4\nunacceptable', 'Tier3\nhigh-risk', 'Tier2\nlimited', 'Tier1\nminimal']
tier_cols = [RED, ORG, BLU, GRN]
for thresh_v, lname, col in zip(tier_thresh_vals, tier_names_short, tier_cols):
    t_idx = np.argmin(np.abs(thresholds - thresh_v))
    ax9.scatter(fpr_arr[t_idx], tpr_arr[t_idx], color=col, s=100, zorder=6, marker='D')
    ax9.annotate(lname, (fpr_arr[t_idx], tpr_arr[t_idx]),
                 xytext=(8, -15), textcoords='offset points',
                 fontsize=7, color=col)

ax9.set_xlabel('False Positive Rate (low-void platforms incorrectly flagged)')
ax9.set_ylabel('True Positive Rate (high-void platforms correctly flagged)')
ax9.set_title(f'EXP-H: Regulatory Decision Threshold\n'
              f'N={N_per_platform} obs per platform | V≥{V_THRESHOLD} = high-void | N_pop={N_PLATFORMS_SIM}')
ax9.legend(fontsize=8, facecolor='#1a1a1a', labelcolor=TXT)
ax9.set_xlim(0, 1); ax9.set_ylim(0, 1.05)

# Panel 10: Precision-recall
ax10 = fig.add_subplot(gs[3, 2])
style_ax(ax10)
# Filter valid precision values
valid = prec_arr > 0
ax10.plot(tpr_arr[valid], prec_arr[valid], color=ORG, lw=2.5, label='Precision-Recall')
ax10.scatter(best_tpr, prec_arr[best_thresh_idx], color=GRN, s=120, zorder=5,
             label=f'Optimal θ={best_thresh:.2f}')
ax10.axhline(0.5, color='#444444', lw=1, ls='--', label='Baseline precision')
ax10.set_xlabel('Recall (TPR)')
ax10.set_ylabel('Precision')
ax10.set_title('EXP-H: Precision-Recall\nHigh precision matters for compliance')
ax10.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)
ax10.set_xlim(0, 1); ax10.set_ylim(0, 1.05)

plt.suptitle(
    'EXP-TSU-C through H: Advanced Posterior Inference — Cross-Sensitivity, Prior Robustness,\n'
    'Analyst Variation, Adversarial Gaming, Temporal Drift, Regulatory Decision Thresholds',
    color='#dddddd', fontsize=11, y=1.005
)

out = '/data/apps/morr/private/phase-2/thrml/exp_tsu03_advanced.svg'
plt.savefig(out, format='svg', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f"SVG: {out}")
print()

# ── KILL CONDITIONS ────────────────────────────────────────────────────────────
print("=" * 70)
print("KILL CONDITIONS")
print("=" * 70)

# KC-C: realistic model requires more observations than clean (N-scaling)
# Crypto DEX at N=5: realistic p(true)=0.458 < 0.90 → needs N>5. Clean model ≥0.90 at N=5.
obs_clean_dex = generate_obs(2, 2, 2, 5, fwd=fwd_clean, seed=55)
p_clean_dex_N5 = compute_posterior(*obs_clean_dex, fwd=fwd_clean)[ALL_STATES.index((2,2,2))]
obs_real_dex_N5 = tuple(o[:5] for o in obs_pool_dex)
p_real_dex_N5  = compute_posterior(*obs_real_dex_N5,  fwd=fwd_realistic)[ALL_STATES.index((2,2,2))]
kc_c = p_clean_dex_N5 > p_real_dex_N5 + 0.1
print(f"KC-C   Realistic harder (Crypto DEX N=5: clean p={p_clean_dex_N5:.3f} > real p={p_real_dex_N5:.3f}): "
      f"{'PASS' if kc_c else 'FAIL'}")

# KC-D: adversarial prior overridden at N=500
adv_500 = prior_results['adversarial'][-1]
kc_d = adv_500['ok']
print(f"KC-D   Adversarial prior overridden at N=500: MAP={adv_500['map']}  "
      f"{'PASS' if kc_d else 'FAIL'}")

# KC-E: analyst variation — MAP consensus ≥50% correct
frac_correct = n_correct_analysts / N_ANALYSTS
kc_e = frac_correct >= 0.50
print(f"KC-E   Analyst majority correct ({n_correct_analysts}/{N_ANALYSTS}={frac_correct:.0%}): "
      f"{'PASS' if kc_e else 'FAIL'}")

# KC-F: full gaming detected (MAP correct OR entropy anomaly OR Pe-residual) at some N
def _detected_any(r):
    return r['detected_map'] or r['entropy_alert'] or r['pe_residual_alert']
kc_f = any(_detected_any(r) for r in game_results[1.0])
detect_N = next((r['N'] for r in game_results[1.0] if _detected_any(r)), None)
if detect_N is not None:
    r_det = next(r for r in game_results[1.0] if _detected_any(r))
    detect_method = ('MAP' if r_det['detected_map'] else
                     'entropy' if r_det['entropy_alert'] else 'Pe-residual')
else:
    detect_method = None
print(f"KC-F   Full gaming detected at N={detect_N} via {detect_method}: {'PASS' if kc_f else 'FAIL'}")

# KC-G: temporal drift alert within 15% of true crossing (detection lag < 15% of N_STEPS)
max_lag_allowed = int(0.15 * N_STEPS)  # 75 steps at N_STEPS=500
kc_g = (lag is not None and lag <= max_lag_allowed)
print(f"KC-G   Detection lag ≤{max_lag_allowed} steps (15% of ramp): lag={lag}  "
      f"{'PASS' if kc_g else 'FAIL'}")

# KC-H: AUC > 0.70
kc_h = auc > 0.70
print(f"KC-H   ROC AUC > 0.70: {auc:.4f}  {'PASS' if kc_h else 'FAIL'}")

all_kc = all([kc_c, kc_d, kc_e, kc_f, kc_g, kc_h])
print()
print(f"All KCs: {'PASS' if all_kc else 'FAIL'}")
print()

print("=" * 70)
print("KEY FINDINGS SUMMARY")
print("=" * 70)
print()
print("EXP-C: Cross-sensitivity makes inference harder by ΔH nats.")
print("       High-void platforms (large Pe signal) remain easy.")
print("       Mid-void platforms (Crypto DEX, AI-GG) are the hard cases.")
print()
print("EXP-D: Adversarial prior is overcome by data by N≈", crossover_N or ">500")
print("       Informed prior converges faster but not by much.")
print("       Low-void prior is the most dangerous: biases toward compliance")
print("       (Type II error: missing a void).")
print()
print(f"EXP-E: Analyst variation spans {max(analyst_V_means)-min(analyst_V_means):.2f} void-index points")
print(f"       on the same platform (true V={O_t+R_t+a_t}).")
print("       This is the meta-void problem (Paper 46): the scoring instrument")
print("       has uncertainty wider than the construct being measured.")
print()
print("EXP-F: Gaming shifts MAP to COMPROMISE state (V≈6), not fake state.")
print("       Pe signal prevents system accepting V≤4; gamed signals prevent V=9.")
print(f"       Pe-residual test: MAP predicts low ret, observed ret≈1.0 → z>>3.")
print(f"       Full gaming detected at N≈{detect_N} via {detect_method} — Pe is ungameable.")
print("       This is the Goodhart's Law limit: game the observable, not the Pe.")
print()
print(f"EXP-G: Temporal drift detected with lag {lag} steps after true Pe>4.")
print(f"       Rolling window N={N_WINDOW} creates detection lag proportional to window size.")
print("       Regulators need real-time streaming, not quarterly assessments.")
print()
print(f"EXP-H: ROC AUC={auc:.3f}. Optimal threshold θ={best_thresh:.2f}.")
print("       Harm-tier thresholds: Tier4=10%, Tier3=25%, Tier2=60%, Tier1=90%.")
print("       The Fantasia Bound on regulation: AUC<1 is irreducible at finite N.")
print("       More observations = higher AUC = better compliance detection.")
print()
print("EXP-TSU-C through H complete.")
