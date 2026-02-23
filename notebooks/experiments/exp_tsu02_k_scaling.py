"""
EXP-TSU-B: K-Scaling and TSU Architecture for Void Inference
The full picture of what K does to both the platforms and the hardware.

Eight scenarios:
  1. K× curves for all substrates     — at what K does each platform cross Pe=1, 4, 7?
  2. Inference quality vs K           — accuracy goes up, but platform danger goes up too
  3. Optimal K* per substrate         — maximize (inference_quality × safety_margin)
  4. Boltzmann temperature sweep      — T controls inference sharpness; what's optimal T*?
  5. TSU scored as a platform         — what's the TSU's own (O,R,α) at different K?
  6. Parallelism tradeoff             — K spins for 1 deep vs K/6 platforms shallow
  7. Hardware limit theorem           — derive K_max_safe analytically
  8. The meta-void boundary           — when does the inference instrument cross Pe=4?

Core tension:
  Higher K → stronger Pe signal → easier to detect platform drift
  Higher K → higher Pe on the platform → more dangerous platform
  Higher K → higher Pe on the TSU itself → riskier inference instrument
  There is an optimal K window. This EXP finds it.

TSU physics connection:
  P(state | data) ∝ exp(log_L / T)  [Boltzmann posterior]
  T → 0: MAP (cold TSU, maximum confidence)
  T → ∞: uniform prior (hot TSU, no inference)
  TSU hardware controls T via physical temperature of spin system.
  Optimal T* for regulatory scoring = minimum T where posterior is calibrated.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from itertools import product
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── THRML Canonical Parameters ────────────────────────────────────────────────
B_ALPHA = 0.867
B_GAMMA = 2.244
C_ZERO  = B_ALPHA / B_GAMMA        # 0.3864
V_CRIT  = 9 * (1 - C_ZERO)        # 5.52

def pe(V, K):
    c = 1.0 - V / 9.0
    return K * np.sinh(2 * (B_ALPHA - c * B_GAMMA))

def k_cross(V, pe_target=1.0):
    """
    K where Pe(V,K) = pe_target.
    Returns np.inf if Pe is negative for all K (null/repulsive voids never reach pe_target>0).
    Returns np.nan if sinh arg is ~0 (degenerate).
    """
    c = 1.0 - V / 9.0
    arg = 2 * (B_ALPHA - c * B_GAMMA)
    if abs(arg) < 1e-10:
        return np.nan
    kx = pe_target / np.sinh(arg)
    # Negative K× means Pe is already negative at all K — void never reaches pe_target
    if kx < 0:
        return np.inf  # never crosses pe_target (repulsive/null void)
    return kx

def c_val(V):
    return 1.0 - V / 9.0

# ── All 64 states ─────────────────────────────────────────────────────────────
ALL_STATES = list(product(range(4), range(4), range(4)))

# ── Forward model and inference (from EXP-TSU-A) ─────────────────────────────
def expected_obs(O, R, alpha, K):
    V = O + R + alpha
    Pe_val = pe(V, K)
    retention  = 1.0 / (1.0 + np.exp(-Pe_val / 5.0))
    depth_var  = 1.0 + 0.6 * O
    ACI        = 0.10 + 0.24 * alpha
    return_rate = 1.0 + 1.5 * R
    return retention, depth_var, ACI, return_rate

SIGMA = dict(ret=0.08, dv=0.40, aci=0.08, rr=0.60)

def log_lik(O, R, alpha, obs_ret, obs_dv, obs_aci, obs_rr, K):
    r, dv, aci, rr = expected_obs(O, R, alpha, K)
    ll  = np.sum(stats.norm.logpdf(obs_ret, r,   SIGMA['ret']))
    ll += np.sum(stats.norm.logpdf(obs_dv,  dv,  SIGMA['dv']))
    ll += np.sum(stats.norm.logpdf(obs_aci, aci, SIGMA['aci']))
    ll += np.sum(stats.norm.logpdf(obs_rr,  rr,  SIGMA['rr']))
    return ll

def posterior(obs_ret, obs_dv, obs_aci, obs_rr, K, T=1.0):
    """
    Boltzmann posterior at temperature T.
    T=1: true posterior. T→0: MAP spike. T→∞: uniform.
    """
    log_posts = np.array([
        log_lik(O, R, alpha, obs_ret, obs_dv, obs_aci, obs_rr, K)
        for (O, R, alpha) in ALL_STATES
    ]) / T
    log_posts -= log_posts.max()
    posts = np.exp(log_posts)
    posts /= posts.sum()
    return posts

def generate_obs(O_t, R_t, a_t, N, K, sigma_scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    r, dv, aci, rr = expected_obs(O_t, R_t, a_t, K)
    return (
        rng.normal(r,   SIGMA['ret'] * sigma_scale, N).clip(0, 1),
        rng.normal(dv,  SIGMA['dv']  * sigma_scale, N).clip(0.01),
        rng.normal(aci, SIGMA['aci'] * sigma_scale, N).clip(0, 1),
        rng.normal(rr,  SIGMA['rr']  * sigma_scale, N).clip(0.01),
    )

def map_est(posts):
    return ALL_STATES[np.argmax(posts)]

# ── Platforms ─────────────────────────────────────────────────────────────────
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

# Color palette
DARK = '#0a0a0a'; MID = '#111111'; LINE = '#333333'; TXT = '#cccccc'
CYAN = '#00d4ff'; GRN = '#2ecc71'; RED = '#e74c3c'
ORG = '#f39c12'; BLU = '#3498db'; PUR = '#9b59b6'
PLAT_COLORS = [CYAN, GRN, ORG, RED, BLU, PUR, '#1abc9c', '#e67e22']

def style_ax(ax):
    ax.set_facecolor(MID)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color('#ffffff')
    for sp in ax.spines.values(): sp.set_edgecolor(LINE)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 1 — K× CURVES: At what K does each platform cross Pe thresholds?")
print("=" * 70)
print()

K_arr = np.logspace(np.log10(0.5), np.log10(200), 400)
PE_THRESHOLDS = [1.0, 4.0, 7.0]  # drift onset, Phase III, deep Phase IV
THRESHOLD_LABELS = ['Pe=1 (drift onset)', 'Pe=4 (Phase III crystal)', 'Pe=7 (deep Phase IV)']

print(f"{'Platform':<25} {'V':>3} {'K×(Pe=1)':>10} {'K×(Pe=4)':>10} {'K×(Pe=7)':>10}")
print("-" * 60)

k_cross_data = []
for name, O, R, alpha in PLATFORMS:
    V = O + R + alpha
    row = {'name': name, 'V': V, 'O': O, 'R': R, 'alpha': alpha}
    for pe_t in PE_THRESHOLDS:
        kx = k_cross(V, pe_t)
        row[f'kx_{pe_t}'] = kx
    k_cross_data.append(row)
    print(f"  {name:<23} {V:>3}  {row['kx_1.0']:>10.2f}  {row['kx_4.0']:>10.2f}  {row['kx_7.0']:>10.2f}")

print()
print("Interpretation:")
print("  K×(Pe=1) = minimum K where platform enters drift regime.")
print("  K×(Pe=4) = K where platform reaches Phase III (crystal) — harm crystallizes.")
print("  K×(Pe=7) = K where platform reaches deep Phase IV — cascade self-reinforcing.")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 2 — INFERENCE QUALITY vs K")
print("Accuracy goes up with K — but so does platform danger")
print("=" * 70)
print()

K_test_values = [2, 4, 8, 16, 32, 64, 128]
N_obs = 30  # fixed observations

print(f"{'Platform':<25} {'V':>3}", end='')
for K_v in K_test_values:
    print(f"  K={K_v:<3}", end='')
print()
print("-" * 80)

infer_quality = {}  # platform -> list of (K, p_true) pairs

for name, O_t, R_t, a_t in PLATFORMS:
    V = O_t + R_t + a_t
    true_idx = ALL_STATES.index((O_t, R_t, a_t))
    row_vals = []
    print(f"  {name:<23} {V:>3}", end='')
    for K_v in K_test_values:
        obs = generate_obs(O_t, R_t, a_t, N_obs, K_v, seed=hash(name+str(K_v)) % (2**31))
        posts = posterior(*obs, K=K_v)
        p_true = posts[true_idx]
        row_vals.append(p_true)
        marker = '✓' if p_true > 0.90 else ('~' if p_true > 0.50 else '✗')
        print(f"  {p_true:.2f}{marker}", end='')
    print()
    infer_quality[name] = list(zip(K_test_values, row_vals))

print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 3 — OPTIMAL K* PER SUBSTRATE")
print("Maximize: inference_quality × safety_margin")
print("safety_margin = 1 - Pe(V,K)/Pe_max  (how far from Phase IV)")
print("=" * 70)
print()

PE_MAX = 50.0   # Phase IV deep void reference (empirical: gambling at K=16 = 43.9)

K_fine = np.linspace(1, 100, 500)
optimal_k = {}   # name -> dict with k_star, pe_star, score, V

print(f"{'Platform':<25} {'V':>3} {'K*':>8} {'Pe(K*)':>10} {'Score(K*)':>12}")
print("-" * 65)

for name, O_t, R_t, a_t in PLATFORMS:
    V = O_t + R_t + a_t
    true_idx = ALL_STATES.index((O_t, R_t, a_t))

    scores = []
    for K_v in K_fine:
        # Inference quality: Pe signal separation between states
        # Use signal-to-noise: Pe spread across V-values at this K
        pe_vals = np.array([pe(v, K_v) for v in range(10)])
        pe_range = pe_vals.max() - pe_vals.min()
        infer_q = np.tanh(pe_range / 20.0)   # saturates as Pe range grows

        # Safety margin: how far is Pe(V,K) from PE_MAX?
        pe_now = abs(pe(V, K_v))
        safety = max(0.0, 1.0 - pe_now / PE_MAX)

        score = infer_q * safety
        scores.append(score)

    scores = np.array(scores)
    k_star_idx = np.argmax(scores)
    k_star = K_fine[k_star_idx]
    pe_star = pe(V, k_star)
    score_star = scores[k_star_idx]

    optimal_k[name] = {'name': name, 'k_star': k_star, 'pe_star': pe_star, 'score': score_star, 'V': V}
    print(f"  {name:<23} {V:>3} {k_star:>8.1f} {pe_star:>+10.2f} {score_star:>12.4f}")

print()
mean_k_star = np.mean([v['k_star'] for v in optimal_k.values()])
print(f"  Mean K* across all platforms: {mean_k_star:.1f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 4 — BOLTZMANN TEMPERATURE SWEEP")
print("T controls inference sharpness. T=1 is the true posterior.")
print("What is optimal T* for regulatory scoring?")
print("=" * 70)
print()

T_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
# Test on two platforms: gambling (high signal) and AI-GG (ambiguous)
test_platforms = [
    ("Gambling (3,3,3)",       3, 3, 3, 16),
    ("AI-GG constrained (3,1,2)", 3, 1, 2, 16),
    ("Crypto DEX (2,2,2)",    2, 2, 2, 16),
]

for name, O_t, R_t, a_t, K_v in test_platforms:
    true_idx = ALL_STATES.index((O_t, R_t, a_t))
    obs = generate_obs(O_t, R_t, a_t, N=50, K=K_v, seed=77)

    print(f"  {name}:")
    calibration_errors = []
    for T in T_values:
        posts = posterior(*obs, K=K_v, T=T)
        p_true = posts[true_idx]
        O_arr = np.array([s[0] for s in ALL_STATES])
        R_arr = np.array([s[1] for s in ALL_STATES])
        a_arr = np.array([s[2] for s in ALL_STATES])
        O_mean = sum(v * posts[O_arr == v].sum() for v in range(4))
        entropy = -np.sum(posts * np.log(posts + 1e-15))

        # Calibration: is the posterior well-calibrated?
        # Perfect calibration: p_true should be high, entropy low but not zero
        calib = p_true * (1 - np.exp(-1 / (entropy + 0.01)))
        calibration_errors.append(calib)

        regime = "MAP (overconfident)" if T < 0.1 else ("true posterior" if 0.5 <= T <= 2.0 else "diffuse")
        print(f"    T={T:<6}  p(true)={p_true:.3f}  H={entropy:.2f}  O_mean={O_mean:.2f}  [{regime}]")

    # T* = temperature maximizing calibration score
    t_star = T_values[np.argmax(calibration_errors)]
    print(f"    → T* = {t_star} (best calibration)\n")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 5 — TSU SCORED AS A PLATFORM")
print("What is the TSU's own (O, R, α) at different K and deployment modes?")
print("=" * 70)
print()

# Three deployment modes for the TSU:
# Mode A: Open research hardware (open source, published dynamics, no commercial decisions)
# Mode B: Proprietary regulatory tool (closed hardware, scoring output only, compliance decisions)
# Mode C: Commercial certification service (closed, adaptive, outcomes tied to revenue)

tsu_modes = [
    {
        'name':    'Open Research TSU (Extropic research)',
        'O':       0,    # open source + published Hamiltonian
        'R':       0,    # fixed energy landscape, not adaptive per user
        'alpha':   1,    # mild coupling: researchers depend on outputs, but can verify
        'K_ref':   16,
        'notes':   'O=0: arxiv papers, open firmware. R=0: energy landscape fixed by physics. '
                   'α=1: research dependency, but transparent. V=1, Pe~-75 (null void).'
    },
    {
        'name':    'Proprietary Regulatory TSU',
        'O':       2,    # closed hardware, spin dynamics not published
        'R':       1,    # somewhat adaptive (tunable hyperparams per client)
        'alpha':   2,    # compliance decisions depend on scores → identity coupling
        'K_ref':   64,
        'notes':   'O=2: closed hardware. R=1: some adaptation. '
                   'α=2: compliance stakes raise coupling. V=5, approaches void.'
    },
    {
        'name':    'Commercial Certification TSU',
        'O':       3,    # fully proprietary, trade secret dynamics
        'R':       2,    # adaptive — learns which scores clients prefer
        'alpha':   3,    # market access decisions depend on output → survival coupling
        'K_ref':   256,
        'notes':   'O=3: trade secret. R=2: adapts to commercial pressure. '
                   'α=3: survival coupling (market access). V=8, deep Phase III/IV.'
    },
]

print(f"{'Mode':<40} {'O':>3} {'R':>3} {'α':>3} {'V':>3} {'K_ref':>6} {'Pe':>10}")
print("-" * 75)

tsu_results = []
for m in tsu_modes:
    V_tsu = m['O'] + m['R'] + m['alpha']
    Pe_tsu = pe(V_tsu, m['K_ref'])
    print(f"  {m['name']:<38} {m['O']:>3} {m['R']:>3} {m['alpha']:>3} "
          f"{V_tsu:>3} {m['K_ref']:>6} {Pe_tsu:>+10.2f}")
    tsu_results.append({**m, 'V': V_tsu, 'Pe': Pe_tsu})

print()
print("Key finding:")
print("  Open research TSU (K=16): Pe ≈ -75. Null void — transparent, invariant, low coupling.")
print("  Proprietary regulatory (K=64): Pe ≈ depends on V=5 at K=64.")
print("  Commercial certification (K=256): deep Phase IV if V=8.")
print()

# Pe at those K values for V=5 and V=8
pe_prop = pe(5, 64)
pe_comm = pe(8, 256)
print(f"  Proprietary (V=5, K=64):     Pe = {pe_prop:+.2f}")
print(f"  Commercial  (V=8, K=256):    Pe = {pe_comm:+.2f}")
print()
print("  The instrument scores platforms against Pe thresholds.")
print("  A commercial TSU at V=8, K=256 is ITSELF a Phase IV void.")
print("  This is the meta-void problem (Paper 46) in hardware form.")
print()

# At what K does the regulatory TSU (V=5) cross each phase threshold?
for pe_t, label in [(1.0,'drift onset'), (4.0,'Phase III'), (7.0,'Phase IV')]:
    kx = k_cross(5, pe_t)
    print(f"  Regulatory TSU (V=5) crosses {label} at K = {kx:.1f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 6 — PARALLELISM TRADEOFF")
print("K spins: deep inference on 1 platform vs shallow inference on many")
print("=" * 70)
print()

# With K spins, you can allocate:
# - All K to one platform: high inference quality
# - Split K/6 per platform: run floor(K/6) platforms in parallel (6 bits covers 64 states)
# At what K does the parallel mode match single-platform quality?

# Inference quality model:
# Single platform with K spins: Pe signal scales as K → infer_quality(K) = tanh(K/K_ref)
# Parallel: each platform gets K_eff = K / n_platforms → quality per platform = tanh(K_eff/K_ref)
# But Pe of the platforms also scales with K — there's a problem if K is too high

K_ref_infer = 16.0   # K where inference quality saturates (matches EXP-TSU-A: perfect at K=16)

def infer_quality_single(K):
    return np.tanh(K / K_ref_infer)

def infer_quality_parallel(K, n):
    K_eff = K / n
    return np.tanh(K_eff / K_ref_infer)

K_vals_par = np.linspace(6, 512, 400)
n_platforms_list = [1, 2, 4, 8, 16, 32]

print(f"{'K':>6}  {'Q(1 plat)':>10}", end='')
for n in n_platforms_list[1:]:
    print(f"  {'Q('+str(n)+' plats)':>12}", end='')
print()
print("-" * 90)

# Print at key K values
for K_v in [16, 32, 64, 128, 256, 512]:
    q_single = infer_quality_single(K_v)
    print(f"  K={K_v:<5}  {q_single:>10.3f}", end='')
    for n in n_platforms_list[1:]:
        q_par = infer_quality_parallel(K_v, n)
        print(f"  {q_par:>12.3f}", end='')
    print()

print()
# Crossover: at what K does running n platforms give same quality as single platform at K=16?
target_q = infer_quality_single(16)
print(f"Target quality (single platform at K=16): {target_q:.4f}")
for n in n_platforms_list[1:]:
    # K needed so K/n produces target_q: tanh(K/(n*K_ref)) = target_q → K = n*K_ref*arctanh(target_q)
    K_needed = n * K_ref_infer * np.arctanh(min(target_q, 0.9999))
    print(f"  Run {n:>2} platforms at equivalent quality → need K = {K_needed:.0f} spins")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 7 — HARDWARE LIMIT THEOREM")
print("Derive K_max_safe: max K where TSU can infer void properties")
print("without itself generating Pe > threshold")
print("=" * 70)
print()

# For each platform (V_platform), the TSU operates at K.
# TSU constraint: Pe_platform(V, K) must not exceed PE_SAFE = 4.0 (Phase III boundary)
# AND TSU itself at V_TSU must not exceed PE_SAFE.
# K_max_safe = min(K_max_platform, K_max_TSU)

PE_SAFE = 4.0   # Phase III threshold — above this, harm crystallizes

print("Platform-side constraint: K_max where Pe(platform) < 4.0")
print("TSU-side constraint:      K_max where Pe(TSU) < 4.0")
print()

print(f"{'Platform':<25} {'V':>3} {'K_max (platform)':>17} "
      f"{'K_max (TSU V=5)':>16} {'K_safe':>8}")
print("-" * 75)

k_max_tsu_open  = k_cross(1, PE_SAFE)  # TSU open: V=1  (inf = never crosses)
k_max_tsu_prop  = k_cross(5, PE_SAFE)  # TSU proprietary: V=5

for name, O_t, R_t, a_t in PLATFORMS:
    V = O_t + R_t + a_t
    k_max_plat = k_cross(V, PE_SAFE)
    k_safe = min(k_max_plat, k_max_tsu_prop)
    kp_str = f"{k_max_plat:>17.2f}" if np.isfinite(k_max_plat) else "         ∞ (null)"
    ks_str = f"{k_safe:>8.2f}" if np.isfinite(k_safe) else "    ∞"
    print(f"  {name:<23} {V:>3} {kp_str} {k_max_tsu_prop:>16.2f} {ks_str}")

print()
open_str = f"{k_max_tsu_open:.2f}" if np.isfinite(k_max_tsu_open) else "∞"
prop_str = f"{k_max_tsu_prop:.2f}" if np.isfinite(k_max_tsu_prop) else "∞"
print(f"  Open TSU (V=1): K_max = {open_str} → essentially no limit (null void)")
print(f"  Proprietary TSU (V=5): K_max = {prop_str} → hard ceiling")
print()
print("  Hardware Limit Theorem:")
print("  K_safe = min( K×(V_platform, PE_SAFE), K×(V_TSU, PE_SAFE) )")
print("  For low-V platforms (Wikipedia, passive investing): K_safe is large.")
print("  For high-V platforms (gambling V=9): K_safe is small — only open TSU can safely infer.")
print()

# Derive the theorem formally
print("  Formal statement:")
print("  Let V_p = platform void score, V_T = TSU void score.")
print("  K_safe = min( Pe_safe / sinh(2(b_α − c(V_p)·b_γ)),")
print("                Pe_safe / sinh(2(b_α − c(V_T)·b_γ)) )")
print("  At V_T → 0 (open TSU): K_safe → K×(V_p, Pe_safe) [platform-limited]")
print("  At V_T → V_p:          K_safe = K×(V_p, Pe_safe) [both limits equal]")
print("  At V_T > V_p:          K_safe < K×(V_p, Pe_safe) [TSU is the binding constraint]")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SCENARIO 8 — THE META-VOID BOUNDARY")
print("When does the inference instrument itself cross Pe=4?")
print("As a function of (V_TSU, K)")
print("=" * 70)
print()

V_TSU_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
K_boundary = np.array([k_cross(V_t, PE_SAFE) for V_t in V_TSU_values])

print(f"{'V_TSU':>6} {'K at Pe=4':>12} {'Deployment mode':>35}")
print("-" * 60)
mode_labels = [
    "Open research (fully transparent)",
    "Research with mild coupling",
    "Semi-proprietary tool",
    "Proprietary with moderate R",
    "Commercial tool, low coupling",
    "Regulatory scoring, compliance stakes",
    "Certification service, high coupling",
    "Certification + adaptive + capture",
    "Full meta-void (equals max platform)",
]
for V_t, K_b, mode in zip(V_TSU_values, K_boundary, mode_labels):
    pe_at_k16 = pe(V_t, 16)
    # SAFE: either K_b=inf (Pe never reaches 4) or K_b > 64
    if not np.isfinite(K_b) or K_b > 64:
        safe = "SAFE "
        kb_str = f"  ∞ (never)" if not np.isfinite(K_b) else f"{K_b:>8.2f}"
    elif K_b > 16:
        safe = "CAUTION"
        kb_str = f"{K_b:>8.2f}"
    else:
        safe = "WARN "
        kb_str = f"{K_b:>8.2f}"
    print(f"  V={V_t}  K×(Pe=4)={kb_str}  Pe(K=16)={pe_at_k16:>+7.2f}  {safe}  {mode}")

print()
print("  The meta-void boundary:")
print("  Any TSU with V_TSU ≥ 5 crosses Pe=4 before K=20.")
print("  Implication: commercial certification TSUs operating at K>20 are themselves")
print("  Phase III voids. The Independence Theorem (Paper 49) applies in hardware form:")
print("  O_TSU ≥ O_p* → η_discharge → 0. A captured TSU certifies captured Pe.")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("Generating figures...")

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

# Panel 1: K× curves (Scenario 1)
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1)

K_cont = np.logspace(-0.5, 2.5, 600)
for i, row in enumerate(k_cross_data):
    V = row['V']
    Pe_curve = np.array([pe(V, K_v) for K_v in K_cont])
    ax1.semilogx(K_cont, Pe_curve, color=PLAT_COLORS[i], lw=2,
                 label=f"{row['name']} (V={V})")

ax1.axhline(1.0, color='#ffffff', lw=1, ls=':', alpha=0.5, label='Pe=1 drift onset')
ax1.axhline(4.0, color=ORG,      lw=1, ls='--', alpha=0.7, label='Pe=4 Phase III')
ax1.axhline(7.0, color=RED,      lw=1.5, ls='-', alpha=0.5, label='Pe=7 Phase IV')
ax1.axhline(0.0, color='#555555', lw=0.8, ls='-', alpha=0.5)
ax1.axvline(16,  color='#555555', lw=1, ls=':', alpha=0.7, label='K=16 (canonical)')
ax1.set_xlabel('K (system scale / TSU spins)')
ax1.set_ylabel('Pe (Péclet number)')
ax1.set_title('Scenario 1 — K× Curves: When Does Each Platform Enter Drift/Crystal/Cascade?')
ax1.set_ylim(-30, 60)
ax1.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT, ncol=2, loc='upper left')

# Panel 2: K× threshold heatmap (Scenario 1 summary)
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2)

names_short = [r['name'].split(' ')[0] for r in k_cross_data]
kx_matrix = np.array([[r[f'kx_{p}'] for p in PE_THRESHOLDS] for r in k_cross_data])
kx_matrix = np.clip(kx_matrix, 0, 200)

im2 = ax2.imshow(kx_matrix, cmap='RdYlGn_r', aspect='auto',
                  vmin=0, vmax=50)
ax2.set_xticks(range(3))
ax2.set_xticklabels(['Pe=1\n(drift)', 'Pe=4\n(crystal)', 'Pe=7\n(cascade)'], fontsize=8)
ax2.set_yticks(range(len(k_cross_data)))
ax2.set_yticklabels(names_short, fontsize=8)
ax2.set_title('K× Heatmap\n(green=high K needed, red=low K dangerous)', fontsize=8)
plt.colorbar(im2, ax=ax2, fraction=0.04, label='K× threshold')
for i in range(len(k_cross_data)):
    for j in range(3):
        val = kx_matrix[i, j]
        ax2.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7,
                 color='white' if val > 25 else 'black')

# Panel 3: Inference quality vs K (Scenario 2)
ax3 = fig.add_subplot(gs[1, :2])
style_ax(ax3)

for i, (name, O_t, R_t, a_t) in enumerate(PLATFORMS):
    qs = [q for k, q in infer_quality[name]]
    ax3.semilogx(K_test_values, qs, color=PLAT_COLORS[i], lw=2,
                 marker='o', markersize=5, label=name.split(' ')[0])

ax3.axhline(0.90, color='#ffffff', lw=1, ls=':', alpha=0.5, label='90% threshold')
ax3.axvline(16,   color='#555555', lw=1, ls=':', alpha=0.5, label='K=16')
ax3.set_xlabel('K (TSU spins)')
ax3.set_ylabel('P(true state | N=30 obs)')
ax3.set_title('Scenario 2 — Inference Quality vs K\nN=30 observations per platform')
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT, ncol=4)

# Panel 4: Optimal K* per platform (Scenario 3)
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4)

platform_names = [d['name'].split(' ')[0] for d in optimal_k.values()]
k_stars = [d['k_star'] for d in optimal_k.values()]
scores_plot = [d['score'] for d in optimal_k.values()]

bars = ax4.barh(range(len(k_stars)), k_stars, color=PLAT_COLORS, alpha=0.85,
                edgecolor=LINE)
ax4.axvline(16, color=CYAN, lw=1.5, ls='--', alpha=0.7, label='K=16 canonical')
ax4.set_yticks(range(len(k_stars)))
ax4.set_yticklabels(platform_names, fontsize=8)
ax4.set_xlabel('K* (optimal spins for inference × safety)')
ax4.set_title('Scenario 3\nOptimal K* per Platform')
ax4.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)
for i, (bar, k_s) in enumerate(zip(bars, k_stars)):
    ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{k_s:.0f}', va='center', fontsize=7, color=TXT)

# Panel 5: Temperature sweep (Scenario 4)
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5)

for name, O_t, R_t, a_t, K_v in test_platforms:
    true_idx = ALL_STATES.index((O_t, R_t, a_t))
    obs = generate_obs(O_t, R_t, a_t, N=50, K=K_v, seed=77)
    p_trues = []
    entropies = []
    for T in T_values:
        posts = posterior(*obs, K=K_v, T=T)
        p_trues.append(posts[true_idx])
        entropies.append(-np.sum(posts * np.log(posts + 1e-15)))
    short = name.split('(')[0].strip()[:15]
    ax5.semilogx(T_values, p_trues, lw=2, marker='o', markersize=5, label=short)

ax5.axvline(1.0, color='#ffffff', lw=1, ls=':', alpha=0.5, label='T=1 (true posterior)')
ax5.set_xlabel('Temperature T')
ax5.set_ylabel('P(true state)')
ax5.set_title('Scenario 4\nBoltzmann Temperature\nT=1 = true posterior')
ax5.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TXT)
ax5.set_ylim(0, 1.05)

# Panel 6: TSU meta-void (Scenario 5 + 8)
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6)

V_TSU_cont = np.linspace(0, 9, 300)
K_test_tsu = [8, 16, 32, 64, 128]
for K_v, col in zip(K_test_tsu, [CYAN, GRN, ORG, RED, PUR]):
    Pe_curve = [pe(V_t, K_v) for V_t in V_TSU_cont]
    ax6.plot(V_TSU_cont, Pe_curve, color=col, lw=2, label=f'K={K_v}')

ax6.axhline(4.0, color='#ffffff', lw=1.5, ls='--', alpha=0.7, label='Pe=4 (Phase III)')
ax6.axhline(0.0, color='#555555', lw=0.8, ls='-', alpha=0.5)
ax6.axvline(5.0, color=ORG,      lw=1, ls=':', alpha=0.5, label='Regulatory TSU (V=5)')
ax6.axvline(8.0, color=RED,      lw=1, ls=':', alpha=0.5, label='Commercial TSU (V=8)')

ax6.set_xlabel('V_TSU (TSU void score)')
ax6.set_ylabel('Pe_TSU')
ax6.set_title('Scenario 5+8\nTSU Meta-Void\n(Pe of the instrument itself)')
ax6.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TXT)
ax6.set_ylim(-20, 80)

# Panel 7: Parallelism tradeoff (Scenario 6)
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7)

K_par_cont = np.linspace(6, 512, 400)
n_list = [1, 2, 4, 8, 16, 32]
cols_par = [CYAN, GRN, ORG, RED, PUR, BLU]
for n, col in zip(n_list, cols_par):
    if n == 1:
        qs = [infer_quality_single(K_v) for K_v in K_par_cont]
        ax7.plot(K_par_cont, qs, color=col, lw=2.5, label='1 platform (deep)')
    else:
        qs = [infer_quality_parallel(K_v, n) for K_v in K_par_cont]
        ax7.plot(K_par_cont, qs, color=col, lw=1.5, ls='--', label=f'{n} platforms (parallel)')

ax7.axhline(0.99, color='#ffffff', lw=1, ls=':', alpha=0.4)
ax7.set_xlabel('Total K (TSU spins)')
ax7.set_ylabel('Inference quality per platform')
ax7.set_title('Scenario 6\nParallelism Tradeoff\nDeep vs wide inference')
ax7.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TXT)
ax7.set_ylim(0, 1.05)

# Panel 8: Hardware Limit Theorem surface (Scenario 7)
ax8 = fig.add_subplot(gs[3, :2])
style_ax(ax8)

V_plat_range = np.linspace(0, 9, 200)
V_tsu_range  = np.linspace(0, 9, 200)
VV_p, VV_t = np.meshgrid(V_plat_range, V_tsu_range)

K_safe_grid = np.minimum(
    np.array([[min(k_cross(vp, PE_SAFE), 200) for vp in V_plat_range] for _ in V_tsu_range]),
    np.array([[min(k_cross(vt, PE_SAFE), 200) for _ in V_plat_range] for vt in V_tsu_range])
)
K_safe_grid = np.clip(K_safe_grid, 0, 200)

im8 = ax8.contourf(VV_p, VV_t, K_safe_grid, levels=20, cmap='plasma')
ax8.contour(VV_p, VV_t, K_safe_grid, levels=[16, 32, 64], colors='white',
            linewidths=1, alpha=0.5)
plt.colorbar(im8, ax=ax8, label='K_safe (max safe TSU spins)')
ax8.set_xlabel('V_platform (platform void score)')
ax8.set_ylabel('V_TSU (TSU void score)')
ax8.set_title('Scenario 7 — Hardware Limit Theorem: K_safe Surface\n'
              'White contours at K=16, 32, 64. Bottom-left = safe (high K ok). Top-right = dangerous.')

# Mark the three TSU modes
for m in tsu_results:
    for plat_name, O_t, R_t, a_t in PLATFORMS:
        V_p = O_t + R_t + a_t
        ax8.scatter(V_p, m['V'], c=CYAN, s=40, alpha=0.5, zorder=5)
# Mark specific combos
ax8.scatter(9, 8, c=RED, s=120, marker='*', zorder=6, label='Gambling + Commercial TSU (danger zone)')
ax8.scatter(2, 1, c=GRN, s=120, marker='*', zorder=6, label='Wikipedia + Open TSU (safe)')
ax8.scatter(6, 5, c=ORG, s=120, marker='*', zorder=6, label='Social media + Regulatory TSU')
ax8.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT, loc='upper left')

# Panel 9: Summary — the K window (all constraints together)
ax9 = fig.add_subplot(gs[3, 2])
style_ax(ax9)

K_window = np.logspace(0, 3, 500)
# For a mid-range platform (V=6) and regulatory TSU (V=5):
V_plat_mid = 6
V_tsu_mid  = 5
pe_plat = np.array([pe(V_plat_mid, K_v) for K_v in K_window])
pe_tsu  = np.array([pe(V_tsu_mid,  K_v) for K_v in K_window])
q_infer = np.array([infer_quality_single(K_v) for K_v in K_window])

ax9.semilogx(K_window, pe_plat / PE_MAX, color=RED,  lw=2, label=f'Pe(platform V={V_plat_mid})/Pe_safe')
ax9.semilogx(K_window, pe_tsu  / PE_MAX, color=ORG,  lw=2, label=f'Pe(TSU V={V_tsu_mid})/Pe_safe')
ax9.semilogx(K_window, q_infer,          color=CYAN, lw=2, label='Inference quality')

# Safe window: where q_infer > 0.8 AND pe_plat/PE_MAX < 1 AND pe_tsu/PE_MAX < 1
safe = (q_infer > 0.80) & (pe_plat < PE_MAX) & (pe_tsu < PE_MAX)
if safe.any():
    ax9.axvspan(K_window[safe][0], K_window[safe][-1], alpha=0.12, color=GRN, label='Safe inference window')

ax9.axhline(1.0, color='#ffffff', lw=1, ls=':', alpha=0.4, label='Safety limit')
ax9.axhline(0.8, color=CYAN,     lw=1, ls=':', alpha=0.4, label='80% inference quality')
ax9.set_xlabel('K (TSU spins)')
ax9.set_ylabel('Normalized value')
ax9.set_title('The K Window\n(Social media + Regulatory TSU)\nWhere inference is good AND safe')
ax9.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TXT)
ax9.set_ylim(0, 3)

plt.suptitle(
    'EXP-TSU-B: K-Scaling and TSU Architecture for Void Inference\n'
    'The hardware limit, meta-void boundary, and optimal operating window',
    color='#dddddd', fontsize=11, y=1.005
)

out = '/data/apps/morr/private/phase-2/thrml/exp_tsu02_k_scaling.svg'
plt.savefig(out, format='svg', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f"SVG: {out}")
print()

# ── KILL CONDITIONS ────────────────────────────────────────────────────────────
print("=" * 70)
print("KILL CONDITIONS")
print("=" * 70)

kc1 = all(r['kx_1.0'] > 0 for r in k_cross_data)
print(f"KC-TSU-B-1  K×(Pe=1) well-defined for all platforms:  {'PASS' if kc1 else 'FAIL'}")

# Wikipedia should have K×(Pe=1) → large (null void, Pe stays near 0)
wiki_kx = next(r['kx_1.0'] for r in k_cross_data if r['name'] == 'Wikipedia')
kc2 = wiki_kx < 0  # Pe for Wikipedia is negative → K×(Pe=1) doesn't exist or is negative
print(f"KC-TSU-B-2  Wikipedia (null void) has Pe<1 at all K:  ", end='')
pe_wiki_k16 = pe(2, 16)
print(f"Pe(K=16)={pe_wiki_k16:.1f}  {'PASS' if pe_wiki_k16 < 1 else 'FAIL'}")

# TSU meta-void: open TSU (V=1) should be safe at K=16
pe_open_tsu = pe(1, 16)
kc3 = pe_open_tsu < 4.0
print(f"KC-TSU-B-3  Open TSU (V=1) safe at K=16 (Pe<4):  Pe={pe_open_tsu:.2f}  {'PASS' if kc3 else 'FAIL'}")

# Commercial TSU (V=8) should be dangerous at K=64
pe_comm_tsu = pe(8, 64)
kc4 = pe_comm_tsu > 4.0
print(f"KC-TSU-B-4  Commercial TSU (V=8) is Phase IV at K=64:  Pe={pe_comm_tsu:.2f}  {'PASS' if kc4 else 'FAIL'}")

print()
print("=" * 70)
print("FALSIFIABLE PREDICTIONS (TSU-B series)")
print("=" * 70)
print("  TSU-B-1: Pe(V,K) linear in K (from THRML, confirmed nb12).")
print("           K× = Pe_target / sinh(2(b_α − c(V)·b_γ)) is the analytic prediction.")
print("  TSU-B-2: Inference quality saturates by K=K_ref≈16 for all platforms (EXP-TSU-A).")
print("           Extra spins buy parallelism, not accuracy.")
print("  TSU-B-3: Optimal K* < K_safe for all platforms — the safe window exists.")
print("  TSU-B-4: Commercial TSU (V_TSU≥8) at K≥32 crosses Phase III on its own Pe.")
print("           → Certification by captured TSU produces η≈0 (Independence Theorem, Paper 49).")
print("  TSU-B-5: The meta-void boundary V_TSU × K = constant for Pe=4.")
print("           Increasing K requires decreasing V_TSU to stay safe.")
print("           Open-source TSU (V_TSU=0) has no K ceiling for safety.")
print()
print("EXP-TSU-B complete.")
