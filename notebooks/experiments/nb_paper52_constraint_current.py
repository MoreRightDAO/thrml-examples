# %% [markdown]
# # nb_paper52 — The Constraint Current: Pe Gradient Analysis
#
# **Companion to Paper 52: "The Constraint Current: Void-Gradient Capital Flows"**
#
# ### Research Questions
# - **Q1.** Does institutional void score (V) predict longevity across 10 historical mechanisms?
# - **Q2.** Does Pe predict effectiveness (lives saved, economic outcomes) more than longevity alone?
# - **Q3.** What is the analytical form of the Constraint Current J(V₁, V₂)?
#
# ### THRML Physics (canonical — EXP-001, **never refit**)
# ```
#   b_α = 0.867, b_γ = 2.244, C_ZERO = 0.3866
#   V3 bridge:  c = 1 − V_raw / 9   (nb26, Spearman=0.910)
#   Pe = K · sinh(2 · (b_α − c · b_γ))
# ```
#
# ### Novel derivation in this notebook
# The **Constraint Current** J is defined by analogy with Fourier's law:
#
#   J = −σ · ∇_V Pe(V)
#
# where σ > 0 is the void conductivity (positive = extraction circuit,
# negative = constraint current depending on capital motivation).
#
# dPe/dV is computed analytically below.

# %% Setup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0d0d0d', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444', 'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0', 'xtick.color': '#aaa',
    'ytick.color': '#aaa', 'grid.color': '#333',
    'axes.titleweight': 'bold', 'font.size': 10,
    'legend.facecolor': '#1a1a2e', 'legend.edgecolor': '#444',
})

# ── Canonical THRML parameters (EXP-001, never refit) ─────────────────────────
B_ALPHA = 0.867    # drift bias
B_GAMMA = 2.244    # constraint bias
C_ZERO  = 0.3866   # Pe=0 boundary (K-invariant)
K_STD   = 16.0     # standard institutional K

# ── V3 bridge and Pe functions (faithful to sim.rs) ───────────────────────────
def c_from_V(V_raw: float, V_max: float = 9.0) -> float:
    """V3 bridge: c = 1 - V/9  (nb26, empirically confirmed over V1/V2)"""
    return 1.0 - V_raw / V_max

def bnet(c: float) -> float:
    return B_ALPHA - c * B_GAMMA

def Pe(V_raw: float, K: float = K_STD) -> float:
    """Pe from raw void index via V3 bridge."""
    return K * np.sinh(2.0 * bnet(c_from_V(V_raw)))

def dPe_dV(V_raw: float, K: float = K_STD) -> float:
    """Analytical derivative of Pe with respect to V_raw.

    Pe(V) = K · sinh(2 · (b_α − (1 − V/9) · b_γ))
    Let f(V) = 2 · (b_α − (1 − V/9) · b_γ)
            = 2 · b_α − 2·b_γ + 2·b_γ·V/9
    df/dV   = 2·b_γ/9

    dPe/dV  = K · cosh(f(V)) · df/dV
            = K · cosh(2·b_net(V)) · (2·b_γ / 9)
    """
    f_V = 2.0 * bnet(c_from_V(V_raw))
    return K * np.cosh(f_V) * (2.0 * B_GAMMA / 9.0)

# ── Constraint Current ─────────────────────────────────────────────────────────
def constraint_current(V_A: float, V_B: float,
                        sigma: float = 1.0, K: float = K_STD) -> float:
    """J_{A→B} = -σ · (Pe_B - Pe_A)

    Positive J: net capital flow from A to B.
    σ > 0, V_A > V_B (high-Pe → low-Pe): J > 0 → extraction circuit
    σ < 0, V_A > V_B: J < 0 → constraint current (capital moves away from void)

    The sign convention: extraction capital is attracted to high-Pe (σ_extr > 0).
    Constraint capital flees high-Pe in favour of low-Pe nodes (σ_cstr < 0).
    """
    return -sigma * (Pe(V_B, K) - Pe(V_A, K))

# Boundary checks
assert abs(c_from_V(0.0) - 1.0) < 1e-9, "c(V=0)=1 violated"
assert abs(c_from_V(9.0) - 0.0) < 1e-9, "c(V=9)=0 violated"
# Pe=0 boundary in V space: c=C_ZERO when V = 9*(1-C_ZERO) ≈ 5.52
V_ZERO = 9.0 * (1.0 - C_ZERO)   # ≈ 5.52 — Pe=0 isoline in V coordinates

print(f"Pe(V=3, K=16) = {Pe(3):.2f}")
print(f"Pe(V=9, K=16) = {Pe(9):.2f}")
print(f"V_ZERO = 9*(1-C_ZERO) = {V_ZERO:.3f}")
print(f"Pe(V_ZERO, K=16) = {Pe(V_ZERO):.4f}  (should ≈ 0)")
print(f"dPe/dV at V_ZERO = {dPe_dV(V_ZERO):.3f}  [minimum gradient at Pe=0 line]")

# %% Historical Institutions Dataset
# O/R/α each scored 0–3. V_raw = O+R+α ∈ {0..9}.
# Sources: void-gradient-economics-historical-evidence.md (2026-02-24)

@dataclass
class Institution:
    name: str
    O: float; R: float; alpha: float
    longevity_years: float   # effective operational lifespan
    effectiveness: int       # 0=failed, 1=weak, 2=moderate, 3=strong, 4=exceptional
    note: str = ""

    @property
    def V_raw(self): return self.O + self.R + self.alpha
    @property
    def c(self): return c_from_V(self.V_raw)
    @property
    def Pe_inst(self): return Pe(self.V_raw)
    @property
    def dPe_dV_inst(self): return dPe_dV(self.V_raw)

institutions: List[Institution] = [
    # Low-void (constraint current regime)
    Institution("Global Fund",          O=1, R=1, alpha=1,
                longevity_years=22, effectiveness=4,
                note="$78B, ~65M lives saved. Board structure prevents donor capture."),
    Institution("Bretton Woods",        O=1, R=1, alpha=1,
                longevity_years=27, effectiveness=4,
                note="27yr stable intl monetary order. Failed when R→3 (US Nixon shock)."),
    Institution("UNGA",                 O=0, R=2, alpha=1,
                longevity_years=79, effectiveness=2,
                note="79yr and counting. Low Pe = durability even with bloc politics."),
    Institution("Marshall Plan",        O=1, R=2, alpha=1,
                longevity_years=4,  effectiveness=4,
                note="4yr designed termination. GDP recovered to pre-war by 1952."),
    Institution("PEPFAR",               O=1, R=2, alpha=1,
                longevity_years=22, effectiveness=3,
                note="~$7B/yr bilateral, results-based. R=2 (US politics) survivable."),
    # Mid-void (diffusion zone)
    Institution("Young Plan (BIS)",     O=2, R=2, alpha=2,
                longevity_years=3,  effectiveness=1,
                note="Lasted 3yr. Hoover Moratorium 1931 ended it. Depression collapse."),
    # High-void (extraction circuit regime)
    Institution("Dawes Plan / JPMorgan", O=2, R=2, alpha=3,
                longevity_years=5,  effectiveness=1,
                note="Circular flow: US→Germany→Allies→US. Extraction, not constraint."),
    Institution("League of Nations",    O=2, R=3, alpha=2,
                longevity_years=15, effectiveness=0,
                note="Eff. dead 1935 (unanimity veto). Formally dissolved 1946."),
    Institution("IMF SAPs",             O=2, R=3, alpha=2,
                longevity_years=40, effectiveness=0,
                note="Ongoing despite chronic failure. Selection pathology confirmed."),
    Institution("Versailles",           O=3, R=3, alpha=3,
                longevity_years=5,  effectiveness=0,
                note="Keynes 1919 predicted failure. Catastrophic in 5yr → WWII arc."),
]

print("\n── Institutional Pe Table ──────────────────────────────────────────────")
print(f"{'Institution':<26} {'V':>4} {'c':>6} {'Pe':>8}  {'Longevity':>9}  {'Effect.':>7}")
print("─" * 72)
for inst in institutions:
    print(f"{inst.name:<26} {inst.V_raw:>4.0f} {inst.c:>6.3f} {inst.Pe_inst:>+8.1f}"
          f"  {inst.longevity_years:>8.0f}yr  {inst.effectiveness:>7d}")

# %% Spearman Correlations
V_vals        = np.array([i.V_raw        for i in institutions])
Pe_vals       = np.array([i.Pe_inst      for i in institutions])
longevity     = np.array([i.longevity_years for i in institutions])
effectiveness = np.array([i.effectiveness  for i in institutions])

rho_V_lon, p_V_lon   = spearmanr(V_vals,  longevity)
rho_Pe_lon, p_Pe_lon = spearmanr(Pe_vals, longevity)
rho_V_eff, p_V_eff   = spearmanr(V_vals,  effectiveness)
rho_Pe_eff, p_Pe_eff = spearmanr(Pe_vals, effectiveness)

print("\n── Spearman Correlations (N=10) ────────────────────────────────────────")
print(f"  V vs longevity:      ρ = {rho_V_lon:+.4f}  p = {p_V_lon:.4f}")
print(f"  Pe vs longevity:     ρ = {rho_Pe_lon:+.4f}  p = {p_Pe_lon:.4f}")
print(f"  V vs effectiveness:  ρ = {rho_V_eff:+.4f}  p = {p_V_eff:.4f}")
print(f"  Pe vs effectiveness: ρ = {rho_Pe_eff:+.4f}  p = {p_Pe_eff:.4f}")

# %% Constraint Current Derivation (analytical)
# ─────────────────────────────────────────────────────────────────────────────
# DERIVATION: Constraint Current J
#
# Define Pe(V) via V3 bridge:
#   Pe(V) = K · sinh(2·b_net(V))
#   b_net(V) = b_α − c(V)·b_γ = b_α − (1 − V/9)·b_γ
#
# Differentiating:
#   dPe/dV = K · cosh(2·b_net(V)) · (2·b_γ / 9)
#
# The void-gradient force on capital is:
#   F_V = −∇_V Pe = −dPe/dV   (capital pushed along Pe gradient)
#
# For two jurisdictions A (high-Pe) and B (low-Pe):
#   J_{A→B} = −σ · ΔPe / Δx   (generalised Fourier-Fick law)
#
# where σ is the "void conductivity":
#   σ_extraction > 0   → capital moves TO high-Pe (profit maximisation)
#   σ_constraint < 0   → capital moves AWAY from high-Pe (measurement mandate)
#
# At Pe gradient V_A=9 (Versailles) → V_B=3 (Global Fund):
#   ΔPe = Pe(3) − Pe(9)   [large, negative]
#   J = −σ · ΔPe / 1  (ΔV = 6, normalise to unit gradient)

V_range = np.linspace(0.01, 8.99, 300)
Pe_curve      = np.array([Pe(v)       for v in V_range])
dPe_dV_curve  = np.array([dPe_dV(v)  for v in V_range])

# Illustrative constraint currents between jurisdiction pairs
pairs = [
    ("Versailles → Global Fund",     9.0, 3.0),
    ("League → Marshall Plan",       7.0, 4.0),
    ("IMF SAPs → PEPFAR",            7.0, 4.0),
    ("Young Plan → Bretton Woods",   6.0, 3.0),
    ("UNSC → UNGA",                  7.0, 3.0),
]

SIGMA_EXTRACTION = +1.0   # extraction conductivity (normalised)
SIGMA_CONSTRAINT = -1.0   # constraint conductivity (normalised)

print("\n── Constraint Current Table (J per unit σ) ─────────────────────────────")
print(f"{'Pair':<38} {'ΔPe':>8}  {'J_extraction':>13}  {'J_constraint':>13}")
print("─" * 82)
for label, vA, vB in pairs:
    dPe = Pe(vB) - Pe(vA)
    J_ex  = constraint_current(vA, vB, sigma=SIGMA_EXTRACTION)
    J_cst = constraint_current(vA, vB, sigma=SIGMA_CONSTRAINT)
    print(f"{label:<38} {dPe:>+8.1f}  {J_ex:>+13.2f}  {J_cst:>+13.2f}")

# %% Figure — 4-panel institutional analysis
fig = plt.figure(figsize=(16, 12))
fig.suptitle("nb_paper52 — The Constraint Current: Pe Gradient Analysis",
             fontsize=13, y=0.98, color='#e0e0e0')

grid = gs.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Panel 1: Pe(V) curve + Pe=0 line ──────────────────────────────────────────
ax1 = fig.add_subplot(grid[0, 0])
ax1.plot(V_range, Pe_curve, color='#ff6b35', lw=2.5, label='Pe(V)')
ax1.axhline(0, color='#3ddc84', lw=1.5, ls='--', label='Pe=0 (c_zero)')
ax1.axvline(V_ZERO, color='#3ddc84', lw=1, ls=':', alpha=0.6)
ax1.fill_between(V_range, Pe_curve, 0,
                 where=Pe_curve>0, alpha=0.15, color='#ff6b35',
                 label='Drift-dominated')
ax1.fill_between(V_range, Pe_curve, 0,
                 where=Pe_curve<0, alpha=0.15, color='#3ddc84',
                 label='Constraint-dominant')
# Overlay institution dots
COLOUR_EFF = {0:'#ff1100', 1:'#ff6b35', 2:'#ffd700', 3:'#7ed957', 4:'#3ddc84'}
for inst in institutions:
    ax1.scatter(inst.V_raw, inst.Pe_inst, s=70,
                color=COLOUR_EFF[inst.effectiveness],
                zorder=5, edgecolors='white', lw=0.5)
    ax1.annotate(inst.name.split('/')[0].split('(')[0].strip(),
                 (inst.V_raw, inst.Pe_inst), fontsize=6.5,
                 xytext=(4, 4), textcoords='offset points', color='#ccc')
ax1.set_xlabel('V_raw (void index, 0–9)')
ax1.set_ylabel('Péclet Number Pe')
ax1.set_title('Pe(V) via V3 Bridge — 10 Mechanisms', fontsize=10)
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Panel 2: Pe vs longevity scatter ──────────────────────────────────────────
ax2 = fig.add_subplot(grid[0, 1])
for inst in institutions:
    ax2.scatter(inst.Pe_inst, inst.longevity_years, s=80,
                color=COLOUR_EFF[inst.effectiveness],
                zorder=5, edgecolors='white', lw=0.5)
    ax2.annotate(inst.name.split('/')[0].split('(')[0].strip()[:14],
                 (inst.Pe_inst, inst.longevity_years), fontsize=6.5,
                 xytext=(4, 4), textcoords='offset points', color='#ccc')
ax2.axvline(0, color='#3ddc84', lw=1, ls='--', alpha=0.7)
rho_lbl = f"ρ = {rho_Pe_lon:+.3f}  p = {p_Pe_lon:.3f}"
ax2.text(0.05, 0.93, rho_lbl, transform=ax2.transAxes,
         color='#ffd700', fontsize=9, va='top')
ax2.set_xlabel('Pe (institutional)')
ax2.set_ylabel('Effective longevity (years)')
ax2.set_title('Pe vs Longevity  [Spearman]', fontsize=10)
ax2.grid(True, alpha=0.3)

# ── Panel 3: Pe vs effectiveness scatter ──────────────────────────────────────
ax3 = fig.add_subplot(grid[1, 0])
for inst in institutions:
    ax3.scatter(inst.Pe_inst, inst.effectiveness, s=80,
                color=COLOUR_EFF[inst.effectiveness],
                zorder=5, edgecolors='white', lw=0.5)
    ax3.annotate(inst.name.split('/')[0].split('(')[0].strip()[:14],
                 (inst.Pe_inst, inst.effectiveness), fontsize=6.5,
                 xytext=(4, 2), textcoords='offset points', color='#ccc')
ax3.axvline(0, color='#3ddc84', lw=1, ls='--', alpha=0.7)
rho_lbl2 = f"ρ = {rho_Pe_eff:+.3f}  p = {p_Pe_eff:.3f}"
ax3.text(0.05, 0.93, rho_lbl2, transform=ax3.transAxes,
         color='#ffd700', fontsize=9, va='top')
ax3.set_xlabel('Pe (institutional)')
ax3.set_ylabel('Effectiveness (0=fail, 4=exceptional)')
ax3.set_yticks([0, 1, 2, 3, 4])
ax3.set_yticklabels(['0 fail', '1 weak', '2 mod.', '3 strong', '4 excep.'])
ax3.set_title('Pe vs Effectiveness  [Spearman]', fontsize=10)
ax3.grid(True, alpha=0.3)

# ── Panel 4: Constraint Current J(V_A → V_B) heat-map ────────────────────────
ax4 = fig.add_subplot(grid[1, 1])
V_nodes  = np.linspace(0.5, 8.5, 50)
VA_grid, VB_grid = np.meshgrid(V_nodes, V_nodes)
J_grid = -1.0 * (np.vectorize(Pe)(VB_grid) - np.vectorize(Pe)(VA_grid))
# σ=−1 (constraint capital): current magnitude, negative = constraint flows
im = ax4.imshow(J_grid, origin='lower', aspect='auto',
                extent=[0.5, 8.5, 0.5, 8.5],
                cmap='RdYlGn', vmin=-J_grid.max(), vmax=J_grid.max())
ax4.set_xlabel('V_A (source jurisdiction)')
ax4.set_ylabel('V_B (destination jurisdiction)')
ax4.set_title('Constraint Current J(V_A→V_B)  [σ=−1]', fontsize=10)
ax4.axvline(V_ZERO, color='white', lw=1, ls=':', alpha=0.5)
ax4.axhline(V_ZERO, color='white', lw=1, ls=':', alpha=0.5)
ax4.text(1.5, 7.5, 'EXTRACTION\nCIRCUIT\n(J<0)', color='#ff4444',
         fontsize=8, ha='center', va='center')
ax4.text(7.0, 1.5, 'CONSTRAINT\nCURRENT\n(J>0)', color='#3ddc84',
         fontsize=8, ha='center', va='center')
plt.colorbar(im, ax=ax4, label='J (capital flux, normalised)')

plt.savefig('ops/lab/experiments/nb_paper52_output.png', dpi=140,
            bbox_inches='tight', facecolor='#0d0d0d')
plt.show()
print("\n── Figure saved to ops/lab/experiments/nb_paper52_output.png")

# %% Analytical summary — numbers for Paper 52 §5
print("\n═══ ANALYTICAL RESULTS FOR PAPER 52 §5 ═══════════════════════════════")
print(f"\nV3 Bridge (canonical): c = 1 − V/9")
print(f"Pe(V):  Pe = {K_STD:.0f} · sinh(2·(b_α − (1−V/9)·b_γ))")
print(f"dPe/dV: {K_STD:.0f} · cosh(2·b_net(V)) · (2·{B_GAMMA}/{9}) = {K_STD:.0f}·cosh(f)·{2*B_GAMMA/9:.4f}")
print(f"  — always positive: Pe strictly increasing in V.")
print(f"  — at V=0 (Pe≈−77):    dPe/dV = {dPe_dV(0.01):.2f}")
print(f"  — at V={V_ZERO:.2f} (Pe=0): dPe/dV = {dPe_dV(V_ZERO):.2f}  [minimum gradient, Pe=0 boundary]")
print(f"  — at V=9 (Pe≈+44.9): dPe/dV = {dPe_dV(8.99):.2f}  [steepest extraction gradient]")

print(f"\nConstraint Current J = −σ · ΔPe")
print(f"  Extraction conductivity σ>0: capital ATTRACTED to high-Pe (Versailles, Dawes)")
print(f"  Constraint conductivity σ<0: capital FLEES high-Pe toward low-Pe (Global Fund)")

print(f"\nSpearman results (N=10 historical mechanisms):")
print(f"  V vs longevity:      ρ = {rho_V_lon:+.4f}  p = {p_V_lon:.4f}")
print(f"  Pe vs longevity:     ρ = {rho_Pe_lon:+.4f}  p = {p_Pe_lon:.4f}")
print(f"  V vs effectiveness:  ρ = {rho_V_eff:+.4f}  p = {p_V_eff:.4f}")
print(f"  Pe vs effectiveness: ρ = {rho_Pe_eff:+.4f}  p = {p_Pe_eff:.4f}")

print(f"\nKey Pe values (K=16):")
for inst in sorted(institutions, key=lambda i: i.Pe_inst):
    print(f"  {inst.name:<28} V={inst.V_raw:.0f}  Pe={inst.Pe_inst:+7.1f}"
          f"  lon={inst.longevity_years:.0f}yr  eff={inst.effectiveness}")
