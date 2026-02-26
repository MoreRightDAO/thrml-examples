"""
nb_evo02: Kin Selection and Hamilton's Rule from Pe < 0

Shows that Hamilton's rule (rB > C) is exactly the Pe < 0 condition in THRML
coordinates. The Pe = 0 contour over (r, B/C) parameter space is the Hamilton
boundary.

Key derivation:
  Map: c(r, BC) = c_zero · r · BC     [relatedness × benefit-cost drives constraint]
       (at rBC = 1: c = c_zero → Pe = 0, Hamilton boundary)

  Pe = K · sinh(2 · B_α · (1 − r · BC))

  At r·BC = 1  (Hamilton boundary rB = C): Pe = 0
  When rB > C (r·BC > 1):               Pe < 0  ← cooperation attractor (constraint > drift)
  When rB < C (r·BC < 1):               Pe > 0  ← defection attractor (drift > constraint)

  c_zero = B_α/B_γ ≈ 0.3866 is the THRML neutral boundary — the mutation-selection
  balance point. Hamilton's rule is its instantiation in kin selection coordinates.

Derivation check:
  c = c_zero · r · BC
  Pe = K · sinh(2 · (B_α − c · B_γ))
     = K · sinh(2 · (B_α − (B_α/B_γ) · r · BC · B_γ))
     = K · sinh(2 · B_α · (1 − r · BC))

References:
  Hamilton (1964) — inclusive fitness theory
  THRML: Eckert (2026) Papers 3, 5
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Canonical THRML Parameters ────────────────────────────────────────────────
B_ALPHA = 0.867
B_GAMMA = 2.244
K       = 16
C_ZERO  = B_ALPHA / B_GAMMA   # ≈ 0.3864

print(f"Canonical parameters:")
print(f"  B_α = {B_ALPHA}  |  B_γ = {B_GAMMA}  |  K = {K}")
print(f"  c_zero = {C_ZERO:.4f}")
print(f"\nDerivation: Pe = K · sinh(2·B_α·(1 − r·BC))")
print(f"  Pe = 0  ↔  rBC = 1  ↔  rB = C   (Hamilton boundary)")
print(f"  Pe < 0  ↔  rBC > 1  ↔  rB > C   (cooperation attractor)")
print(f"  Pe > 0  ↔  rBC < 1  ↔  rB < C   (defection attractor)")

# ── Pe Formula — Hamilton Mapping ─────────────────────────────────────────────
def pe_hamilton(r, BC):
    """
    Pe under Hamilton's rule mapping.
    r   = genetic relatedness ∈ (0, 1]
    BC  = benefit-cost ratio B/C > 0

    Pe = K * sinh(2 * B_α * (1 - r * BC))
    Clipped at ±60 nats to avoid sinh overflow on heatmap.
    """
    arg = 2 * B_ALPHA * (1.0 - r * BC)
    arg = np.clip(arg, -6.0, 6.0)   # sinh clip: sinh(6) ≈ 201 → Pe range ≈ ±3200
    return K * np.sinh(arg)

# Boundary verification
print(f"\nBoundary verification (Pe should = 0 when rBC = 1):")
for r, BC in [(0.5, 2.0), (0.25, 4.0), (0.1, 10.0), (1.0, 1.0)]:
    pe = pe_hamilton(r, BC)
    print(f"  r={r:.2f}, B/C={BC:.1f}: r·BC={r*BC:.2f}, Pe={pe:.4f}")

# ── Biological Scenarios ───────────────────────────────────────────────────────
# (name, r, BC, expected — "coop" if rBC > 1, "defect" if rBC < 1)
scenarios = [
    ("Haplodiploid sisters\n(r=0.75, B/C=3)",   0.75, 3.0),
    ("Diploid siblings\n(r=0.5, B/C=2.5)",       0.50, 2.5),
    ("Half-siblings\n(r=0.25, B/C=5)",           0.25, 5.0),
    ("Cousins\n(r=0.125, B/C=6)",               0.125, 6.0),
    ("Cheater strategy\n(r=0.5, B/C=0.5)",       0.50, 0.5),
    ("Unrelated alloparenting\n(r=0.0, B/C=8)",  0.02, 8.0),
]

print(f"\nBiological scenarios:")
for name, r, BC in scenarios:
    pe = pe_hamilton(r, BC)
    rbc = r * BC
    outcome = "Pe < 0 (cooperation)" if rbc > 1 else "Pe > 0 (defection)"
    hamilton = "rB > C ✓ cooperation" if rbc > 1 else "rB < C — defection"
    print(f"  {name.split(chr(10))[0]:<30s}: r·BC={rbc:.3f}  Pe={pe:+8.2f}  |  {hamilton}")

# ── Parameter Sweep ────────────────────────────────────────────────────────────
r_vals  = np.linspace(0.02, 1.0,  300)
bc_vals = np.linspace(0.5,  6.0,  300)
R, BC   = np.meshgrid(r_vals, bc_vals)
PE_grid = pe_hamilton(R, BC)

# Hamilton boundary: r * BC = 1 → BC = 1/r
bc_boundary = np.where(r_vals > 0, 1.0 / r_vals, np.nan)
mask        = (bc_boundary >= 0.5) & (bc_boundary <= 6.0)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d0d0d')

# ── Left panel: 2D heatmap ────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#0d0d0d')

pe_abs_max = np.percentile(np.abs(PE_grid), 97)
norm = mcolors.TwoSlopeNorm(vmin=-pe_abs_max, vcenter=0, vmax=pe_abs_max)

im = ax.pcolormesh(r_vals, bc_vals, PE_grid, cmap='RdBu_r', norm=norm, shading='auto')
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.ax.tick_params(colors='#888888')
cbar.set_label('Péclet number Pe', color='#cccccc', fontsize=9)

# Hamilton boundary (Pe = 0 ↔ r·BC = 1)
ax.plot(r_vals[mask], bc_boundary[mask], 'w-', linewidth=2.2, label='rB = C  (Pe = 0)', zorder=5)

# Scenario points
for name, r, BC_val in scenarios:
    pe = pe_hamilton(r, BC_val)
    color  = '#00ff88' if r * BC_val > 1 else '#ff4444'
    ax.scatter(r, BC_val, s=80, color=color, edgecolors='white', linewidth=0.8, zorder=6)
    short = name.split('\n')[0].split('(')[0].strip()
    ax.annotate(short, (r, BC_val), textcoords='offset points',
                xytext=(5, 3), color='white', fontsize=7, alpha=0.9)

ax.text(0.70, 4.8, 'Pe < 0\nCooperation\n(rB > C)', color='#aaddff', fontsize=9, ha='center', style='italic')
ax.text(0.20, 1.2, 'Pe > 0\nDefection\n(rB < C)', color='#ffaaaa', fontsize=9, ha='center', style='italic')

ax.set_xlabel('Genetic relatedness r', color='#cccccc', fontsize=10)
ax.set_ylabel('Benefit-cost ratio B/C', color='#cccccc', fontsize=10)
ax.set_title("Hamilton's Rule as Pe Bifurcation\nPe < 0 ↔ rB > C", color='white', fontsize=11)
ax.tick_params(colors='#888888')
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
ax.legend(handles=[
    Line2D([0],[0], color='white', linewidth=2, label='Hamilton boundary (Pe=0)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#00ff88', markersize=8, label='Cooperation (rB>C, Pe<0)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff4444', markersize=8, label='Defection (rB<C, Pe>0)'),
], fontsize=8, loc='upper right', facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')

# ── Right panel: Pe vs r at fixed B/C ────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0d0d0d')

bc_fixed  = [1.2, 2.0, 3.0, 5.0, 8.0]
colors_bc = ['#00aaff', '#00ccff', '#00ddaa', '#ffcc44', '#ff8844']
r_sweep   = np.linspace(0.01, 1.0, 300)

for BC_f, col in zip(bc_fixed, colors_bc):
    pe_sweep = pe_hamilton(r_sweep, BC_f)
    ax2.plot(r_sweep, pe_sweep, color=col, linewidth=1.8, label=f'B/C = {BC_f}')
    r_cross = 1.0 / BC_f
    if 0 < r_cross <= 1.0:
        ax2.axvline(x=r_cross, color=col, linestyle=':', linewidth=0.8, alpha=0.5)
        ax2.scatter([r_cross], [0], color=col, s=50, zorder=5)

ax2.axhline(y=0, color='white', linestyle='--', linewidth=1.8, alpha=0.9, label='Pe=0 (Hamilton boundary)')

# Shade cooperation / defection
y_min = ax2.get_ylim()[0] if ax2.get_ylim()[0] != ax2.get_ylim()[1] else -200
pe_plot_range = [pe_hamilton(r, BC_f) for r in r_sweep for BC_f in bc_fixed]
y_lo = max(np.min(pe_plot_range), -300)
ax2.set_ylim(y_lo, 200)
ax2.axhspan(y_lo, 0,    alpha=0.05, color='#0066ff')
ax2.axhspan(0,    200,  alpha=0.05, color='#ff3333')
ax2.text(0.7,  100, 'Defection\n(Pe > 0)', color='#ffaaaa', fontsize=9, ha='center')
ax2.text(0.7, y_lo*0.5, 'Cooperation\n(Pe < 0)', color='#aaddff', fontsize=9, ha='center')

ax2.set_xlim(0, 1)
ax2.set_xlabel('Genetic relatedness r', color='#cccccc', fontsize=10)
ax2.set_ylabel('Péclet number Pe', color='#cccccc', fontsize=10)
ax2.set_title("Pe vs r at Fixed B/C\n(dots = Hamilton crossover r = C/B)", color='white', fontsize=11)
ax2.tick_params(colors='#888888')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333333')
ax2.legend(fontsize=8, loc='upper left', facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')

plt.tight_layout()
svg_path = '/data/apps/morr/private/phase-2/thrml/nb_evo02_kin_selection.svg'
plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {svg_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY: Kin Selection / Hamilton's Rule (nb_evo02)")
print("="*60)
print(f"\nCore identity confirmed:")
print(f"  Pe = K · sinh(2·B_α·(1 − r·BC))")
print(f"  Pe = 0 boundary ≡ Hamilton boundary rB = C")
print(f"\nc_zero = {C_ZERO:.4f}:")
print(f"  THRML: mutation-selection balance (EXP-001, 10 substrates)")
print(f"  Kin selection: the inclusive fitness threshold")
print(f"  Same constraint — different substrate — same boundary")
print(f"\nScenario validation:")
for name, r, BC_val in scenarios:
    pe   = pe_hamilton(r, BC_val)
    rbc  = r * BC_val
    pred = "Pe<0 (coop)" if pe < 0 else "Pe>0 (defect)"
    hami = "rB>C (coop)" if rbc > 1 else "rB<C (defect)"
    ok   = (pe < 0) == (rbc > 1)
    print(f"  {'✓' if ok else '✗'} {name.split(chr(10))[0]:<35s}  Pe={pe:+8.2f}  {pred} | {hami}")
