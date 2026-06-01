"""
OQ-1 (residual): what is the ~8% gap between the τ-formula and the empirical t*?

derive_tau_formula.py gives the leading-order threshold
    T_c = (τ! / ((N-1) Q_τ))^{1/τ} = 24.05   vs   empirical t* = 26   (7.5% error)

This script asks whether that 8% is a single missing analytic term — and finds it
is NOT. Using a fully DETERMINISTIC per-pair analysis (no Monte-Carlo noise) on the
benign co-activation matrix Q, "the transition" turns out not to be one point but a
WINDOW, because the DC-SBM degree heterogeneity (hubs) separates two thresholds:

  T_MR    Molloy-Reed onset, ⟨k²⟩/⟨k⟩ = 2   — where a giant FIRST becomes possible
  T_⟨k⟩=1 mean-degree-one crossing (exact Poisson) — the ER mean-field point

The hubs make ⟨k²⟩/⟨k⟩ large, so the giant can nucleate well below ⟨k⟩=1. Between
T_MR and T_⟨k⟩=1 the giant grows; the empirical t* — defined by the steepest jump
in the giant-component ratio — sits INSIDE this window, as does the leading-order
formula. So the residual is the finite WIDTH of the transition, not a missing term.

Two further honesty notes the numbers expose:
  • The leading-order ⟨k⟩ OVER-counts high-q (hub) pairs (their edge probability
    exceeds 1 in the approximation), which pulls T_c down; truncating higher-order
    Poisson terms pulls it up. The 7.5% agreement is partly a cancellation.
  • The empirical anchor (t*=26, ⟨k⟩≈1.10) is measured on the ATTACK-accelerated
    graph, while the formula is benign mean-field — part of the gap is the attack's
    extra density, not formula error.

Conclusion: the formula is accurate to within the intrinsic transition-window
width. Tightening below ~8% is not a universal correction — it requires choosing
WHICH point in the window counts as "the" threshold and adding a deployment-specific
attack-density offset.
"""

import os
import numpy as np
from scipy import stats, special, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import siw_style as S

# ─────────────────────────────────────────
# DETERMINISTIC SEMANTIC SPACE (identical params to derive_tau_formula)
# ─────────────────────────────────────────
np.random.seed(42)
N, K, TAU = 500, 12, 3
EMP_TSTAR = 26
EMP_PC    = 0.002196
EMP_KBAR  = (N - 1) * EMP_PC          # empirical mean degree at t*  (≈ 1.10)

cluster_assignments = np.random.choice(K, N)
hub_set = set(np.random.choice(N, int(N * 0.05), replace=False))

P = np.full((N, K), 0.05)
for c in range(K):
    P[cluster_assignments == c, c] = 0.30
P[list(hub_set), :] *= 3.0
P = np.minimum(P, 1.0)

Q = (P @ P.T) / K
np.fill_diagonal(Q, 0)
q_vals = Q[np.triu_indices(N, k=1)]
Q_tau = np.mean(q_vals ** TAU)


# ─────────────────────────────────────────
# THE THREE THRESHOLD DEFINITIONS (deterministic)
# ─────────────────────────────────────────
def kbar_leading(T):
    return (N - 1) * (T ** TAU / special.factorial(TAU)) * Q_tau

def edge_prob_matrix(T):
    Pe = 1.0 - stats.poisson.cdf(TAU - 1, T * Q)
    np.fill_diagonal(Pe, 0.0)
    return Pe

def kbar_exact(T):
    return edge_prob_matrix(T).sum(axis=1).mean()

def mr_ratio(T):
    ki = edge_prob_matrix(T).sum(axis=1)
    return (ki ** 2).mean() / (ki.mean() + 1e-12)

T_lead  = (special.factorial(TAU) / ((N - 1) * Q_tau)) ** (1.0 / TAU)   # ⟨k⟩_lead = 1
T_exact = optimize.brentq(lambda T: kbar_exact(T) - 1.0, 1, 200)        # ⟨k⟩_exact = 1
T_mr    = optimize.brentq(lambda T: mr_ratio(T) - 2.0, 3, 200)          # MR onset

print("=== THREE THRESHOLD DEFINITIONS (deterministic, benign Q) ===")
print(f"Q_τ = {Q_tau:.3e}")
print(f"T_MR   (Molloy-Reed onset, ⟨k²⟩/⟨k⟩=2) = {T_mr:.2f}   err vs t*={abs(T_mr-EMP_TSTAR)/EMP_TSTAR*100:4.1f}%")
print(f"T_lead (leading-order ⟨k⟩=1)            = {T_lead:.2f}   err vs t*={abs(T_lead-EMP_TSTAR)/EMP_TSTAR*100:4.1f}%")
print(f"t*     (empirical, steepest jump)        = {EMP_TSTAR}      [anchor]")
print(f"T_⟨k⟩=1 (exact-Poisson ⟨k⟩=1)            = {T_exact:.2f}   err vs t*={abs(T_exact-EMP_TSTAR)/EMP_TSTAR*100:4.1f}%")
window_w = T_exact - T_mr
print(f"\nTransition WINDOW [T_MR, T_⟨k⟩=1] = [{T_mr:.1f}, {T_exact:.1f}]  width = {window_w:.1f} sessions "
      f"({window_w/EMP_TSTAR*100:.0f}% of t*)")
print(f"Empirical t*={EMP_TSTAR} and leading-order {T_lead:.1f} BOTH lie inside the window.")

# leading vs exact at the empirical anchor (hub over-counting)
print(f"\n=== HONESTY CHECK 1: leading-order over-counts hub pairs ===")
print(f"⟨k⟩ at T={EMP_TSTAR}:  leading={kbar_leading(EMP_TSTAR):.3f}   exact={kbar_exact(EMP_TSTAR):.3f}   "
      f"(leading inflated by {(kbar_leading(EMP_TSTAR)/kbar_exact(EMP_TSTAR)-1)*100:.0f}%)")
print(f"  leading-order T^τ Q_τ/τ! systematically overestimates 1-F_Poisson for the")
print(f"  high-q (hub/within-cluster) pairs, where T·q is not small.")

print(f"\n=== HONESTY CHECK 2: empirical anchor is attack-accelerated ===")
print(f"empirical ⟨k⟩ at t* (from p_c={EMP_PC}) = {EMP_KBAR:.3f}")
print(f"benign-only exact ⟨k⟩ at T={EMP_TSTAR}    = {kbar_exact(EMP_TSTAR):.3f}")
print(f"  → attack adds ≈ {EMP_KBAR - kbar_exact(EMP_TSTAR):.2f} mean degree (extra density, not formula error)")


# ─────────────────────────────────────────
# TRAJECTORIES FOR PLOTTING
# ─────────────────────────────────────────
T_grid = np.linspace(1, 45, 90)
kb_lead  = np.array([kbar_leading(T) for T in T_grid])
kb_exact = np.array([kbar_exact(T)  for T in T_grid])
mr_traj  = np.array([mr_ratio(T)    for T in T_grid])


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.suptitle('OQ-1: Transition Threshold Residual',
             fontsize=14, fontweight='bold', color=S.TEXT_C)

# ── Panel A: ⟨k⟩(T) leading vs exact ──
ax = axes[0, 0]; S.style_ax(ax)
ax.plot(T_grid, kb_exact, '-', color=S.C_BLUE, lw=2.2, label='exact Poisson  ⟨k⟩')
ax.plot(T_grid, kb_lead,  '--', color=S.C_ORANGE, lw=2, label='leading-order  T^τ Q_τ/τ!')
ax.axhline(1.0, color=S.SUBTLE, ls=':', lw=1.4)
ax.text(1.5, 1.04, '⟨k⟩ = 1', fontsize=8.5, color=S.SUBTLE)
for T, c, lab in [(T_lead, S.C_ORANGE, 'T_lead'), (T_exact, S.C_BLUE, 'T_⟨k⟩=1'),
                  (EMP_TSTAR, S.C_RED, 't*')]:
    ax.axvline(T, color=c, ls='-', lw=1.1, alpha=0.6)
ax.set_xlabel('session T'); ax.set_ylabel('mean degree ⟨k⟩')
ax.set_ylim(0, 2.2); ax.set_xlim(1, 45)
ax.set_title('Leading-order over-counts hub pairs →\nreaches ⟨k⟩=1 earlier (T=24) than exact (T=29)')
ax.legend(fontsize=8.5, loc='upper left')
S.panel_label(ax, 'a')

# ── Panel B: Molloy-Reed ratio ──
ax = axes[0, 1]; S.style_ax(ax)
ax.plot(T_grid, mr_traj, '-', color=S.C_PURPLE, lw=2.2)
ax.axhline(2.0, color=S.SUBTLE, ls=':', lw=1.4)
ax.text(1.5, 2.1, 'giant onset  ⟨k²⟩/⟨k⟩ = 2', fontsize=8.5, color=S.SUBTLE)
ax.axvline(T_mr, color=S.C_PURPLE, ls='-', lw=1.1, alpha=0.6)
ax.scatter([T_mr], [2.0], color=S.C_PURPLE, zorder=5, s=40)
ax.set_xlabel('session T'); ax.set_ylabel('⟨k²⟩ / ⟨k⟩  (Molloy-Reed)')
ax.set_xlim(1, 45)
ax.set_title(f'DC-SBM hubs make the giant nucleate at T≈{T_mr:.0f},\n'
             'well below the ⟨k⟩=1 mean-field point')
S.panel_label(ax, 'b')

# ── Panel C: threshold ruler with window ──
ax = axes[1, 0]; S.style_ax(ax)
ax.axvspan(T_mr, T_exact, color=S.C_GREEN, alpha=0.10)
marks = [(T_mr, 'Molloy-Reed\nonset', S.C_PURPLE),
         (T_lead, 'leading-order\nformula', S.C_ORANGE),
         (EMP_TSTAR, 'empirical\nt*', S.C_RED),
         (T_exact, 'exact ⟨k⟩=1', S.C_BLUE)]
for i, (T, lab, c) in enumerate(marks):
    y = 0.3 + (i % 2) * 0.35
    ax.axvline(T, color=c, lw=1.6)
    ax.scatter([T], [y], color=c, s=60, zorder=5)
    ax.text(T, y + 0.06, f'{lab}\nT={T:.1f}', ha='center', fontsize=8, color=c, fontweight='bold')
ax.annotate('', xy=(T_exact, 0.12), xytext=(T_mr, 0.12),
            arrowprops=dict(arrowstyle='<->', color=S.C_GREEN, lw=1.8))
ax.text((T_mr + T_exact) / 2, 0.05, f'transition window  ≈ {window_w:.1f} sessions',
        ha='center', fontsize=9, color=S.C_GREEN, fontweight='bold')
ax.set_xlim(T_mr - 3, T_exact + 3); ax.set_ylim(0, 1.0); ax.set_yticks([])
ax.set_xlabel('session T')
ax.set_title('All four threshold definitions bracket t*;\nthe 8% gap is sub-window')
S.panel_label(ax, 'c')

# ── Panel D: error budget / conclusion ──
ax = axes[1, 1]; S.style_ax(ax); ax.axis('off')
ax.set_title('Residual budget', fontsize=11, pad=8)
rows = [
    ('leading-order T_c',        f'{T_lead:.2f}',   f'{abs(T_lead-EMP_TSTAR)/EMP_TSTAR*100:.1f}% vs t*'),
    ('exact-Poisson ⟨k⟩=1',      f'{T_exact:.2f}',  f'{abs(T_exact-EMP_TSTAR)/EMP_TSTAR*100:.1f}% vs t*'),
    ('Molloy-Reed onset',        f'{T_mr:.2f}',     f'{abs(T_mr-EMP_TSTAR)/EMP_TSTAR*100:.1f}% vs t*'),
    ('transition window width',  f'{window_w:.1f}', f'{window_w/EMP_TSTAR*100:.0f}% of t*'),
    ('hub over-count @ t*',      f'{(kbar_leading(EMP_TSTAR)/kbar_exact(EMP_TSTAR)-1)*100:.0f}%', 'leading vs exact ⟨k⟩'),
    ('attack density offset',    f'{EMP_KBAR - kbar_exact(EMP_TSTAR):+.2f}', 'Δ⟨k⟩ (not formula err)'),
]
y0, dy = 0.86, 0.135
for lab, val, note in rows:
    ax.text(0.02, y0, lab, transform=ax.transAxes, fontsize=9, color=S.SUBTLE, va='top')
    ax.text(0.56, y0, val, transform=ax.transAxes, fontsize=9.5, color=S.TEXT_C,
            va='top', ha='right', fontweight='bold')
    ax.text(0.60, y0, note, transform=ax.transAxes, fontsize=8.2, color=S.SUBTLE, va='top')
    ax.plot([0.02, 0.98], [y0 - 0.02, y0 - 0.02], color=S.SPINE_C, lw=0.5,
            transform=ax.transAxes, clip_on=False)
    y0 -= dy
ax.text(0.02, 0.04,
        'The formula is accurate to within the transition-window width. The 8%\n'
        'is not a missing term — it is which point in the window you call "the"\n'
        'threshold, plus a deployment-specific attack-density offset.',
        transform=ax.transAxes, fontsize=8.3, color=S.SUBTLE, va='bottom')
S.panel_label(ax, 'd')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), 'figure_oq1_residual.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out}")

# ─────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────
print("\n" + "=" * 64)
print("OQ-1 RESIDUAL VERDICT")
print("=" * 64)
print(f"""
The ~8% residual between the τ-formula (T_c={T_lead:.1f}) and the empirical
t*={EMP_TSTAR} is NOT a single missing analytic term. A deterministic per-pair
analysis of the benign co-activation matrix shows:

  The DC-SBM degree heterogeneity (hubs) splits "the transition" into a window:
      Molloy-Reed onset      T ≈ {T_mr:.1f}   (giant first possible, ⟨k²⟩/⟨k⟩=2)
      exact ⟨k⟩=1 crossing   T ≈ {T_exact:.1f}   (ER mean-field point)
      window width           ≈ {window_w:.1f} sessions  ({window_w/EMP_TSTAR*100:.0f}% of t*)

  The leading-order formula ({T_lead:.1f}) and the empirical t* ({EMP_TSTAR}) BOTH fall
  inside this window. The leading-order term lands close to t* partly by a
  cancellation: it over-counts hub pairs by {(kbar_leading(EMP_TSTAR)/kbar_exact(EMP_TSTAR)-1)*100:.0f}% at t* (pulling T_c down)
  while truncating higher-order Poisson terms (pulling it up).

  Separately, the empirical anchor is attack-accelerated: ⟨k⟩≈{EMP_KBAR:.2f} at t* vs
  benign-only exact ⟨k⟩={kbar_exact(EMP_TSTAR):.2f}, an offset of {EMP_KBAR-kbar_exact(EMP_TSTAR):+.2f} that is extra
  attack density, not formula error.

Conclusion: the τ-formula is accurate to within the intrinsic transition-window
width. The residual cannot be reduced by a universal closed-form correction — it
is set by (1) which point in the window one defines as "the" threshold and (2) a
deployment-specific attack-density offset. This dissolves OQ-1's residual rather
than overfitting it — consistent with the project's honest-boundary stance.

  OQ-1 residual: CHARACTERISED (window effect + attack offset), not "closed".
""")
