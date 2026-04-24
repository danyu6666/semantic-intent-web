"""
OQ-1 (sub): Derive analytical p_c formula for SIW with co-activation threshold τ

Key insight from DC-SBM analysis:
  - COACT_THRESHOLD = τ means edges require τ joint session appearances
  - This is NOT independent edge formation (as DC-SBM/ER assume)
  - Edge probability at session T: P(C_ij(T) ≥ τ) = 1 - F_Poisson(τ-1; T × q_ij)

Derivation strategy:
  1. Compute q_ij: per-session co-activation probability for each pair
  2. Derive P(edge by T): exact Poisson approximation
  3. Show edge density p(T) ≈ T^τ × Q_τ / τ!   (power-law in T)
  4. Threshold condition: E[d(T)] = 1 → T_c = (τ! / (N × Q_τ))^{1/τ}
  5. Community correction: DC-SBM shifts T_c via λ_max(B)
  6. Verify against empirical t* = 26
"""

import numpy as np
from scipy import stats, special
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────
# SIMULATION PARAMETERS (identical)
# ─────────────────────────────────────────
N   = 500
K   = 12
TAU = 3          # COACT_THRESHOLD
T   = 300        # N_SESSIONS

cluster_assignments = np.random.choice(K, N)
hub_features        = np.random.choice(N, int(N * 0.05), replace=False)
hub_set             = set(hub_features)

EMPIRICAL_T_STAR = 26
EMPIRICAL_PC     = 0.002196
MAX_EDGES        = N * (N - 1) / 2

# ─────────────────────────────────────────
# STEP 1: DERIVE q_ij ANALYTICALLY
# ─────────────────────────────────────────
print("=== STEP 1: q_ij ANALYTICAL FORMULA ===")

# p_i(c) = activation probability of feature i given session cluster c
def p_act(feat, sess_cluster):
    base = 0.3 if cluster_assignments[feat] == sess_cluster else 0.05
    if feat in hub_set:
        base *= 3.0
    return min(base, 1.0)

# q_ij = E_c[p_i(c) × p_j(c)] = (1/K) Σ_c p_i(c) × p_j(c)
# For a random session cluster c drawn uniformly from {0,...,K-1}

# Precompute p_i(c) for all features and clusters
P_mat = np.zeros((N, K))  # P_mat[i, c] = p_i(c)
for i in range(N):
    for c in range(K):
        P_mat[i, c] = p_act(i, c)

# q_ij = (1/K) Σ_c P_mat[i,c] × P_mat[j,c]
# For all pairs: Q = (1/K) × P_mat @ P_mat.T  (matrix of q_ij)
Q_mat = (P_mat @ P_mat.T) / K  # N×N matrix, Q_mat[i,j] = q_ij
np.fill_diagonal(Q_mat, 0)

q_values = Q_mat[np.triu_indices(N, k=1)]  # upper triangle, all pairs

# Summary by pair type
same_cluster = (cluster_assignments[:, None] == cluster_assignments[None, :])
hub_mask = np.zeros(N, dtype=bool)
hub_mask[list(hub_set)] = True
hub_pair = hub_mask[:, None] | hub_mask[None, :]

q_within = Q_mat[same_cluster & ~np.eye(N, dtype=bool)]
q_cross  = Q_mat[~same_cluster]

print(f"q_ij within-cluster: mean={np.mean(q_within):.5f}, max={np.max(q_within):.5f}")
print(f"q_ij cross-cluster:  mean={np.mean(q_cross):.5f}")
print(f"q_in / q_out ratio:  {np.mean(q_within)/np.mean(q_cross):.2f}×")
print(f"Hub-containing pairs: mean q = {np.mean(q_values[hub_pair[np.triu_indices(N,k=1)]]):.5f}")
print(f"Non-hub pairs:        mean q = {np.mean(q_values[~hub_pair[np.triu_indices(N,k=1)]]):.5f}")

# ─────────────────────────────────────────
# STEP 2: EDGE PROBABILITY AS FUNCTION OF T
# ─────────────────────────────────────────
print("\n=== STEP 2: EDGE FORMATION RATE p(T) ===")

# P(edge (i,j) by session T) = P(Poisson(T × q_ij) ≥ τ)
# For small λ = T × q_ij:  P ≈ λ^τ / τ!   (leading Poisson term)
# For large λ:              P ≈ 1

def edge_prob(q, t, tau):
    lam = t * q
    return 1.0 - stats.poisson.cdf(tau - 1, lam)

def edge_prob_approx(q, t, tau):
    """Leading-order approximation: valid for T*q << 1"""
    lam = t * q
    return lam ** tau / special.factorial(tau)

# Verify approximation at T = T_c
q_mean = np.mean(q_values)
lam_at_tc = EMPIRICAL_T_STAR * q_mean
print(f"λ̄ = T_c × q̄ = {EMPIRICAL_T_STAR} × {q_mean:.5f} = {lam_at_tc:.4f}")
print(f"Approximation valid (λ̄ << 1): {'YES' if lam_at_tc < 0.5 else 'NO, use exact'}")

# Expected edge density as function of T
# Sample q_values for speed (5000 pairs representative)
T_range = np.arange(1, T + 1)
rng_idx = np.random.choice(len(q_values), size=5000, replace=False)
q_sample = q_values[rng_idx]

p_T_exact  = np.array([np.mean(1.0 - stats.poisson.cdf(TAU - 1, t * q_sample))
                        for t in T_range])
p_T_approx = np.array([(t ** TAU / special.factorial(TAU)) * np.mean(q_values ** TAU)
                        for t in T_range])

print(f"\nEdge density at T* = {EMPIRICAL_T_STAR}:")
print(f"  Exact:       p = {p_T_exact[EMPIRICAL_T_STAR-1]:.6f}")
print(f"  Empirical:   p = {EMPIRICAL_PC:.6f}")
print(f"  Approx:      p = {p_T_approx[EMPIRICAL_T_STAR-1]:.6f}")

# Mean degree at T*
d_at_tc = p_T_exact[EMPIRICAL_T_STAR-1] * (N - 1)
print(f"  Mean degree: E[d] = {d_at_tc:.4f}")

# ─────────────────────────────────────────
# STEP 3: ANALYTICAL T_c FORMULA
# ─────────────────────────────────────────
print("\n=== STEP 3: ANALYTICAL T_c DERIVATION ===")

# The mean degree at session T (from Poisson approximation):
#   E[d(T)] = (N-1) × E_ij[P(edge by T)]
#           ≈ (N-1) × (T^τ / τ!) × E[q_ij^τ]
#
# Phase transition condition (ER mean-field):
#   E[d(T_c)] = 1
#
# Solving for T_c:
#   T_c^τ ≈ τ! / ((N-1) × E[q_ij^τ])
#   T_c   ≈ (τ! / ((N-1) × Q_τ))^{1/τ}
#
# where Q_τ = E[q_ij^τ] = mean of q^τ over all pairs

Q_tau = np.mean(q_values ** TAU)
T_c_formula = (special.factorial(TAU) / ((N - 1) * Q_tau)) ** (1.0 / TAU)

print(f"Q_τ = E[q^τ] = {Q_tau:.8f}")
print(f"τ! = {int(special.factorial(TAU))}")
print(f"T_c (formula) = ({int(special.factorial(TAU))} / ({N-1} × {Q_tau:.2e}))^{{1/{TAU}}}")
print(f"T_c (formula) = {T_c_formula:.2f}")
print(f"T_c (empirical t*) = {EMPIRICAL_T_STAR}")
print(f"Error: {abs(T_c_formula - EMPIRICAL_T_STAR)/EMPIRICAL_T_STAR*100:.1f}%")

# Corresponding p_c from the formula
p_c_formula = (T_c_formula ** TAU / special.factorial(TAU)) * Q_tau
print(f"\np_c (formula, approx) = {p_c_formula:.6f}")
print(f"p_c (empirical)       = {EMPIRICAL_PC:.6f}")
print(f"p_c (ER: 1/N)         = {1/N:.6f}")
print(f"Formula error: {abs(p_c_formula - EMPIRICAL_PC)/EMPIRICAL_PC*100:.1f}%")

# ─────────────────────────────────────────
# STEP 4: COMMUNITY CORRECTION TO T_c
# ─────────────────────────────────────────
print("\n=== STEP 4: COMMUNITY CORRECTION ===")

# The mean-field threshold E[d] = 1 ignores community structure.
# DC-SBM branching process gives the TRUE threshold:
#   ρ_c × λ_max(T_DC-SBM) = 1
#
# In the temporal model, this becomes:
#   T_c^τ × λ_max(T_DC-SBM)^τ × Q_τ / τ! = 1
#   T_c = (τ! / (λ_max(T_DC-SBM)^τ × Q_τ))^{1/τ}
#
# But empirically: ER-correction dominates because community structure
# is moderate (p_in/p_out = 7.62×) and τ amplifies it to (7.62)^3 ≈ 442×
# However the community correction in T_c:
#   ΔT_c / T_c ≈ ln(p_in/p_out) × τ / K  (log-ratio effect)

ratio = np.mean(q_within) / np.mean(q_cross)
community_correction = 1.0 / (ratio ** (1/TAU))  # T_c scales as Q_τ^{-1/τ}

# Q_τ for within vs cross cluster
Q_tau_within = np.mean(q_within ** TAU)
Q_tau_cross  = np.mean(q_cross ** TAU)
Q_tau_mixed  = np.mean(q_values ** TAU)

print(f"Q_τ within-cluster: {Q_tau_within:.2e}")
print(f"Q_τ cross-cluster:  {Q_tau_cross:.2e}")
print(f"Q_τ overall:        {Q_tau_mixed:.2e}")
print(f"Q_within / Q_cross: {Q_tau_within/Q_tau_cross:.1f}×")
print(f"(q_in/q_out)^τ:     {ratio**TAU:.1f}×  (τ amplifies community separation by ×{TAU})")
print(f"\nFraction of within-cluster edges at T_c:")
within_contribution = Q_tau_within * len(q_within)
cross_contribution  = Q_tau_cross  * len(q_cross)
frac = within_contribution / (within_contribution + cross_contribution)
print(f"  Within:  {frac*100:.1f}%  of expected edges")
print(f"  Cross:   {(1-frac)*100:.1f}%  of expected edges")
print(f"  (Actual G(t*): 39.4% within, 60.6% cross)")

# ─────────────────────────────────────────
# STEP 5: FULL p_c FORMULA SUMMARY
# ─────────────────────────────────────────
print("\n=== STEP 5: COMPLETE p_c FORMULA ===")

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│           SIW p_c FORMULA WITH THRESHOLD τ                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Exact:   p_c = E_{{ij}}[ 1 - F_Poisson(τ-1; T_c × q_{{ij}}) ]  │
│                                                                  │
│  Leading-order approximation (T × q << 1):                      │
│                                                                  │
│         p_c ≈ T_c^τ × Q_τ / τ!                                  │
│                                                                  │
│  where T_c = (τ! / ((N-1) × Q_τ))^{{1/τ}}                       │
│        Q_τ = E[q_{{ij}}^τ] = mean co-activation rate raised to τ  │
│        q_{{ij}} = (1/K) Σ_c p_i(c) × p_j(c)                     │
│                                                                  │
│  Combining: p_c ≈ 1/(N-1) × 1                                   │
│  → reduces to ER formula p_c ≈ 1/N  (independent of τ!)         │
│                                                                  │
│  BUT T_c depends on τ:                                           │
│         T_c ∝ (τ!)^{{1/τ}} / Q_τ^{{1/τ}}                        │
│                                                                  │
│  Numerical results:                                              │
│         T_c (formula)  = {T_c_formula:.2f}  sessions              │
│         T_c (empirical)= {EMPIRICAL_T_STAR}    sessions              │
│         p_c (formula)  = {p_c_formula:.6f}                        │
│         p_c (empirical)= {EMPIRICAL_PC:.6f}                        │
│         p_c (ER: 1/N)  = {1/N:.6f}                        │
└─────────────────────────────────────────────────────────────────┘

Key insight:
  p_c = T_c^τ × Q_τ / τ! = 1/(N-1) regardless of τ (to leading order).
  The threshold τ shifts WHEN (T_c) but NOT WHERE (p_c) the transition occurs.

  τ amplifies community structure in Q_τ:
    Q_τ_within / Q_τ_cross = {Q_tau_within/Q_tau_cross:.0f}×  vs  q_in/q_out = {ratio:.1f}×
  This means: hub nodes reach TAU co-activations much faster (dominate early graph).
  Community structure is more strongly enforced at τ > 1.
""")

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('OQ-1: Analytical p_c Formula with Co-activation Threshold τ',
             fontsize=13, fontweight='bold')

# ── Plot 1: p(T) exact vs approximation ──
ax = axes[0, 0]
ax.plot(T_range, p_T_exact,  'b-', lw=2, label='Exact (Poisson CDF)')
ax.plot(T_range, p_T_approx, 'r--', lw=1.5, label=f'Approx: T^τ Q_τ/τ!')
ax.axvline(EMPIRICAL_T_STAR, color='green', ls='--', lw=2,
           label=f't* = {EMPIRICAL_T_STAR} (empirical)')
ax.axvline(T_c_formula, color='orange', ls=':', lw=2,
           label=f'T_c = {T_c_formula:.1f} (formula)')
ax.axhline(EMPIRICAL_PC, color='grey', ls=':', lw=1, alpha=0.6)
ax.set_xlabel('Session T')
ax.set_ylabel('Mean edge density p(T)')
ax.set_title('Edge Density Growth: Exact vs Approximation')
ax.legend(fontsize=8)
ax.set_xlim(0, 80)
ax.set_ylim(0, EMPIRICAL_PC * 3)
ax.grid(True, alpha=0.3)

# ── Plot 2: q_ij distribution by pair type ──
ax = axes[0, 1]
ax.hist(q_within, bins=40, alpha=0.7, color='red',
        label=f'Within-cluster (mean={np.mean(q_within):.4f})', density=True)
ax.hist(q_cross,  bins=40, alpha=0.5, color='blue',
        label=f'Cross-cluster (mean={np.mean(q_cross):.4f})',  density=True)
ax.set_xlabel('q_ij (per-session co-activation probability)')
ax.set_ylabel('Density')
ax.set_title('q_ij Distribution by Pair Type')
ax.legend(fontsize=9)
ax.set_xlim(0, max(np.percentile(q_within, 99), np.percentile(q_cross, 99)))

# ── Plot 3: log p(T) vs log T — shows T^τ power law ──
ax = axes[1, 0]
log_T = np.log(T_range[:60])
log_p = np.log(np.maximum(p_T_exact[:60], 1e-10))
ax.plot(log_T, log_p, 'b-', lw=2, label='log p(T) exact')
# Fit slope
valid = np.isfinite(log_p) & (log_p > -20)
if valid.sum() > 5:
    slope, intercept, r, _, _ = stats.linregress(log_T[valid], log_p[valid])
    ax.plot(log_T, intercept + slope * log_T, 'r--', lw=1.5,
            label=f'Fit slope = {slope:.2f} (expect τ={TAU})')
ax.axvline(np.log(EMPIRICAL_T_STAR), color='green', ls='--', lw=1.5,
           label=f'log(t*) = {np.log(EMPIRICAL_T_STAR):.2f}')
ax.set_xlabel('log T')
ax.set_ylabel('log p(T)')
ax.set_title(f'Power-law: p(T) ∝ T^τ  (τ={TAU})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 4: p_c formula summary ──
ax = axes[1, 1]
ax.axis('off')
summary = f"""
ANALYTICAL RESULT

q_ij = (1/K) Σ_c p_i(c) × p_j(c)
     within-cluster:  {np.mean(q_within):.5f}
     cross-cluster:   {np.mean(q_cross):.5f}
     q_in/q_out:      {ratio:.2f}×

Q_τ = E[q_ij^τ] = {Q_tau:.2e}

T_c = (τ! / ((N-1) × Q_τ))^{{1/τ}}
    = ({int(special.factorial(TAU))} / ({N-1} × {Q_tau:.1e}))^{{1/{TAU}}}
    = {T_c_formula:.2f}  sessions
    (empirical t* = {EMPIRICAL_T_STAR})

p_c ≈ T_c^τ × Q_τ / τ! = 1/(N-1)
    = {p_c_formula:.6f}
    (empirical = {EMPIRICAL_PC:.6f})
    (ER = {1/N:.6f})

τ effect on T_c:
  τ=1: T_c ≈ {(1/((N-1)*np.mean(q_values)))**1:.1f} sessions
  τ=2: T_c ≈ {(2/((N-1)*np.mean(q_values**2)))**0.5:.1f} sessions
  τ=3: T_c ≈ {T_c_formula:.1f} sessions  ← current
  τ=5: T_c ≈ {(special.factorial(5)/((N-1)*np.mean(q_values**5)))**0.2:.1f} sessions

τ amplifies community sep: (q_in/q_out)^τ = {ratio**TAU:.0f}×
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Formula Summary', pad=10)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), 'figure_tau_formula.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Plot saved: {out}")
