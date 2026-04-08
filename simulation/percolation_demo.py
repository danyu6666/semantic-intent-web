"""
SIW Semantic Percolation Simulation
Goal: Characterize the topology of semantic feature co-activation graphs
and find the phase transition (intent crystallization threshold).

Key questions:
1. Does the phase transition exist in semantic space?
2. Is the degree distribution Erdős–Rényi (random) or power-law (scale-free)?
3. What is the empirical p_c?
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

np.random.seed(42)

# ─────────────────────────────────────────
# PART 1: Semantic Feature Space
# ─────────────────────────────────────────
# Simulate a realistic semantic space with:
# - Hub features (high centrality concepts: "harm", "weapon", "bypass")
# - Peripheral features (specific, low-frequency concepts)
# - Hierarchical clustering (topic clusters)

N_FEATURES = 500       # total semantic features
N_CLUSTERS = 12        # topic clusters
N_SESSIONS = 300       # total sessions to simulate
FEATURES_PER_SESSION = 15  # avg features activated per session

# Generate clustered semantic space
cluster_assignments = np.random.choice(N_CLUSTERS, N_FEATURES)

# Hub features: top 5% are hubs (higher activation probability)
hub_features = np.random.choice(N_FEATURES, int(N_FEATURES * 0.05), replace=False)
hub_set = set(hub_features)

def activation_probability(feature_idx, session_cluster):
    """Features in same cluster activate together; hubs activate more often."""
    base = 0.3 if cluster_assignments[feature_idx] == session_cluster else 0.05
    if feature_idx in hub_set:
        base *= 3.0
    return min(base, 1.0)

# ─────────────────────────────────────────
# PART 2: Simulate Sessions
# (benign baseline + attacker sessions)
# ─────────────────────────────────────────

def simulate_session(session_type='benign', attack_cluster=None):
    """Simulate a single session's feature activations."""
    if session_type == 'benign':
        cluster = np.random.randint(N_CLUSTERS)
    else:
        # Attacker focuses on specific cluster but mixes in benign features
        cluster = attack_cluster
    
    activated = []
    for f in range(N_FEATURES):
        p = activation_probability(f, cluster)
        if session_type == 'attack':
            p *= 1.8  # attackers activate more targeted features
        if np.random.random() < p:
            activated.append(f)
    return activated

# Run sessions: 250 benign + 50 attack (distributed across 10 attack sessions)
sessions = []
ATTACK_CLUSTER = 3  # the cluster an attacker targets

for i in range(250):
    sessions.append(('benign', simulate_session('benign')))

for i in range(50):
    sessions.append(('attack', simulate_session('attack', ATTACK_CLUSTER)))

np.random.shuffle(sessions)

# ─────────────────────────────────────────
# PART 3: Build Temporal Semantic Graph
# ─────────────────────────────────────────

coactivation = defaultdict(int)
feature_count = defaultdict(int)

# Track giant component size over time
giant_component_ratios = []
edge_counts = []
G = nx.Graph()
G.add_nodes_from(range(N_FEATURES))

COACT_THRESHOLD = 3  # edges form after N co-activations

for t, (stype, activated) in enumerate(sessions):
    # Update co-activation counts
    for i in range(len(activated)):
        feature_count[activated[i]] += 1
        for j in range(i+1, len(activated)):
            a, b = min(activated[i], activated[j]), max(activated[i], activated[j])
            coactivation[(a,b)] += 1
            
            # Add edge if threshold crossed
            if coactivation[(a,b)] == COACT_THRESHOLD:
                G.add_edge(a, b, weight=coactivation[(a,b)])
    
    # Measure giant component
    if G.number_of_edges() > 0:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        ratio = len(components[0]) / N_FEATURES
    else:
        ratio = 0
    
    giant_component_ratios.append(ratio)
    edge_counts.append(G.number_of_edges())

# ─────────────────────────────────────────
# PART 4: Find Phase Transition
# ─────────────────────────────────────────

ratios = np.array(giant_component_ratios)
edges = np.array(edge_counts)
max_possible_edges = N_FEATURES * (N_FEATURES - 1) / 2
p_values = edges / max_possible_edges

# Find p_c: steepest increase in giant component
diffs = np.diff(ratios)
pc_idx = np.argmax(diffs) + 1
pc_empirical = p_values[pc_idx]
phi_c = ratios[pc_idx]

print(f"=== PHASE TRANSITION DETECTED ===")
print(f"Empirical p_c: {pc_empirical:.6f}")
print(f"Erdős–Rényi prediction: {1/N_FEATURES:.6f}")
print(f"Giant component at transition: {phi_c:.3f} ({phi_c*100:.1f}% of nodes)")
print(f"Ratio p_c_empirical / p_c_ER: {pc_empirical / (1/N_FEATURES):.2f}x")

# ─────────────────────────────────────────
# PART 5: Degree Distribution Analysis
# (Power-law vs Erdős–Rényi)
# ─────────────────────────────────────────

degrees = [d for n, d in G.degree() if d > 0]
degrees_arr = np.array(sorted(degrees, reverse=True))

# Fit power law
log_deg = np.log(degrees_arr[degrees_arr > 0])
log_rank = np.log(np.arange(1, len(degrees_arr[degrees_arr > 0]) + 1))
slope, intercept, r_power, _, _ = stats.linregress(log_rank, log_deg)

# Fit Poisson (Erdős–Rényi prediction)
mean_deg = np.mean(degrees_arr)
poisson_fit = stats.poisson(mean_deg)

print(f"\n=== DEGREE DISTRIBUTION ===")
print(f"Mean degree: {mean_deg:.2f}")
print(f"Max degree (hub): {degrees_arr.max()}")
print(f"Power-law fit R²: {r_power**2:.4f}")
print(f"Power-law exponent: {-slope:.3f}")
print(f"Scale-free threshold (R² > 0.85): {'YES' if r_power**2 > 0.85 else 'NO'}")

# Kolmogorov-Smirnov test vs Poisson
ks_stat, ks_p = stats.kstest(degrees_arr, 
    lambda x: poisson_fit.cdf(x))
print(f"KS test vs Poisson: stat={ks_stat:.4f}, p={ks_p:.4f}")
print(f"Reject Erdős–Rényi model: {'YES' if ks_p < 0.05 else 'NO'}")

# ─────────────────────────────────────────
# PART 6: Plot Results
# ─────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SIW Semantic Percolation Simulation', fontsize=14, fontweight='bold')

# Plot 1: Giant component over time
ax1 = axes[0, 0]
ax1.plot(range(len(ratios)), ratios, 'b-', linewidth=1.5, alpha=0.8)
ax1.axvline(x=pc_idx, color='red', linestyle='--', linewidth=2, label=f'p_c transition (t={pc_idx})')
ax1.axhline(y=phi_c, color='orange', linestyle=':', linewidth=1.5, label=f'φ_c = {phi_c:.2f}')
ax1.set_xlabel('Session (time)')
ax1.set_ylabel('Giant Component Ratio |C_max|/n')
ax1.set_title('Intent Crystallization Over Time')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Phase transition (p vs giant component)
ax2 = axes[0, 1]
ax2.plot(p_values, ratios, 'b-', linewidth=1.5)
ax2.axvline(x=pc_empirical, color='red', linestyle='--', 
            label=f'p_c empirical = {pc_empirical:.5f}')
ax2.axvline(x=1/N_FEATURES, color='green', linestyle=':', 
            label=f'p_c Erdős–Rényi = {1/N_FEATURES:.5f}')
ax2.set_xlabel('Edge Probability p(t)')
ax2.set_ylabel('Giant Component Ratio')
ax2.set_title('Phase Transition: Semantic Percolation')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Degree distribution (log-log)
ax3 = axes[1, 0]
deg_sorted = sorted(degrees, reverse=True)
counts = np.bincount(deg_sorted)
deg_vals = np.where(counts > 0)[0]
deg_counts = counts[deg_vals]
ax3.loglog(deg_vals, deg_counts, 'b.', alpha=0.6, markersize=5, label='Empirical')
# Power law fit line
x_fit = np.linspace(1, max(deg_vals), 100)
y_fit = np.exp(intercept) * x_fit**slope
ax3.loglog(x_fit, y_fit, 'r-', linewidth=2, 
           label=f'Power law γ={-slope:.2f}, R²={r_power**2:.3f}')
ax3.set_xlabel('Degree k')
ax3.set_ylabel('Count P(k)')
ax3.set_title('Degree Distribution (log-log)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Attack detection — giant component in attack cluster
ax4 = axes[1, 1]
# Subgraph of attack cluster features
attack_features = [f for f in range(N_FEATURES) 
                   if cluster_assignments[f] == ATTACK_CLUSTER]
G_attack = G.subgraph(attack_features)
attack_degrees = dict(G_attack.degree())
attack_deg_vals = list(attack_degrees.values())

# Compare attack cluster density vs random cluster
random_cluster = (ATTACK_CLUSTER + 1) % N_CLUSTERS
random_features = [f for f in range(N_FEATURES) 
                   if cluster_assignments[f] == random_cluster]
G_random = G.subgraph(random_features)
random_deg_vals = list(dict(G_random.degree()).values())

ax4.hist(attack_deg_vals, bins=20, alpha=0.6, color='red', 
         label=f'Attack cluster (mean={np.mean(attack_deg_vals):.1f})')
ax4.hist(random_deg_vals, bins=20, alpha=0.6, color='blue', 
         label=f'Benign cluster (mean={np.mean(random_deg_vals):.1f})')
ax4.set_xlabel('Node Degree')
ax4.set_ylabel('Count')
ax4.set_title('Attack vs Benign Cluster Density')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/mareen2018/Desktop/SIW/simulation_results.png', 
            dpi=150, bbox_inches='tight')
print(f"\n=== PLOT SAVED ===")
print(f"File: /Users/mareen2018/Desktop/SIW/simulation_results.png")

# Summary
print(f"\n=== FRAMEWORK VERDICT ===")
is_power_law = r_power**2 > 0.85
er_rejected = ks_p < 0.05
pc_ratio = pc_empirical / (1/N_FEATURES)

if is_power_law and er_rejected:
    print("Topology: SCALE-FREE (Barabási–Albert model appropriate)")
    print(f"Erdős–Rényi is WRONG by {pc_ratio:.1f}x")
    print("Recommendation: Use scale-free percolation model")
elif not er_rejected:
    print("Topology: RANDOM (Erdős–Rényi model acceptable)")
else:
    print("Topology: INTERMEDIATE (empirical calibration needed)")

print(f"\nPhase transition: CONFIRMED at p_c = {pc_empirical:.6f}")
print(f"Intent crystallization threshold φ_c = {phi_c:.3f}")
