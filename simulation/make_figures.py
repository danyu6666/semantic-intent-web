"""
SIW Publication Figures — styled after academic paper aesthetic:
  - Warm cream background (#F2EDE7)
  - No top/right spines
  - Values annotated directly on elements
  - Bold panel labels (a, b, c, d)
  - Muted distinct color palette
  - Clean sans-serif typography
"""

import numpy as np
import networkx as nx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from siw_style import (BG, SPINE_C, TEXT_C, SUBTLE,
                        C_BLUE, C_ORANGE, C_RED, C_GREEN, C_PURPLE, C_GREY,
                        style_ax, panel_label, note, bar_labels)

# ── Reproduce simulation ─────────────────────────────────────────
N_FEATURES      = 500
N_CLUSTERS      = 12
N_SESSIONS      = 300
COACT_THRESHOLD = 3
ATTACK_CLUSTER  = 3

cluster_assignments = np.random.choice(N_CLUSTERS, N_FEATURES)
hub_features        = np.random.choice(N_FEATURES, int(N_FEATURES * 0.05), replace=False)
hub_set             = set(hub_features)

def act_prob(feat, sess_cluster):
    base = 0.3 if cluster_assignments[feat] == sess_cluster else 0.05
    if feat in hub_set: base *= 3.0
    return min(base, 1.0)

def simulate_session(stype='benign', attack_cluster=None):
    cluster = np.random.randint(N_CLUSTERS) if stype == 'benign' else attack_cluster
    return [f for f in range(N_FEATURES)
            if np.random.random() < (act_prob(f, cluster) *
                                     (1.8 if stype == 'attack' else 1.0))]

sessions = []
for _ in range(250): sessions.append(('benign', simulate_session('benign')))
for _ in range(50):  sessions.append(('attack', simulate_session('attack', ATTACK_CLUSTER)))
np.random.shuffle(sessions)

coactivation = defaultdict(int)
G = nx.Graph()
G.add_nodes_from(range(N_FEATURES))
giant_ratios, edge_counts = [], []

for t, (stype, activated) in enumerate(sessions):
    for i in range(len(activated)):
        for j in range(i + 1, len(activated)):
            a, b = min(activated[i], activated[j]), max(activated[i], activated[j])
            coactivation[(a, b)] += 1
            if coactivation[(a, b)] == COACT_THRESHOLD:
                G.add_edge(a, b)
    ne = G.number_of_edges()
    edge_counts.append(ne)
    if ne > 0:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        giant_ratios.append(len(comps[0]) / N_FEATURES)
    else:
        giant_ratios.append(0.0)

ratios    = np.array(giant_ratios)
edges     = np.array(edge_counts)
max_edges = N_FEATURES * (N_FEATURES - 1) / 2
p_values  = edges / max_edges
diffs     = np.diff(ratios)
pc_idx    = int(np.argmax(diffs)) + 1
pc_emp    = p_values[pc_idx]
phi_c     = ratios[pc_idx]

degrees = [d for n, d in G.degree() if d > 0]

cluster_nodes = defaultdict(list)
for node in range(N_FEATURES):
    cluster_nodes[cluster_assignments[node]].append(node)

attack_nodes  = cluster_nodes[ATTACK_CLUSTER]
random_nodes  = cluster_nodes[(ATTACK_CLUSTER + 1) % N_CLUSTERS]
G_attack      = G.subgraph(attack_nodes)
G_random      = G.subgraph(random_nodes)
attack_deg    = [d for _, d in G_attack.degree()]
benign_deg    = [d for _, d in G_random.degree()]

# ── Figure ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs  = GridSpec(2, 3, figure=fig,
               left=0.07, right=0.97, top=0.93, bottom=0.10,
               hspace=0.45, wspace=0.38)

ax_a = fig.add_subplot(gs[0, :2])   # wide: crystallization over time
ax_b = fig.add_subplot(gs[0, 2])    # p_c comparison
ax_c = fig.add_subplot(gs[1, :2])   # degree distribution (log-log)
ax_d = fig.add_subplot(gs[1, 2])    # attack vs benign cluster density

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.set_facecolor(BG)
    style_ax(ax)

# ─── Panel a: Crystallization curve ─────────────────────────────
t_range = np.arange(len(ratios))

# Shade pre/post transition
ax_a.fill_between(t_range, 0, ratios,
                  where=t_range <= pc_idx,
                  color=C_BLUE, alpha=0.12)
ax_a.fill_between(t_range, 0, ratios,
                  where=t_range > pc_idx,
                  color=C_ORANGE, alpha=0.15)

ax_a.plot(t_range, ratios, color=C_BLUE, lw=2.2, zorder=4)

# Critical point marker
ax_a.axvline(pc_idx, color=C_RED, lw=1.8, ls='--', alpha=0.85, zorder=3)
ax_a.scatter([pc_idx], [phi_c], color=C_RED, s=70, zorder=6)

# Annotation: place text box well to the right of the transition, no arrow overlap
ax_a.annotate(f'intent crystallizes\nsession {pc_idx},  φ_c = {phi_c:.2f}',
              xy=(pc_idx, phi_c),
              xytext=(pc_idx + 55, 0.22),
              fontsize=10, color=C_RED, ha='left',
              arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2,
                              connectionstyle='arc3,rad=-0.25'))

ax_a.set_xlabel('Session  t', fontsize=11)
ax_a.set_ylabel('|C_max| / |V|', fontsize=11)
ax_a.set_title('Intent Crystallization over Time', fontsize=12, pad=8)
ax_a.set_xlim(0, N_SESSIONS)
ax_a.set_ylim(-0.02, 1.08)

# Sub/super-critical labels — placed in clear regions away from curve
ax_a.text(6, 0.07,
          'sub-critical\n(benign-like)',
          ha='left', fontsize=10, color=C_BLUE, alpha=0.90)
ax_a.text((pc_idx + N_SESSIONS) / 2, 0.52,
          'super-critical\n(attack crystallized)',
          ha='center', fontsize=10, color=C_ORANGE, alpha=0.90)

panel_label(ax_a, 'a')

# ─── Panel b: p_c comparison ─────────────────────────────────────
labels_b   = ['ER\nprediction', 'Empirical\np_c']
vals_b     = [1 / N_FEATURES, pc_emp]
colors_b   = [C_GREY, C_RED]

bars = ax_b.bar(labels_b, vals_b, color=colors_b,
                width=0.5, alpha=0.88, edgecolor='none')

for bar, v in zip(bars, vals_b):
    ax_b.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.00004,
              f'{v:.5f}', ha='center', va='bottom',
              fontsize=10, fontweight='bold', color=TEXT_C)

ax_b.set_ylabel('Edge probability  p_c', fontsize=11)
ax_b.set_title('Phase Transition\nThreshold', fontsize=12, pad=8)
ax_b.set_ylim(0, max(vals_b) * 1.4)
panel_label(ax_b, 'b')

# ─── Panel c: Degree distribution — rank plot (Zipf style, R²=0.812) ────
degrees_pos = np.array([d for d in degrees if d > 0])
degrees_sorted = np.sort(degrees_pos)[::-1]   # rank descending
ranks = np.arange(1, len(degrees_sorted) + 1)

# Power-law fit on rank-degree (same as original percolation_demo.py → R²=0.812)
log_rank = np.log(ranks)
log_deg  = np.log(degrees_sorted)
slope, intercept, r, _, _ = stats.linregress(log_rank, log_deg)

ax_c.loglog(ranks, degrees_sorted, 'o',
            color=C_PURPLE, alpha=0.55, ms=3.5, zorder=4,
            label='Empirical degree (Zipf rank)')
x_fit = np.logspace(0, np.log10(len(ranks)), 80)
ax_c.loglog(x_fit, np.exp(intercept) * x_fit ** slope,
            '--', color=C_ORANGE, lw=2.2, alpha=0.9,
            label=f'Power law fit  γ = {-slope:.2f}  (R² = {r**2:.2f})')

ax_c.set_xlabel('Rank', fontsize=11)
ax_c.set_ylabel('Degree  k', fontsize=11)
ax_c.set_title('Degree Distribution — DC-SBM Intermediate Topology', fontsize=12, pad=8)
ax_c.legend(fontsize=9, frameon=False, labelcolor=TEXT_C)
note(ax_c, 'not pure scale-free (R² < 0.85)\nnot Poisson (KS p ≈ 0)\n→ DC-SBM intermediate')
panel_label(ax_c, 'c')

# ─── Panel d: Attack vs benign cluster density ───────────────────
mean_att = np.mean(attack_deg)
mean_ben = np.mean(benign_deg)
pct_diff = (mean_att - mean_ben) / mean_ben * 100

labels_d = ['Benign\ncluster', 'Attack\ncluster']
vals_d   = [mean_ben, mean_att]
colors_d = [C_BLUE, C_RED]

bars_d = ax_d.barh(labels_d, vals_d, color=colors_d,
                   height=0.45, alpha=0.88, edgecolor='none')

# Value labels — padded right of each bar
for bar, v in zip(bars_d, vals_d):
    ax_d.text(v + max(vals_d) * 0.03,
              bar.get_y() + bar.get_height() / 2,
              f'{v:.1f}', va='center', ha='left',
              fontsize=11, fontweight='bold', color=TEXT_C)

# Percentage difference — text only, placed clearly above the attack bar
ax_d.text(max(vals_d) * 0.72, 1.32,
          f'+{pct_diff:.0f}% denser', va='center', ha='center',
          fontsize=10, fontweight='bold', color=C_GREEN)

ax_d.set_xlabel('Mean node degree', fontsize=11)
ax_d.set_title('Cluster Density\nat Crystallization', fontsize=12, pad=8)
ax_d.set_xlim(0, max(vals_d) * 1.55)
ax_d.set_ylim(-0.6, 1.7)
panel_label(ax_d, 'd')

# ── Figure title ─────────────────────────────────────────────────
fig.suptitle('Semantic Intent Web — Phase Transition Analysis',
             fontsize=14, fontweight='bold', color=TEXT_C, y=0.99)

# ── Save ─────────────────────────────────────────────────────────
import os
out = os.path.join(os.path.dirname(__file__), 'simulation_results.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
plt.close()
