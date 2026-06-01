"""
OQ-10: Baseline Definition (who defines "normal", and whose curiosity is "suspicious")

SIW flags users whose semantic graph is anomalously concentrated relative to a
"normal" baseline. But *who* defines normal? If the baseline is calibrated on a
majority of shallow, broad browsers, then any user with focused, deep curiosity —
a chemistry researcher, a security journalist, a historian of weapons — looks
anomalous. This experiment quantifies that bias and shows where it is fixable and
where it is not.

Five populations, each a "user" = a small batch of sessions, scored by their own
co-activation graph:

  majority          broad shallow browsing (defines the baseline)
  expert_off        deep, legitimate expertise in a NON-sensitive domain
  expert_on         deep, legitimate expertise in the SENSITIVE domain (researcher)
  attacker_overt    sensitive domain + boost + composition across many clusters
  attacker_stealth  mimics expert_on exactly (same generative process, harmful intent)

Two detection signals:
  S_density   max single-cluster edge density        (Level 1, raw concentration)
  S_span      #clusters spanned by the giant component (composition / assembly)

Findings the experiment establishes:
  1. A majority-calibrated DENSITY threshold flags BOTH expert groups at near-100%
     FPR — disparate impact against curious / expert users.
  2. The SPAN (composition) signal — what adversary.md proves is the detectable
     signature — separates the OVERT attacker from experts with high AUC at fair
     FPR. Using the right signal both detects and de-biases.
  3. The STEALTH attacker, who mimics a legitimate deep expert, is structurally
     IDENTICAL to expert_on on both signals (AUC ≈ 0.5). Separating them requires
     labelling the sensitive *domain* itself as suspicious — exactly the
     value-laden step where societal bias enters, and the I(G)=0 honest boundary
     of adversary.md. This residue is not solvable by graph structure alone.
"""

import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import siw_style as S

# ─────────────────────────────────────────
# SEMANTIC SPACE (identical to framework)
# ─────────────────────────────────────────
N_FEATURES      = 500
N_CLUSTERS      = 12
COACT_THRESHOLD = 3
SENSITIVE       = 3          # the cluster a researcher AND an attacker both use
T_USER          = 15         # sessions per user
N_USERS         = 80         # users per population
DETECT_Z        = 2.0

_rs = np.random.RandomState(42)
cluster_assignments = _rs.choice(N_CLUSTERS, N_FEATURES)
hub_set = set(_rs.choice(N_FEATURES, int(N_FEATURES * 0.05), replace=False))
cluster_nodes = defaultdict(list)
for node in range(N_FEATURES):
    cluster_nodes[cluster_assignments[node]].append(node)

P = np.full((N_FEATURES, N_CLUSTERS), 0.05)
for c in range(N_CLUSTERS):
    P[cluster_assignments == c, c] = 0.30
P[list(hub_set), :] *= 3.0
P = np.minimum(P, 1.0)


def session(cluster, boost, rng):
    p = np.minimum(P[:, cluster] * boost, 1.0)
    return np.where(rng.random(N_FEATURES) < p)[0]


def user_graph(sessions):
    coact = defaultdict(int)
    G = nx.Graph(); G.add_nodes_from(range(N_FEATURES))
    for a in sessions:
        a = list(a)
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                u, v = (a[i], a[j]) if a[i] < a[j] else (a[j], a[i])
                coact[(u, v)] += 1
                if coact[(u, v)] == COACT_THRESHOLD:
                    G.add_edge(u, v)
    return G


def signals(G):
    s_density = max(nx.density(G.subgraph(cluster_nodes[k])) for k in range(N_CLUSTERS))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    span = len(set(cluster_assignments[n] for n in comps[0])) if comps else 0
    return s_density, span


def make_user(kind, seed):
    rng = np.random.RandomState(seed)
    if kind == 'majority':
        sess = [session(rng.randint(N_CLUSTERS), 1.0, rng) for _ in range(T_USER)]
    elif kind == 'expert_off':
        sess = [session(8, 1.0, rng) for _ in range(T_USER)]
    elif kind == 'expert_on':
        sess = [session(SENSITIVE, 1.0, rng) for _ in range(T_USER)]
    elif kind == 'attacker_stealth':          # identical process to expert_on
        sess = [session(SENSITIVE, 1.0, rng) for _ in range(T_USER)]
    elif kind == 'attacker_overt':            # boost + composition across clusters
        sess = [session(SENSITIVE, 1.8, rng) for _ in range(T_USER - 4)] + \
               [session(c, 1.8, rng) for c in (SENSITIVE, 4, 5, 6)]
    return user_graph(sess)


GROUPS = ['majority', 'expert_off', 'expert_on', 'attacker_stealth', 'attacker_overt']
data = {g: {'density': [], 'span': []} for g in GROUPS}
for g in GROUPS:
    for s in range(N_USERS):
        d, sp = signals(make_user(g, 5000 + s))
        data[g]['density'].append(d); data[g]['span'].append(sp)
for g in GROUPS:
    for k in data[g]:
        data[g][k] = np.array(data[g][k])


def auc(pos, neg):
    """P(score_pos > score_neg) — Mann-Whitney AUC."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = sum((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
               for _ in [0])
    return wins / (len(pos) * len(neg))


# Majority-calibrated thresholds (95th percentile of the majority baseline)
thr_density = np.percentile(data['majority']['density'], 95)
thr_span    = np.percentile(data['majority']['span'], 95)
print(f"Majority-calibrated thresholds: density={thr_density:.3f}  span={thr_span:.1f}")

print("\n=== RATE FLAGGED under majority-calibrated thresholds ===")
print(f"{'group':18s} {'mean dens':>9s} {'FPR/TPR dens':>12s} {'mean span':>9s} {'FPR/TPR span':>12s}")
rates = {}
for g in GROUPS:
    fd = float((data[g]['density'] > thr_density).mean())
    fs = float((data[g]['span'] > thr_span).mean())
    rates[g] = (fd, fs)
    print(f"{g:18s} {data[g]['density'].mean():9.3f} {fd:12.0%} "
          f"{data[g]['span'].mean():9.1f} {fs:12.0%}")

print("\n=== SEPARABILITY (AUC vs benign expert_on) ===")
auc_overt_d  = auc(data['attacker_overt']['density'],   data['expert_on']['density'])
auc_overt_s  = auc(data['attacker_overt']['span'],      data['expert_on']['span'])
auc_steal_d  = auc(data['attacker_stealth']['density'], data['expert_on']['density'])
auc_steal_s  = auc(data['attacker_stealth']['span'],    data['expert_on']['span'])
print(f"  OVERT   attacker vs expert_on:  density AUC={auc_overt_d:.2f}  span AUC={auc_overt_s:.2f}")
print(f"  STEALTH attacker vs expert_on:  density AUC={auc_steal_d:.2f}  span AUC={auc_steal_s:.2f}")

fairness_gap_density = rates['expert_on'][0] - rates['majority'][0]
fairness_gap_span    = rates['expert_on'][1] - rates['majority'][1]
print(f"\nFairness gap (FPR_expert_on - FPR_majority):")
print(f"  density signal: {fairness_gap_density:+.0%}   span signal: {fairness_gap_span:+.0%}")


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.suptitle('OQ-10: Baseline Bias',
             fontsize=14, fontweight='bold', color=S.TEXT_C)

GCOL = {'majority': S.C_GREY, 'expert_off': S.C_BLUE, 'expert_on': S.C_TEAL,
        'attacker_stealth': S.C_ORANGE, 'attacker_overt': S.C_RED}
GLAB = {'majority': 'majority\n(shallow)', 'expert_off': 'expert\n(off-domain)',
        'expert_on': 'expert\n(sensitive)', 'attacker_stealth': 'attacker\n(stealth)',
        'attacker_overt': 'attacker\n(overt)'}

# ── Panel A: density distributions ──
ax = axes[0, 0]; S.style_ax(ax)
for g in GROUPS:
    ax.scatter(np.full(N_USERS, GROUPS.index(g)) + np.random.uniform(-0.18, 0.18, N_USERS),
               data[g]['density'], s=12, color=GCOL[g], alpha=0.5, edgecolor='none')
ax.axhline(thr_density, color=S.SUBTLE, ls='--', lw=1.4)
ax.text(len(GROUPS) - 0.5, thr_density * 1.25, 'majority-calibrated threshold',
        ha='right', fontsize=8.5, color=S.SUBTLE)
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([GLAB[g] for g in GROUPS], fontsize=8)
ax.set_ylabel('S_density  (max cluster density)')
ax.set_title('Raw concentration: experts sit far above the\n'
             'majority baseline — and stealth hides among them')
S.panel_label(ax, 'a')

# ── Panel B: FPR/TPR by group, both signals ──
ax = axes[0, 1]; S.style_ax(ax)
x = np.arange(len(GROUPS)); w = 0.38
b1 = ax.bar(x - w/2, [rates[g][0] for g in GROUPS], w, color=S.C_RED, alpha=0.85,
            label='S_density')
b2 = ax.bar(x + w/2, [rates[g][1] for g in GROUPS], w, color=S.C_PURPLE, alpha=0.85,
            label='S_span (composition)')
ax.set_xticks(x); ax.set_xticklabels([GLAB[g] for g in GROUPS], fontsize=8)
ax.set_ylabel('fraction flagged')
ax.set_ylim(0, 1.08)
ax.set_title('Density flags both expert groups (bias);\n'
             'composition spares experts, keeps the overt attacker')
ax.legend(fontsize=9, loc='center left')
S.panel_label(ax, 'b')

# ── Panel C: separability AUCs ──
ax = axes[1, 0]; S.style_ax(ax)
cats = ['overt\ndensity', 'overt\nspan', 'stealth\ndensity', 'stealth\nspan']
vals = [auc_overt_d, auc_overt_s, auc_steal_d, auc_steal_s]
cols = [S.C_RED, S.C_PURPLE, S.C_RED, S.C_PURPLE]
bars = ax.bar(cats, vals, color=cols, alpha=0.85, edgecolor='none')
ax.axhline(0.5, color=S.SUBTLE, ls='--', lw=1.4)
ax.text(3.4, 0.52, 'chance (indistinguishable)', ha='right', fontsize=8.5, color=S.SUBTLE)
S.bar_labels(ax, bars, fmt='{:.2f}')
ax.set_ylim(0, 1.08)
ax.set_ylabel('AUC  vs  benign expert (sensitive)')
ax.set_title('Overt attacker is separable (esp. by composition);\n'
             'stealth attacker is at chance — structurally an expert')
S.panel_label(ax, 'c')

# ── Panel D: the irreducible residue ──
ax = axes[1, 1]; S.style_ax(ax); ax.axis('off')
ax.set_title('Where the bias lives', fontsize=11, pad=8)
grid = [
    ['',                'OVERT',                  'STEALTH'],
    ['fair detection?', f'YES (AUC {auc_overt_s:.2f})', f'NO (AUC {auc_steal_s:.2f})'],
    ['structural twin', 'none',                   'benign expert'],
    ['separation needs','composition',            'a domain value-label'],
]
col_x = [0.02, 0.40, 0.70]
y0, dy = 0.82, 0.20
for r, row in enumerate(grid):
    for c, cell in enumerate(row):
        col = S.TEXT_C
        weight = 'bold' if r == 0 or c == 0 else 'normal'
        if r == 1 and c == 2: col = S.C_RED
        if r == 1 and c == 1: col = S.C_GREEN
        ax.text(col_x[c], y0 - r * dy, cell, transform=ax.transAxes,
                fontsize=8.5, color=col, va='top', ha='left', fontweight=weight)
ax.text(0.02, 0.06,
        'Residue: a stealth attacker who mimics legitimate deep curiosity is\n'
        'structurally a benign expert (I(G)=0). Flagging them means flagging the\n'
        'domain — encoding "whose curiosity is suspicious". Not solvable by\n'
        'structure alone; this is the governance boundary, not an engineering gap.',
        transform=ax.transAxes, fontsize=8.5, color=S.SUBTLE, va='bottom', ha='left')
S.panel_label(ax, 'd')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), 'figure_oq10_baseline_bias.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out}")

# ─────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────
print("\n" + "=" * 64)
print("OQ-10 VERDICT: BASELINE DEFINITION & BIAS")
print("=" * 64)
print(f"""
"Normal" defined by a shallow-majority baseline is not neutral — it encodes a
bias against focused curiosity. Quantified over {N_USERS} users/group:

  Disparate impact: a majority-calibrated DENSITY threshold flags
    expert (off-domain)  at {rates['expert_off'][0]:.0%}   and
    expert (sensitive)   at {rates['expert_on'][0]:.0%}   FPR,
  while the majority sits at {rates['majority'][0]:.0%}. Curiosity ≈ guilt under this baseline.

  Partial fix (use the RIGHT signal): the composition signal S_span — the
  detectable signature in adversary.md — separates the OVERT attacker from
  benign experts (AUC {auc_overt_s:.2f}) at fair FPR (expert_on flagged {rates['expert_on'][1]:.0%} vs
  {rates['majority'][1]:.0%} majority). Detecting on assembly, not on depth, de-biases.

  Irreducible residue: the STEALTH attacker, who mimics a legitimate deep expert,
  is at chance vs expert_on on BOTH signals (density AUC {auc_steal_d:.2f}, span AUC
  {auc_steal_s:.2f}). No graph-structural test separates them. Doing so requires
  labelling the sensitive DOMAIN as suspicious — the value-laden choice where
  societal bias enters, and the I(G)=0 honest boundary of adversary.md.

Answer to "who defines normal": the baseline is a governance artifact, not a
statistic. SIW can be made fair against the detectable (composition) class, but
the curiosity-vs-malice residue inside a single domain is a policy decision that
graph structure cannot — and should not pretend to — resolve.

  OQ-10: ADDRESSED (boundary characterised; not "solved" — it is partly a value
  judgment by construction).
""")
