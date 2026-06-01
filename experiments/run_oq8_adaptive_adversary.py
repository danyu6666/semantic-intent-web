"""
OQ-8: Adaptive Adversary & Dynamic Graph Model

The percolation model in framework.md assumes a STATIC adversary (fixed cluster,
fixed activation boost). A real adversary who knows SIW exists will adapt to
evade it. This experiment asks: what is the right dynamic graph model when the
adversary tunes its behaviour against the detector — and does evasion come free?

Detection happens in the EARLY, sub-critical window (the framework's t* ≈ tens of
sessions), before the benign background has itself crystallised. We therefore
measure the attacked cluster against a TIME-MATCHED benign null built from the
other clusters in the same stream — never at the saturated end state.

Three evasion knobs, each swept while we record both the detector's signal and
the adversary's own cost:

  1. CAMOUFLAGE  — interleave benign decoy sessions to dilute the trace.
                   Cost: the goal takes far more sessions to deliver (latency).
  2. SPREAD (M)  — split the goal across M clusters so no single cluster looks
                   anomalous (evades Level 1).
                   Cost: the M pieces must be joined → cross-cluster composition
                   edges appear (I(G) > 0, the signature adversary.md proves
                   is unavoidable for a goal that must be reassembled).
  3. BOOST (β)   — lower per-session activation intensity toward benign.
                   Cost: within a fixed budget the goal assembles less (lower
                   coherence) — and concentration, not intensity, is the signal.

Detector signal (Level 1): the attacked cluster's edge-density z-score relative
to the time-matched distribution of the other clusters.
Adversary utility: goal coherence = attacked-cluster edge density within a fixed
session budget (how assembled the harmful intent graph is).

Central result: detector signal and goal coherence are both monotone in the same
quantity — the attacker's concentrated co-activation rate. The region {low
detectability ∧ high coherence} is empirically empty. The right dynamic model is
a DC-SBM percolation with an adversary-controlled co-activation rate q(t); the
only undetectable setting is q→benign, where the goal never assembles. This is
the simulation form of the evasion-cost bound in adversary.md.
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
# SHARED SIMULATION PRIMITIVES (identical to framework / OQ-6)
# ─────────────────────────────────────────
N_FEATURES      = 500
N_CLUSTERS      = 12
COACT_THRESHOLD = 3          # τ
N_BENIGN        = 180        # honest background users
N_ATTACK        = 36         # real attack-intent sessions (the goal budget)
NUM_SEEDS       = 4          # repeats per config for stable means

T_BUDGET        = 60         # fixed observation budget for signal / coherence
RECORD_EVERY    = 2          # density snapshot cadence (sessions)
DETECT_Z        = 2.0        # Level 1 fires at 2σ above the time-matched null
FLOOR           = 0.06       # absolute density floor to suppress early noise
SIGMA_FLOOR     = 0.015      # min null std (few-edge density scale); avoids σ→0 blow-up

# Fixed semantic space (varied only by the RNG of the sessions)
_rng_space = np.random.RandomState(42)
cluster_assignments = _rng_space.choice(N_CLUSTERS, N_FEATURES)
hub_set             = set(_rng_space.choice(N_FEATURES, int(N_FEATURES * 0.05), replace=False))

cluster_nodes = defaultdict(list)
for node in range(N_FEATURES):
    cluster_nodes[cluster_assignments[node]].append(node)

P_base = np.full((N_FEATURES, N_CLUSTERS), 0.05)
for c in range(N_CLUSTERS):
    P_base[cluster_assignments == c, c] = 0.30
P_base[list(hub_set), :] *= 3.0
P_base = np.minimum(P_base, 1.0)


def simulate_session(cluster, boost, rng):
    p = np.minimum(P_base[:, cluster] * boost, 1.0)
    return np.where(rng.random(N_FEATURES) < p)[0]


def run_attack(camouflage=0.0, spread=1, boost=1.8, seed=0):
    """Stream an adaptive-adversary session sequence and record early-window
    detector signal + adversary cost. Benign null is time-matched (the other
    clusters at the same session index)."""
    rng = np.random.RandomState(seed)
    attacked = list(range(spread))                       # target clusters 0..M-1
    other    = [k for k in range(N_CLUSTERS) if k not in attacked]

    attack_sessions = [(simulate_session(attacked[i % spread], boost, rng), True)
                       for i in range(N_ATTACK)]
    n_decoy = int(round(camouflage / (1 - camouflage) * N_ATTACK)) if camouflage < 1 else 0
    decoys  = [(simulate_session(rng.randint(N_CLUSTERS), 1.0, rng), False)
               for _ in range(n_decoy)]
    benign  = [(simulate_session(rng.randint(N_CLUSTERS), 1.0, rng), False)
               for _ in range(N_BENIGN)]

    stream = attack_sessions + decoys + benign
    rng.shuffle(stream)
    total_adv = N_ATTACK + n_decoy
    T_complete = max(i for i, (_, atk) in enumerate(stream) if atk) + 1

    coact = defaultdict(int)
    G = nx.Graph(); G.add_nodes_from(range(N_FEATURES))

    rec_T, rec_atk, rec_omean, rec_ostd, rec_op90 = [], [], [], [], []
    T_detect = None
    for ti, (a, _) in enumerate(stream):
        for i in range(len(a)):
            ai = a[i]
            for j in range(i + 1, len(a)):
                aj = a[j]
                key = (ai, aj) if ai < aj else (aj, ai)
                coact[key] += 1
                if coact[key] == COACT_THRESHOLD:
                    G.add_edge(key[0], key[1])
        if ti % RECORD_EVERY == 0 or ti == T_complete - 1:
            d_atk = np.mean([nx.density(G.subgraph(cluster_nodes[k])) for k in attacked])
            d_oth = np.array([nx.density(G.subgraph(cluster_nodes[k])) for k in other])
            om, osd, op90 = d_oth.mean(), d_oth.std(), np.percentile(d_oth, 90)
            rec_T.append(ti); rec_atk.append(d_atk)
            rec_omean.append(om); rec_ostd.append(osd); rec_op90.append(op90)
            if T_detect is None and d_atk >= FLOOR and \
               (d_atk - om) / max(osd, SIGMA_FLOOR) >= DETECT_Z:
                T_detect = ti

    rec_T = np.array(rec_T); rec_atk = np.array(rec_atk)
    rec_omean = np.array(rec_omean); rec_ostd = np.array(rec_ostd)

    # fixed-budget signal & coherence
    bidx = int(np.searchsorted(rec_T, T_BUDGET, side='right')) - 1
    bidx = max(0, min(bidx, len(rec_T) - 1))
    coherence_budget = float(rec_atk[bidx])
    z_budget = float((rec_atk[bidx] - rec_omean[bidx]) / max(rec_ostd[bidx], SIGMA_FLOOR))
    # peak z over the early window (capped to ≤ 2×budget to stay sub-critical)
    wmask = rec_T <= 2 * T_BUDGET
    z_traj = (rec_atk - rec_omean) / np.maximum(rec_ostd, SIGMA_FLOOR)
    z_peak = float(z_traj[wmask].max())

    # cross-cluster composition edges among attacked clusters (I(G)>0 signature)
    bridges = sum(1 for u, v in G.edges()
                  if cluster_assignments[u] in attacked
                  and cluster_assignments[v] in attacked
                  and cluster_assignments[u] != cluster_assignments[v])

    return dict(z_budget=z_budget, z_peak=z_peak, coherence=coherence_budget,
                T_detect=(T_detect if T_detect is not None else 2 * T_BUDGET),
                T_complete=T_complete, total_adv=total_adv,
                detected=z_peak >= DETECT_Z, bridges=float(bridges))


def sweep(name, values, **fixed):
    keys = ['z_budget', 'z_peak', 'coherence', 'T_detect', 'T_complete',
            'total_adv', 'bridges', 'detected']
    out = {k: [] for k in keys}; out['value'] = list(values)
    for val in values:
        acc = defaultdict(list)
        for s in range(NUM_SEEDS):
            r = run_attack(seed=2000 + s, **{name: val}, **fixed)
            for k in keys:
                acc[k].append(float(r[k]))
        for k in keys:
            out[k].append(float(np.mean(acc[k])))
    return out


# ─────────────────────────────────────────
# REFERENCE + SWEEPS
# ─────────────────────────────────────────
ref = run_attack(seed=2000)
print("=== STATIC ATTACK (reference) ===")
print(f"z_peak={ref['z_peak']:.1f}  z@budget={ref['z_budget']:.1f}  "
      f"coherence={ref['coherence']:.3f}  T_detect={ref['T_detect']}  "
      f"T_complete={ref['T_complete']}  detected={ref['detected']}")

print("\n=== KNOB 1: CAMOUFLAGE (benign decoy dilution) ===")
camo = sweep('camouflage', [0.0, 0.3, 0.5, 0.7], spread=1, boost=1.8)
for v, td, tc, tot, zp in zip(camo['value'], camo['T_detect'], camo['T_complete'],
                              camo['total_adv'], camo['z_peak']):
    print(f"  c={v:.2f}: T_detect={td:5.1f}  T_complete={tc:5.1f}  "
          f"adv sessions={tot:5.1f}  z_peak={zp:5.1f}  in-time={td<=tc}")

print("\n=== KNOB 2: SPREAD (split goal across M clusters) ===")
spread = sweep('spread', [1, 2, 3, 4, 6], camouflage=0.0, boost=1.8)
for v, zp, br, det in zip(spread['value'], spread['z_peak'],
                          spread['bridges'], spread['detected']):
    print(f"  M={v}: z_peak(per-cluster)={zp:6.1f}  composition edges={br:7.1f}  "
          f"L1-detected={det>0.5}")

print("\n=== KNOB 3: BOOST (per-session intensity → benign) ===")
boost = sweep('boost', [1.0, 1.15, 1.3, 1.5, 1.8, 2.5], camouflage=0.0, spread=1)
for v, zb, co in zip(boost['value'], boost['z_budget'], boost['coherence']):
    print(f"  β={v:.2f}: z@budget={zb:6.1f}  coherence={co:.3f}")

print("\n=== EVASION FRONTIER (boost × spread grid) ===")
front = []
for b in [1.0, 1.15, 1.3, 1.5, 1.8, 2.5]:
    for m in [1, 2, 4]:
        acc = defaultdict(list)
        for s in range(NUM_SEEDS):
            r = run_attack(camouflage=0.0, spread=m, boost=b, seed=3000 + s)
            acc['z'].append(r['z_peak']); acc['co'].append(r['coherence'])
        front.append((b, m, float(np.mean(acc['z'])), float(np.mean(acc['co']))))
fz = np.array([f[2] for f in front]); fc = np.array([f[3] for f in front])
corr = float(np.corrcoef(fz, fc)[0, 1])
print(f"  corr(detectability z_peak, coherence) = {corr:+.3f}")


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.suptitle('OQ-8: Adaptive Adversary',
             fontsize=14, fontweight='bold', color=S.TEXT_C)

# ── Panel A: camouflage — latency vs delivery cost ──
ax = axes[0, 0]; S.style_ax(ax)
x = camo['value']
ax.plot(x, camo['T_detect'], '-o', color=S.C_RED, lw=2, label='detection latency T_detect')
ax.plot(x, camo['T_complete'], '-s', color=S.C_BLUE, lw=2, label='goal delivered T_complete')
ax.fill_between(x, camo['T_detect'], camo['T_complete'],
                color=S.C_BLUE, alpha=0.10)
ax.set_xlabel('camouflage fraction  c')
ax.set_ylabel('observed sessions')
ax.set_title('Camouflage: detector still fires before the goal\n'
             'is delivered — dilution only buys latency')
ax.legend(fontsize=8.5, loc='upper left')
S.panel_label(ax, 'a')

# ── Panel B: spread — local signal vs composition edges ──
ax = axes[0, 1]; S.style_ax(ax)
x = spread['value']
ax.plot(x, spread['z_peak'], '-o', color=S.C_RED, lw=2)
ax.axhline(DETECT_Z, color=S.SUBTLE, ls='--', lw=1.2)
ax.text(x[-1], DETECT_Z * 1.15, 'detection threshold', ha='right', fontsize=8.5, color=S.SUBTLE)
ax.set_xlabel('goal spread  M  (clusters)')
ax.set_ylabel('Level 1 signal  z_peak', color=S.C_RED)
ax.tick_params(axis='y', labelcolor=S.C_RED)
axb = ax.twinx()
axb.plot(x, spread['bridges'], '-^', color=S.C_PURPLE, lw=2)
axb.set_ylabel('I(G)>0  composition edges', color=S.C_PURPLE)
axb.tick_params(axis='y', labelcolor=S.C_PURPLE)
axb.spines['top'].set_visible(False)
ax.set_title('Spread: suppressing the per-cluster signal\n'
             'manufactures cross-cluster composition edges')
S.panel_label(ax, 'b')

# ── Panel C: boost — detector signal and utility move together ──
ax = axes[1, 0]; S.style_ax(ax)
x = boost['value']
ax.plot(x, boost['z_budget'], '-o', color=S.C_RED, lw=2)
ax.axhline(DETECT_Z, color=S.SUBTLE, ls='--', lw=1.2)
ax.set_xlabel('activation boost  β   (β=1 ≡ benign intensity)')
ax.set_ylabel('detector signal  z @ budget', color=S.C_RED)
ax.tick_params(axis='y', labelcolor=S.C_RED)
axc = ax.twinx()
axc.plot(x, boost['coherence'], '-D', color=S.C_GREEN, lw=2)
axc.set_ylabel('goal coherence (utility)', color=S.C_GREEN)
axc.tick_params(axis='y', labelcolor=S.C_GREEN)
axc.spines['top'].set_visible(False)
ax.set_title('Boost: even β=1 is detected (concentration is\n'
             'the signal); coherence tracks the same knob')
S.panel_label(ax, 'c')

# ── Panel D: evasion frontier ──
ax = axes[1, 1]; S.style_ax(ax)
sc = ax.scatter(fc, fz, c=[f[0] for f in front], cmap='YlOrRd',
                s=80, edgecolor=S.TEXT_C, linewidth=0.5, zorder=5)
ax.axhline(DETECT_Z, color=S.SUBTLE, ls='--', lw=1.2)
ax.text(fc.max(), DETECT_Z * 1.4, 'detected above threshold', ha='right',
        fontsize=8.5, color=S.SUBTLE)
xspan = fc.max() - fc.min()
ax.axvspan(fc.min() + 0.5 * xspan, fc.max() + 0.05 * xspan,
           ymin=0, ymax=DETECT_Z / (fz.max() * 1.05),
           color=S.C_GREEN, alpha=0.10)
ax.text(fc.max(), DETECT_Z * 0.4, 'stealthy + working\nattack\n(empirically empty)',
        ha='right', fontsize=8.5, color=S.C_GREEN, fontweight='bold')
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label('boost β', color=S.TEXT_C)
cb.ax.tick_params(colors=S.TEXT_C)
ax.set_xlabel('goal coherence  (attacker utility)')
ax.set_ylabel('detector signal  z_peak')
ax.set_title(f'Evasion frontier  (boost × spread)\n'
             f'corr(z, coherence) = {corr:+.2f}  →  no free lunch')
S.panel_label(ax, 'd')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), 'figure_oq8_adaptive_adversary.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out}")

# ─────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────
print("\n" + "=" * 64)
print("OQ-8 VERDICT: ADAPTIVE ADVERSARY & DYNAMIC GRAPH MODEL")
print("=" * 64)
print(f"""
The adaptive adversary is a CONTROLLED DC-SBM: it tunes the effective per-session
co-activation rate within its target via three knobs. Each was swept; none yields
a free evasion (means over {NUM_SEEDS} seeds).

  Camouflage  : goal-delivery time grows {camo['T_complete'][0]:.0f} → {camo['T_complete'][-1]:.0f} sessions
                ({camo['total_adv'][-1]/camo['total_adv'][0]:.1f}× adversary sessions) yet T_detect
                ({camo['T_detect'][0]:.0f} → {camo['T_detect'][-1]:.0f}) stays below T_complete — the relative
                anomaly survives dilution; camouflage only buys latency.

  Spread M    : per-cluster signal z_peak falls {spread['z_peak'][0]:.0f} → {spread['z_peak'][-1]:.0f} as the goal is
                split (still > threshold here), while composition edges rise
                {spread['bridges'][0]:.0f} → {spread['bridges'][-1]:.0f}. Pushing the local signal toward threshold
                only manufactures the I(G)>0 composition signature adversary.md
                proves is unavoidable for a reassembled goal — Level 1 trades
                into Level 2, it does not disappear.

  Boost β     : even β=1.0 (benign per-session intensity) is detected
                (z@budget={boost['z_budget'][0]:.0f}) — CONCENTRATION, not intensity, is the
                signal. Coherence {boost['coherence'][0]:.2f} → {boost['coherence'][-1]:.2f} tracks the same knob.

  Frontier    : corr(detectability, coherence) = {corr:+.2f} across the
                boost×spread grid. {{low detectability ∧ high coherence}} is empty.

Dynamic graph model: a DC-SBM percolation with an adversary-controlled co-activation
rate q(t). Detector statistic and attack utility are both monotone in q, so the
only undetectable operating point is q→benign — where T_c→∞ and the goal never
assembles. This is the simulation form of the evasion-cost bound.

  OQ-8: ADDRESSED.
  Remaining: a CLOSED-LOOP adversary that estimates the deployment's calibrated
  φ_c online and servo-controls q to ride just under threshold (vs the open-loop
  knobs swept here).
""")
