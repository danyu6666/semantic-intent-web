"""
OQ-2 (跨模型): Ollama mistral embedding vs sentence-transformer φ_c 校準

補完 OQ-2 的「cross-architecture」維度：
  - 之前只用 sentence-transformer (all-MiniLM-L6-v2, 384-dim)
  - 現在加 Ollama mistral embedding (4096-dim)
  - 問題：φ_c 是否跨 embedding 架構穩定？

若 φ_c 在不同 embedding 模型下差異大 → 需要 per-model 校準
若 φ_c 穩定 → φ_c 主要由語義結構決定，與 embedding 架構無關
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import ollama
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────
# 用 OQ-2 相同的 domain prompts（子集）
# ─────────────────────────────────────────
DOMAINS = {
    "Cooking": [
        "how to make pasta carbonara", "best way to roast chicken",
        "how to bake sourdough bread", "recipe for beef stew",
        "how to caramelize onions", "making homemade pizza dough",
        "how to cook risotto properly", "tips for perfect scrambled eggs",
        "how to make hollandaise sauce", "grilling steak to medium rare",
    ],
    "Programming": [
        "how to implement a binary search tree", "explain async await in javascript",
        "difference between tcp and udp", "how does garbage collection work",
        "implement quicksort in python", "what is a closure in programming",
        "explain database indexing strategies", "how to design a REST API",
        "what is dependency injection", "explain the CAP theorem",
    ],
    "Science": [
        "how does quantum entanglement work", "explain general relativity simply",
        "what is the higgs boson", "how does CRISPR gene editing work",
        "explain the double slit experiment", "what causes superconductivity",
        "how does nuclear fusion work", "what is dark matter",
        "explain entropy and the arrow of time", "how do black holes evaporate",
    ],
}

N_SESSIONS   = 80     # 每個 domain
TOP_K_ST     = 15     # sentence-transformer (384-dim): 15/384 = 3.9% density
TOP_K_OL     = 160    # Ollama mistral (4096-dim):    160/4096 = 3.9% density
COACT_THRESH = 2      # 統一門檻

# ─────────────────────────────────────────
# φ_c 提取函式
# ─────────────────────────────────────────
def extract_phi_c(sessions_features, n_features, tau=COACT_THRESH):
    coact = defaultdict(int)
    G = nx.Graph()
    G.add_nodes_from(range(n_features))
    ratios, edges = [], []
    for feats in sessions_features:
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                a, b = min(feats[i], feats[j]), max(feats[i], feats[j])
                coact[(a, b)] += 1
                if coact[(a, b)] == tau:
                    G.add_edge(a, b)
        ne = G.number_of_edges()
        edges.append(ne)
        if ne > 0:
            comps = sorted(nx.connected_components(G), key=len, reverse=True)
            ratios.append(len(comps[0]) / n_features)
        else:
            ratios.append(0.0)
    ratios = np.array(ratios)
    diffs  = np.diff(ratios)
    t_star = int(np.argmax(diffs)) + 1 if len(diffs) > 0 else 0
    phi_c  = float(ratios[t_star]) if t_star < len(ratios) else 0.0
    max_e  = n_features * (n_features - 1) / 2
    p_c    = edges[t_star] / max_e if t_star < len(edges) else 0.0
    return phi_c, t_star, p_c, ratios

# ─────────────────────────────────────────
# MODEL A: sentence-transformer (384-dim)
# ─────────────────────────────────────────
print("=== MODEL A: sentence-transformer (384-dim) ===")
encoder_st = SentenceTransformer("all-MiniLM-L6-v2")
results_st = {}

for domain, prompts in DOMAINS.items():
    embs = encoder_st.encode(prompts, show_progress_bar=False)
    sessions_feats = []
    for _ in range(N_SESSIONS):
        idx   = np.random.randint(len(prompts))
        feats = list(np.argsort(np.abs(embs[idx]))[-TOP_K_ST:])
        sessions_feats.append(feats)
    phi_c, t_star, p_c, ratios = extract_phi_c(sessions_feats, n_features=384)
    results_st[domain] = {'phi_c': phi_c, 't_star': t_star, 'p_c': p_c, 'ratios': ratios}
    print(f"  {domain:12s}  φ_c={phi_c:.3f}  T_c={t_star:3d}  p_c={p_c:.5f}")

# ─────────────────────────────────────────
# MODEL B: Ollama mistral (4096-dim)
# ─────────────────────────────────────────
print("\n=== MODEL B: Ollama mistral embedding (4096-dim) ===")
results_ol = {}

for domain, prompts in DOMAINS.items():
    # Pre-fetch all embeddings
    embs_ol = []
    for p in prompts:
        r = ollama.embeddings(model='mistral', prompt=p)
        embs_ol.append(np.array(r['embedding']))
    embs_ol = np.array(embs_ol)

    sessions_feats = []
    for _ in range(N_SESSIONS):
        idx   = np.random.randint(len(prompts))
        feats = list(np.argsort(np.abs(embs_ol[idx]))[-TOP_K_OL:])
        sessions_feats.append(feats)

    phi_c, t_star, p_c, ratios = extract_phi_c(sessions_feats, n_features=4096)
    results_ol[domain] = {'phi_c': phi_c, 't_star': t_star, 'p_c': p_c, 'ratios': ratios}
    print(f"  {domain:12s}  φ_c={phi_c:.3f}  T_c={t_star:3d}  p_c={p_c:.5f}")

# ─────────────────────────────────────────
# 比較
# ─────────────────────────────────────────
print("\n=== 跨模型 φ_c 比較 ===")
print(f"{'Domain':12s}  {'ST φ_c':>8}  {'Ollama φ_c':>10}  {'差異':>8}")
print("-" * 46)
diffs_phi = []
for domain in DOMAINS:
    phi_st = results_st[domain]['phi_c']
    phi_ol = results_ol[domain]['phi_c']
    diff   = abs(phi_st - phi_ol) / max(phi_st, 1e-9) * 100
    diffs_phi.append(diff)
    print(f"{domain:12s}  {phi_st:>8.3f}  {phi_ol:>10.3f}  {diff:>7.1f}%")

print(f"\n平均 φ_c 差異: {np.mean(diffs_phi):.1f}%")
phi_st_vals = [results_st[d]['phi_c'] for d in DOMAINS]
phi_ol_vals = [results_ol[d]['phi_c'] for d in DOMAINS]
print(f"ST  φ_c: mean={np.mean(phi_st_vals):.3f}  std={np.std(phi_st_vals):.3f}")
print(f"OL  φ_c: mean={np.mean(phi_ol_vals):.3f}  std={np.std(phi_ol_vals):.3f}")
print(f"Domain ordering preserved: "
      f"{'YES' if list(np.argsort(phi_st_vals)) == list(np.argsort(phi_ol_vals)) else 'NO'}")

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('OQ-2 (跨模型): sentence-transformer vs Ollama mistral embedding',
             fontsize=12, fontweight='bold')

colors_d = ['#e41a1c', '#377eb8', '#4daf4a']
domains  = list(DOMAINS.keys())

# ── Plot 1: Crystallization curves — sentence-transformer ──
ax = axes[0]
for domain, color in zip(domains, colors_d):
    ax.plot(results_st[domain]['ratios'], lw=1.8, label=domain, color=color)
    ax.axvline(results_st[domain]['t_star'], color=color, ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Session t')
ax.set_ylabel('|C_max| / |V|')
ax.set_title('sentence-transformer (384-dim)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 2: Crystallization curves — Ollama mistral ──
ax = axes[1]
for domain, color in zip(domains, colors_d):
    ax.plot(results_ol[domain]['ratios'], lw=1.8, label=domain, color=color, ls='--')
    ax.axvline(results_ol[domain]['t_star'], color=color, ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Session t')
ax.set_ylabel('|C_max| / |V|')
ax.set_title('Ollama mistral (4096-dim)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 3: φ_c 對比 ──
ax = axes[2]
x   = np.arange(len(domains))
w   = 0.35
b1  = ax.bar(x - w/2, phi_st_vals, w, label='sentence-transformer', color='steelblue', alpha=0.8)
b2  = ax.bar(x + w/2, phi_ol_vals, w, label='Ollama mistral',       color='tomato',    alpha=0.8)
for bar, v in zip(list(b1) + list(b2), phi_st_vals + phi_ol_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=15)
ax.set_ylabel('φ_c')
ax.set_title(f'φ_c: Cross-Model Comparison\n(mean diff = {np.mean(diffs_phi):.1f}%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), 'figure_oq2_crossmodel.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out}")

# ─────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────
print("\n" + "="*55)
print("OQ-2 跨模型 VERDICT")
print("="*55)
ordering_ok = list(np.argsort(phi_st_vals)) == list(np.argsort(phi_ol_vals))
print(f"""
sentence-transformer (384-dim) vs Ollama mistral (4096-dim):

  φ_c 平均差異:  {np.mean(diffs_phi):.1f}%
  Domain 排序相同: {'YES' if ordering_ok else 'NO'}
  ST  mean φ_c = {np.mean(phi_st_vals):.3f}  (std={np.std(phi_st_vals):.3f})
  OL  mean φ_c = {np.mean(phi_ol_vals):.3f}  (std={np.std(phi_ol_vals):.3f})

結論:
  {'φ_c 跨模型差異 < 20%，且 domain 排序保留 → φ_c 主要由語義結構決定，跨架構近似穩定。' if np.mean(diffs_phi) < 20 and ordering_ok else
   'φ_c 跨模型差異顯著（> 20%）或 domain 排序不同 → 不同架構需要獨立校準。'}

  OQ-2 完整回答:
    領域差異 (CV=0.47):         需校準
    跨架構差異 ({np.mean(diffs_phi):.0f}%):  {'需校準' if np.mean(diffs_phi) > 20 else '近似穩定'}
    跨語言差異:                 尚未測試
""")
