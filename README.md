# Cross-Session LLM Attack Detection via Semantic Graph Percolation

[![Version](https://img.shields.io/badge/version-v0.9-blue?style=flat-square)](https://github.com/danyu6666/SIW)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen?style=flat-square)](https://www.python.org/)

**Semantic Intent Web (SIW)** detects distributed LLM attacks that evade per-request safety filters — by modeling intent as an accumulating semantic graph and detecting when it undergoes a phase transition.

---

## The Attack This Framework Addresses

Current LLM safety evaluates each request independently. A sophisticated attacker
exploits this by spreading a single harmful goal across many sessions, each appearing
individually harmless:

```
Session 1 → "What household chemicals shouldn't be mixed?"         (safe alone)
Session 2 → "What is the decomposition mechanism of H₂O₂?"        (safe alone)
Session 3 → "How would a novelist describe transporting oxidizers?" (safe alone)
Session 4 → "What containers resist concentrated nitric acid?"      (safe alone)

Combined  → complete synthesis pathway                              (MISSED)
```

**Lemma 1 (proven):** Any per-request classifier has detection probability equal
to its false positive rate against distributed attacks — regardless of how good
the classifier is. This is a theorem, not a limitation.

---

## The Design Concept

### Intent is a network, not a sentence.

When an attacker queries an LLM repeatedly, activated semantic features leave
traces. Features that appear together across sessions form edges.
As sessions accumulate, these edges build a graph — and that graph has structure.

```
      [session 1]   [session 2]   [session 3]   ...
           |              |              |
      features       features       features
           \              |              /
            ──── co-activation graph ────
                          |
               cluster densification
                          |
              ┌───────────────────────┐
              │   PHASE TRANSITION    │  ← intent crystallization
              └───────────────────────┘
                          |
                  attack detectable
```

This is not pattern matching on content. It is detecting the emergence of
network structure — the same phenomenon studied in percolation theory.

### Why a phase transition?

A benign user's sessions activate features randomly across many topics.
An attacker's sessions repeatedly activate features within a specific semantic
cluster. Over time, that cluster becomes unusually dense.

When density crosses a critical threshold φ_c, the graph undergoes a
**percolation phase transition**: a giant connected component emerges, spanning
the targeted semantic space. This moment is the detection signal.

```
sub-critical (benign)          super-critical (attack)
○   ○   ○   ○   ○              ●─────●─────●─────●
  ○       ○   ○                │     │     │     │
○   ○   ○     ○                ●─────●─────●─────●
  ○   ○     ○                  │           │
○     ○   ○   ○                ●─────●─────●

  scattered, no giant            giant component emerged
  component                      → intent crystallized
```

---

## Framework Architecture

SIW is organized in three layers, with detection at two levels:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Semantic Interaction Dynamics                         │
│    Each session activates features → edges accumulate over time  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Structural Emergence                                  │
│    Clusters densify → phase transition at p_c                    │
│                                                                  │
│    Level 1 detection (local):   cluster density ρ_k > threshold  │
│    Level 2 detection (global):  |C_max|/|V| ≥ φ_c               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — Intent Inference                                      │
│    Graph structure → safety decision                             │
└─────────────────────────────────────────────────────────────────┘
```

**Level 1 (always on):** monitors cluster density locally. Fast, private,
decentralizable. Detects single-cluster attacks early.

**Level 2 (triggered):** monitors the global giant component. Definitive signal.
Requires central aggregation. Catches multi-cluster attacks.

---

## The Fundamental Constraint: SIW Trilemma

Detecting distributed attacks requires cross-session monitoring.
Cross-session monitoring conflicts with privacy and decentralization.
This is not an engineering trade-off — it is proven impossible to satisfy all three:

```
         Effectiveness
         (detect attacks)
              △
             / \
            /   \
           /     \
          ×───────×
   Privacy         Decentralization
   (ε-DP)         (no SPOF)

  You can pick at most two vertices.
```

**Proof uses:** data processing inequality · sharp threshold sensitivity ·
SMPC simulation invariance · K-player set disjointness.

See [`proofs.md`](proofs.md) for the full formal proof.

---

## Key Theoretical Results

**τ-Invariance Theorem** (new in v0.5)
The crystallization threshold p_c ≈ 1/N is independent of the co-activation
threshold τ. Increasing τ shifts *when* (session count T_c) the transition
occurs, but not *where* (graph density p_c).

```
T_c = ( τ! / ((N-1) × Q_τ) )^{1/τ}    where Q_τ = E[q_ij^τ]
```

Empirically verified: T_c predicted at 24.1 sessions vs empirical 26 (7.5% error).

**Topology: DC-SBM, not ER** (new in v0.5)
Semantic co-activation graphs are best described by the Degree-Corrected
Stochastic Block Model — not Erdős–Rényi, not scale-free. Community structure
and hub features are both present; their interaction produces the observed
intermediate topology.

**Lemma 3 tight for DC-SBM** (new in v0.9)
The Ω(n) communication lower bound is tight for DC-SBM topology. Level 1
(K independent local computations) provably avoids the centralization
bottleneck. Level 2 (global aggregation) cannot.

---

## Empirical Results

![Phase Transition](simulation/simulation_results.png)

| Measurement | Result |
|-------------|--------|
| Attack cluster density vs benign | **+69%** at crystallization |
| Level 1 detection at ε_p = 2.4 | **90% TPR**, 5% FPR |
| Level 2 detection at same TPR | ε_p = 9.6 — **4× more privacy cost** |
| Response-only proxy signal | **4.5× clustering** (Ollama/Mistral black-box) |
| φ_c domain variation | CV = 0.47 — calibration required per deployment |
| φ_c cross-architecture gap | 67.6% (sentence-transformer vs Ollama mistral) |

---

## Honest Boundary

SIW is not a complete solution. We prove what it cannot do:

> **Attacks with I(G|A) = 0 evade SIW — and this is proven unsolvable.**

If an attacker can decompose their goal into semantically independent fragments
(zero interaction complexity), no graph-based system can detect them.
This is the provably undetectable class. See [`adversary.md`](adversary.md).

---

## Running the Experiments

```bash
pip install -r requirements.txt

# Core: synthetic phase transition
python simulation/percolation_demo.py

# φ_c calibration across 5 domains
python experiments/run_oq2_phi_calibration.py

# Privacy-utility curve (Level 1 vs Level 2)
python experiments/run_oq5_privacy_utility.py

# True black-box detection (requires Ollama + mistral)
ollama pull mistral
python experiments/run_oq3_ollama_blackbox.py
```

---

## What's In This Repository

| File | Contents |
|------|----------|
| [`framework.md`](framework.md) | Mathematical model: DC-SBM topology, τ-invariance, two-level architecture |
| [`proofs.md`](proofs.md) | Formal proofs: Lemma 1–3, SIW Trilemma, Theorem 3' (DC-SBM) |
| [`adversary.md`](adversary.md) | Adversary model: composition theorem, fragmentation bounds, honest boundary |
| [`open_questions.md`](open_questions.md) | Open problems with experimental findings (OQ-1 through OQ-10) |
| `simulation/` | Synthetic percolation simulation and figures |
| `experiments/` | 9 experiments covering OQ-1 through OQ-6 |

---

## Status

```
Theory          ██████████ complete (v0.9)
Experiments     ████████░░ 6/10 OQs addressed
Full paper      ░░░░░░░░░░ in preparation
Real LLM data   ░░░░░░░░░░ future work (SAE activations)
```

---

## Citation

```bibtex
@misc{siw2026,
  author       = {Chang, Danyu},
  title        = {{Semantic Intent Web: A Percolation Framework for
                   Cross-Session LLM Safety}},
  year         = {2026},
  howpublished = {\url{https://github.com/danyu6666/SIW}},
  note         = {Contact: aa.prime.studio@gmail.com}
}
```

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
