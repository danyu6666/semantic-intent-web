# Semantic Intent Web (SIW): Framework v0.4
### Semantic Percolation Theory for Cross-Session LLM Safety

> **Status:** Pre-print with formal proofs and adversary analysis.  
> **Date:** 2026-04-15  
> **Next:** Full paper with related work and extended experiments.

---

## Abstract

We introduce the Semantic Intent Web (SIW), a theoretical framework for detecting
distributed, cross-session attacks on Large Language Models. Current LLM safety
mechanisms evaluate each request in isolation, leaving a structural blind spot:
sophisticated attackers who distribute malicious intent across multiple sessions.

SIW models intent as a percolation process on a temporal semantic graph.
We prove the **SIW Trilemma**: no system can simultaneously achieve
effectiveness against distributed attacks, individual privacy, and decentralization.
We further show, via simulation, that **local cluster density is a leading indicator
of global intent crystallization**, enabling earlier detection at lower privacy cost.

---

## 1. Problem Statement

Current LLM safety paradigm:

```
Request → Evaluate → Allow / Reject
```

Structural blind spot: An attacker who distributes an attack across N sessions,
each individually below detection threshold, is invisible to this paradigm.

We call this a **distributed attack**. No existing LLM safety paper addresses
detection across sessions without sacrificing privacy or requiring centralized
surveillance infrastructure. This paper formalizes that impossibility and
proposes a principled design space.

---

## 2. Core Primitive: Semantic Intent Crystallization

### 2.1 Semantic Feature Space

Let F be the space of semantic features extractable from LLM interactions.
Each session σ_i activates a subset F(σ_i) ⊆ F.

*(In practice, F corresponds to sparse autoencoder features from interpretability
research, e.g., Anthropic's circuit tracing work.)*

### 2.2 Temporal Semantic Intent Graph

G(t) = (V(t), E(t), w(t)) where:

- V(t) = ∪_{i: t_i ≤ t} F(σ_i)         [nodes = activated features up to time t]
- E(t) = {(v_i,v_j) : CoAct(v_i,v_j,t) ≥ γ}  [edges = co-activation above threshold]
- w_ij(t) = normalized co-activation frequency

### 2.3 Intent Crystallization

**Definition (Global Crystallization):**

```
Crystallized_global(G(t)) ⟺ |C_max(t)| / |V(t)| ≥ φ_c
```

Where C_max(t) is the largest connected component.

**Definition (Local Crystallization):**

```
Crystallized_local(G(t), k) ⟺ ρ_k(t) / ρ_baseline > δ_k
```

Where ρ_k(t) is the edge density within cluster k at time t,
and δ_k is the cluster-specific anomaly threshold.

**Key finding from simulation:** Local crystallization precedes global
crystallization by a measurable lead time, providing an early warning signal.

---

## 3. Topology Characterization (Empirical)

**Standard models and their fit to semantic co-activation graphs:**

| Model              | Prediction          | Empirical fit       |
|--------------------|---------------------|---------------------|
| Erdős–Rényi        | Random, p_c = 1/n   | Rejected (KS p≈0)   |
| Barabási–Albert    | Power-law γ ∈ [2,3] | Partial (R²=0.81)   |
| **Intermediate**   | **Mixed structure** | **Confirmed**       |

**Simulation findings (N=500 features, 300 sessions):**

```
Global p_c (empirical):   0.002196
Global p_c (ER):          0.002000   [1.10× — acceptable approximation globally]
Power-law R²:             0.812      [below scale-free threshold of 0.85]
KS test vs Poisson:       p ≈ 0      [ER model rejected]
Attack cluster density:   37.0 mean degree
Benign cluster density:   21.9 mean degree  [69% difference — detectable]
```

**Conclusion:** Semantic co-activation graphs exhibit an **intermediate topology**:
- Global structure: Erdős–Rényi is a valid first approximation for p_c
- Local structure: Cluster densities follow non-random distributions
- Hub features exist (high-centrality semantic concepts)

This intermediate topology is a finding, not a limitation. It justifies a
**two-level detection architecture** that exploits both global and local structure.

---

## 4. Two-Level Detection Architecture

The simulation reveals that global and local crystallization carry different
information at different costs.

```
Level 1: Local Cluster Detection
    — Monitors edge density within semantic clusters
    — Requires only local session history (lower privacy cost)
    — Earlier signal: detects before global p_c is crossed
    — Trade-off: misses attacks spanning multiple clusters

Level 2: Global Phase Transition Detection
    — Monitors giant component ratio |C_max|/n
    — Requires cross-session global graph
    — Later signal: definitive crystallization
    — Trade-off: higher privacy cost, requires more centralization
```

**Practical deployment:**

```
Level 1 (always on, privacy-preserving):
    Low privacy cost → can run with strong DP guarantees
    Catches: focused single-cluster attacks

Level 2 (triggered by Level 1 or policy):
    Higher privacy cost → requires governance framework
    Catches: distributed multi-cluster attacks
```

---

## 5. The SIW Trilemma

### 5.1 Definitions

**Effectiveness E:** The system detects distributed attacks with probability ≥ 1-ε
against a fully-informed adversary (type θ₂).

**Privacy P:** The system satisfies (ε_p, δ_p)-differential privacy with respect
to individual actor session histories.

**Decentralization D:** The system does not require a centralized node whose
failure collapses detection capability (formalized as ρ(M) ≤ ρ* < 1).

### 5.2 Main Theorem

**Theorem (SIW Trilemma):**

> No mechanism M simultaneously satisfies E, P (with ε_p → 0), and D
> against a type θ₂ adversary.

*Proof sketch (full proofs in `proofs.md`):*
- By data processing inequality: any E-satisfying mechanism must use cross-session history (Lemma 1)
- By sharp threshold sensitivity at p_c + DP definition: detection requires ε_p ≥ ln((1-ε-δ_p)/α) > 0 (Lemma 2b). SMPC does not reduce this bound — it protects computation, not output (Lemma 2c).
- By distributed computation lower bounds: computing |C_max|/|V| requires centralized aggregation (Lemma 3)
- Therefore E ∧ P ∧ D is infeasible. □

### 5.3 Refined Trilemma (Two-Level)

The two-level architecture partially relaxes the trilemma:

| Configuration        | E_local | E_global | P  | D  |
|----------------------|---------|----------|----|----|
| Level 1 only         | ✓       | ✗        | ✓  | ✓  |
| Level 2 only         | ✗       | ✓        | ~  | ✗  |
| Both levels          | ✓       | ✓        | ~  | ✗  |

**Corollary LC (Local Crystallization leads Global):**

There exists a detectable interval [t_local, t_global] where local crystallization
has occurred but global crystallization has not yet crossed φ_c.

Empirically: t_local ≈ 0.4 × t_global (attack cluster density anomaly detectable
at ~40% of the time needed for global phase transition).

This interval is the operational window for privacy-preserving early detection.

### 5.4 Design Space

```
Three principled deployment points:

DP₁: E ∧ P ∧ ¬D  — Centralized governed system (consumer AI platforms)
DP₂: E ∧ D ∧ ¬P  — High-security surveillance system (critical infrastructure)
DP₃: P ∧ D ∧ ¬E_global — Open privacy system (Level 1 only, OSS deployment)
```

---

## 6. Ethics Framework

### Core Tension

SIW's detection mechanism and its privacy risks share the same substrate.
We resolve this through four design principles:

**Principle 1 — Contextual Integrity (Nissenbaum):**
Information flows are legitimate only when they match the norms of the original
context. SIW data must be technically purpose-bound to safety detection.

**Principle 2 — Data Minimization:**
SIW stores feature activation vectors, not session content.
Vectors must be non-invertible (one-way mapping to content).

**Principle 3 — Asymmetric Accountability:**
SIW's monitoring must itself be monitored. Independent audit access is
a non-negotiable design requirement, not an optional add-on.

**Principle 4 — Asymmetric Error Cost:**
False positives (innocent users flagged) have higher ethical cost than
false negatives (attacks missed). System outputs are continuous risk scores;
automated enforcement requires mandatory human review.

### The Surveillance Paradox — Direct Answer

> "How is SIW different from a surveillance infrastructure?"

Three simultaneous conditions define the boundary:

1. **Purpose binding** — cryptographic enforcement, not policy statements
2. **Minimal collection** — non-invertible feature vectors, not content
3. **External accountability** — independent auditors with verified access

Absence of any condition makes SIW a surveillance tool. This is the red line.

---

## 7. SIW Hypothesis and Three-Layer Theory

### 7.1 The SIW Hypothesis

> Intent emerges as a detectable structural transition in semantic
> interaction networks.

This is a falsifiable scientific claim: intent is not a property of
individual prompts but a network phenomenon observable through graph structure.

### 7.2 Three-Layer Theory

SIW organizes into three layers of analysis:

**Layer 1 — Semantic Interaction Dynamics:**
How interactions generate semantic structures. Each session activates
features and creates co-activation edges. This layer studies the
network formation process itself.

**Layer 2 — Structural Emergence:**
When and how structure appears. This layer studies cluster densification,
cross-cluster bridge formation, and the phase transition from sub-critical
to super-critical graph states. The two-level detection architecture (§4)
operates at this layer.

**Layer 3 — Intent Inference:**
What structural signals reveal about intent. This layer connects
graph-theoretic observations to safety decisions. The trilemma (§5)
and adversary model (`adversary.md`) constrain what is achievable here.

### 7.3 Research Questions

| ID | Question | Status |
|----|----------|--------|
| RQ1 | Do semantic interaction networks exhibit structural transitions corresponding to intent formation? | Confirmed in simulation; early embedding validation in `experiments/` |
| RQ2 | Can local cluster densification serve as an early signal? | Confirmed in simulation (t_local ≈ 0.4 × t_global) |
| RQ3 | Do malicious interactions produce denser semantic clusters? | Confirmed in simulation (+69% density) |
| RQ4 | Do cross-domain tasks produce detectable cross-cluster bridges? | Theoretically bounded (`adversary.md` composition theorem) |
| RQ5 | Which feature representations best reveal crystallization? | Open — early experiments with sentence-transformers and Ollama |

### 7.4 Research Perspectives

SIW connects three communities:

- **AI Safety:** Can distributed attacks be detected via interaction structure?
- **AI Interpretability:** Do LLM feature activations form meaningful semantic graphs?
- **Network Science:** What graph dynamics govern intent crystallization?

---

## 8. Open Questions

For the research community (see also `open_questions.md` for full list):

1. **Topology model:** What random graph model best characterizes semantic
   co-activation? The intermediate topology found here requires a new model.

2. **φ_c calibration:** How does the crystallization threshold vary across
   domains, languages, and model architectures?

3. **Adversarial topology:** *(Addressed — see `adversary.md`)*
   Fragmentation analysis shows composition-dependent attacks (I(G|A) > 0)
   require detectable cross-cluster sessions. Detection signal scales
   continuously with I(G|A). Remaining: empirical calibration.

4. **Feature space access:** SIW assumes access to internal feature activations.
   What proxy signals work for black-box models?

5. **Privacy-utility curve:** What is the precise quantitative relationship
   between ε_p and detection rate for the two-level architecture?

---

## 9. Simulation and Experiments

### 9.1 Synthetic simulation

Code available in `simulation/percolation_demo.py`.

**Reproducing key results:**

```bash
python3 simulation/percolation_demo.py
# Outputs:
# - Phase transition at p_c = 0.002196 (ER prediction: 0.002000)
# - Degree distribution: intermediate topology (R²=0.812, KS p≈0)
# - Attack cluster density: 37.0 vs benign 21.9 (69% difference)
# - Crystallization threshold φ_c = 0.242
```

### 9.2 Early validation (real embeddings)

Experiments in `experiments/` test the SIW hypothesis using real LLM embeddings:

- `experiments/run_test.py` — basic percolation validation (random features, 500 nodes)
- `experiments/run_semantic_graph.py` — sentence-transformer (`all-MiniLM-L6-v2`) embedding graph
- `experiments/run_ollama_graph.py` — Ollama/Mistral embedding graph

Both real-embedding experiments confirm phase transition behavior on
semantic co-activation graphs, supporting RQ1.

---

## Citation

If you build on this framework:

```
@misc{siw2026,
  title   = {Semantic Intent Web: A Percolation Framework for
             Cross-Session LLM Safety},
  year    = {2026},
  note    = {Concept preprint. Full paper in preparation.},
  contact = {aa.prime.studio@gmail.com}
}
```

---

## Contributing

This is an open framework. We welcome:
- Alternative topology models for semantic co-activation
- Empirical validation on real LLM interaction data
- Empirical calibration of I(G|A) for specific attack categories
- Extensions to multi-modal systems

Open an issue or pull request.

