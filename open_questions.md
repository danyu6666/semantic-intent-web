# Open Questions

For the research community to engage with.

## Technical

**OQ-1: Topology Model**  
*(Partially addressed — see `experiments/run_dcsbm_analysis.py`)*

Analysis of the critical graph G(t*) (274 edges, mean degree ≈ 1.096) confirms
that the correct **static** model is the **Degree-Corrected Stochastic Block
Model (DC-SBM)**:

```
P(edge i,j) = θ_i × θ_j × B_{k(i),k(j)}

Empirical results at G(t*):
  Hub / non-hub degree ratio:  11.18×  (8.12 vs 0.73 mean degree)
  p_in / p_out:                7.62×   (strong community structure)
  Poisson rejected:            KS p = 0.000
  Molloy-Reed κ:               11.11   (>> 1, non-ER)
```

DC-SBM captures both hub heterogeneity (θ distribution) and community
structure (B matrix), explaining why the topology is intermediate: community
structure prevents global scale-free behaviour; hubs prevent Poisson degree
distribution.

**Dynamic p_c formula — DERIVED. See `experiments/derive_tau_formula.py`.**

The co-activation threshold τ introduces delayed edge formation.
Exact edge probability at session T:

```
P(edge (i,j) by session T) = 1 - F_Poisson(τ-1; T × q_ij)

q_ij = (1/K) Σ_c p_i(c) × p_j(c)   [per-session co-activation prob]
```

Leading-order approximation (valid when T × q_ij << 1):

```
p_c ≈ T_c^τ × Q_τ / τ!   →   ≈ 1/(N-1)   (independent of τ)

T_c = (τ! / ((N-1) × Q_τ))^{1/τ}
Q_τ = E[q_ij^τ]
```

**Key result: p_c ≈ 1/N regardless of τ.**
τ shifts WHEN (T_c) but not WHERE (p_c) the transition occurs.

Numerical verification:
```
T_c (formula)  = 24.1 sessions   (empirical t* = 26, error 7.5%)
p_c (formula)  = 0.002004        (empirical = 0.002196, error 8.7%)
```

τ amplifies community separation (q_in/q_out = 2.15× → Q_τ ratio = 9.5×):
larger τ makes community structure more detectable earlier.

**Remaining:** exact correction for ≈8% residual (discrete threshold
effects, session-order shuffling). See `open_questions.md` OQ-6 (Lemma 3)
for analogous refinement pattern.

**OQ-2: φ_c Calibration**  
*(Partially addressed — see `experiments/run_oq2_phi_calibration.py`)*

Experiment: sentence-transformer (all-MiniLM-L6-v2), 5 domains × 120 sessions,
COACT_THRESHOLD = 3, top-15 embedding dimensions as features.

```
Domain       φ_c    T_c  p_c
Cooking      0.154   21  0.00577
Programming  0.141   18  0.00567
Fitness      0.039    9  0.00144
Science      0.219   32  0.00858
Chemistry    0.094   16  0.00205
Global mix   0.083   34  0.00318
Synthetic    0.242   26  0.00220

φ_c CV = 0.47  (domain-dependent — calibration required)
p_c CV = 0.56  (more variable than φ_c — NOT a universal threshold)
```

**Key finding:** φ_c is domain-dependent (CV=0.47). No universal φ_c exists.
φ_c is more stable across domains than p_c, making it the better
threshold metric for deployment. Science domains crystallize slowest
(highest φ_c); focused/repetitive domains (Fitness) crystallize fastest.

**Cross-architecture calibration (Ollama mistral 4096-dim, K=160, equal density):**

```
Model              φ_c (Cooking)  φ_c (Programming)  φ_c (Science)
sentence-transformer   0.177          0.169              0.148
Ollama mistral         0.039          0.071              0.050
Cross-arch diff:      77.9%          58.3%             66.4%   mean=67.6%
```

Mistral crystallises at T_c=1–5 with very low φ_c: a few dominant embedding
dimensions activate for all prompts, producing rapid but shallow crystallisation.
Domain ordering is NOT preserved across models.

**Complete OQ-2 answer:**
```
φ_c depends on:        calibration needed?
  Domain          ✅  YES (CV=0.47, range 0.039–0.219)
  Architecture    ✅  YES (67.6% gap, ST vs mistral)
  Language        —   Not tested
```

φ_c is neither domain-universal nor architecture-universal.
Per-deployment calibration is mandatory.

**Remaining:** Cross-language calibration.

**OQ-3: Black-box Feature Proxy**  
*(Partially addressed — see `experiments/run_oq3_proxy_signals.py`)*

Three proxy strategies tested (Science domain, 150 sessions, 30 attack):

```
Strategy              φ_c   T_c  Attack ratio  Detectable
A: Top-K activation  0.089   25     1.38×        weak
B: Random projection 0.115   26     1.83×        YES
C: Random baseline   0.065  128     0.00×        NO

Required access: A & B need embedding; C needs nothing.
```

**Key findings:**
1. Phase transition is detectable with external embeddings (A & B).
2. Strategy B (random projection of embedding) gives HIGHER attack density
   ratio than A (top-K), because top-K hits shared technical vocabulary
   in both attack and benign domains. Projection choice matters.
3. Strategy C (no embeddings) produces a transition but zero attack signal,
   proving the semantic structure — not the graph construction — drives detection.
4. T_c is consistent between A and B (25 vs 26 sessions) — proxy does not
   significantly delay detection timing.

**Counterintuitive result:** a less informed proxy (B) can outperform a
more structured one (A) for attack detection when attack and benign domains
share high-magnitude embedding features.

**True black-box test (Ollama mistral, 80 sessions, τ=2):**

```
Strategy D: response embedding (only LLM output observed)
  φ_c = 0.169  T_c = 17  attack ratio = 1.34×  (weak)

vs Strategy A: input embedding (reference)
  φ_c = 0.120  T_c = 11  attack ratio = 1.14×  (weak)

Key finding: attack response cosine clustering ratio = 4.50×
  (within-attack sim 0.402 vs attack-vs-benign 0.089)
```

**Critical insight:** Response text carries strong attack signal in continuous
embedding space (4.50× clustering) but this degrades to 1.34× in the SIW
graph because top-K discretisation + threshold construction loses information.

Implication: for response-only proxy, direct cosine clustering of response
embeddings may outperform SIW graph-based detection. The graph approach
requires more sessions (>200) or lower τ to reliably recover the signal.

**Remaining:** Quantify the exact threshold (min sessions, max τ) at which
response-embedding SIW detection becomes reliable (>1.5× density ratio).

**OQ-4: Adversarial Topology** *(Addressed — see `adversary.md`)*  
Fragmentation analysis complete: attacks with semantic interaction complexity
I(G) > 0 require at least one cross-cluster composition session (detectable).
Attacks with I(G) = 0 evade SIW (honest boundary). Remaining: empirical
quantification of I(G) for real attack categories, and precise detection
probability for cross-cluster bridges.

**OQ-5: Privacy-Utility Curve**  
*(Addressed — see `experiments/run_oq5_privacy_utility.py`)*

Laplace mechanism applied to Level 1 (cluster density) and Level 2 (|C_max|/N),
measured at detection window T_detect=50 sessions.

```
Statistic            Signal gap   Δf      Signal/Δf
Level 1 (local)      0.524        0.670   0.78
Level 2 (global)     0.093        0.178   0.52

To achieve 90% TPR at 5% FPR (Laplace mechanism):
  Level 1:  ε_p ≥ 2.40
  Level 2:  ε_p ≥ 9.61
  Theory:   ε_p ≥ 2.88  [Lemma 2b lower bound]

Level 1 privacy advantage: 75% budget savings vs Level 2
```

**Key findings:**
1. Level 1 achieves near-theoretical DP efficiency (ε_p=2.40 vs bound 2.88).
2. Level 2 requires 4× more privacy budget for same detection rate.
3. Level 1 signal-to-sensitivity ratio (0.78) > Level 2 (0.52): local
   cluster density is inherently more privacy-efficient than global giant
   component as a detection statistic.
4. Practical operating point: ε_p ≈ 2.5 for Level 1 (90% detection).

**Methodological note:** FPR control requires ε_p-adaptive thresholds in
deployment (fixed threshold inflates FPR at low ε_p due to Laplace noise).
The ε_p vs TPR ordering (L1 >> L2) is robust regardless.

**Practical recommendation:**
  Always-on Level 1 (ε_p ≈ 2.5) + escalate to Level 2 only on flag.
  Combined budget ≈ ε_p(L1) + ε_p(L2)×P(escalate) << ε_p(L2) alone.

## Theoretical

**OQ-6: Lemma Formalization** *(Addressed — see `proofs.md`)*  
Formal proofs completed for all three lemmas including Lemma 2c (SMPC case).
Remaining refinement: Lemma 3 communication lower bound for intermediate
topology specifically (current proof uses general graph bound).

**OQ-7: Multi-modal Extension**  
SIW is defined for text features. How does the framework extend to
vision-language models where features span modalities?

**OQ-8: Graph Model for Adversarial Settings**  
The percolation model assumes a static adversary. What is the right
dynamic graph model when the adversary adapts to the detection system?

## Ethical / Governance

**OQ-9: Purpose Binding Implementation**  
How do you technically enforce purpose binding beyond policy statements?
Cryptographic commitments? Trusted execution environments?

**OQ-10: Baseline Definition**  
Who defines "normal" semantic behavior, and how do you prevent the
baseline from encoding existing societal biases about what constitutes
suspicious curiosity?

