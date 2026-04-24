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
How does the crystallization threshold vary across domains, languages,
model sizes, and architectures? Is there a universal φ_c?

**OQ-3: Black-box Feature Proxy**  
SIW assumes access to internal feature activations. What proxy signals
(output distributions, latency, confidence scores) work for closed models?

**OQ-4: Adversarial Topology** *(Addressed — see `adversary.md`)*  
Fragmentation analysis complete: attacks with semantic interaction complexity
I(G) > 0 require at least one cross-cluster composition session (detectable).
Attacks with I(G) = 0 evade SIW (honest boundary). Remaining: empirical
quantification of I(G) for real attack categories, and precise detection
probability for cross-cluster bridges.

**OQ-5: Privacy-Utility Curve**  
Quantify the exact relationship between ε_p and detection rate for the
two-level architecture. Where is the practical operating point?

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

