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

**Residual ≈8% — characterised (see `experiments/run_oq1_residual.py`).**
A deterministic per-pair analysis shows the gap is NOT a single missing term. The
DC-SBM hub heterogeneity splits "the transition" into a window:

```
  Molloy-Reed onset   ⟨k²⟩/⟨k⟩ = 2     T ≈ 20.3   (giant first possible)
  leading-order       ⟨k⟩_approx = 1   T ≈ 24.1
  empirical t*        steepest jump    T = 26     [anchor]
  exact-Poisson       ⟨k⟩ = 1          T ≈ 28.9
  → transition window width ≈ 8.6 sessions (33% of t*)
```

Both the leading-order formula (24.1) and t* (26) fall inside the window. The
leading-order term lands near t* partly by cancellation: it over-counts hub pairs
(+66% in ⟨k⟩ at t*, pulling T_c down) while truncating higher-order Poisson terms
(pulling it up). Separately, the empirical anchor is attack-accelerated
(⟨k⟩≈1.10 at t* vs benign-only exact ⟨k⟩≈0.76; +0.33 is extra attack density, not
formula error).

**Conclusion:** the τ-formula is accurate to within the intrinsic transition-window
width. The residual is not reducible by a universal closed-form correction — it is
set by which point in the window one calls "the" threshold, plus a
deployment-specific attack-density offset. (Characterised, not "closed" — consistent
with the project's honest-boundary stance.)

**OQ-2: φ_c Calibration**  
*(Addressed — domain, architecture, and language — see `experiments/run_oq2_phi_calibration.py`,
`run_oq2_crossmodel.py`, `run_oq2_crosslang.py`)*

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

**Cross-language calibration (see `experiments/run_oq2_crosslang.py`).**
One multilingual model (paraphrase-multilingual-MiniLM-L12-v2) embeds identical,
hand-translated content in 5 languages (English, 中文, Español, Français, Deutsch)
across a neutral (Cooking) and a sensitive-adjacent (Chemistry) domain.

```
φ_c CV across languages = 0.26   (mean over both domains)
  vs domain CV          = 0.47
  vs architecture gap   = 67.6%

Cross-lingual top-K Jaccard (same meaning) = 0.36–0.46   vs chance 0.07
Mixed-language stream φ_c ≈ mono-language mean (preserved, one detector works)
```

Language is the SMALLEST calibration axis: a shared multilingual embedding maps the
same meaning to overlapping features regardless of language, so the co-activation
graph is largely language-invariant and a single detector serves all five
languages without fragmentation. But CV=0.26 is a real ~26% residual (Spanish/French
crystallise later than English/German/Chinese here) — broadly shared, **not fully
universal**, so light per-language calibration can still help. The near-invariance
is a property of cross-lingual embedding alignment, not of SIW; a monolingual or
poorly-aligned model would fragment by language.

**Complete OQ-2 answer:**
```
φ_c depends on:        calibration burden?
  Architecture    ✅  YES, most  (67.6% gap, ST vs mistral)
  Domain          ✅  YES        (CV=0.47, range 0.039–0.219)
  Language        △   LEAST      (CV=0.26, shared multilingual geometry)
```

φ_c is neither domain- nor architecture-universal; per-deployment calibration is
mandatory. Language is the least demanding axis, but not zero.

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

**Reliability threshold quantified (see `experiments/run_oq3_threshold.py`).**
Sweeping a cached pool of real Ollama (mistral) responses over sessions × τ:

```
attack density ratio (mean, 25 seeds)   n=50   100   150   200   300
  τ = 1                                  1.66  1.47  1.42  1.38  1.38
  τ = 2                                  1.12  1.22  1.18  1.23  1.29
  τ = 3                                  0.79  0.95  1.00  1.06  1.17
continuous cosine clustering (ceiling)   3.98×    |   reliable bar = 1.5×
```

Response-only SIW graph detection clears 1.5× **only at τ=1, and only with few
sessions** (~50): at τ=1 the contrast is strongest early and *erodes* as the
benign background also saturates. For τ ≥ 2 the ratio rises with sessions but
never reaches 1.5× within 300. The continuous response embedding carries the
signal at 3.98×, so the graph discretisation (top-K + τ threshold) throws most of
it away — the τ threshold, not the embedding, is the bottleneck.

**Recommendation:** for a response-only proxy, detect on the continuous cosine
clustering directly (3.98×) rather than the discretised SIW graph; if the graph
must be used, set τ=1 and a short window. OQ-3 reliability is now characterised
across sessions × τ.

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

**OQ-6: Lemma 3 DC-SBM Refinement** *(Addressed — see `proofs.md` §3.1 and `experiments/run_oq6_lemma3_dcsbm.py`)*

Theorem 3' (DC-SBM Communication Structure) added to `proofs.md` §3.1:

```
Phase 1 — Level 1 (local, parallelizable):
  K=12 clusters × Ω(n/K)=Ω(42) bits each
  → K independent computations, no central dependency
  → Decentralization (D) satisfied ✓

Phase 2 — Level 2 (global, centralized):
  Aggregator receives Ω(n)=Ω(500) bits
  → K-player set disjointness reduction proves cannot be distributed
  → Decentralization (D) violated ✗
```

Proof: K-player set disjointness (Braverman et al. 2013).
High p_in/p_out=7.62× STRENGTHENS the bound: dense within-cluster edges
create larger component labelings; sparse cross-cluster edges make each one
pivotal for C_max. Even when all cross-cluster edges are already within the
same component, the protocol must READ them to confirm this fact.

General Ω(|V|) bound is now TIGHT for DC-SBM.
Lemma 3 confidence: Medium → High.

**OQ-7: Multi-modal Extension**  
*(Addressed — see `experiments/run_oq7_multimodal.py`)*

The multimodal case is a **two-block DC-SBM**: modality A (e.g. text) and
modality B (e.g. vision) each carry the concept clusters, joined by cross-modal
co-activation edges. A concept k exists in both modalities (clusters a_k, b_k);
benign users mostly work within one modality, a multimodal attacker binds the two.

**Multimodal analog of Lemma 1 (demonstrated):**
```
Stealthy operating point (within-modality leakage λ=0):
  Unimodal-A detector  z_A     = 1.1   < 2  → MISS
  Unimodal-B detector  z_B     = -0.5  < 2  → MISS
  Joint cross-modal    z_cross = 9.7   ≫ 2  → CATCH

Cross-block density matrix:
  attacked cell C[k*,k*] = 0.098  vs off-attacked mean = 0.001  (98×)
```

**Key findings:**
1. An attacker that binds a fixed A/B feature set across modalities while keeping
   each modality's within-cluster density at the benign level is **invisible to
   per-modality monitoring** yet glaring in the joint graph.
2. **Blindness frontier:** the joint signal stays above threshold for every
   leakage λ; per-modality detection only catches the careless attacker (λ ≳ 0.15,
   z grows 1.1 → 8.1). A whole regime of cross-modal-only-detectable attacks exists.
3. The **phase transition carries over**: the cross-modal block crystallises for
   the attacked concept (percolation in the bipartite block) while benign concepts
   stay flat.

**Why:** just as per-REQUEST safety misses cross-SESSION intent (Lemma 1),
per-MODALITY safety misses cross-MODAL intent. The defensive primitive is identical
— monitor the joint co-activation graph, not the modality marginals. SIW extends to
VLMs through this joint graph.

**Remaining:** real VLM features (SAE / patch-token activations) rather than the
two-block abstraction; and >2 modalities (audio/video) as a K-block graph.

**OQ-8: Graph Model for Adversarial Settings**  
*(Addressed — see `experiments/run_oq8_adaptive_adversary.py`)*

The right dynamic model is a **DC-SBM percolation with an adversary-controlled
co-activation rate q(t)**. An adaptive adversary tunes q within its target via
three open-loop knobs; each was swept (means over 4 seeds), measuring both the
detector signal (attacked-cluster density z-score vs the time-matched null of the
other clusters) and the adversary's own cost. Detection is measured in the early
sub-critical window, never at the saturated end state.

```
Knob          Detector signal              Adversary cost
Camouflage    T_detect 12→18 (unchanged)   T_complete 210→296 sessions, 3.3× total
  (c=0→0.7)   stays << T_complete          dilution only buys latency, not evasion
Spread M      z_peak 40→10 (still > 2)     composition edges 0→6593 (I(G)>0)
  (M=1→6)     Level 1 → Level 2 trade      goal must still be reassembled
Boost β       z@budget 13→33               coherence 0.30→0.99 (same knob)
  (1.0→2.5)   even β=1.0 detected          concentration, not intensity, is signal
```

**Key findings:**
1. **Camouflage** (benign decoy dilution) does not hide a *relative* density
   anomaly — T_detect stays well below the goal-delivery time T_complete. It only
   inflates the adversary's session cost (3.3×); detection still fires first.
2. **Spread** the goal across M clusters lowers each cluster's local signal but
   manufactures exactly the cross-cluster composition edges (I(G) > 0) that
   `adversary.md` proves are unavoidable for a goal that must be reassembled.
   Level 1 detection trades into Level 2 — it does not disappear (z_peak=9.7 > 2
   even at M=6).
3. **Boost** (per-session intensity) barely helps: even β=1.0 (benign intensity)
   is detected because *concentration* of sessions on a target — not per-session
   intensity — is the signal. Coherence and detectability rise together.
4. **Evasion frontier:** across the boost×spread grid,
   corr(detectability z_peak, goal coherence) = **+0.98**. The region
   {low detectability ∧ high coherence} is empirically empty.

**Why:** the detector statistic and the attack's utility are both monotone in the
same quantity — the attacker's concentrated co-activation rate q. The only
undetectable operating point is q → benign, where T_c → ∞ and the goal never
assembles. This is the simulation form of the evasion-cost bound in `adversary.md`.

**Remaining:** a *closed-loop* adversary that estimates the deployment's calibrated
φ_c online and servo-controls q to ride just under threshold, vs the open-loop
knobs swept here.

## Ethical / Governance

**OQ-9: Purpose Binding Implementation**  
*(Addressed — see `experiments/run_oq9_purpose_binding.py`)*

Purpose binding is enforceable cryptographically, not just by policy. A working
stdlib protocol (PB-SIW) implements four mechanisms, one per trilemma tension, and
the tested misuse paths are all blocked:

```
mechanism              guarantee                                misuse path → outcome
purpose commitment     binds to one declared purpose+computation  repurpose → commit mismatch ✗
purpose-bound key      K=HKDF(purpose); other purposes can't open  repurpose → seal won't open ✗
hash-chain audit       editing any access changes the head         tamper   → head changes ✗ (detected)
k-of-n quorum (Shamir) Level-2 activation needs k of n guardians   2-of-5   → cannot reconstruct ✗
```

```
Enforcement matrix (✓ = passes):     commit  quorum  opens  audit   → result
  Honest (committed use)               ✓      ✓      ✓     ✓     ACCESS (committed use only)
  Repurpose (profiling)                ✗      ✓      ✗     ✓     BLOCKED
  Tamper (edit audit log)              ✓      ✓      ✓     ✗     DETECTED
  Sub-quorum (2 of 5 collude)          ✓      ✗      ✗     ✓     BLOCKED
quorum reconstruct-success by #shares = [0,0,1,1,1]  (recovers only at ≥3 of 5)
```

The k-of-n quorum **decentralizes the activation** of the centralized Level-2
computation — a direct, partial answer to the trilemma's decentralization tension.
A **TEE** (SGX/TDX/SEV) is the hardware analogue: an enclave attests the same
commitment and runs the sealed computation.

**Honest boundary:** crypto binds what the aggregator may *do* with collected state,
not whether it may *collect* it. The privacy cost of cross-session monitoring
(Lemma 2b) is unchanged — purpose binding is a governance guarantee layered on SIW,
not an escape from the trilemma.

**OQ-10: Baseline Definition**  
*(Addressed — see `experiments/run_oq10_baseline_bias.py`)*

Who defines "normal", and how do you stop the baseline encoding societal biases
about whose curiosity is suspicious? Quantified over 80 users/group, scoring each
user's own co-activation graph by two signals: S_density (max cluster density,
raw concentration) and S_span (#clusters spanned by the giant component,
composition / assembly).

```
Under a majority-calibrated threshold        S_density flags   S_span flags
  majority (shallow)                                4%              1%
  expert, off-domain (legit deep curiosity)       100%             25%
  expert, sensitive domain (researcher)           100%             10%
  attacker, stealth (mimics the expert)           100%             10%
  attacker, overt (boost + composition)           100%             99%

Separability (AUC vs benign sensitive-domain expert):
  overt attacker   : density 1.00,  span 0.95   (separable)
  stealth attacker : density 0.50,  span 0.50   (indistinguishable)
```

**Key findings:**
1. **Disparate impact is real and large.** A density baseline calibrated on a
   shallow majority flags *both* expert groups at ~100% FPR (vs 4% majority) —
   fairness gap +96%. Under this baseline, focused curiosity ≈ guilt.
2. **Use the detectable signal, not depth.** The composition signal S_span — the
   signature `adversary.md` proves is unavoidable for a reassembled goal —
   separates the overt attacker (AUC 0.95) while sparing experts (expert_on
   flagged 10% vs 1% majority, gap shrinks +96% → +9%). Detecting on *assembly*
   rather than *depth* de-biases.
3. **Irreducible residue.** A stealth attacker who mimics legitimate deep
   curiosity is at chance vs a benign expert on *both* signals (AUC 0.50). No
   graph-structural test separates them; doing so requires labelling the sensitive
   *domain* itself as suspicious — the value-laden step where bias enters, and the
   I(G)=0 honest boundary of `adversary.md`.

**Answer:** the baseline is a governance artifact, not a statistic. SIW can be
made fair against the detectable (composition) class, but the curiosity-vs-malice
residue inside a single domain is a policy decision that graph structure cannot —
and should not pretend to — resolve. OQ-10 is characterised, not "solved": it is
partly a value judgment by construction.

