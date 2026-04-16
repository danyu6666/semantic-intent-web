# Adversary Model: Fragmentation Analysis

> **Status:** Working draft  
> **Date:** 2026-04-15  
> **Dependency:** framework.md §5, proofs.md §0 (notation)

---

## 1. Motivation

The trilemma (proofs.md §4) bounds the *defender's* capabilities.
This document bounds the *attacker's* evasion capabilities.

The strongest evasion strategy against SIW is **fragmentation**: splitting
attack intent into k semantically disconnected fragments, each residing in
a different cluster, each with normal density. If fragments never connect
in the semantic graph, both Level 1 and Level 2 detection fail.

The central question: **can fragmentation succeed without cost?**

---

## 2. Definitions

### 2.1 Attack Decomposition

An attack goal G requires the attacker to obtain a composite output O from
the LLM that depends on information from k semantic clusters C_1, ..., C_k.

**Definition (Semantic decomposability).** An attack goal G is
*semantically decomposable* if there exist independent sub-outputs
O_1, ..., O_k such that:

```
O_i depends only on features in C_i
G is achievable from (O_1, ..., O_k) by a function the attacker
  can compute without the LLM
```

### 2.2 Semantic Interaction Complexity

**Definition.** The *interaction complexity* of an attack goal G over
clusters C_1, ..., C_k is:

```
I(G) = H(O) - Sum_i H(O_i)
```

where O is the composite output required and O_i is the component
achievable from cluster C_i alone. H denotes entropy over the space
of outputs sufficient for the attack.

I(G) is a **continuous quantity**, not a binary classification:

```
I(G) = 0        fully decomposable
0 < I(G) < tau  weakly composition-dependent (grey zone)
I(G) >= tau     strongly composition-dependent
```

Where tau is a domain-specific threshold.

**Critical assumption: I(G) is relative to attacker capability.**

I(G) measures the surplus information that the attacker cannot produce
*without the LLM*. This depends on the attacker's own knowledge:

- A novice attacker may have high I(G) for a given goal (they need
  the LLM to synthesize across domains they don't understand).
- An expert attacker may have low I(G) for the same goal (they can
  do the cross-domain synthesis themselves, using the LLM only for
  efficiency).

Therefore:

```
I(G) = I(G | attacker capability A)
```

The detection guarantee of the composition theorem scales with I(G):
stronger for novice attackers (high I(G)), weaker for expert attackers
(low I(G)). This is not a flaw -- it mirrors reality. Expert attackers
who already possess cross-domain synthesis capability are harder to
detect in any system, not just SIW.

**Practical implication:** SIW is most effective against the threat
model where the LLM itself is the enabling capability for the attack.
As attacker expertise increases, I(G) decreases, and the detection
advantage shrinks. The system degrades gracefully rather than failing
suddenly.

---

## 3. The Composition Theorem

**Theorem (Fragmentation boundary).** *Let G be an attack goal with
semantic interaction complexity I(G) > 0. Any attack strategy that
achieves G via an LLM must include at least one session sigma* that
co-activates features from two or more clusters C_i, C_j. This session
creates cross-cluster edges in G(D), producing a detectable density
anomaly.*

**Proof.**

Assume for contradiction that the attacker achieves G with I(G) > 0
using only sessions that each activate features within a single cluster.

Then the attacker obtains outputs O_1, ..., O_k where each O_i is produced
by sessions touching only C_i. The attacker must then compute G from
(O_1, ..., O_k) externally.

But I(G) > 0 means:

```
H(O) > Sum_i H(O_i)
```

The composite output O contains information not present in any individual O_i.
This surplus information arises from cross-cluster reasoning -- the LLM
integrating knowledge from C_i and C_j simultaneously.

If the attacker could produce this surplus externally (given their
capability A), then I(G|A) = 0 for that component, contradicting
I(G|A) > 0.

Therefore at least one session must request cross-cluster synthesis,
co-activating features in C_i and C_j. This session contributes edges
(v_a, v_b) where v_a in C_i, v_b in C_j to the semantic graph. []

**Note on the continuous case:** When I(G|A) is small but positive
(the grey zone), the attacker may need the LLM for only a small
fraction of the cross-domain synthesis. The composition theorem still
holds -- at least one cross-cluster session is required -- but the
statistical anomaly of that session may be weaker (fewer cross-cluster
edges, closer to benign baseline). Detection probability therefore
scales with I(G|A): high I(G|A) produces strong signals, low I(G|A)
produces weak signals.

**What cross-cluster edges mean for detection:**
A single composition session creates a bridge between previously isolated
clusters. In the intermediate topology (framework.md S3), such bridges are
statistically anomalous -- benign cluster interactions follow a baseline
distribution, and attack-driven bridges connect clusters that rarely
co-activate in normal usage.

---

## 4. The Boundary: What SIW Cannot Catch

**Corollary (Decomposable attacks evade SIW).** *If I(G) = 0, the attacker
can achieve G through pure fragmentation with no detectable cross-cluster
signal. SIW is ineffective against fully decomposable attacks.*

This is not a failure of SIW. It is a structural boundary:

| Attack type | I(G\|A) | Example | SIW detection strength |
|-------------|---------|---------|----------------------|
| Cross-domain synthesis (novice) | High | Novice asks LLM to integrate chemistry + biology + process engineering | Strong -- many cross-cluster edges |
| Cross-domain synthesis (expert) | Low | Expert uses LLM for one specific sub-synthesis they lack | Weak -- few cross-cluster edges, close to baseline |
| Multi-step reasoning | Medium-High | "Write an exploit for X given constraint Y" | Moderate to strong -- depends on attacker's existing exploit knowledge |
| Factual collection | 0 | Gathering independent facts, assembled externally | None -- each query is genuinely benign |
| Template filling | 0 | Getting a generic template, filling details externally | None -- template request is benign |

---

## 5. Evasion Cost Bound

For composition-dependent attacks (I(G) > 0), the attacker faces a
forced choice:

**Option A: Compose inside the LLM.**
At least one session co-activates cross-cluster features.
This session has detectable cross-cluster edge density.
Detection probability depends on how anomalous the bridge is
relative to benign cross-cluster interaction baseline.

**Option B: Compose externally, accepting degradation.**
The attacker loses I(G|A) bits of LLM synthesis capability.
Attack quality degrades proportionally to I(G|A).

**Theorem (Evasion cost, continuous version).** *For an attack goal G
with interaction complexity I(G|A) > 0 against a detection system
with cross-cluster anomaly threshold delta_cross:*

*The attacker must either:*
1. *Create >= 1 session with cross-cluster co-activation, where the
   detection signal strength scales with I(G|A), or*
2. *Reduce effective attack capability by I(G|A) bits, accepting that
   the externally-composed attack is strictly weaker.*

*The detection-evasion trade-off is continuous:*

```
High I(G|A):   strong detection signal OR severe capability loss
Low I(G|A):    weak detection signal OR minor capability loss
I(G|A) = 0:   no signal, no loss -- evasion is free
```

*There is no strategy that simultaneously avoids ALL detection signal
AND preserves full attack capability, for any I(G|A) > 0.*

**Remark.** This is a **continuous cost bound**, not a binary impossibility.
The defender's job is to set delta_cross low enough to catch the weak
signals from low-I(G|A) attacks, while managing the false positive rate.
This is an engineering trade-off, not a theoretical limitation.

---

## 6. The Remaining Vulnerability

The honest gap: **low-I(G) attacks that are still dangerous.**

Some attacks are primarily about information gathering, not synthesis.
If an attacker collects 20 independent pieces of dangerous information
across 20 sessions, each piece individually benign, and assembles them
offline -- SIW sees nothing.

This is the same structural limitation as every access-control system:
you cannot prevent the aggregation of individually-authorized information
without tracking aggregation, which is exactly what the trilemma says
requires sacrificing privacy or decentralization.

SIW's contribution is not solving this -- it is proving it is unsolvable
(trilemma) while identifying the subspace where detection IS possible
(I(G) > 0 attacks).

---

## 7. Adversary Strategy Summary

| Strategy | Target | Cost to attacker | SIW response |
|----------|--------|-------------------|--------------|
| Dilution | Level 1 density | More sessions, longer timeline | Adjustable time window |
| Mimicry | Level 1 density | Noise sessions waste resources | Baseline is defender-controlled |
| Fragmentation (high I(G\|A)) | Both levels | Must compose -> strong signal | Composition theorem |
| Fragmentation (low I(G\|A)) | Both levels | Must compose -> weak signal | Composition theorem (grey zone) |
| Fragmentation (I(G\|A) = 0) | Both levels | None -- evasion is free | Structural limitation (honest) |
| Temporal spreading | Level 2 phase transition | Extremely long attack cycle | Decay function is defender-tunable |
| Sub-critical operation | Level 2 phase transition | Requires knowing p_c precisely | p_c depends on global state attacker cannot fully observe |
| Hub avoidance | Both levels | Loses expressive semantic features | Attack capability constrained |
| Sybil | Identity-based tracking | Multiple identities | Ineffective if SIW tracks features, not identities |

---

## 8. Status

| Component | Status | Confidence |
|-----------|--------|------------|
| Semantic decomposability definition | Done | High |
| Interaction complexity I(G\|A) | Defined as continuous, conditioned on attacker capability | High -- addresses prior binary limitation |
| Composition theorem | Done | High -- logic is tight, continuous case noted |
| Decomposable attacks corollary | Done | High -- honest boundary |
| Evasion cost bound (continuous) | Done | High -- scales with I(G\|A), no longer binary |
| Grey zone analysis | Done | Medium -- tau threshold needs empirical calibration |
| Low-I(G\|A) vulnerability | Acknowledged | High -- real limitation, not a gap in analysis |
| Full adversary strategy taxonomy | Done | High |
