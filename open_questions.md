# Open Questions

For the research community to engage with.

## Technical

**OQ-1: Topology Model**  
What random graph model best characterizes semantic co-activation in LLM
interactions? Simulation shows intermediate topology (R²=0.812 for power-law,
ER rejected). A new model may be needed.

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

