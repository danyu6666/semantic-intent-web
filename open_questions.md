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

**OQ-4: Adversarial Topology**  
Can a θ₂ adversary deliberately shape their semantic footprint to maintain
cluster density below the anomaly threshold while achieving attack goals?
What is the cost of this evasion strategy?

**OQ-5: Privacy-Utility Curve**  
Quantify the exact relationship between ε_p and detection rate for the
two-level architecture. Where is the practical operating point?

## Theoretical

**OQ-6: Lemma Formalization**  
The three lemmas (minimax, DP composition, decentralization) are stated
as sketches. Full formal proofs are needed, particularly Lemma 2 (SMPC case).

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

