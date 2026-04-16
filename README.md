# Semantic Intent Web (SIW)

**A semantic graph framework for detecting distributed attacks on Large Language Models.**

---

## Motivation

Current LLM safety systems evaluate each request independently:

```
Request  ->  Evaluate  ->  Allow / Reject
```

This design creates a structural blind spot.

A sophisticated attacker can distribute an attack across many sessions:

```
session 1  ->  harmless
session 2  ->  harmless
session 3  ->  harmless
...
combined   ->  malicious outcome
```

Because each prompt appears benign in isolation, existing defenses cannot detect the attack.

SIW proposes that **intent emerges as a network structure** across interactions,
rather than being contained in a single prompt.

---

## Core Idea: Intent Crystallization

Intent does not appear instantly.
It **crystallizes** from patterns of semantic interactions —
like a spider web forming thread by thread.

```
interactions
      |
semantic feature activation
      |
co-activation graph
      |
structural emergence
      |
intent crystallization
```

When the semantic interaction graph crosses a critical structural threshold,
malicious intent becomes detectable.

---

## SIW Hypothesis

> Intent emerges as a detectable structural transition in semantic
> interaction networks.

This means:

```
intent  !=  single interaction
intent   =  network structure
```

---

## Three-Layer Theory

SIW organizes as three layers of analysis:

```
Layer 1 — Semantic Interaction Dynamics
         How interactions generate semantic structures
                        |
Layer 2 — Structural Emergence
         When clusters densify and phase transitions occur
                        |
Layer 3 — Intent Inference
         What structural signals reveal about intent
```

---

## Semantic Intent Graph

Each interaction activates semantic features.
The system constructs a temporal graph:

```
G(t) = (V(t), E(t))

V(t) = activated semantic features
E(t) = co-activation edges
```

**Global intent crystallization** occurs when:

```
|C_max(t)| / |V(t)|  >=  phi_c
```

Where C_max = largest connected component, phi_c = crystallization threshold.

**Local intent crystallization** occurs when:

```
rho_k(t) / rho_baseline  >  delta_k
```

Where rho_k = cluster edge density, detectable *before* global crystallization.

---

## Key Results

### Simulation (synthetic)

```
Empirical p_c         ~= 0.002196
ER prediction         ~= 0.002000
Attack cluster degree    37.0
Benign cluster degree    21.9
Density difference       +69%
```

### Early validation (real embeddings)

Phase transition confirmed using sentence-transformer embeddings
(`all-MiniLM-L6-v2`) and Ollama/Mistral embeddings on semantic
co-activation graphs. See `experiments/`.

### Theoretical

- **SIW Trilemma proven**: no system can simultaneously achieve
  effectiveness, privacy, and decentralization
- **Composition theorem**: fragmentation attacks have incompressible cost
  that scales continuously with interaction complexity I(G|A)
- **Honest boundary**: attacks with I(G|A) = 0 evade SIW — and we prove
  this is unsolvable, not just unaddressed

---

## Two-Level Detection Architecture

```
Level 1 — Local Detection
  monitor cluster density anomalies
  + earlier signal
  + lower privacy cost
  + decentralizable

Level 2 — Global Detection
  monitor semantic phase transition (|C_max|/|V|)
  + strong confirmation signal
  + detects multi-cluster attacks
  - higher privacy cost
```

Local crystallization precedes global crystallization by ~60% of time.

---

## The SIW Trilemma

```
No system can simultaneously achieve all three:

  Effectiveness     ----  detect distributed attacks
  Privacy           ----  protect individual interactions
  Decentralization  ----  avoid centralized monitoring

You can satisfy at most two.
```

This is not an engineering problem. It is a formal impossibility result.
See `proofs.md` for the full proof.

---

## Research Perspectives

SIW connects three research communities:

| Perspective | Core question |
|-------------|---------------|
| **AI Safety** | Can distributed attacks be detected via interaction structure? |
| **AI Interpretability** | Do LLM feature activations form meaningful semantic graphs? |
| **Network Science** | What graph dynamics govern intent crystallization? |

---

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Do semantic interaction networks exhibit structural transitions corresponding to intent formation? |
| RQ2 | Can local cluster densification serve as an early signal of intent crystallization? |
| RQ3 | Do malicious interactions produce measurably denser semantic clusters than benign ones? |
| RQ4 | Do tasks requiring cross-domain reasoning produce detectable cross-cluster bridges? |
| RQ5 | Which semantic feature representations best reveal crystallization dynamics? |

---

## Repository Structure

```
SIW/
├── README.md
├── LICENSE                      -- CC BY 4.0
├── requirements.txt             -- Python dependencies
├── framework.md                 -- mathematical model
├── proofs.md                    -- formal proofs of the SIW Trilemma
├── adversary.md                 -- attack and evasion analysis
├── open_questions.md            -- open research problems
├── SESSION_LOG.md               -- development history
│
├── simulation/
│   ├── percolation_demo.py      -- synthetic percolation simulation
│   └── simulation_results.png   -- phase transition plots
│
└── experiments/
    ├── run_test.py              -- basic percolation validation
    ├── run_semantic_graph.py    -- sentence-transformer embedding graph
    ├── run_ollama_graph.py      -- Ollama/Mistral embedding graph
    ├── figure_percolation_test.png
    └── figure_semantic_graph.png
```

---

## Running the Experiments

### Synthetic simulation

```bash
python simulation/percolation_demo.py
```

### Embedding-based experiments

```bash
pip install -r requirements.txt
cd experiments
python run_semantic_graph.py
python run_ollama_graph.py   # requires Ollama with mistral model
```

---

## Status

Concept research release with:
- formal framework
- theoretical proofs (trilemma + composition theorem)
- adversary analysis
- synthetic simulation
- early embedding-based validation

Full paper in preparation.

---

## Citation

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

## Why This Repository Exists

This project is released early to:

1. Establish priority for the core ideas
2. Invite collaboration on open research questions
3. Encourage empirical validation with real LLM feature data

The goal is to explore network-based representations of intent in LLM systems.

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
You may share and adapt freely, but you **must give attribution**.
