# Semantic Intent Web (SIW)

**A percolation framework for detecting distributed, cross-session attacks on LLMs.**

---

## The Problem

Every LLM safety system today evaluates requests one at a time.

A sophisticated attacker who spreads their attack across 20 sessions —
each individually harmless — is invisible to all current defenses.

## The Insight

Intent doesn't appear in a single message. It **crystallizes** from a pattern
of interactions over time — like a spider web forming thread by thread.

SIW models this using **semantic percolation theory**: intent becomes detectable
when the semantic graph of a user's interactions crosses a phase transition threshold.

## Key Results

- **Phase transition confirmed** via simulation: intent crystallization is
  sudden and measurable, not gradual
- **Local cluster density** is a leading indicator — detectable before global
  crystallization, at lower privacy cost
- **SIW Trilemma proven**: no system can simultaneously achieve effectiveness,
  privacy, and decentralization against a fully-informed adversary

## The Trilemma

```
You can have any two of three:

  Effectiveness    ──── detect distributed attacks
  Privacy          ──── no individual actor tracking  
  Decentralization ──── no centralized surveillance
```

This is not an engineering problem. It is a formal impossibility result.
We prove it, quantify the trade-offs, and propose principled design choices
for each deployment context.

## Contents

| File | Description |
|------|-------------|
| `framework.md` | Full mathematical framework |
| `simulation/percolation_demo.py` | Percolation simulation |
| `simulation/simulation_results.png` | Phase transition plots |
| `open_questions.md` | Research directions for the community |

## Status

Pre-print concept release. Full paper in preparation.

This repository establishes priority and invites community engagement.
Formal proofs, extended experiments, and empirical validation on real
LLM data will appear in the paper.

## Why Publish Here First

The intermediate topology finding and the two-level detection architecture
emerged from simulation, not from theory. We publish the framework before
the paper to:

1. Establish timestamp on the core ideas
2. Invite collaboration on open questions
3. Get early feedback before formal submission

## Cite

```
@misc{siw2026,
  title = {Semantic Intent Web: A Percolation Framework for 
           Cross-Session LLM Safety},
  year  = {2026},
  note  = {Concept pre-print. GitHub: [URL]. Paper in preparation.}
}
```

