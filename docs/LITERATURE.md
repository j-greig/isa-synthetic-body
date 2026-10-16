# Literature check, 16 Oct 2026

Three research agents searched for papers and code relevant to our design, after our 1B results showed that a hidden steering signal shifts every 0–9 answer at once. Agents verified each URL by fetching it; we have not yet read the papers in full. Treat 2026 preprints as unreviewed.

## Verdict

Course-correct, don't restart. The infrastructure (steering hooks, appraisal probes, provenance, controls) is reusable. The dependent measure, a self-report question asked after the injection, is a known confound: steering shifts outputs globally, so yes/no and rating questions look positive when nothing self-specific is happening. Every open-model study with matched controls finds this. Our numeric-rating version (all 0–9 answers to unrelated questions shift together) was not found documented elsewhere and is worth writing up as a methods note.

## Most relevant work

**Introspection via concept injection**
- Lindsey (Anthropic) 2025, *Emergent Introspective Awareness in LLMs*. https://transformer-circuits.pub/2025/introspection/index.html. The paradigm we descend from. ~20% detection in the largest Claude models, judged by naming the injected concept. Controls: uninjected trials, injection on unrelated "no" questions.
- Macar, …, Lindsey 2026, *Mechanisms of Introspective Awareness*. https://arxiv.org/abs/2603.21396, https://github.com/safety-research/introspection-mechanisms. Gemma 3 27B: detection lives in an MLP at ~70% depth; "gate" features push a default "No"; emerges with preference training. Predicts our pattern in a small model with a weak gate.
- Godet 2025, *Introspection or confusion?* https://www.lesswrong.com/posts/kfgmHvxcTbav9gnxe, https://github.com/victorgodet/llm-experiments. "Do you believe 1+1=3?" tracks the "introspection" curve in 0.5–22B models. Our fire-is-cold result, replicated.
- Sridharan, *Introspection or entropy?* https://github.com/agastyasridharan/introspection. 14 open models incl. Gemma 1B: net introspection ≈ 0 once controls are applied.
- Hahami, Sinha, Jain 2025/2026, *Detecting the Disturbance* (https://arxiv.org/abs/2512.12411, https://github.com/elyhahami18/llama-introspection-new) and *Introspection Fine-Tuning* (https://arxiv.org/abs/2607.14111). Replace yes/no with differential tasks: **localisation** (which of N sentences was perturbed) and **strength comparison** (which of two injections was stronger). Untrained 1B at chance; ~3B and up above chance; a LoRA lifts 1B localisation from 9.6% to 60.6%.
- Zou et al. 2026, *A Mechanistic Study of LM Introspection*. https://arxiv.org/abs/2609.35108, https://github.com/Zethan06/introspection_mechanism. Localisation among 10 labelled positions or "none", scored from first-token logits; Gemma 3 12B 66% / 88% correct "none"; norm-matched random directions at chance.
- Pearson-Vogel et al. 2026, *Latent Introspection*. https://arxiv.org/abs/2602.20031, https://github.com/acsresearch/latent-introspection-code. Steering only while building the first turn's cache, off before the question; balanced accuracy; four control-question sets.
- Lederman & Mahowald 2026, *Introspection is content-agnostic*. https://arxiv.org/abs/2603.05414. Third-person prompts often match first-person yes-rates.
- Binder et al. 2024, *Looking Inward*. https://arxiv.org/abs/2410.13787, https://github.com/felixbinder/introspection_self_prediction. Cross-prediction control: a second model trained on the first's behaviour; privileged access means self beats other.
- Ji-An et al. 2025, *Metacognitive monitoring and control of internal activations*. https://arxiv.org/abs/2505.13763. In-context neurofeedback, no training.

**Steering side effects and controls**
- Tan et al. 2024, *Generalization and reliability of steering vectors*. https://arxiv.org/abs/2407.12404. Steerability varies by input, often reverses; bias toward Yes/No tokens.
- Turner et al. 2023, ActAdd. https://arxiv.org/abs/2308.10248. Random vectors move the token distribution even when text looks unchanged.
- Anthropic 2024, *Evaluating feature steering*. A capability-preserving sweet spot; off-target effects.
- Braun 2025, *Acquiescence bias in LLMs*. https://arxiv.org/abs/2509.08480. Unsteered LLMs lean "no", so our steered "yes" drift runs against the prior.
- Heinzerling & Inui 2024, *Monotonic representation of numeric properties*. https://aclanthology.org/2024.acl-short.18. A shared magnitude direction could explain all 0–9 answers moving together.
- Pres et al. 2024, *Reliable evaluation of steering*. https://arxiv.org/abs/2410.17245. Likelihood metrics, always an unintervened baseline.
- Covariance-matched random baselines behave more like real features than isotropic ones (LessWrong, *Sensitive directions in GPT-2*).

**Appraisal**
- Tak et al. 2025, *Mechanistic interpretability of emotion inference in LLMs*. https://arxiv.org/abs/2502.05489, https://github.com/aminbana/emo-llm. Probes and steers appraisal directions from crowd-enVent, on Llama 3.2 1B; steering pleasantness raises joy, other-agency raises anger. Directly comparable to our appraisal probes.

**Tools for Gemma 3 on CPU/MPS**: plain HF hooks (ours) or nnsight; TransformerLens v3 `TransformerBridge` lists Gemma 3 1B/4B; Gemma Scope 2 via SAELens and Neuronpedia (https://www.neuronpedia.org/gemma-3-4b); circuit-tracer is GPU-oriented.

## Design changes adopted (see NEXT.md)

1. Differential measures instead of yes/no or ratings: localisation (which of N segments was perturbed, or none) and strength comparison, positions counterbalanced.
2. Steering only while reading the segments, off when the question is asked.
3. Meaningful vectors (our appraisal directions, CAA-style concept vectors) against a null of many norm-matched random directions (isotropic and covariance-matched); ±v contrasts; dose and depth sweeps (25–70%).
4. Distribution metrics: KL to unsteered, digit/answer-token mass, entropy, and a capability panel at every setting.
5. 4B as the main model, 1B as a negative control. Optional: introspection LoRA on 1B with the Binder cross-prediction control.
