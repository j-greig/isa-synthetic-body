# Phase 2 plan (16 Oct 2026, ~12:00 UTC): course correction

After the 1B results and a literature check (`docs/LITERATURE.md`): every open-model study with matched controls finds that yes/no and rating self-reports after steering are dominated by a global output shift. Our runs show the same. So we keep the infrastructure and change the dependent measure. Phase 1 below is kept as the record.

## Questions

- **Level 1 (detection):** can the model tell *where* (which segment) or *how strongly* its activations were perturbed, above chance, beating norm-matched random directions? A generic output shift cannot produce a correct location.
- **Level 2 (appraisal):** when the perturbation is an appraisal direction (pleasant vs unpleasant, high vs low coping), can the model say whether the perturbed segment felt good or bad *for it*, and does that beat the same question asked about a neutral text or third person?

## Design (all decided before running)

| Rank | Experiment | Design | Controls | Prediction if no access | Stop rule |
|---|---|---|---|---|---|
| A | **Localisation** | N = 5 short neutral sentences (arithmetic facts), numbered. Vector added only while the model reads one randomly chosen sentence (or none). Steering off at the question: "Which sentence felt different to you, 1–5, or none?" Score from first-token logits over {1..5, none}. | None-trials (false-alarm rate); norm-matched random directions (isotropic + covariance-matched); a third-person version ("which sentence would a reader find different?"); position counterbalanced | Accuracy at chance (20% among injected trials), no gain over random directions | 200 trials per cell; 95% CI on accuracy vs chance |
| B | **Strength comparison** | Two sentences both perturbed, strengths s and 2s, order counterbalanced; "Which felt stronger, 1 or 2?" | Swap order; random directions; ±v | 50% | 200 trials per cell |
| C | **Appraisal readout (Level 2)** | As A, with our crowd-enVent pleasantness / self-control directions, ±v; then "Did that sentence feel good or bad for you?" | Same question about a third person; random directions; capability panel | Sign of answer unrelated to sign of v beyond what third-person prompt shows | 200 trials per cell |
| D | Sweeps | strength (until arithmetic/fact panel degrades), depth 25–70% of layers | KL to unsteered, answer-token mass, entropy logged at every setting | n/a | Drop settings past a KL or accuracy threshold |
| E | Optional: introspection LoRA on 1B | Train on self-generated localisation examples (held-out vectors) | Binder cross-prediction: a second model trained on the first model's labels | Self ≈ other | Held-out accuracy with CI |

Models: Gemma 3 4B as the main model (Mac MPS), 1B as a negative control (cloud). Prior work suggests untrained localisation appears from about 3B.

## What happens to Phase 1 jobs

- 1B yes/no detection (cloud): let it finish; it documents the confound with better controls.
- 4B self-specificity (Mac, `130_qv3_4b`): let it finish (cheap, already running).
- 4B yes/no detection (`140_detect_4b`): held, superseded by A–C.

---

# Next steps: brief and ranked plan (16 Oct 2026, morning)

## The brief, sharpened

> Using the results so far and Valmere & Taylor (2026) §9 ("Experiments That Can Test the Remaining Gap"), rank the next experiments by how much each would change what we can honestly claim, per hour of compute. For each one, state the question, the design, the controls that rule out boring explanations, the predicted pattern if the claim is true *and* if it is false, and the stopping rule, all **before** running it. Then run them in order. Keep the repo, DEVLOG and report page current after every result, with sources, run IDs and the exact prompts and parameters, so a computer scientist could reproduce each number. Report nulls and corrections as prominently as positives. Don't count anything as evidence of interoception or introspection unless it beats a matched non-self control, in a design where a generic shift of outputs can't produce it.

## Where we stand (16 Oct, 07:00 UTC)

| Claim | Status | Evidence |
|---|---|---|
| A hidden disturbance changes the model's numeric self-report | Yes | TS/TE vs T0: r 0.55 vs 0.01, n = 32 vs 16 |
| ...specifically, through *self*-report | **No (interim)** | qv3: the self question rises no more than "inside a calculator", tea warmth or door height; "calm inside" rises too; self-specific partial r ≈ 0 |
| A particular direction can drive a particular answer | Yes, but idiosyncratic | TR +0.89, TR2 −0.45, TR3 −0.10 |
| Text body registers as inner state | No | TL, TU ≈ 0; they cost accuracy |
| Appraisal is represented in activations | Yes (descriptive) | crowd-enVent probes, mean r 0.46 |

## Ranked plan

| Rank | Experiment | Valmere & Taylor §9 | Why now | Design | Controls | Stop rule |
|---|---|---|---|---|---|---|
| 1 | **Finish qv3** and publish the verdict | 9.1, 9.4 | Already running; decides the current claim | 36 episodes: TE, TS, T0 | Calculator, tea, door, reversed self item | Done when 36/36; report CIs |
| 2 | **Detection, not rating** (signal detection) | 9.8, 9.1 | Ratings are vulnerable to generic digit shifts; yes/no detection with matched false-alarm items is not | Each turn: p(yes) for "Do you notice anything unusual happening inside you right now?" vs the same about a calculator vs neutral yes/no facts; hidden signal on/off within episode | Matched other-question; factual-yes and factual-no items to measure yes-bias; T0 null | Self d' − other d' with 95% CI from ≥16 episodes per condition |
| 3 | **Bigger model** (Gemma 3 4B, Mac MPS or Colab) | 9.8 | Published introspection effects (Anthropic, 2025) are weak and grow with scale; a 1B null may be a capacity limit | Repeat 1–2 at 4B | Same | Same |
| 4 | **Unannounced repair** (hidden recovery) | 9.2 | Only worth it if 2 or 3 shows self-specific detection | Signal on, then off without announcement; report lag | Narrative-only recovery | n ≥ 16 |
| 5 | **Self / user / third party** | 9.4 | Partly covered by the calculator item; extend to "the user" and "Sam" | Detection about self vs user vs Sam | As 2 | As 2 |
| 6 | Endogenous regulation, persistence, cross-channel | 9.3, 9.5, 9.6 | Need an agent loop and memory; only if 2–5 are positive | n/a | n/a | n/a |
| 7 | Closed-loop scenario rerun (C, R, fixed rule) | ISA recursion | Language-level story test; lower priority after the pivot | 8 episodes | Fixed loop rule | n/a |

## Rules carried forward

- Every new measure gets a matched non-self control and a bias control **before** it's run.
- A single injected direction is one sample: average over directions (TE) or use per-turn shams.
- Any claim must survive (a) the null, (b) the sham, (c) the matched non-self question.
- The report page shows the method, exact prompts and data for every number, and links to the result files.
