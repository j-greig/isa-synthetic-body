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
