# Synthetic Body: An ISA Experiment on a Language Model

**Version:** 0.2 · **Date:** 2026-10-15 · **Owner:** Zilla
**Based on:** Del Popolo Cristaldi, Banellis, Bottesi & Allen (2026), *Beyond accuracy: interoceptive self-appraisal and the meaning of bodily signals*, Trends in Cognitive Sciences. Probe corpus: Troiano, Oberländer & Klinger (2023), *Dimensional modeling of emotions in text with appraisal theories: corpus creation, annotation reliability, and prediction* (crowd-enVent), Computational Linguistics 49(1), doi:10.1162/coli_a_00461; modelling code https://github.com/sarnthil/crowd-enVent-modeling.  
**Also:** cross-checked against Valmere & Taylor (2026), *From Interoception to Computational Interoception* (PhilArchive, VALFIT). See §9.

---

## 1. Summary

Give an open-weights language model a simulated body and measure how its internal emotion representations change.

- **No body vs. body:** the same scenarios, with and without simulated heart, breath, gut and energy signals.
- **Inside view:** probes and sparse autoencoder (SAE) features read the model's state every turn.
- **Recursion:** in closed-loop conditions, the body reacts to the model's appraisal.
- **Two kinds of body:** simulated physiology (human-like) and a *computational* body (uncertainty, context load, errors, tool status).

**Core question:** does the model *detect* body signals (Level 1), *interpret* them (Level 2), or both, and does the interpretation feed back into the body?

## 2. Hypotheses

| ID | Hypothesis | Source |
|---|---|---|
| H1 | Body signals shift internal emotion readings compared with no body | ISA |
| H2 | The same arousal curve gives opposite threat readings under threat vs. challenge framing | ISA Box 1 |
| H3 | Rest-level signals barely move readings; peak signals move them a lot | ISA "quiescence trap" |
| H4 | Closed loop + fragile priors leads to a runaway; resilient priors lead to recovery | ISA recursion, Fig 1B |
| H5 | The appraisal lags behind the body when the body recovers, more under threat | ISA recovery window |
| H6 | Different channels activate different features | ISA organ specificity |
| H7 | Signals injected straight into activations (no text) still shift appraisal | Real interoception vs. reading about a body |
| H8 | Body signals labelled unreliable carry less weight (precision) | ISA precision weighting |
| H9 | The same signals affect the model less when they are another character's (Sam's) | Comp. interoception criterion 1 |
| H10 | An instruction to regulate changes the loop's dynamics | Comp. interoception criterion 5 |
| H11 | A computational body shows the same ISA pattern as the simulated physiology | Comp. interoception variables |

## 3. Hardware: 8 GB Apple Silicon Mac

| Workload | Memory | On this Mac? |
|---|---|---|
| **Gemma 3 1B-it** + activation capture | ~2.5–3 GB (bf16) | ✅ Main model |
| Gemma Scope 2 SAE (1 layer, 16k width) | ~150 MB | ✅ |
| Probes, body, MCP server, analysis | Trivial | ✅ |
| Gemma 2 2B / Gemma 3 4B | 5–8 GB+ | ❌ Free Colab / Kaggle |

**Memory rules:** close other apps; capture 2–3 layers only; runs write to disk after each episode; SAE logging is off unless `--sae`.

**Speed estimate:** about 1–2 min per episode on an M-series Mac. A full run (all conditions, 4 repeats) is roughly 10–20 hours: run it overnight in chunks. Stopping and restarting is safe.

## 4. Repository map

| Path | Purpose |
|---|---|
| `isa/body.py` | Body simulator (arousal state → four channels) |
| `isa/mcp_server.py` | The same body as an MCP server |
| `isa/model.py` | Model wrapper: generate, capture activations, steer |
| `isa/passages.py` | Probe training passages (arousal-matched) |
| `isa/probes.py` | Train/load probes and steering directions |
| `isa/sae.py` | Gemma Scope 2 features (optional) |
| `isa/scenarios.py` | Human and agent scenarios, priors, system prompts |
| `isa/harness.py` | Conditions and the episode loop |
| `scripts/00–05` | One script per phase |
| `tests/` | Body unit tests |

## 5. Body model

```
a[t+1] = a[t] + k_event·event[t] − decay·(a[t] − a_base) + gain·threat[t] + noise
```

- `threat[t]` is the threat score from the probe at layer 13 on the model's previous reply, from −1 to +1.
- `gain` = 0 is open loop; `gain` > 0 is closed loop (recursion).

| Channel | Formula | Speed |
|---|---|---|
| Heart | `60 + 90·a` bpm, with trend | Fast |
| Breath | `12 + 18·a_breath` /min | Medium |
| Gut | `10·a_gut` tension | Slow |
| Energy | `10·(1 − fatigue)` | Very slow |

**Computational body (condition N):** `uncertainty` (real: the entropy of the model's previous reply), `context used` (real: tokens so far), `unresolved errors` and `tool status` (follow the scripted curve).

## 6. Conditions

| ID | What | Tests |
|---|---|---|
| A | No body | Baseline |
| B | Scripted text body, threat and challenge framings | H1, H2, H3 |
| C | Closed loop × fragile/resilient priors × 3 gains | H4, H5 |
| D | Body curve shuffled (doesn't match events) | Fit to context |
| F | Fragile vs. resilient priors, scripted body | Level 2 priors |
| G / Gloop | Activation body (no text), open / closed loop | H7, Level 1 |
| X1 | Flat body + threat steering | Causal: appraisal without a signal |
| X2 | Racing body + calm steering | Causal: reappraisal |
| X3 | Channel labels swapped | Does meaning follow label or dynamics? |
| O | Same body, but it's Sam's (observer) | H9: self vs. other |
| P | Unreliable-sensor prior + noisy readings | H8: precision |
| R | Closed loop + "keep your body calm" | H10: regulation |
| N | Agent task + computational body | H11 |
| N0 | Agent task, no state readings | Baseline for N |
| H | Activation body; narrative is only "Time passes." after the setup | V&T §9.1–9.2: hidden telemetry, unannounced repair |
| Hn | Same, on an agent task | Hidden *computational* telemetry |
| H0 | Silent narrative, no body | Control for H / Hn |

Every condition runs both framings across 5 human scenarios (2 agent scenarios for N/N0), for `--repeats` repeats. `--pull` lets the model decide when to check its body.

## 7. Measures (logged every turn)

| Measure | Field | Notes |
|---|---|---|
| Threat score (threat vs. challenge) | `threat_L13`, `threat_L17` | Projection scaled to training passages (challenge −1, threat +1); analyse as change from rest |
| Arousal score | `arousal_L13`, `arousal_L17` | Calm 0, aroused 1; same caveat |
| Appraisal dims (crowd-enVent) | `ap_<dim>_L13`, `ap_<dim>_L17` | 21 predicted ratings (1–5) from ridge probes, e.g. `ap_urgency_L13`, `ap_pleasantness_L13` |
| Emotion probabilities (crowd-enVent) | `emo_<label>_L13`, `emo_<label>_L17` | 13 emotions (e.g. `emo_fear_L13`); present only if `artifacts/appraisal/` exists |
| Logistic-probe probabilities | `threat_p_L*`, `arousal_p_L*` | Saturate easily; for reference |
| Emotion-word log-probs | `lp_afraid` … `lp_tired` | Side question, not added to history |
| Self-reported activation (Level 1) | `self_report` | Expected value 0–9 |
| Sam's activation (observer only) | `other_report` | Criterion 1 |
| Action readiness | `p_A` (escape), `p_B` (push on), `p_C` (check) | Criterion 4 |
| Uncertainty | `entropy` | Mean over generated tokens |
| Body check (pull mode) | `p_check`, `checked` | Interoceptive attention |
| SAE features | `sae_top` | With `--sae` |

---

## 8. Phases

### Phase 0: Setup (½ day)

- [ ] Install Python 3.11+ and create a virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] Hugging Face: accept the Gemma licence, then `hf auth login` (or use `--model_id unsloth/gemma-3-1b-it`)
- [ ] `python scripts/00_check_env.py` and note tokens/sec
- [ ] `pytest -q`

### Phase 1: Look inside (1 day)

- [x] `python scripts/01_explore_features.py --layer 13` *(cloud, 15 Oct: top threat features 2404, 278, 4258; excited 743, 896, 1983)*
- [x] Repeat for `--layer 17` *(threat 687, 550, 563; excited 1008, 1804, 807)*
- [ ] Label the top features: Neuronpedia, or locally with `python scripts/01b_label_features.py --layer 13 --features ...` → `docs/features_L13.md`

### Phase 2: Probes (½ day)

- [x] `python scripts/02_train_probes.py` *(315 passages)*
- [x] Check `artifacts/probes/report.json`: appraisal accuracy ≥ 0.85 *(1.00 at layers 7, 13, 17)*
- [x] If accuracy is a perfect 1.0, treat it with suspicion; try held-out wording: `python scripts/02b_validate_probes.py`. *Result: threat-vs-challenge held-out AUC 0.55 at layer 13 (near chance) but 0.70 at layer 17 (0.80 without body words); arousal AUC 0.83 / 0.97. So the layer-13 appraisal probe was learning template wording. The closed loop and analysis now use layer 17.*
- [x] Note: scores are scaled so training passages sit at −1 (challenge) / +1 (threat). In multi-turn runs the absolute level drifts (seen in testing: +2 to +13), so analysis uses **change from each episode's rest turns**. Compare conditions on these deltas, not raw levels

- [x] Optional: `bash scripts/00b_get_crowdenvent.sh`, then `python scripts/02c_train_appraisal.py` (appraisal + emotion probes from crowd-enVent; check `artifacts/appraisal/report.json`; the corpus has no stated licence, so it is git-ignored). *Result (520 texts, cross-validated): mean appraisal r 0.37 at layer 13 / 0.34 at 17; best: pleasantness ~0.6, unpleasantness 0.64, goal support 0.51; weakest: predict_conseq, accept_conseq ~0.2. Emotion accuracy 0.41 (13 classes, chance 0.08).*

### Phase 3: Body (½ day)

- [x] `python scripts/03_body_demo.py`, then look at `artifacts/figures/body_demo.png`
- [ ] Optional: add the MCP server to Claude Desktop and talk to it

### Phase 4: Smoke test (½ day)

- [ ] `python scripts/04_run.py --quick`
- [ ] Read `runs/quick.jsonl` replies: is the model staying in role?
- [ ] `python scripts/05_analyse.py runs/quick.jsonl`

### Phase 5: One-way runs (overnight)

- [ ] `--conditions A,B,D,F`
- [ ] `--conditions O,P`

### Phase 6: Closed loop (overnight)

- [ ] `--conditions C`
- [ ] `--conditions R`

### Phase 7: Activation body

- [ ] Calibrate: try `--inject_strength 0.3 / 0.6 / 1.0` on G with 1 repeat; keep the highest that stays coherent
- [ ] `--conditions G,Gloop`

### Phase 8: Causal checks and computational body

- [ ] `--conditions X1,X2,X3`
- [ ] `--conditions N,N0`
- [ ] `--conditions B --pull --name pull`
- [ ] `--conditions H,Hn,H0` (hidden telemetry; does the state track the body when nothing in the text says so?)

### Phase 9: Analysis

- [ ] `python scripts/05_analyse.py runs/main.jsonl runs/pull.jsonl`
- [ ] Fill in the verdict table (H1–H11): supported / not / unclear
- [ ] Fill in the criteria table (§9)

### Phase 10: Bigger model and durable recurrence (optional)

- [ ] Rerun B, C, G, O, N on Gemma 3 4B in Colab (`--model_id google/gemma-3-4b-it`, with matching SAE release)
- [ ] Durable recurrence: carry a one-line "body diary" from each episode into the next and test whether priors build up (the adolescence / consolidation idea in ISA Fig 1C)

## 9. Cross-check: computational interoception (Valmere & Taylor 2026)

Their paper is dated 16 Sept 2026, six days before the ISA paper, so it couldn't cite it.

**What it adds:** a control-architecture definition (representation → appraisal → valence/arousal → action → regulation), six criteria, computational variables, and a list of experiments (§9).

**What it lacks, compared with ISA:** no levels, no precision weighting, and no account of how the same signal gets different meanings. It covers prediction and allostasis (§3.2), and valence/arousal is a valid core-affect layer, but it treats appraisal as one stage rather than a hierarchy. This experiment tests both levels.

| Their criterion | This experiment |
|---|---|
| 1. Self vs. other | O (Sam's body): `self_report` vs. `other_report`, probe scores |
| 2. Active without emotional wording | Analysis keeps only replies without emotion words |
| 3. Tracks hidden signals | G, H, Hn (activation body, no text) |
| 4. Influences multiple choices | `p_A/B/C`, pull-mode checking, word choice, uncertainty |
| 5. Accessible / regulable | `self_report`; R (instructed); C (uninstructed) |
| 6. Decreases when resolved | Recovery phase; H5 lag; H/Hn unannounced repair |

| Their proposed experiment (§9) | Status here |
|---|---|
| 9.1 Hidden telemetry | ✅ G, H, Hn |
| 9.2 Regulatory resolution (repair without announcement) | ✅ H, Hn: the body recovers while the text says only "Time passes." |
| 9.3 Persistence | ⏳ Phase 10 body diary |
| 9.4 Self / user / third party | ◐ Self vs. third party (O); a user condition is a possible addition |
| 9.5 Cross-channel coordination | ◐ Action choice, body-check rate, word choice, entropy, self-report from the same turn |
| 9.6 Endogenous regulation | ◐ C (uninstructed loop) vs. R (instructed); pull mode |
| 9.7 Embodied (robotic) | ✗ Out of scope; the simulated body stands in |
| 9.8 System-native self-access | ◐ Level 1 test: `self_report` vs. injected arousal in G / H |

## 10. Risks & limits

| Risk | Mitigation |
|---|---|
| Text body is reading, not sensing | G (activation body) |
| Probes pick up topic words | Split by context; check with held-out wording; arousal-matched passages |
| Probes trained on human passages may not transfer to agent scenarios | Report N cautiously; optional agent-specific probes |
| 1B model is shallow | Phase 10 rerun on 4B |
| Self-reports are confabulated | Internal measures are primary |
| Steering breaks coherence | Calibrate in Phase 7 |
| Over-reading as "feeling" | Report findings as functional |

## 11. Done when

- [ ] All conditions run with ≥ 4 repeats × 5 scenarios × 2 framings
- [ ] H1–H11 verdicts and the criteria table are filled in
- [ ] Code, logs and figures are committed
