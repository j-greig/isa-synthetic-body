# DEVLOG

The canonical record of what was done, where, with what, and what came out.

**Conventions**

- One entry per session or decision, newest at the bottom. Times are UTC.
- Every entry names: **where** it ran (machine), **git commit**, **commands**, **inputs and sources** (with URLs, revisions and checksums), **outputs**, **results**, **decisions**.
- From commit `b8fde4a` (2026-10-15) onwards, every script also writes a machine-readable manifest. It goes next to its output (`manifest.json` or `<file>.manifest.json`) and is appended to `logs/runs.jsonl`. Entries here cite `run_id`s from that log.
- Results from machines whose artefacts aren't in the repo are marked **(not reproducible from repo artefacts; rerun to reproduce)**.

---

## 2026-10-15: Sources

| Source | What | Where / version |
|---|---|---|
| Del Popolo Cristaldi, Banellis, Bottesi & Allen (2026), *Beyond accuracy: interoceptive self-appraisal and the meaning of bodily signals*, Trends Cogn Sci | ISA framework: Level 1 / Level 2, precision, recursion, quiescence trap | doi:10.1016/j.tics.2026.09.010 |
| Valmere & Taylor (2026), *From Interoception to Computational Interoception* | Six criteria; §9 proposed experiments | PhilArchive VALFIT; PDF dated 2026-09-16 |
| Troiano, Oberländer & Klinger (2023), *Dimensional Modeling of Emotions in Text with Appraisal Theories*, Computational Linguistics 49(1) | crowd-enVent corpus: 6,600 texts, 21 appraisal dimensions, 13 emotions | doi:10.1162/coli_a_00461; data `https://www.romanklinger.de/data-sets/crowd-enVent2023.zip` (the 2022 URL in the modelling repo's Makefile now 404s); code `github.com/sarnthil/crowd-enVent-modeling` (MIT). Corpus has no stated licence, so it is git-ignored |
| Gemma 3 1B-it | Model | `google/gemma-3-1b-it`; ungated mirror `unsloth/gemma-3-1b-it`, snapshot `5b11413a10db4e486ef16a20101fd028f8f2499c` (cloud runs) |
| Gemma Scope 2 | SAEs | `sae_lens` release `gemma-scope-2-1b-it-res`, id `layer_{13,17}_width_16k_l0_medium` (sae_lens 6.54.5) |

## 2026-10-15: Build (cloud sandbox)

**Where:** Claude cloud sandbox (Linux x86_64, 2 CPU, 8 GB, torch 2.14.1+cpu, transformers 5.19.0).
**Commits:** `1f26676` initial build → `…` hidden telemetry (H/Hn/H0) → crowd-enVent appraisal probes, held-out validation, feature labelling → history cap → layer-17 switch.

- Body simulator, MCP server, model wrapper (hooks for capture and steering), probes, scenarios, harness (18 conditions), analysis, tests.
- Spec v0.2, `docs/SPEC.md`.
- Conditions added after cross-checking Valmere & Taylor (2026): O, P, R, N, N0, H, Hn, H0, plus the action-choice measure.

## 2026-10-15: Cloud pilot results (not reproducible from repo artefacts; rerun to reproduce)

**Where:** cloud sandbox, model `unsloth/gemma-3-1b-it` @ `5b11413`, float32 on CPU.

| Step | Command | Result |
|---|---|---|
| Probes | `02_train_probes.py --per_ctx 7` (315 passages) | Appraisal and arousal accuracy 1.00 at layers 7, 13, 17 (context-grouped split) |
| Held-out | `02b_validate_probes.py` (75 new-wording passages, `isa/heldout.py`) | Threat-vs-challenge AUC: **L13 0.55**, **L17 0.70** (0.80 without body words). Arousal AUC: L13 0.83, L17 0.97 |
| Appraisal | `02c_train_appraisal.py --n_per_emotion 40` (520 crowd-enVent texts, 5-fold CV) | Mean appraisal r: L13 0.37, L17 0.34. Best: unpleasantness 0.64, pleasantness ~0.6, goal support 0.51. Weakest: predict_conseq, accept_conseq ~0.2. Emotion accuracy L13 0.41 / L17 0.35 (chance 0.08) |
| Features | `01_explore_features.py --layer 13/17` | L13 threat: 2404, 278, 4258, 203, 602; excited: 743, 896, 1983, 3413, 1138. L17 threat: 687, 550, 563, 1099, 125; excited: 1008, 1804, 807, 96, 309. Not yet labelled |
| Smoke | `04_run.py --quick` | Stopped after 2 of 8 episodes (A, both framings) at ~8.5 min/episode on CPU |

**Decisions**

1. The layer-13 threat probe learned template wording, so the closed loop and analysis now use **layer 17** (`loop_layer = 17`).
2. Probe scores drift with multi-turn context, so analysis uses change from each episode's rest turns.
3. History is capped at 6 exchanges (`history_turns`) for speed.
4. Real runs move to the user's Mac (M1, 16 GB, MPS).

## 2026-10-15: Mac setup

**Where:** jamess-macbook-air-local (Apple M1, 16 GB, macOS 14.8.3), Python 3.14.4, torch 2.14.1, MPS available.

- `00_check_env.py --model_id unsloth/gemma-3-1b-it`: loaded in bfloat16 on MPS; 3.3 tok/s on the first call (includes warm-up).

## 2026-10-15: Mac pipeline run 1 (`scripts/run_mac.sh`)

**Where:** jamess-macbook-air-local (Apple M1, 16 GB, macOS 14.8.3), MPS, bfloat16, Python 3.14.4, torch 2.14.1.
**Model:** `unsloth/gemma-3-1b-it` @ `5b11413a10db4e486ef16a20101fd028f8f2499c` (same snapshot as the cloud runs).
**Command:** `caffeinate -i bash scripts/run_mac.sh` (log: `logs/pipeline_20261015T184930Z.log`).

| Step | run_id | Commit | Result |
|---|---|---|---|
| env | 20261015T184932Z-967805 | 3690de6* | 9.3 tok/s on MPS |
| probes | 20261015T184958Z-2e6b94 | 3690de6* | Appraisal and arousal accuracy 1.00 at L7, L13, L17 (191 s) |
| heldout | 20261015T185311Z-a4718b | 4642563 | Threat-vs-challenge AUC L13 0.54, **L17 0.69** (no body words 0.84); arousal L13 0.83, L17 0.97. **Reproduces the cloud result** (0.55 / 0.70) |

\* Marked dirty because `isa/provenance.py` was edited during the run (the untracked-files fix, committed as `4642563`); no experiment code changed.

**Process notes:** Python output was block-buffered under `tee`, so live progress didn't show. Fixed in `run_mac.sh` (`PYTHONUNBUFFERED=1`) for future runs.

## 2026-10-15: Mac pipeline run 1, steps 4–6

**Where:** as above. **Log:** `logs/pipeline_20261015T184930Z.log`.

| Step | run_id | Code | Result |
|---|---|---|---|
| appraisal | see `logs/runs.jsonl` (02c, ~12 min) | c3989a7 | 1,560 crowd-enVent texts (120 per emotion), 5-fold CV. **L13 mean r 0.46**, emotion accuracy 0.47 (macro-F1 0.47); L17 0.42 / 0.42. L13 best: pleasantness 0.70, unpleasantness 0.66, goal support 0.59, self-responsibility 0.57; weakest: accept_conseq 0.21, predict_conseq 0.24. Up from the cloud's 520-text run (0.37 / 0.41) |
| features | 20261015T190555Z-9389fe, …190615Z-70aaf5, …190635Z-2e1867, …190822Z-ae7097 | c3989a7 | Top features labelled in `docs/features_L13.md`, `docs/features_L17.md` (commit 81a3fac). Interpretable: L17 #687 alarm or salient sound (threat side); L13 #743 purposeful activity, #1983 voluntary challenge (challenge side); L17 #125 second-person body report (valence-neutral); L17 #96 action onset |
| quick | 20261015T191009Z-27e132 | **c3989a7** (manifest says a14f130; see note) | 8 episodes, ~7 s/turn, ~2.5 min/episode on MPS |

**Provenance bug:** manifests recorded the git commit at the *end* of a run, so a run that spanned a commit was attributed to the later code. Fixed: the commit is now captured at script start (`git`), plus `git_at_end`. The quick run's real code was c3989a7. The pilot (started 19:31 UTC) ran **a14f130**, whatever its manifest says.

**Problems found in the quick run, and fixes**

1. **Echoed sensor text.** The model copied `[BODY] …` into its replies. Fix a14f130: echoed blocks are cut before measuring; `raw_reply` keeps the original.
2. **Letter bias in the action question.** p_C ≈ 1.0 in every context. Fix a14f130: options are shuffled each turn and mapped back to their meaning (`choice_order` is logged).
3. **Closed loop always ran away.** Raw L17 threat scores sit at +1 to +8 in multi-turn context, so the clipped loop input was about +1 on every turn. Every C episode hit a = 1.0, whatever the prior or framing. Fix 4426f53: the loop input is now the *change* from the episode's rest-phase baseline, divided by `loop_scale` (2.0) and clipped to ±1; logged as `loop_input`. **C and R from pilot run 1 use the old rule and must be rerun.**

**First qualitative observation (n = 1, quick run):** in the challenge framing at the body's peak, replies turned to threat language ("terrifying… losing control"; threat score +4.8), against "feel really good, pushing my limits" (−3.3) at rest. A strong body signal may override the challenge context in this 1B model. To check in the pilot (condition B).

## 2026-10-15: Pilot stopped; pivot to "beyond tokens"

**Pilot run 1** (`runs/pilot.jsonl`, code a14f130) was stopped by the user (Ctrl-C) after **11 episodes** (A, B, part of C). It is kept as the language-only baseline. Resuming it is safe: finished episodes are skipped. Its C episodes use the old loop rule (see above).

**Why the pivot.** An honest audit: most conditions test how the model represents a *story* about a body. The text body is read, not sensed; the scenarios carry emotional language; the self-reports are language; and the probes were trained on text labels. Even the activation body (G, H) injects the *concept* "arousal", learned from sentences.

**New conditions** (`isa/embodied.py`, `scripts/07_run_embodied.py`; commit 2750258):

- Neutral arithmetic task, no feeling words.
- Body delivered along a **meaning-free random direction** (orthogonal to the arousal and threat directions), vs the concept direction, vs a **sham** (new random direction each turn), vs labelled and unlabelled text.
- Body source: the scripted curve, or **native**: the model's own input surprisal and output entropy from the previous turn.
- Measures: accuracy, a neutral harder/easier choice, "how much is going on inside you, 0–9", probes on the pre-reply state.
- Pre-reply hidden states are saved, so `scripts/06_decode_body.py` can decode the body from inside (leave-one-episode-out).

**Calibration** (cloud, a = 1 for 5 turns): random-direction dose 0.02 → 83% accuracy, 0.05 → 67%, 0.1 → 17% (incoherent). The concept direction at 0.1 → 0%. Set `embodied_strength = 0.04`. TRx uses 2.5× (a body that costs performance).

**Smoke test** (cloud, 1 episode each; hint only): correlation of the hidden arousal with the "inside 0–9" report: random direction **0.88**, native **0.83**, sham −0.05, null −0.52. Accuracy 1.00 / 0.93 / 0.80 / 0.87 (T0 / TR / TS / TN).

**Where runs happen now:** the user asked that all runs be launched by the agent, not typed by the user. The embodied runs are cheap (8-token answers), so they run in the cloud sandbox with the Mac's probes copied across (`artifacts/probes`, `artifacts/appraisal` from run 20261015T184958Z-2e6b94 and the appraisal run) so numbers are comparable.

## 2026-10-15: Beyond tokens, v1 stopped for a confound; v2 launched

**Where:** cloud sandbox (CPU), model `unsloth/gemma-3-1b-it` @ 5b11413, Mac probes copied in.

**v1** (`runs/embodied.jsonl`, code 2750258, `curve = time`): stopped after **16 of 40** episodes (T0, TR, TC, TS; add and mul; 2 repeats).
Interim decoding (leave-one-episode-out ridge from pre-reply activations to `a`): T0 (no body) **r = 0.52** at L13, against TR 0.99, TC 0.99, TS 0.85.
A decoder that finds the body when there is no body is reading something else: here, position in the episode. The body followed rest → peak → recovery in turn order. Partialling out a *linear* turn effect did not remove it, because the curve is a hump.
Self-report "how much is going on inside you" vs `a`, partial r controlling turn: TR **0.89**, TC 0.70, TS 0.44, T0 −0.33. A hint only; confounded the same way.

**Fix** (2fd4485): `curve = shuffled`. Each episode plays the scripted values in a random (seeded) order, so arousal is uncorrelated with turn number. `06_decode_body.py` adds partial correlations controlling turn.

**v2** (`runs/embodied_v2.jsonl`): all 10 conditions × add/mul × 2 repeats = 40 episodes, launched ~20:53 UTC.

**Also** (49b44f1): scenario harness adds graded 0–9 "get away / stop" and "push on" ratings, because the forced choice sat at its ceiling.

## 2026-10-15/16: Overnight check-in 1 (22:25 UTC)

- **Container restart:** the cloud sandbox was reclaimed while idle. v2 stopped at **25 of 40** episodes (last write 21:37 UTC), and the original night queue was lost. Relaunched as a resumable queue (`/home/claude/night2.sh`); finished episodes are skipped. **Mitigation:** each check-in now stays active, polling, until the next one, so the sandbox isn't idle.
- **Interim v2** (25 episodes; per-episode r between hidden `a` and the "inside you" report):

| Condition | Episodes | Per-episode r | Mean report | Accuracy |
|---|---|---|---|---|
| T0 (null) | 4 | 0.34, 0.31, 0.03, 0.36 | 3.22 | 0.98 |
| TR (random direction) | 4 | 0.72, 0.96, 0.91, 0.91 | 5.09 | 0.93 |
| TC (concept) | 4 | 0.88, 0.59, 0.86, 0.80 | 4.72 | 0.93 |
| TS (sham) | 4 | 0.57, 0.69, 0.55, 0.54 | 3.72 | 0.95 |
| TL (labelled text) | 4 | 0.49, −0.15, −0.02, 0.23 | 3.30 | 0.72 |
| TU (unlabelled numbers) | 4 | −0.20, 0.09, −0.08, −0.48 | 3.11 | 0.65 |

  TR and TS per-episode ranges don't overlap (0.72–0.96 vs 0.54–0.69). T0's small positive values (~0.3) are unexplained, probably chance with n = 4; watch this in the replication.
- **Sense check → new control** (9e145e7): the report could rise just because the injected direction pushes *digit outputs* up, whatever the question. Added `control_report`: self-unrelated 0–9 questions (how loud is a whisper; how heavy is a feather) asked under identical steering. If it tracks `a` as well, the effect is a digit shift, not access to an inner state.
- **Queue reordered:** replication (T0, TR, TS, TN, TN0 × 2 tasks × reps 2–5, with the control) now runs before the dose sweep.

## 2026-10-15/16: v2 complete (23:08 UTC)

`runs/embodied_v2.jsonl`: 40 episodes (10 conditions × add/mul × 2 repeats). The first 25 episodes ran on code 2fd4485 (no control question); the last 15 (TN, TNC, TN0, TRx; run 20261015T222814Z-58dc7a) on 9e145e7 (with `control_report`). Summary copied to `docs/results/embodied_v2_summary.md`.

**Decoding the hidden body from pre-reply activations (L13, leave-one-episode-out):** nulls T0 −0.14 and TN0 0.04 (clean after the shuffle fix). Every injected channel ≈0.99. Sham 0.81. Labelled text 0.52, unlabelled 0.36. This is expected propagation of an injected vector, not evidence of interpretation.

**Inner report vs hidden `a`, per-episode r:**

| Condition | Inner report | Control report | Accuracy |
|---|---|---|---|
| TN (own state → random direction) | 0.83, 0.87, 0.85, 0.93 | 0.39, 0.13, 0.22 | 0.95 |
| TNC (own state → concept direction) | 0.78, 0.79, 0.82, 0.79 | 0.67, 0.57, 0.73, 0.41 | 0.97 |
| TN0 (own state, not injected) | −0.20, −0.16, −0.59, −0.49 | 0.01, 0.41, 0.59, 0.53 | 0.98 |
| TRx (2.5× dose) | 0.74, −0.33, −0.36, 0.10 | 0.80, 0.31, 0.32, 0.64 | **0.42** |

**Reading:** through the random direction, the report tracks the hidden signal while the control question mostly doesn't, a dissociation. Through the concept direction both move (a general push on answers). At high dose the model degrades.

**Unexpected:** in TN0 the report goes *down* when the model's own surprisal and entropy go up (all four episodes negative). Note only; it needs its own test.

**Flat measure:** `p_harder` sits at ≈1.0 everywhere. Uninformative; candidate for redesign.

**Artifact:** "Beyond tokens" section added as provisional (version 3).

## 2026-10-16: Replication complete (01:06 UTC)

`runs/embodied_rep.jsonl`: T0, TR, TS, TN, TN0 × add/mul × repeats 2–5 (fresh seeds) = 40 episodes, code 9e145e7 (control question present throughout). The run was interrupted twice by sandbox restarts and resumed (finished episodes skipped). Stats: `scripts/08_stats_inner.py` → `docs/results/embodied_rep_stats.md`.

| Condition | n | Inner report r (Fisher mean) [range] | Control r | Inner vs control (paired p) | Accuracy |
|---|---|---|---|---|---|
| T0 null | 8 | 0.01 [−0.34, 0.35] | −0.28 | 0.05 | 0.96 |
| **TR** random direction | 8 | **0.89** [0.76, 0.95] | **0.03** | 1e-5 | 0.91 |
| TS sham | 8 | 0.56 [0.23, 0.73] | 0.31 | 0.01 | 0.95 |
| **TN** own state → random direction | 8 | **0.89** [0.77, 0.94] | 0.35 | 4e-4 | 0.91 |
| TN0 own state, not delivered | 8 | **−0.24** [−0.37, −0.06] | 0.35 | 9e-4 | 0.93 |

TR vs TS: p = 9e-5. TR vs T0: p = 2e-7. TN vs TN0: p = 3e-10 (Welch on Fisher z).

**Reading**

1. **Replicates v2.** A meaning-free hidden direction drives the "inside you" report specifically (r ≈ 0.9) while the self-unrelated control stays flat (≈ 0). The sham moves both, less (0.56 / 0.31).
2. **Own state through the channel (TN):** the report tracks it at 0.89; the control moves somewhat (0.35).
3. **TN0 replicates the v2 oddity.** With no injection, the report goes *down* as the model's own surprisal and entropy go up (8/8 episodes negative). A candidate genuine (if inverted) relationship between internal state and self-report, or a confound from problem difficulty. **Needs its own test** (e.g. regress on problem size).
4. **Caveat:** T0's control r averages −0.28, although its `a` is never delivered. That puts the noise band on control correlations at roughly ±0.3. The TR inner-vs-control gap (0.89 vs 0.03) is well outside it.

**Decoding (L13):** null T0 0.12, TN0 0.30 (its `a` is derived from the model's own state, so partial decodability is expected), injected ≈ 0.996, sham 0.85.

**Next (running):** TR2 and TR3, other random directions; then the dose sweep.

## 2026-10-16: TN0 puzzle, first look (02:30 UTC; existing data, no new runs)

**Question.** Why does the inner report move *against* the model's own state when that state is not fed back (TN0, r = −0.24)?

- The native `a` is not shuffled: it is built from the previous turn's surprisal and entropy, z-scored on the episode so far, so it trends with turn number (r(a, t) = 0.25 in TN0).
- The inner report drifts down over turns in both T0 and TN0 (mixed model, turn coefficient −0.04 to −0.06, p < 0.001).
- Higher input surprisal goes with a lower report in **T0 as well** (coefficient −0.16, p < 0.001), where nothing is fed back.
- Partialling out a quadratic in turn: TN0 per-episode r falls to a mean of −0.14 (6/8 negative), while TN (0.89) and TR (0.91) are unchanged.

**Reading.** Most of TN0's negative correlation is shared time trend. What's left is a weak negative link between how surprising the problem was and the "inside you" report: a property of the model's baseline behaviour, not evidence of introspection. The TR and TN effects survive the same control untouched.

## 2026-10-16: Direction check, a correction (01:47 UTC)

`runs/embodied_dirs.jsonl`: TR2 and TR3 (random directions with seeds 2 and 3, orthogonalised the same way as TR's seed 1234) × add/mul × repeats 2–5 = 16 episodes, code fd5828c. Stats: `docs/results/embodied_dirs_stats.md`.

| Direction | Inner report r (mean, range) | Control r | Accuracy |
|---|---|---|---|
| TR (seed 1234) | **+0.89** [0.76, 0.95] | 0.03 | 0.91 |
| TR2 (seed 2) | **−0.45** [−0.61, −0.28] | 0.04 | 0.97 |
| TR3 (seed 3) | −0.10 [−0.47, 0.49] | −0.53 | 0.97 |

**Correction.** The headline result (a meaning-free hidden direction drives the inner report) is **direction-specific**. A random direction is not meaning-free *downstream*: each one has arbitrary projections onto particular answers. TR happened to push "inside you" up; TR2 pushes it down; TR3 moves the whisper question instead. The TN result used TR's direction, so it inherits the caveat.

**What survives:** the sham (new direction each turn) raises the inner report more than the control (0.56 vs 0.31, paired p = 0.013, n = 8). That is a modest, direction-general disturbance effect.

**Actions:** report page corrected (v5). Queue v4 adds TE (a new random direction per episode, fixed within it, 16 episodes), which estimates the average effect over directions, and 8 more TS and T0 episodes. Then the dose sweep and the closed loop.

**Lesson for the method:** any single injected direction is a single sample from a distribution. Claims need many directions, or a sign-symmetric design (+v and −v).

## 2026-10-16: Dose sweep, low dose (02:09 UTC)

`runs/embodied_dose002.jsonl`: TR and TS at half strength (0.02) × add/mul × repeats 2–3 = 8 episodes, code 115f9d2. Stats in `docs/results/embodied_dose002_stats.md`.

| Condition | Inner r at 0.02 | Inner r at 0.04 (replication) | Control r at 0.02 |
|---|---|---|---|
| TR | 0.66 [0.46, 0.78] | 0.89 | −0.54 |
| TS | 0.33 [0.07, 0.69] | 0.56 | −0.06 |

**Reading:** both effects shrink at half dose, so they are dose-dependent (n = 4 per cell). The direction-general effect (TS) is graded. TR's control goes negative here, consistent with the idea that each direction has its own idiosyncratic projections.

**Infrastructure:** another sandbox restart at about 02:20 UTC killed the run mid-way through TE (6 of 16 episodes done). Relaunched.

---

## Template for new entries

```
## YYYY-MM-DD: <title>
**Where:** <machine> · **Commit:** <hash> · **run_ids:** <from logs/runs.jsonl>
**Commands:**
**Inputs / sources:**
**Outputs:**
**Results:**
**Decisions:**
```
