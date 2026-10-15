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
