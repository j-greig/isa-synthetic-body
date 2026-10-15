# Overnight operating brief

Standing instructions for unattended check-ins. Each check-in reads this file first and follows it. Written 2026-10-15 ~22:40 BST at the user's request ("check in periodically, update what needs updating, check results, sense-check that the work is useful, course-correct, update the artifact; multiple rounds").

## Goal for the night

Turn the beyond-tokens hints into results that survive scrutiny, and leave the user a clear, honestly sourced morning summary. Quality over quantity. One solid, replicated finding beats ten new conditions.

## The question we are answering

Does a signal that never appears as text, and has no meaning in language, change the model's internal state, report and behaviour *more* than structureless perturbation does? And does the model's *own* state (surprisal, uncertainty), fed back through that channel, behave like a body? This is ISA's Level 1 (detection) tested from inside, with Level 2 (meaning) to follow.

## Each check-in, in order

1. **Status.** Read the tail of every active log in `/home/claude/` (`emb*.log`, `night*.log`) and `runs/*.jsonl` row counts. Is something running? Has it stalled (no new rows for >20 min)? Crashed (Traceback)?
2. **Health.** For new data: accuracy collapse (<50% where it shouldn't be); replies that are not numbers; NaNs; identical replies; a null condition (T0, TN0) that "decodes" (|r| > 0.3), which signals a confound like the turn-order one we hit.
3. **Results.** Run `scripts/06_decode_body.py` on any run that finished since the last check. Compare against the pre-registered predictions below. Write what changed, including null and negative results.
4. **Sense check.** Ask: is the next queued job still the most informative thing to run? Is any measure at ceiling or floor (uninformative)? Is any result explainable by something boring (turn number, reply length, accuracy, text length, perturbation size)? If yes, design the control and queue it ahead of new conditions.
5. **Act.** Fix bugs (commit with a message saying what was seen), queue or reorder jobs, never delete data (mark superseded runs in the DEVLOG instead).
6. **Record.** DEVLOG entry: time, what ran (run file, commit, conditions, n), result, decision. Commit in the cloud repo. Sync to the Mac repo via bundle (ff-only).
7. **Report page.** Update the Synthetic Body artifact only when a result is firm (finished run, analysed). Every number on the page names its source: run file, machine, date. Label provisional results as provisional.
8. **Message.** 2–4 lines to the user: what finished, what it showed, what is running next.
9. **Schedule** the next check-in (if the chain is not already scheduled).

## Pre-registered predictions (write down before looking)

| Test | Prediction if the channel acts like a sense | Boring explanation to rule out |
|---|---|---|
| Inner report vs `a`: TR vs TS | TR > TS clearly, replicated across tasks and repeats | Perturbation size alone (TS controls this) |
| Dose-response, TR | Inner report slope rises with dose; accuracy falls only at high dose | Ceiling at 9 or floor effects |
| Native (TN vs TN0) | TN's inner report tracks the model's own surprisal more than TN0's does | TN0 shows the same: then it's the input, not the channel |
| Text body (TL) | Inner report does *not* track; accuracy drops | Reading load (TU controls labels vs numbers) |
| Decoding null (T0, TN0) | |r| < 0.2 | Any larger value means a confound |

## Priorities (in order)

1. Finish v2 (`runs/embodied_v2.jsonl`) and analyse it.
2. Dose-response for TR and TS (doses 0.5×, 1×, 2× `embodied_strength`).
3. Replicate the key comparison (T0, TR, TS, TN, TN0) with 4 repeats and a third task variant if time allows.
4. Closed-loop scenario conditions C and R with the fixed loop rule (slow in the cloud: one scenario, one repeat).
5. Only then: new ideas.

## Rules

- Never claim more than the data shows. n is small: say so. Show nulls and negatives.
- Never delete runs or rewrite history; supersede and record.
- Keep the cloud and Mac repos in sync (bundles, ff-only). Never touch the user's `.venv`.
- Don't push to GitHub (not possible from here). Leave the push line for the morning.
- If something is badly wrong and can't be fixed safely, stop the queue and say so plainly in the morning summary.

## Morning summary (last check-in, ~08:30 BST)

Top of chat: what ran, three key results with numbers and sources, what failed or was inconclusive, what changed in the code, the push line, and suggested next steps.
