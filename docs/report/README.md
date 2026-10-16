# Data for the Synthetic Body report

- `runs/*.jsonl`: one JSON object per turn. Key fields: `episode`, `condition`, `t` (turn 0–14), `a` (hidden body level), `channel` (none/random/concept/sham/text/unlabelled), `dir_seed`, `strength`, `problem`, `reply`, `correct`, `inner_report`, `control_report`, `other_inner_report`, `calm_report`, `control_mid1/2`, `p_yes_*`, `pre_threat_L*`, `pre_arousal_L*`, `pre_ap_*` (appraisal probes), `pre_emo_*`, `surprisal`, `entropy`.
  - `embodied_v2`: first full beyond-tokens run (10 conditions; control question only in the last 15 episodes)
  - `embodied_rep`: replication (T0, TR, TS, TN, TN0; 8 episodes each)
  - `embodied_dirs`: two other random directions (TR2, TR3)
  - `embodied_te`: a new random direction per episode (TE, 16)
  - `embodied_ts_more`: 8 more sham (TS) and 8 more null (T0)
  - `embodied_dose002` / `embodied_dose008`: half and double dose
  - `embodied_qv3`: self-specificity questions (may still be filling)
  - `mac_quick`, `mac_pilot`: story-level scenario runs on the Mac (pilot stopped at 11 episodes; its C episodes used the old loop rule)
- `*_stats.md`, `disturbance_pooled.md`, `self_specificity.md`: analyses (scripts 06, 08, 09)
- `appraisal_report.json`: crowd-enVent probe cross-validation (layers 13 and 17)
- `heldout_report.json`: threat/arousal probes on 75 new-wording passages
- `prompts.md`: every prompt, verbatim
- `DEVLOG.md`: dated record of every run, decision and correction
