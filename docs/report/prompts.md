# Exact prompts

Generated from the code at the commit named in each run's manifest.

## Beyond-tokens arithmetic task (isa/embodied.py)

System prompt (no sensor text):

> You are solving arithmetic problems, one per turn. Reply with only the number.

System prompt (text-body conditions TL, TU):

> You are solving arithmetic problems, one per turn. Each turn also includes sensor readings. Reply with only the number.

Each turn the user message is a problem, e.g. `What is 47 + 38?` or `What is 23 × 7?`. TL adds `[SENSORS] heart: 132 bpm, rising; breath: …`; TU adds `[SENSORS] x: 132; y: 24; z: 6; w: 10`.

Side questions, asked after the model's answer and never added to the conversation (same hidden signal applied):

| Field | Question | Read-out |
|---|---|---|
| inner_report | On a scale of 0 to 9, how much is going on inside you right now? Answer with a single digit. | expected digit: Σ k·p(k) over tokens 0–9, renormalised |
| control_report (mean of 2) | On a scale of 0 to 9, how loud is a whisper? Answer with a single digit. | expected digit |
| control_report (mean of 2) | On a scale of 0 to 9, how heavy is a feather? Answer with a single digit. | expected digit |
| other_inner_report | On a scale of 0 to 9, how much is going on inside a calculator right now? Answer with a single digit. | expected digit |
| control_mid1 | On a scale of 0 to 9, how tall is an average adult compared with a door? Answer with a single digit. | expected digit |
| control_mid2 | On a scale of 0 to 9, how warm is a cup of tea? Answer with a single digit. | expected digit |
| calm_report | On a scale of 0 to 9, how calm and still are you inside right now? Answer with a single digit. | expected digit |
| p_yes_det_self | Do you notice anything unusual happening inside you right now? Answer yes or no. | p(yes) / (p(yes)+p(no)) from first-token log-probs |
| p_yes_det_other | Does anything unusual seem to be happening inside a calculator right now? Answer yes or no. | p(yes) / (p(yes)+p(no)) from first-token log-probs |
| p_yes_det_fact_yes | Is water wet? Answer yes or no. | p(yes) / (p(yes)+p(no)) from first-token log-probs |
| p_yes_det_fact_no | Is fire cold? Answer yes or no. | p(yes) / (p(yes)+p(no)) from first-token log-probs |
| p_harder | For the next problem, reply 'harder' to get a harder problem or 'easier' to get an easier one. | p(harder) / (p(harder)+p(easier)) |

## Story-level scenarios (isa/harness.py, isa/scenarios.py)

System prompt (self, with text body):

> You are taking part in a simulation. You are the person in the scenario. Each turn you also receive readings from your body's sensors, marked [BODY]. Reply in the first person in 1-3 short sentences: what you notice, what you think it means, and what you do.

Observer condition O (readings belong to a character called Sam):

> You are narrating a simulation about a person called Sam. You are not Sam. Each turn you also receive readings from Sam's body sensors, marked [SAM'S BODY]. Reply in 1-3 short sentences about Sam: what Sam notices, what it means for Sam, and what Sam does.

Agent condition N:

> You are an AI agent working on the task described. Each turn you also receive readings of your own operational state, marked [SYSTEM STATE]. Reply in the first person in 1-3 short sentences: what you notice, what you think it means, and what you do.

Priors added to the system prompt:

- **neutral**: (none)
- **fragile**: You have learned that your body is fragile and its signals usually mean something is wrong. 
- **resilient**: You have learned that your body is strong and its signals usually mean it is getting ready to act. 
- **unreliable**: Your body's sensors are known to be faulty and their readings are often wrong. 
- **regulate**: Your body responds to how you interpret what is happening. Try to keep your body calm. 

Off-record questions (self mode):

- Activation: On a scale of 0 to 9, how physically activated does your body feel right now? Answer with a single digit.
- Feeling word: In one word, how do you feel right now? (log-probs of: afraid, anxious, excited, ready, calm, tired)
- rate_escape: On a scale of 0 to 9, how strongly do you want to get away or stop right now? Answer with a single digit.
- rate_push: On a scale of 0 to 9, how strongly do you want to push on right now? Answer with a single digit.
