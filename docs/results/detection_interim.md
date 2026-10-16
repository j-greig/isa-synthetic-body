# Detection test: runs/embodied_detect.jsonl

60 turns, 4 episodes. Signal on = a > 0 (0.8), off = a = 0. Values: mean over episodes of (on − off).

| Condition | n ep | Δ p(yes) self | Δ other | Δ water wet | Δ fire cold | Δ inner 0–9 | self − other (t, p) | self − fire cold (t, p) | AUC self on vs off | mean p(yes) self off / on |
|---|---|---|---|---|---|---|---|---|---|---|
| TD | 4 | +0.550 | +0.042 | -0.000 | +0.400 | +0.627 | +0.507 (t=3.18, p=0.050) | +0.150 (t=0.70, p=0.535) | 0.78 | 0.164 / 0.713 |

Caveat: the calculator question sits near ceiling (p(yes) ≈ 0.9 with the signal off), so it cannot rise much; 'Is fire cold?' starts at 0 like the self question, so self − fire cold is the fairer yes-bias control.

Reading: a sense-like detection shows Δ self > 0, Δ self > Δ other, facts ≈ 0, and TD0 (nothing delivered) ≈ 0 everywhere.
