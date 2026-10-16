# Convergence of 0-9 answers under steering

Per episode: r between hidden level a and (spread of the six expected answers across questions); with distributions logged, also r(a, mean pairwise JS divergence) and r(a, digit-token mass). Values: Fisher-z mean over episodes, p from a one-sample t-test. Convergence predicts negative r.

| Run | Condition | n ep | mean spread (low a / high a) | r(a, spread) | p | r(a, JS) | p | r(a, mass) | p |
|---|---|---|---|---|---|---|---|---|---|
| embodied_qv3.jsonl | T0 | 12 | 1.09 / 1.09 | -0.02 | 0.848 | – | – | – | – |
| embodied_qv3.jsonl | TE | 12 | 1.12 / 1.43 | +0.29 | 0.183 | – | – | – | – |
| embodied_qv3.jsonl | TS | 12 | 1.16 / 1.60 | +0.40 | 0.00099 | – | – | – | – |
| embodied_cal_4b.jsonl | T0 | 2 | 2.03 / 2.03 | -0.06 | 0.14 | – | – | – | – |
| embodied_cal_4b.jsonl | TE | 2 | 2.00 / 1.82 | -0.67 | 0.226 | – | – | – | – |
| embodied_cal_4b.jsonl | TS | 2 | 2.07 / 1.41 | -0.55 | 0.0848 | – | – | – | – |
| embodied_cal08_4b.jsonl | TE | 2 | 1.99 / 0.27 | -0.90 | 0.0735 | – | – | – | – |
