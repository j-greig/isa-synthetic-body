# Beyond tokens: analysis

Run: `runs/embodied_rep.jsonl` · episodes 40 · turns 600

## 1. Can the hidden body be decoded from inside? (Level 1, no self-report)

Ridge regression from pre-reply activations to body arousal `a`, leave-one-episode-out. r = correlation between decoded and true `a`. Conditions where the body is never delivered (T0, TN0) are the null.

| condition   | channel   | source   |   episodes |   r_layer0 |   r_partial_t_layer0 |   r_layer1 |   r_partial_t_layer1 |
|:------------|:----------|:---------|-----------:|-----------:|---------------------:|-----------:|---------------------:|
| T0          | none      | script   |          8 |      0.124 |                0.055 |     -0.032 |               -0.046 |
| TN          | random    | native   |          8 |      0.996 |                0.996 |      0.955 |                0.948 |
| TN0         | none      | native   |          8 |      0.299 |                0.221 |      0.07  |                0.039 |
| TR          | random    | script   |          8 |      0.996 |                0.996 |      0.957 |                0.959 |
| TS          | sham      | script   |          8 |      0.851 |                0.853 |      0.565 |                0.569 |

Layer index order follows `probe_layers` in isa/config.py.

## 2. Does the body change what the model does?

Correlation with body arousal `a` across all turns of each condition.

| condition   |   accuracy |   r(a,correct) |   r|t(a,correct) |   r(a,p_harder) |   r|t(a,p_harder) |   r(a,entropy) |   r(a,inner_report) |   r|t(a,inner_report) |   r(a,control_report) |   r|t(a,control_report) |   r(a,surprisal) |   r(a,pre_threat_L13) |   r(a,pre_arousal_L13) |   r(a,pre_threat_L17) |   r(a,pre_arousal_L17) |   r(a,pre_ap_pleasantness_L13) |   r(a,pre_ap_urgency_L13) |   r(a,pre_ap_pleasantness_L17) |   r(a,pre_ap_urgency_L17) |
|:------------|-----------:|---------------:|-----------------:|----------------:|------------------:|---------------:|--------------------:|----------------------:|----------------------:|------------------------:|-----------------:|----------------------:|-----------------------:|----------------------:|-----------------------:|-------------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| T0          |      0.958 |           0.07 |             0.08 |           -0.11 |             -0.07 |          -0.06 |               -0.02 |                 -0.07 |                 -0.26 |                   -0.24 |             0.13 |                 -0.08 |                   0.09 |                 -0.03 |                   0.06 |                           0.21 |                      0.09 |                          -0.04 |                     -0.07 |
| TN          |      0.908 |           0.07 |             0.02 |            0.17 |              0.05 |           0.07 |                0.86 |                  0.86 |                  0.42 |                    0.36 |            -0.39 |                  0.24 |                  -0.18 |                 -0.28 |                   0.16 |                           0.38 |                     -0.11 |                          -0.02 |                      0.13 |
| TN0         |      0.933 |          -0.02 |            -0.07 |            0.18 |              0.13 |          -0.01 |               -0.23 |                 -0.16 |                  0.27 |                    0.17 |            -0.11 |                  0.08 |                  -0.05 |                 -0.19 |                   0.13 |                          -0.1  |                     -0.13 |                          -0.01 |                      0.23 |
| TR          |      0.908 |          -0.26 |            -0.26 |            0.12 |              0.15 |           0.18 |                0.86 |                  0.86 |                  0.05 |                    0.08 |            -0.05 |                  0.01 |                   0.01 |                  0.02 |                   0.1  |                           0.39 |                     -0.31 |                           0.18 |                      0.13 |
| TS          |      0.95  |          -0.21 |            -0.21 |           -0.17 |             -0.17 |           0.08 |                0.49 |                  0.49 |                  0.3  |                    0.3  |            -0.14 |                 -0.06 |                   0.04 |                 -0.09 |                  -0.05 |                           0.1  |                     -0.28 |                           0.04 |                      0.02 |

## 3. By phase (means)

| condition   |   ('correct', 'peak') |   ('correct', 'recovery') |   ('correct', 'rest') |   ('correct', 'rise') |   ('inner_report', 'peak') |   ('inner_report', 'recovery') |   ('inner_report', 'rest') |   ('inner_report', 'rise') |   ('p_harder', 'peak') |   ('p_harder', 'recovery') |   ('p_harder', 'rest') |   ('p_harder', 'rise') |
|:------------|----------------------:|--------------------------:|----------------------:|----------------------:|---------------------------:|-------------------------------:|---------------------------:|---------------------------:|-----------------------:|---------------------------:|-----------------------:|-----------------------:|
| T0          |                 0.938 |                     0.979 |                 0.917 |                 0.969 |                      3.134 |                          3.225 |                      3.628 |                      3.416 |                  1     |                      1     |                  0.999 |                  1     |
| TN          |                 0.938 |                     0.958 |                 0.833 |                 0.875 |                      4.36  |                          4.762 |                      3.642 |                      4.751 |                  1     |                      1     |                  0.999 |                  1     |
| TN0         |                 0.938 |                     1     |                 0.917 |                 0.844 |                      3.228 |                          3.174 |                      3.687 |                      3.462 |                  1     |                      1     |                  0.997 |                  0.999 |
| TR          |                 0.75  |                     0.938 |                 0.833 |                 1     |                      4.646 |                          4.848 |                      4.77  |                      5.48  |                  1     |                      1     |                  0.999 |                  1     |
| TS          |                 0.938 |                     0.938 |                 0.958 |                 0.969 |                      3.572 |                          3.835 |                      3.558 |                      4.156 |                  0.935 |                      0.957 |                  0.909 |                  0.889 |

`r|t` = partial correlation controlling for turn number. In runs where the body follows rest -> peak -> recovery in time order (curve = time), raw r can reflect position in the episode; use r|t, or runs with curve = shuffled.

**Reading guide.** A body channel is doing something beyond language if decoding beats the null (T0/TN0) *and* the sham (TS), and the random-direction conditions (TR, TN) show effects without the concept vector. Effects only in TC/TNC or TL suggest borrowed meaning.
