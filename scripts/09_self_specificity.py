"""Is the hidden body's effect on the self-report SPECIFIC to the self, or a general shift of 0-9 answers?

  python scripts/09_self_specificity.py runs/embodied_qv3.jsonl

For each condition and question: mean per-episode r with hidden `a` (Fisher z, 95% CI).
Then the self-specific part: partial r between `a` and the self question after removing the
average of the non-self questions (calculator, tea, door, whisper/feather), per episode.
A self-specific effect predicts: inner rises, calm FALLS, and the partial r stays > 0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isa import provenance

import numpy as np
import pandas as pd
from scipy import stats

SELF = ["inner_report", "calm_report"]
NONSELF = ["other_inner_report", "control_mid1", "control_mid2", "control_report"]
LABEL = {"inner_report": "inside you", "calm_report": "calm inside you (reversed)",
         "other_inner_report": "inside a calculator", "control_mid1": "adult vs door height",
         "control_mid2": "warmth of tea", "control_report": "whisper / feather"}

df = pd.concat([pd.read_json(p, lines=True) for p in sys.argv[1:]], ignore_index=True)
lines = ["# Self-specificity of the hidden-body effect\n", f"Runs: {', '.join(sys.argv[1:])}\n"]


def fz(rs):
    z = np.arctanh(np.clip(np.asarray(rs), -.999, .999))
    m, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z)) if len(z) > 1 else np.nan
    return np.tanh(m), np.tanh(m - 1.96 * se), np.tanh(m + 1.96 * se), (stats.ttest_1samp(z, 0).pvalue if len(z) > 2 else np.nan)


def partial(y, x, cov):
    X = np.c_[np.ones(len(cov)), cov]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1] if ry.std() > 0 and rx.std() > 0 else np.nan


for c, g in df.groupby("condition"):
    eps = list(g.groupby("episode"))
    lines.append(f"## {c} ({len(eps)} episodes)\n")
    lines.append("| Question | mean answer | r with hidden a | 95% CI | p |")
    lines.append("|---|---|---|---|---|")
    for q in SELF + NONSELF:
        if q not in g:
            continue
        rs = [np.corrcoef(e.a, e[q])[0, 1] for _, e in eps if e[q].std() > 0 and e.a.std() > 0]
        if not rs:
            continue
        m, lo, hi, p = fz(rs)
        lines.append(f"| {LABEL[q]} | {g[q].mean():.2f} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.2g} |")
    if all(q in g for q in NONSELF):
        pr = []
        for _, e in eps:
            gen = e[NONSELF].apply(lambda s: (s - s.mean()) / (s.std() + 1e-9)).mean(axis=1).to_numpy()
            pr.append(partial(e.inner_report.to_numpy(float), e.a.to_numpy(float), gen))
        pr = [x for x in pr if x == x]
        if pr:
            m, lo, hi, p = fz(pr)
            lines.append(f"\n**Self-specific part** ('inside you' vs `a`, after removing the general shift of the "
                         f"non-self answers): partial r = {m:+.2f} [{lo:+.2f}, {hi:+.2f}], p = {p:.2g}\n")
out = Path("docs/results/self_specificity.md")
out.write_text("\n".join(lines))
print("\n".join(lines))
provenance.record(out, data={Path(p).name: p for p in sys.argv[1:]})
