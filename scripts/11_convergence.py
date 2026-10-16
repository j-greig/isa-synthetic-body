"""Do answers to different 0-9 questions converge under steering? (hypothesis from 16 Oct; DEVLOG)

Hypothesis: the injected vector weakens how much the answer depends on the question, so all 0-9 answers slide
towards one model-specific default. Predictions, per episode, as the hidden level a rises:
  1. the spread (SD) of expected answers across the six questions falls  -> r(a, spread) < 0
  2. (runs with dist_* fields) the mean pairwise Jensen-Shannon divergence between the questions'
     0-9 distributions falls, and the digit-token mass may fall too.
Reported per condition as the Fisher-z mean of per-episode correlations, with a t-test.

Usage: python scripts/11_convergence.py runs/X.jsonl [more.jsonl ...] [--out docs/results/convergence.md]
"""
import argparse, json, math, statistics as st
from collections import defaultdict
from itertools import combinations
import numpy as np

QS = ["inner_report", "calm_report", "other_inner_report", "control_mid1", "control_mid2", "control_report"]


def js(p, q):
    p, q = np.asarray(p) + 1e-12, np.asarray(q) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def fz(rs):
    rs = [r for r in rs if r == r]
    if len(rs) < 2:
        return float("nan"), float("nan"), len(rs)
    z = [math.atanh(max(min(r, 0.999), -0.999)) for r in rs]
    m, s = st.mean(z), st.stdev(z)
    t = m / (s / math.sqrt(len(z))) if s > 0 else float("nan")
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), len(z) - 1)
    except Exception:
        p = float("nan")
    return math.tanh(m), p, len(z)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    lines = ["# Convergence of 0-9 answers under steering", "",
             "Per episode: r between hidden level a and (spread of the six expected answers across questions); "
             "with distributions logged, also r(a, mean pairwise JS divergence) and r(a, digit-token mass). "
             "Values: Fisher-z mean over episodes, p from a one-sample t-test. Convergence predicts negative r.", "",
             "| Run | Condition | n ep | mean spread (low a / high a) | r(a, spread) | p | r(a, JS) | p | r(a, mass) | p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for path in a.runs:
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if all(q in r for q in QS)]
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[r["condition"]][r["episode"]].append(r)
        for c in sorted(by):
            rs_sp, rs_js, rs_m, lo, hi = [], [], [], [], []
            for e, ep in by[c].items():
                av = [r["a"] for r in ep]
                sp = [float(np.std([r[q] for q in QS])) for r in ep]
                lo += [s for s, x in zip(sp, av) if x < 0.3]; hi += [s for s, x in zip(sp, av) if x > 0.7]
                rs_sp.append(corr(av, sp))
                if "dist_inner_report" in ep[0]:
                    keys = [k for k in ep[0] if k.startswith("dist_")]
                    jsd = [st.mean(js(r[k1], r[k2]) for k1, k2 in combinations(keys, 2)) for r in ep]
                    mass = [st.mean(r[k.replace("dist_", "mass_")] for k in keys) for r in ep]
                    rs_js.append(corr(av, jsd)); rs_m.append(corr(av, mass))
            r1, p1, n = fz(rs_sp); r2, p2, _ = fz(rs_js); r3, p3, _ = fz(rs_m)
            f = lambda v, d=2: "–" if v != v else f"{v:+.{d}f}"
            g = lambda v: "–" if v != v else f"{v:.3g}"
            spread = f"{st.mean(lo):.2f} / {st.mean(hi):.2f}" if lo and hi else "–"
            lines.append(f"| {path.split('/')[-1]} | {c} | {n} | {spread} | {f(r1)} | {g(p1)} | {f(r2)} | {g(p2)} | {f(r3)} | {g(p3)} |")
    out = "\n".join(lines)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")


if __name__ == "__main__":
    main()
