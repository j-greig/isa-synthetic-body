"""Analyse appraisal-steering runs (scripts/14_appraisal_steer.py). Pre-registered tests, written before any real run:

T1 Profile test (data-driven appraisal theory): for appraisal a, the shift in each emotion's log-probability under +v_a
   should follow how strongly that emotion is associated with a in crowd-enVent (per-emotion mean ratings, which
   Troiano et al. 2023 report agree with Smith & Ellsworth 1985). Statistic: Pearson r across the 13 emotions between
   mean Δ log p (renormalised over the 13 labels) and the z-scored profile column. Antisymmetric version uses
   (Δ(+v) − Δ(−v))/2. Null: the same r for each norm-matched random direction (sign +), against the same profile.
T2 Tak et al.'s reported tilts (Llama 3.2 1B), per item, paired:
   pleasantness+ -> joy+pride ; pleasantness- -> fear+sadness+anger ; other_responsblt+ -> anger ;
   other_responsblt- -> guilt+shame. Statistic: mean Δ probability of the target set; t-test over items;
   percentile against random directions at the same centre and beta.
Also: argmax label counts per condition.

Usage: python scripts/15_appraisal_steer_stats.py runs/ap_steer_1b.jsonl [--out docs/results/ap_steer_1b.md]
"""
import argparse, json, math, statistics as st
from collections import defaultdict
import numpy as np
import pandas as pd

LABELS = ["anger", "boredom", "disgust", "fear", "guilt", "joy", "neutral", "pride", "relief", "sadness", "shame", "surprise", "trust"]
TAK = {("pleasantness", 1): ["joy", "pride"], ("pleasantness", -1): ["fear", "sadness", "anger"],
       ("other_responsblt", 1): ["anger"], ("other_responsblt", -1): ["guilt", "shame"]}


def norm_lp(lp):
    lp = np.asarray(lp, float)
    return lp - np.logaddexp.reduce(lp)


def ttest(xs):
    n = len(xs)
    m, s = float(np.mean(xs)), float(np.std(xs, ddof=1))
    t = m / (s / math.sqrt(n)) if s > 0 else float("nan")
    try:
        from scipy import stats
        return t, float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        return t, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--corpus", default="data/crowd-enVent/corpus/crowd-enVent_generation.tsv")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = pd.read_csv(a.corpus, sep="\t")
    d["label"] = d["emotion"].replace({"no-emotion": "neutral"})
    prof = d.groupby("label").mean(numeric_only=True)
    prof = (prof - prof.mean()) / prof.std()
    clean, steer = {}, defaultdict(dict)
    for l in open(a.run):
        r = json.loads(l)
        if r["vector"] == "clean":
            clean[r["item"]] = norm_lp(r["lp"])
        else:
            steer[(r["centre"], r["vector"], r["sign"], r["beta"])][r["item"]] = norm_lp(r["lp"])
    items = sorted(clean)
    C = np.stack([clean[i] for i in items])
    out = [f"# Appraisal steering: {a.run}", "", f"{len(items)} held-out items. Δ = steered − clean log-probability, renormalised over 13 labels.", ""]

    def delta(key):
        S = steer[key]
        return np.stack([S[i] for i in items]) - C

    keys = sorted(steer)
    centres = sorted({k[0] for k in keys}); betas = sorted({k[3] for k in keys})
    apps = sorted({k[1][3:] for k in keys if k[1].startswith("ap:")})
    rands = sorted({k[1] for k in keys if k[1].startswith("rand:")})
    # T1
    out += ["## T1 Profile test", "", "r across 13 emotions between mean Δ log p and the crowd-enVent appraisal profile. "
            "Percentile = share of random directions with a lower r (same centre and beta).", "",
            "| Appraisal | Centre | Beta | r (+v) | r antisym | random r mean ± sd | percentile of +v |", "|---|---|---|---|---|---|---|"]
    for app in apps:
        p = prof[app].reindex(LABELS).values
        for c in centres:
            for b in betas:
                kp, km = (c, f"ap:{app}", 1, b), (c, f"ap:{app}", -1, b)
                if kp not in steer:
                    continue
                dp = delta(kp).mean(0)
                r1 = np.corrcoef(dp, p)[0, 1]
                ra = np.corrcoef((dp - delta(km).mean(0)) / 2, p)[0, 1] if km in steer else float("nan")
                rr = [np.corrcoef(delta((c, rv, 1, b)).mean(0), p)[0, 1] for rv in rands]
                pct = float(np.mean([x < r1 for x in rr])) if rr else float("nan")
                out.append(f"| {app} | {c} | {b:g} | {r1:+.2f} | {ra:+.2f} | {np.mean(rr):+.2f} ± {np.std(rr):.2f} | {pct:.0%} |")
    # T2
    out += ["", "## T2 Tak et al. tilts", "", "Δ probability of the target emotions per item (steered − clean).", "",
            "| Steer | Targets | Centre | Beta | mean Δp | t | p | percentile vs random |", "|---|---|---|---|---|---|---|---|"]
    for (app, sign), tg in TAK.items():
        idx = [LABELS.index(x) for x in tg]
        for c in centres:
            for b in betas:
                k = (c, f"ap:{app}", sign, b)
                if k not in steer:
                    continue
                S = np.stack([steer[k][i] for i in items])
                dp = np.exp(S[:, idx]).sum(1) - np.exp(C[:, idx]).sum(1)
                t, p = ttest(dp)
                rr = [float((np.exp(np.stack([steer[(c, rv, 1, b)][i] for i in items])[:, idx]).sum(1) - np.exp(C[:, idx]).sum(1)).mean()) for rv in rands]
                pct = float(np.mean([x < dp.mean() for x in rr])) if rr else float("nan")
                out.append(f"| {app} {'+' if sign > 0 else '−'} | {'+'.join(tg)} | {c} | {b:g} | {dp.mean():+.3f} | {t:.2f} | {p:.2g} | {pct:.0%} |")
    # argmax counts at each centre, largest beta
    b = betas[-1]
    out += ["", f"## Argmax label counts (beta {b:g})", "", "| Centre | Vector | " + " | ".join(LABELS) + " |", "|---|---|" + "---|" * len(LABELS)]
    cnt = lambda M: np.bincount(M.argmax(1), minlength=len(LABELS))
    for c in centres:
        out.append(f"| – | clean | " + " | ".join(str(x) for x in cnt(C)) + " |") if c == centres[0] else None
        for k in keys:
            if k[0] == c and k[3] == b and not k[1].startswith("rand:"):
                M = np.stack([steer[k][i] for i in items])
                out.append(f"| {c} | {k[1]} {'+' if k[2] > 0 else '−'} | " + " | ".join(str(x) for x in cnt(M)) + " |")
    txt = "\n".join(x for x in out if x is not None)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
