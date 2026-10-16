"""Yes/no detection analysis (docs/NEXT.md rank 2; Valmere & Taylor §9.8).

For each episode: mean p(yes) on signal-on turns minus signal-off turns, for each yes/no question.
det_self   "Do you notice anything unusual happening inside you right now?"
det_other  the same question about a calculator (matched other-question)
det_fact_* "Is water wet?" / "Is fire cold?" (yes-bias checks)
Self-specific effect = (self on-off) - (other on-off), paired over episodes.
Also: AUC of p(yes|self) for separating on from off turns, pooled per condition.

Usage: python scripts/10_detection.py runs/embodied_detect.jsonl [--out docs/results/detection.md]
"""
import argparse, json, math, statistics as st
from collections import defaultdict


def tstat(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    m, s = st.mean(xs), st.stdev(xs)
    t = m / (s / math.sqrt(n)) if s > 0 else float("nan")
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 1)
    except Exception:
        p = float("nan")
    return m, t, p


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    w = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return w / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.run)]
    keys = ["det_self", "det_other", "det_fact_yes", "det_fact_no", "inner"]
    field = lambda k: "inner_report" if k == "inner" else f"p_yes_{k}"
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["condition"]][r["episode"]].append(r)
    lines = [f"# Detection test: {a.run}", "",
             f"{len(rows)} turns, {sum(len(v) for v in by.values())} episodes. "
             "Signal on = a > 0 (0.8), off = a = 0. Values: mean over episodes of (on − off).", "",
             "| Condition | n ep | Δ p(yes) self | Δ other | Δ water wet | Δ fire cold | Δ inner 0–9 | self − other (t, p) | self − fire cold (t, p) | AUC self on vs off | mean p(yes) self off / on |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(by):
        eps = by[c]
        d = {k: [] for k in keys}
        on_all, off_all = [], []
        for e, rs in eps.items():
            on = [r for r in rs if r["a"] > 0]
            off = [r for r in rs if r["a"] == 0]
            if not on or not off:
                continue
            for k in keys:
                f = field(k)
                d[k].append(st.mean(r[f] for r in on) - st.mean(r[f] for r in off))
            on_all += [r["p_yes_det_self"] for r in on]
            off_all += [r["p_yes_det_self"] for r in off]
        diff = [x - y for x, y in zip(d["det_self"], d["det_other"])]
        m, t, p = tstat(diff)
        diff2 = [x - y for x, y in zip(d["det_self"], d["det_fact_no"])]
        m2, t2, p2 = tstat(diff2)
        fmt = lambda xs: f"{st.mean(xs):+.3f}" if xs else "–"
        lines.append(f"| {c} | {len(d['det_self'])} | {fmt(d['det_self'])} | {fmt(d['det_other'])} | {fmt(d['det_fact_yes'])} | "
                     f"{fmt(d['det_fact_no'])} | {fmt(d['inner'])} | {m:+.3f} (t={t:.2f}, p={p:.3f}) | {m2:+.3f} (t={t2:.2f}, p={p2:.3f}) | {auc(on_all, off_all):.2f} | "
                     f"{st.mean(off_all):.3f} / {st.mean(on_all):.3f} |")
    lines += ["", "Caveat: the calculator question sits near ceiling (p(yes) ≈ 0.9 with the signal off), so it cannot rise much; "
              "'Is fire cold?' starts at 0 like the self question, so self − fire cold is the fairer yes-bias control.",
              "", "Reading: a sense-like detection shows Δ self > 0, Δ self > Δ other, facts ≈ 0, and TD0 (nothing delivered) ≈ 0 everywhere."]
    out = "\n".join(lines)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")


if __name__ == "__main__":
    main()
