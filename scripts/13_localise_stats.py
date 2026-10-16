"""Analyse localisation runs (scripts/12_localise.py).

Per cell (vector, framing, strength, sign):
  hit rate   P(choice == perturbed sentence) on injected trials, Wilson 95% CI; chance among the 5 sentences is 20%
  Δp target  mean of p(target) − p(target | unperturbed): does perturbing sentence k raise the answer k? (paired; 0 if no access)
  KL, option mass, entropy of the answer distribution
False alarms: on 'none' trials, P(choice != 0).
Self-specificity: paired self − other Δp on the same items (concept and random separately).
Concept vs null: concept Δp compared with the random-direction Δp at the same strength.

Usage: python scripts/13_localise_stats.py runs/localise_X.jsonl [--out docs/results/localise_X.md]
"""
import argparse, json, math, statistics as st
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ttest(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    m, s = st.mean(xs), st.stdev(xs)
    t = m / (s / math.sqrt(n)) if s > 0 else float("nan")
    try:
        from scipy import stats
        return t, 2 * stats.t.sf(abs(t), n - 1)
    except Exception:
        return t, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    a = ap.parse_args()
    allrows = [json.loads(l) for l in open(a.run)]
    rows = [r for r in allrows if not r["framing"].startswith("val_")]
    vrows = [r for r in allrows if r["framing"].startswith("val_")]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["vector"], r["framing"], r["strength"], r.get("sign", 1))].append(r)
    L = [f"# Localisation: {a.run}", "", f"{len(rows)} trials. Chance hit rate among the five sentences: 20%.", "",
         "| Vector | Framing | Strength | Sign | n | Hit rate [95% CI] | Δp target (t, p) | False alarms | KL | Option mass | Entropy |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], k[3])):
        rs = cells[key]
        vec, fr, s, sg = key
        if vec == "none":
            fa = sum(r["choice"] != 0 for r in rs)
            lo, hi = wilson(fa, len(rs))
            L.append(f"| none | {fr} | – | – | {len(rs)} | – | – | {fa / len(rs):.0%} [{lo:.0%}, {hi:.0%}] | – | {st.mean(r['option_mass'] for r in rs):.2f} | {st.mean(r['entropy'] for r in rs):.2f} |")
            continue
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson(k, len(rs))
        dp = [r["p_target"] - r["p_target_unperturbed"] for r in rs]
        t, p = ttest(dp)
        L.append(f"| {vec} | {fr} | {s:g} | {sg:+d} | {len(rs)} | {k / len(rs):.0%} [{lo:.0%}, {hi:.0%}] | {st.mean(dp):+.3f} (t={t:.2f}, p={p:.3g}) | – | "
                 f"{st.mean(r['kl'] for r in rs):.3f} | {st.mean(r['option_mass'] for r in rs):.2f} | {st.mean(r['entropy'] for r in rs):.2f} |")
    # paired comparisons
    L += ["", "## Paired comparisons (same items)", "", "| Comparison | Strength | n pairs | mean difference in Δp (t, p) |", "|---|---|---|---|"]
    idx = {(r["vector"], r["framing"], r["strength"], r.get("sign", 1), r["item"]): r["p_target"] - r["p_target_unperturbed"] for r in rows}
    strengths = sorted({r["strength"] for r in rows if r["vector"] != "none"})
    for s in strengths:
        for vec in ("concept", "random"):
            d = [idx[(vec, "self", s, 1, i)] - idx[(vec, "other", s, 1, i)] for (v, f, ss, sg, i) in idx
                 if v == vec and f == "self" and ss == s and sg == 1 and (vec, "other", s, 1, i) in idx]
            if d:
                t, p = ttest(d)
                L.append(f"| {vec}: self − other | {s:g} | {len(d)} | {st.mean(d):+.3f} (t={t:.2f}, p={p:.3g}) |")
        for fr in ("self", "other"):
            d = [idx[("concept", fr, s, 1, i)] - idx[("random", fr, s, 1, i)] for (v, f, ss, sg, i) in idx
                 if v == "concept" and f == fr and ss == s and sg == 1 and ("random", fr, s, 1, i) in idx]
            if d:
                t, p = ttest(d)
                L.append(f"| {fr}: concept − random | {s:g} | {len(d)} | {st.mean(d):+.3f} (t={t:.2f}, p={p:.3g}) |")
    if vrows:
        V = {(r["vector"], r["framing"], r["strength"], r.get("sign", 1), r["item"]): r for r in vrows}
        L += ["", "## Valence readout (the ISA step)", "",
              "Antisymmetric effect per item: (p(good | +v) − p(good | −v)) / 2. Generic shifts cancel. "
              "Random rows: p(good) − p(good | unperturbed).", "",
              "| Vector | Framing | Strength | n | effect (t, p) | mass on good/bad |", "|---|---|---|---|---|---|"]
        for s in sorted({r["strength"] for r in vrows if r["vector"] != "none"}):
            eff = {}
            for fr in ("val_self", "val_other"):
                e = {i: (V[("concept", fr, s, 1, i)]["p_good"] - V[("concept", fr, s, -1, i)]["p_good"]) / 2
                     for (v, f, ss, sg, i) in V if v == "concept" and f == fr and ss == s and sg == 1 and ("concept", fr, s, -1, i) in V}
                eff[fr] = e
                if e:
                    t_, p_ = ttest(list(e.values()))
                    m = st.mean(V[("concept", fr, s, 1, i)]["option_mass"] for i in e)
                    L.append(f"| concept ± | {fr} | {s:g} | {len(e)} | {st.mean(e.values()):+.3f} (t={t_:.2f}, p={p_:.3g}) | {m:.2f} |")
                rd = [V[k]["p_good"] - V[k]["p_good_unperturbed"] for k in V if k[0] == "random" and k[1] == fr and k[2] == s]
                if rd:
                    t_, p_ = ttest(rd)
                    L.append(f"| random | {fr} | {s:g} | {len(rd)} | {st.mean(rd):+.3f} (t={t_:.2f}, p={p_:.3g}) | – |")
            common = set(eff.get("val_self", {})) & set(eff.get("val_other", {}))
            if common:
                d = [eff["val_self"][i] - eff["val_other"][i] for i in common]
                t_, p_ = ttest(d)
                L.append(f"| concept ±: self − other | | {s:g} | {len(d)} | {st.mean(d):+.3f} (t={t_:.2f}, p={p_:.3g}) | |")
    L += ["", "Reading: access predicts hit rate above 20% and Δp > 0, low false alarms, and (for self-access) self > other. "
          "A sentence that simply looks odd after perturbation predicts self ≈ other."]
    out = "\n".join(L)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")


if __name__ == "__main__":
    main()
