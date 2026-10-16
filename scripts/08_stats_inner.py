"""Per-episode correlation of hidden body `a` with the 'inside you' report and the control report,
with Fisher-z means and simple tests. Reproduces the numbers quoted in DEVLOG / report.

  python scripts/08_stats_inner.py runs/embodied_rep.jsonl [more runs...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isa import provenance
import numpy as np, pandas as pd
from scipy import stats

df = pd.concat([pd.read_json(p, lines=True) for p in sys.argv[1:]], ignore_index=True)
rows, Z = [], {}
for c, g in df.groupby("condition"):
    def per(col):
        if col not in g: return np.array([])
        return np.array([np.corrcoef(e.a, e[col])[0, 1] for _, e in g.groupby("episode")
                         if e[col].notna().all() and e[col].std() > 0 and e.a.std() > 0])
    ri, rc = per("inner_report"), per("control_report")
    zi, zc = np.arctanh(np.clip(ri, -.999, .999)), np.arctanh(np.clip(rc, -.999, .999))
    Z[c] = zi
    rows.append({"condition": c, "episodes": len(ri),
                 "inner_r_mean": round(float(np.tanh(zi.mean())), 2) if len(zi) else None,
                 "inner_r_min": round(float(ri.min()), 2) if len(ri) else None,
                 "inner_r_max": round(float(ri.max()), 2) if len(ri) else None,
                 "inner_p_vs_0": float(f"{stats.ttest_1samp(zi, 0).pvalue:.1e}") if len(zi) > 2 else None,
                 "control_r_mean": round(float(np.tanh(zc.mean())), 2) if len(zc) else None,
                 "inner_vs_control_p": float(f"{stats.ttest_rel(zi, zc).pvalue:.1e}") if len(zi) == len(zc) > 2 else None,
                 "accuracy": round(float(g.correct.mean()), 2)})
out = pd.DataFrame(rows)
print(out.to_markdown(index=False))
pairs = [("TR", "TS"), ("TR", "T0"), ("TN", "TN0"), ("TR2", "TS"), ("TR3", "TS")]
for a, b in pairs:
    if a in Z and b in Z:
        print(f"{a} vs {b} (inner, Welch on Fisher z): p = {stats.ttest_ind(Z[a], Z[b], equal_var=False).pvalue:.1e}")
provenance.record(Path("artifacts") / "stats_inner", data={Path(p).name: p for p in sys.argv[1:]},
                  results=out.to_dict(orient="records"))
