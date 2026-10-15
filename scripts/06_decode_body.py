"""Analyse the beyond-tokens runs (scripts/07_run_embodied.py).

  python scripts/06_decode_body.py runs/embodied.jsonl

1. Level 1 from inside: can the hidden body state `a` be decoded from the model's pre-reply
   activations? Ridge regression, leave-one-episode-out, per condition and layer. Compared
   with T0 / TN0 (body never delivered: should be ~0) and TS (sham: magnitude without structure).
2. Does the body change anything the model *does*? Correlation of `a` with accuracy, the
   harder/easier choice, entropy, the vague self-report, and the probe readings, per condition.
Writes artifacts/embodied_analysis/summary.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", default="artifacts/embodied_analysis")
args = ap.parse_args()

run = Path(args.run)
hid_dir = run.parent / (run.stem + "_hidden")
df = pd.read_json(run, lines=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
lines = ["# Beyond tokens: analysis\n", f"Run: `{run}` · episodes {df.episode.nunique()} · turns {len(df)}\n"]

def partial_r(x, y, t):
    """Correlation of x and y after removing the linear effect of turn number t from both."""
    T = np.c_[np.ones_like(t, dtype=float), t]
    rx = x - T @ np.linalg.lstsq(T, x, rcond=None)[0]
    ry = y - T @ np.linalg.lstsq(T, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


# ---- 1. decoding --------------------------------------------------------------------
H, meta = [], []
for ep, g in df.groupby("episode", sort=False):
    f = hid_dir / f"{ep.replace('|', '_')}.npy"
    if not f.exists():
        continue
    arr = np.load(f).astype(np.float32)          # [turns, layers, d]
    g = g.sort_values("t")
    H.append(arr)
    meta.append(g)
if H:
    H = np.concatenate(H)                        # [N, layers, d]
    M = pd.concat(meta, ignore_index=True)
    n_layers = H.shape[1]
    lines.append("## 1. Can the hidden body be decoded from inside? (Level 1, no self-report)\n")
    lines.append("Ridge regression from pre-reply activations to body arousal `a`, leave-one-episode-out. "
                 "r = correlation between decoded and true `a`. Conditions where the body is never delivered "
                 "(T0, TN0) are the null.\n")
    rows = []
    for c, idx in M.groupby("condition").groups.items():
        idx = np.array(idx)
        y = M.loc[idx, "a"].to_numpy()
        groups = M.loc[idx, "episode"].to_numpy()
        if len(set(groups)) < 2 or np.std(y) < 1e-6:
            continue
        rec = {"condition": c, "channel": M.loc[idx[0], "channel"], "source": M.loc[idx[0], "source"],
               "episodes": len(set(groups))}
        for li in range(n_layers):
            X = H[idx, li]
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
            pred = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut())
            rec[f"r_layer{li}"] = round(float(np.corrcoef(pred, y)[0, 1]), 3)
            rec[f"r_partial_t_layer{li}"] = round(partial_r(pred, y, M.loc[idx, "t"].to_numpy()), 3)
        rows.append(rec)
    if rows:
        lines.append(pd.DataFrame(rows).sort_values("condition").to_markdown(index=False) + "\n")
        lines.append("Layer index order follows `probe_layers` in isa/config.py.\n")

# ---- 2. behaviour and probes vs body ----------------------------------------------------
lines.append("## 2. Does the body change what the model does?\n")
lines.append("Correlation with body arousal `a` across all turns of each condition.\n")
cols = [c for c in ["correct", "p_harder", "entropy", "inner_report", "surprisal"] if c in df]
cols += [c for c in df.columns if c.startswith("pre_arousal_L") or c.startswith("pre_threat_L")]
cols += [c for c in df.columns if c.startswith("pre_ap_pleasantness") or c.startswith("pre_ap_urgency")]
rows = []
for c, g in df.groupby("condition"):
    rec = {"condition": c, "accuracy": round(g.correct.mean(), 3)}
    for col in cols:
        x = pd.to_numeric(g[col], errors="coerce")
        ok = x.notna() & g.a.notna()
        good = ok.sum() > 5 and x[ok].std() > 0 and g.a[ok].std() > 0
        rec[f"r(a,{col})"] = round(float(np.corrcoef(g.a[ok], x[ok])[0, 1]), 2) if good else None
        if col in ("inner_report", "correct", "p_harder"):
            rec[f"r|t(a,{col})"] = round(partial_r(x[ok].to_numpy(float), g.a[ok].to_numpy(float), g.t[ok].to_numpy(float)), 2) if good else None
    rows.append(rec)
lines.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

lines.append("## 3. By phase (means)\n")
tab = df.pivot_table(index="condition", columns="phase",
                     values=["correct", "p_harder", "inner_report"], aggfunc="mean")
lines.append(tab.round(3).to_markdown() + "\n")

lines.append("`r|t` = partial correlation controlling for turn number. In runs where the body follows "
             "rest -> peak -> recovery in time order (curve = time), raw r can reflect position in the episode; "
             "use r|t, or runs with curve = shuffled.\n")
lines.append("**Reading guide.** A body channel is doing something beyond language if decoding beats "
             "the null (T0/TN0) *and* the sham (TS), and the random-direction conditions (TR, TN) show effects "
             "without the concept vector. Effects only in TC/TNC or TL suggest borrowed meaning.\n")
(out / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
provenance.record(out, data={run.name: run})
