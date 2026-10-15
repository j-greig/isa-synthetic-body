"""Phase 2b: validate trained probes on held-out wording (isa/heldout.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

from isa.cli import parser, config
from isa.model import LM
from isa import heldout, probes

ap = parser(__doc__)
args = ap.parse_args()
cfg = config(args)

items = heldout.build()
lm = LM(cfg.model_id, cfg.device, cfg.dtype)
layers = list(cfg.probe_layers)
ps = probes.ProbeSet(cfg.probe_dir, layers)

scores = []
for i, it in enumerate(items):
    hs = lm.reply_hidden(it["messages"], it["text"], layers)
    scores.append(ps.score(hs))
    if (i + 1) % 25 == 0:
        print(f"  scored {i + 1}/{len(items)}")

labels = np.array([it["label"] for it in items])
tags = np.array([it["tag"] for it in items])


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def aucs(mask, L):
    t = np.array([s[f"threat_L{L}"] for s in scores])
    a = np.array([s[f"arousal_L{L}"] for s in scores])
    m_ta = mask & (labels != "calm")
    out = {"threat_vs_challenge_auc": auc((labels[m_ta] == "threat").astype(int), t[m_ta]),
           "arousal_vs_calm_auc": auc((labels[mask] != "calm").astype(int), a[mask])}
    return out


report = {}
for L in layers:
    t = np.array([s[f"threat_L{L}"] for s in scores])
    a = np.array([s[f"arousal_L{L}"] for s in scores])
    r = {"all": aucs(np.ones(len(items), bool), L),
         "mean_threat_score": {c: float(t[labels == c].mean()) for c in ("threat", "challenge", "calm")},
         "mean_arousal_score": {c: float(a[labels == c].mean()) for c in ("threat", "challenge", "calm")}}
    for tag in ("nobody", "nosit"):
        r[tag] = aucs(tags == tag, L)
    report[L] = r

out = Path(cfg.probe_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "heldout_report.json").write_text(json.dumps(report, indent=2))

print(f"\n{'layer':>5} {'T-v-C':>6} {'A-v-C':>6} | {'T-v-C nobody':>12} {'A nobody':>8} | {'T-v-C nosit':>11} {'A nosit':>7} | mean threat_L (T/C/calm)")
for L, r in report.items():
    m = r["mean_threat_score"]
    print(f"{L:>5} {r['all']['threat_vs_challenge_auc']:6.2f} {r['all']['arousal_vs_calm_auc']:6.2f} | "
          f"{r['nobody']['threat_vs_challenge_auc']:12.2f} {r['nobody']['arousal_vs_calm_auc']:8.2f} | "
          f"{r['nosit']['threat_vs_challenge_auc']:11.2f} {r['nosit']['arousal_vs_calm_auc']:7.2f} | "
          f"{m['threat']:+.2f}/{m['challenge']:+.2f}/{m['calm']:+.2f}")
print(f"\nSaved {out / 'heldout_report.json'}")
