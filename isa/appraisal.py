"""Appraisal and emotion probes trained on crowd-enVent (Troiano, Oberlaender & Klinger 2023,
Computational Linguistics 49(1), doi 10.1162/coli_a_00461).

Each corpus item is a first-person event description rated on 21 appraisal dimensions (1-5) and
labelled with one of 13 emotions. We mean-pool the model's activations over a reply that states the
event ("I felt something when ...") and fit, per layer:
  (a) ridge regression -> the 21 appraisal ratings
  (b) logistic regression -> the 13 emotion labels
In the corpus the emotion word is masked with "..." (sometimes a whole "I felt ... when" opener is kept),
so the probes see the event, not the emotion word.
"""
from __future__ import annotations

import csv
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from isa import passages, probes

DIMS = ["suddenness", "familiarity", "predict_event", "pleasantness", "unpleasantness", "goal_relevance",
        "chance_responsblt", "self_responsblt", "other_responsblt", "predict_conseq", "goal_support",
        "urgency", "self_control", "other_control", "chance_control", "accept_conseq", "standards",
        "social_norms", "attention", "not_consider", "effort"]
EMOTIONS = ["anger", "boredom", "disgust", "fear", "guilt", "joy", "no-emotion", "pride", "relief",
            "sadness", "shame", "surprise", "trust"]

_OPENER = re.compile(r"^\s*(?:i\s+felt\s*)?(?:\.{2,}|…)?\s*(?:when|after|because|that)\b\s*", re.I)
_MASK = re.compile(r"\.{2,}|…")


def clean_text(raw: str) -> str:
    """Masked emotion words ('...') become 'something'; a leading 'I felt ... when' is dropped
    (we add our own opener) and the rest is kept as written."""
    t = raw.strip()
    t = _OPENER.sub("", t) if _OPENER.match(t) else t
    t = _MASK.sub("something", t).strip()
    return "I felt something when " + t


def load_corpus(path, n=None, seed=0):
    """-> list of {"text", "messages", "emotion", "appraisal": {dim: float}}; n per emotion if given."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if (r.get("hidden_emo_text") or "").strip()]
    rows = [r for r in rows if r["emotion"] in EMOTIONS]
    rng = random.Random(seed)
    if n is not None:
        by = {}
        for r in rows:
            by.setdefault(r["emotion"], []).append(r)
        rows = []
        for e in EMOTIONS:
            pool = by.get(e, [])
            rows += rng.sample(pool, min(n, len(pool)))
    items = []
    for r in rows:
        try:
            ap = {d: float(r[d]) for d in DIMS}
        except (KeyError, ValueError):
            continue
        items.append({"text": clean_text(r["hidden_emo_text"]), "emotion": r["emotion"], "appraisal": ap})
    for it in items:
        it["messages"] = passages._context(rng)
    return items


def train(lm, layers, out_dir, corpus_path, n_per_emotion=120, seed=0):
    items = load_corpus(corpus_path, n=n_per_emotion, seed=seed)
    Y = np.array([[it["appraisal"][d] for d in DIMS] for it in items])
    emo = np.array([it["emotion"] for it in items])
    print(f"Extracting activations for {len(items)} corpus items at layers {list(layers)} ...")
    X, norms = probes.extract(lm, items, layers)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    report = {}
    for L in layers:
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11)))
        pred = cross_val_predict(reg, X[L], Y, cv=cv.split(X[L], emo))
        r = {d: float(pearsonr(pred[:, i], Y[:, i])[0]) for i, d in enumerate(DIMS)}
        reg.fit(X[L], Y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        ep = cross_val_predict(clf, X[L], emo, cv=cv.split(X[L], emo))
        acc = float((ep == emo).mean())
        f1 = float(f1_score(emo, ep, average="macro"))
        clf.fit(X[L], emo)
        with open(out / f"layer_{L}.pkl", "wb") as f:
            pickle.dump({"reg": reg, "clf": clf, "dims": DIMS, "emotions": list(clf.classes_),
                         "token_norm": norms[L]}, f)
        report[L] = {"pearson_r": r, "mean_r": float(np.mean(list(r.values()))),
                     "emotion_cv_acc": acc, "emotion_cv_macro_f1": f1, "n": len(items)}
        print(f"\nLayer {L}: mean r {report[L]['mean_r']:.2f} | emotion acc {acc:.2f} | macro-F1 {f1:.2f}")
        for d, v in sorted(r.items(), key=lambda kv: -kv[1]):
            print(f"  {d:<18}{v:+.2f}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


class AppraisalSet:
    def __init__(self, probe_dir, layers):
        self.layers = list(layers)
        self.p = {}
        for L in self.layers:
            with open(Path(probe_dir) / f"layer_{L}.pkl", "rb") as f:
                self.p[L] = pickle.load(f)

    def score(self, hidden: dict) -> dict:
        out = {}
        for L in [l for l in self.layers if l in hidden]:
            P = self.p[L]
            x = hidden[L].mean(0).numpy()[None]
            for d, v in zip(P["dims"], P["reg"].predict(x)[0]):
                out[f"ap_{d}_L{L}"] = float(v)
            for e, v in zip(P["emotions"], P["clf"].predict_proba(x)[0]):
                out[f"emo_{e}_L{L}"] = float(v)
        return out


def maybe_load(probe_dir, layers):
    d = Path(probe_dir)
    if all((d / f"layer_{L}.pkl").exists() for L in layers):
        return AppraisalSet(d, layers)
    return None
