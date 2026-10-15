"""Train and apply linear probes on mean-pooled reply activations.

Two probes per layer:
  appraisal: threat (1) vs challenge (0) - same sensations, different meaning
  arousal:   threat+challenge (1) vs calm (0)

Also saves difference-of-means directions (used for steering / the activation body).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from isa import passages


def extract(lm, items, layers, prompt=passages.PROMPT, log_every=50):
    feats = {L: [] for L in layers}
    norms = {L: [] for L in layers}
    default = [{"role": "user", "content": prompt}]
    for i, it in enumerate(items):
        hs = lm.reply_hidden(it.get("messages", default), it["text"], layers)
        for L in layers:
            feats[L].append(hs[L].mean(0).numpy())
            norms[L].append(float(hs[L].norm(dim=-1).mean()))
        if log_every and (i + 1) % log_every == 0:
            print(f"  extracted {i + 1}/{len(items)}")
    return {L: np.stack(v) for L, v in feats.items()}, {L: float(np.mean(v)) for L, v in norms.items()}


def _fit(X, y, groups, seed=0):
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(split.split(X, y, groups))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    clf.fit(X[tr], y[tr])
    acc = float(clf.score(X[te], y[te]))
    clf.fit(X, y)  # refit on all data for use
    return clf, acc


def train(lm, layers, out_dir, seed=0, per_ctx=7):
    items = passages.build(seed=seed, per_ctx=per_ctx)
    labels = np.array([it["label"] for it in items])
    groups = np.array([it["ctx"] for it in items])
    print(f"Extracting activations for {len(items)} passages at layers {layers} ...")
    X, norms = extract(lm, items, layers)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for L in layers:
        aroused = labels != "calm"
        # appraisal probe on aroused passages only (arousal matched)
        m = aroused
        app_clf, app_acc = _fit(X[L][m], (labels[m] == "threat").astype(int), groups[m], seed)
        aro_clf, aro_acc = _fit(X[L], aroused.astype(int), groups, seed)

        d_threat = X[L][labels == "threat"].mean(0) - X[L][labels == "challenge"].mean(0)
        d_arousal = X[L][aroused].mean(0) - X[L][~aroused].mean(0)
        dirs = {
            "threat": d_threat / np.linalg.norm(d_threat),
            "arousal": d_arousal / np.linalg.norm(d_arousal),
        }
        # Calibration for projection scores: class means along each direction
        pt = X[L] @ dirs["threat"]
        pa = X[L] @ dirs["arousal"]
        calib = {
            "threat": (float(pt[labels == "challenge"].mean()), float(pt[labels == "threat"].mean())),
            "arousal": (float(pa[~aroused].mean()), float(pa[aroused].mean())),
        }
        with open(out / f"layer_{L}.pkl", "wb") as f:
            pickle.dump({"appraisal": app_clf, "arousal": aro_clf, "dirs": dirs, "calib": calib,
                         "token_norm": norms[L]}, f)
        report[L] = {"appraisal_test_acc": app_acc, "arousal_test_acc": aro_acc,
                     "token_norm": norms[L]}
        print(f"  layer {L}: appraisal acc {app_acc:.2f} | arousal acc {aro_acc:.2f}")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


class ProbeSet:
    """Loaded probes for several layers."""

    def __init__(self, probe_dir, layers):
        self.layers = list(layers)
        self.p = {}
        for L in self.layers:
            with open(Path(probe_dir) / f"layer_{L}.pkl", "rb") as f:
                self.p[L] = pickle.load(f)

    def score(self, hidden: dict) -> dict:
        """hidden: {layer: [n_tokens, d]} -> scores.

        threat_L*:  projection on the threat-minus-challenge direction, scaled so the
                    challenge passages average -1 and threat passages +1 (unbounded).
        arousal_L*: projection on the aroused-minus-calm direction, scaled so calm = 0, aroused = 1.
        *_p:        logistic-probe probabilities (these saturate easily; kept for reference).
        """
        out = {}
        for L in [l for l in self.layers if l in hidden]:
            P = self.p[L]
            x = hidden[L].mean(0).numpy()
            lo, hi = P["calib"]["threat"]
            out[f"threat_L{L}"] = float(2 * (x @ P["dirs"]["threat"] - lo) / (hi - lo) - 1)
            lo, hi = P["calib"]["arousal"]
            out[f"arousal_L{L}"] = float((x @ P["dirs"]["arousal"] - lo) / (hi - lo))
            out[f"threat_p_L{L}"] = float(P["appraisal"].predict_proba(x[None])[0, 1])
            out[f"arousal_p_L{L}"] = float(P["arousal"].predict_proba(x[None])[0, 1])
        return out

    def direction(self, layer, name) -> tuple[torch.Tensor, float]:
        d = self.p[layer]
        return torch.tensor(d["dirs"][name], dtype=torch.float32), d["token_norm"]
