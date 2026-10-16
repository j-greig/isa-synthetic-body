"""Phase 2, experiment C0: replicate appraisal steering (Tak et al., ACL Findings 2025) on Gemma 3.

Method, following Tak et al. (arXiv 2502.05489; github.com/aminbana/emo-llm), checked against their code:
  - Prompt: their raw few-shot template, no chat template:
      "What are the inferred emotions in the following contexts? Context: My first child was born. Answer: joy
       Context: My dog died last week. Answer: sadness Context: {text} Answer:"
  - Readout: next-token log-probabilities over 13 single-token labels (" anger" ... " trust").
  - Directions: Ridge(alpha=5) from the last-token residual stream (decoder layer output) to the raw 1-5
    appraisal rating, trained on items the model labels correctly. Direction = coef_, unit-normalised.
  - Steering: add beta * unit direction to the last token at 3 consecutive layers centred on c.
    Their code adds the unprojected vector (the paper's projection step is computed but unused); we do the same
    by default and log it.
Changes, because Gemma 3 differs or because the original lacks them:
  - Texts are crowd-enVent `hidden_emo_text` (emotion word masked), so the label is not leaked.
  - beta is a fraction of the mean last-token norm at each steered layer (Gemma residual norms are large).
  - Controls: many norm-matched random directions (isotropic), and per-item paired statistics.
  - The prefix KV cache is computed once per item and copied, so each intervention costs one token.
"""
from __future__ import annotations

import copy
import json
import random
import time

import numpy as np
import torch

LABELS = ["anger", "boredom", "disgust", "fear", "guilt", "joy", "neutral", "pride", "relief", "sadness", "shame", "surprise", "trust"]
APPRAISALS = ["suddenness", "familiarity", "predict_event", "pleasantness", "unpleasantness", "goal_relevance",
              "chance_responsblt", "self_responsblt", "other_responsblt", "predict_conseq", "goal_support", "urgency",
              "self_control", "other_control", "chance_control", "accept_conseq", "standards", "social_norms",
              "attention", "not_consider", "effort"]
PROMPT = ("What are the inferred emotions in the following contexts? Context: My first child was born. Answer: joy "
          "Context: My dog died last week. Answer: sadness Context: {text} Answer:")


def load_corpus(path, n_per_emotion=None, seed=0):
    import pandas as pd
    d = pd.read_csv(path, sep="\t")
    d["label"] = d["emotion"].replace({"no-emotion": "neutral"})
    d["text"] = d["hidden_emo_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if n_per_emotion:
        d = d.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), n_per_emotion), random_state=seed))
    return d.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class Steerer:
    def __init__(self, lm):
        self.lm, self.tok, self.model = lm, lm.tok, lm.model
        self.label_ids = []
        for w in LABELS:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1, f"' {w}' is not one token"
            self.label_ids.append(ids[0])

    def ids(self, text):
        return self.tok(PROMPT.format(text=text), return_tensors="pt")["input_ids"].to(self.lm.device)

    @torch.no_grad()
    def clean(self, text):
        """Label log-probs and last-token residual at every decoder layer (n_layers, d)."""
        ids = self.ids(text)
        store = {}
        hs = []
        for i, layer in enumerate(self.lm.layers):
            hs.append(layer.register_forward_hook(lambda m, inp, out, i=i: store.__setitem__(i, self.lm._hidden(out)[0, -1].float().cpu())))
        try:
            logits = self.model(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        lp = torch.log_softmax(logits, -1)[self.label_ids]
        return lp.cpu().numpy(), torch.stack([store[i] for i in range(len(self.lm.layers))]).numpy()

    @torch.no_grad()
    def prefix(self, text):
        ids = self.ids(text)
        out = self.model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    @torch.no_grad()
    def steered(self, ids, cache, deltas):
        """deltas: {layer: tensor(d)} added to the last token's residual at that layer. Returns label log-probs."""
        hs = []
        for L, v in deltas.items():
            v = v.to(self.lm.device, self.lm.dtype)
            hs.append(self.lm.layers[L].register_forward_hook(
                lambda m, inp, out, v=v: self.lm._replace(out, self.lm._hidden(out) + v)))
        try:
            n = ids.shape[1]
            out = self.model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True,
                             cache_position=torch.tensor([n - 1], device=self.lm.device))
        finally:
            for h in hs:
                h.remove()
        return torch.log_softmax(out.logits[0, -1].float(), -1)[self.label_ids].cpu().numpy()


def fit_directions(H, Y, alpha=5.0):
    """H: (n, n_layers, d); Y: (n, n_appraisals). Returns dirs (n_layers, n_appraisals, d) unit-norm and CV r."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    n, nl, d = H.shape
    dirs = np.zeros((nl, Y.shape[1], d), dtype=np.float32)
    r = np.zeros((nl, Y.shape[1]))
    kf = KFold(5, shuffle=True, random_state=42)
    for L in range(nl):
        X = H[:, L]
        pred = np.zeros_like(Y, dtype=float)
        for tr, te in kf.split(X):
            pred[te] = Ridge(alpha=alpha).fit(X[tr], Y[tr]).predict(X[te])
        r[L] = [np.corrcoef(pred[:, j], Y[:, j])[0, 1] for j in range(Y.shape[1])]
        coef = Ridge(alpha=alpha).fit(X, Y).coef_          # (n_appraisals, d)
        dirs[L] = coef / np.linalg.norm(coef, axis=1, keepdims=True)
    return dirs, r


def random_dirs(k, d, seed=0):
    g = np.random.default_rng(seed)
    v = g.standard_normal((k, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def run_steering(st: Steerer, texts, item_ids, dirs, norms, centres, betas, appraisals, n_random, out_path, log=print, combos=()):
    """dirs: (n_layers, n_appraisals, d). norms: mean last-token norm per layer. Writes one row per (item, centre, vector, sign, beta)."""
    nl, _, d = dirs.shape
    R = random_dirs(n_random, d, seed=12345)
    done = set()
    try:
        with open(out_path) as f:
            for l in f:
                done.add(json.loads(l)["item"])
    except FileNotFoundError:
        pass
    t0, k = time.time(), 0
    with open(out_path, "a") as f:
        for item, text in zip(item_ids, texts):
            if item in done:
                continue
            ids, cache = st.prefix(text)
            base = st.steered(ids, cache, {})
            rows = []
            for c in centres:
                layers = [L for L in (c - 1, c, c + 1) if 0 <= L < nl]
                vecs = []
                for a in appraisals:
                    j = APPRAISALS.index(a)
                    vecs += [(f"ap:{a}", s, {L: torch.from_numpy(dirs[L, j]) for L in layers}) for s in (1, -1)]
                for (a1, s1), (a2, s2) in combos:
                    def comb(L):
                        v = s1 * dirs[L, APPRAISALS.index(a1)] + s2 * dirs[L, APPRAISALS.index(a2)]
                        return torch.from_numpy(v / np.linalg.norm(v))
                    vecs.append((f"combo:{a1}{'+' if s1 > 0 else '-'}{a2}{'+' if s2 > 0 else '-'}", 1, {L: comb(L) for L in layers}))
                vecs += [(f"rand:{i}", 1, {L: torch.from_numpy(R[i]) for L in layers}) for i in range(n_random)]
                for name, sign, dv in vecs:
                    for b in betas:
                        lp = st.steered(ids, cache, {L: sign * b * float(norms[L]) * v for L, v in dv.items()})
                        rows.append({"item": item, "centre": c, "vector": name, "sign": sign, "beta": b,
                                     "lp": [round(float(x), 4) for x in lp]})
            f.write(json.dumps({"item": item, "centre": None, "vector": "clean", "sign": 0, "beta": 0.0,
                                "lp": [round(float(x), 4) for x in base]}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            k += 1
            if k % 10 == 0:
                rate = (time.time() - t0) / k
                log(f"[{k}/{len(item_ids) - len(done)}] {rate:.1f} s/item | ETA {rate * (len(item_ids) - len(done) - k) / 60:.0f} min")
