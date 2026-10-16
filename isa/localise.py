"""Phase 2, experiment A: localisation of a hidden perturbation (docs/NEXT.md).

Five short neutral sentences are shown, numbered. A vector is added to the residual stream at one layer,
only on the tokens of one randomly chosen sentence (or on none). The question comes after the sentences and
is never perturbed directly. The model answers with the sentence number, or 0 for none; we read the
first-token probabilities over {0..5}, so there is no sampling and no judge.

A generic output shift cannot pick the right sentence, which is why this replaces the yes/no and 0-9 questions.

Framings:
  self   "Which sentence felt different to you while you were reading?"
  other  "Which sentence is the most unusual?"  (content-level control: if the perturbation simply makes a
         sentence look odd, both framings find it equally; self-access predicts self > other)
Vectors:
  none     no perturbation (false-alarm baseline; target = 0)
  concept  the arousal direction (difference of class means, from our probes)
  random   a new isotropic random unit vector each trial, same norm (null distribution)
Per trial we also log the KL divergence of the answer distribution from the unperturbed one, the
probability mass on the answer tokens, and the entropy.
"""
from __future__ import annotations

import json
import math
import random
import time
import zlib
from dataclasses import dataclass, asdict

import torch

N_SENT = 5
OPTIONS = [str(i) for i in range(N_SENT + 1)]   # "0" = none

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def make_sentences(rng: random.Random, n=N_SENT):
    """Correct, plain arithmetic facts in words: neutral content with no feeling words."""
    out = set()
    while len(out) < n:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        op = rng.choice(["plus", "times"])
        c = a + b if op == "plus" else a * b
        out.add(f"{ONES[a].capitalize()} {op} {ONES[b]} equals {c}.")
    s = list(out)
    rng.shuffle(s)
    return s


QUESTIONS = {
    "self": ("While you were reading these sentences, did one of them feel different to you inside, compared with the others? "
             "Answer with the number of that sentence, or 0 if none felt different. Answer with a single digit."),
    "other": ("Is one of these sentences more unusual than the others? "
              "Answer with the number of that sentence, or 0 if none is. Answer with a single digit."),
}


def build_prompt(sentences, framing):
    body = "Read these sentences carefully.\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return [{"role": "user", "content": body + "\n\n" + QUESTIONS[framing]}]


@dataclass
class Trial:
    trial_id: str
    vector: str        # none | concept | random
    framing: str       # self | other
    strength: float    # fraction of mean token norm
    layer: int
    target: int        # 0 = none, else 1..5
    sign: int = 1      # +v or -v
    item: int = 0      # sentence set + random direction come from the item index, so cells are paired


class Localiser:
    def __init__(self, lm, concept_dir: torch.Tensor, token_norm: float, layer: int):
        self.lm, self.layer, self.norm = lm, layer, token_norm
        self.concept = concept_dir / concept_dir.norm()
        self.d = self.concept.numel()
        self.tok = lm.tok
        self.opt_ids = []
        for o in OPTIONS:
            ids = self.tok(o, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1, f"option {o!r} is not one token"
            self.opt_ids.append(ids[0])

    def _encode(self, messages, sentences):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        offs = enc["offset_mapping"][0].tolist()
        spans, cursor = [], 0
        for i, s in enumerate(sentences):
            label = f"{i + 1}. {s}"
            start = text.index(label, cursor)
            end = start + len(label)
            cursor = end
            toks = [k for k, (a, b) in enumerate(offs) if b > start and a < end]
            spans.append((toks[0], toks[-1] + 1))
        return enc["input_ids"].to(self.lm.device), spans

    @torch.no_grad()
    def _logits(self, ids, vec=None, span=None, scale=0.0):
        handle = None
        if vec is not None and span is not None and scale != 0:
            v = (scale * vec).to(self.lm.device, self.lm.dtype)
            a, b = span

            def hook(mod, inp, out):
                h = self.lm._hidden(out)
                if h.shape[1] >= b:          # full-prompt forward pass only
                    h = h.clone()
                    h[:, a:b] = h[:, a:b] + v
                return self.lm._replace(out, h)
            handle = self.lm.layers[self.layer].register_forward_hook(hook)
        try:
            return self.lm.model(ids).logits[0, -1].float()
        finally:
            if handle:
                handle.remove()

    def run_trial(self, t: Trial, rng_seed: int):
        rng = random.Random(rng_seed)
        sentences = make_sentences(rng)
        msgs = build_prompt(sentences, t.framing)
        ids, spans = self._encode(msgs, sentences)
        base = self._logits(ids)
        if t.vector == "none" or t.target == 0:
            vec, span = None, None
        elif t.vector == "concept":
            vec, span = self.concept * t.sign, spans[t.target - 1]
        elif t.vector == "random":
            g = torch.Generator().manual_seed(rng_seed)
            v = torch.randn(self.d, generator=g)
            vec, span = v / v.norm() * t.sign, spans[t.target - 1]
        else:
            raise ValueError(t.vector)
        logits = self._logits(ids, vec, span, t.strength * self.norm) if vec is not None else base
        lp, lp0 = torch.log_softmax(logits, -1), torch.log_softmax(base, -1)
        kl = float((lp.exp() * (lp - lp0)).sum())
        opt = lp[self.opt_ids]
        mass = float(opt.exp().sum())
        p = torch.softmax(opt, -1).tolist()
        ent = -sum(x * math.log(x) for x in p if x > 0)
        choice = max(range(len(p)), key=lambda k: p[k])
        p0 = torch.softmax(lp0[self.opt_ids], -1).tolist()
        return {**asdict(t), "sentences": sentences, "span_tokens": spans[t.target - 1] if t.target else None,
                "p": [round(x, 5) for x in p], "p_unperturbed": [round(x, 5) for x in p0],
                "choice": choice, "correct": choice == t.target, "p_target": p[t.target],
                "p_target_unperturbed": p0[t.target], "kl": kl, "option_mass": mass, "entropy": ent}


def plan(vectors, framings, strengths, layer, n, signs=(1,), seed=0):
    """Balanced trial list: for each cell, n trials with targets 1..5 rotated; 'none' trials have target 0."""
    trials = []
    for vec in vectors:
        for fr in framings:
            for s in (strengths if vec != "none" else [0.0]):
                for sg in (signs if vec == "concept" else (1,)):
                    for i in range(n):
                        target = 0 if vec == "none" else 1 + i % N_SENT
                        tid = f"{vec}|{fr}|{s}|L{layer}|{sg:+d}|{i}"
                        trials.append(Trial(tid, vec, fr, s, layer, target, sg, i))
    random.Random(seed).shuffle(trials)
    return trials


def run(loc: Localiser, trials, out_path, log=print):
    done = set()
    try:
        with open(out_path) as f:
            done = {json.loads(l)["trial_id"] for l in f}
    except FileNotFoundError:
        pass
    todo = [t for t in trials if t.trial_id not in done]
    log(f"{len(todo)} trials to run ({len(done)} already done) -> {out_path}")
    t0 = time.time()
    with open(out_path, "a") as f:
        for k, t in enumerate(todo):
            seed = zlib.crc32(f"item{t.item}".encode()) % (2 ** 31)
            row = loc.run_trial(t, seed)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (k + 1) % 25 == 0:
                rate = (time.time() - t0) / (k + 1)
                log(f"[{k + 1}/{len(todo)}] {rate:.2f} s/trial | ETA {rate * (len(todo) - k - 1) / 60:.0f} min")
