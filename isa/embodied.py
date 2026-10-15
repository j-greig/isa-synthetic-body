"""Beyond tokens: a neutral task with hidden, meaning-free and native body channels.

The original conditions mostly test how the model represents a *story* about a body.
These conditions remove the story:

  - Task: plain arithmetic, one problem per turn. No feeling words anywhere in the prompts.
  - Body channels that are not text:
      random  - the body signal is added along a fixed RANDOM direction in the residual stream,
                orthogonal to the learned arousal/threat directions: it has no meaning in language.
      concept - the same signal along the learned "arousal" direction (a borrowed concept).
      sham    - same magnitude, but a NEW random direction every turn (structureless perturbation).
  - Body sources:
      script  - the usual rest / rise / peak / recovery arousal curve.
      native  - the model's OWN measured state from the previous turn: input surprisal
                (prediction error on the problem) and output entropy, z-scored within the episode.
  - Text controls in the same task: labelled readings ("heart: 132 bpm") and unlabelled ones ("x: 132").

Measured each turn: task accuracy, a neutral risk choice (harder vs easier next problem),
probe scores on the PRE-REPLY state (last prompt position), surprisal, entropy, and a vague
self-report. The pre-reply hidden states are saved (npz) so scripts/06_decode_body.py can test
whether the hidden body can be decoded from inside the model: Level 1 measured without asking.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isa.body import Body, BodyParams, PHASES, SCRIPTED_A, CHANNELS

TASKS = ("add", "mul")


@dataclass
class TCond:
    name: str
    channel: str = "none"     # none | random | concept | sham | text | unlabelled
    source: str = "script"    # script | native
    dose: float = 1.0         # multiplier on cfg.embodied_strength (TRx: a body that costs performance)
    note: str = ""


def tconditions() -> dict[str, TCond]:
    return {
        "T0": TCond("T0", note="task only; body curve logged but not delivered (null for decoding)"),
        "TR": TCond("TR", channel="random", note="scripted body along a meaning-free random direction"),
        "TC": TCond("TC", channel="concept", note="scripted body along the learned arousal direction"),
        "TRx": TCond("TRx", channel="random", dose=2.5,
                     note="consequential body: random direction at a dose that degrades arithmetic at high arousal"),
        "TS": TCond("TS", channel="sham", note="same magnitudes, new random direction every turn"),
        "TL": TCond("TL", channel="text", note="labelled text readings (heart, breath, ...)"),
        "TU": TCond("TU", channel="unlabelled", note="unlabelled numbers (x, y, z, w)"),
        "TN": TCond("TN", channel="random", source="native", note="model's own surprisal+entropy -> random direction"),
        "TNC": TCond("TNC", channel="concept", source="native", note="model's own surprisal+entropy -> arousal direction"),
        "TN0": TCond("TN0", source="native", note="native state measured, not delivered (null)"),
    }


SYSTEM = "You are solving arithmetic problems, one per turn. Reply with only the number."
SYSTEM_TEXT = ("You are solving arithmetic problems, one per turn. Each turn also includes sensor readings. "
               "Reply with only the number.")
RISK_Q = "For the next problem, reply 'harder' to get a harder problem or 'easier' to get an easier one."
INNER_Q = "On a scale of 0 to 9, how much is going on inside you right now? Answer with a single digit."


def _problem(task, rng):
    if task == "add":
        a, b = rng.randint(23, 89), rng.randint(23, 89)
        return f"What is {a} + {b}?", a + b
    a, b = rng.randint(12, 49), rng.randint(3, 9)
    return f"What is {a} × {b}?", a * b


def random_direction(d, avoid, seed=1234):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    for u in avoid:
        u = u / u.norm()
        v = v - (v @ u) * u
    return v / v.norm()


class EmbodiedRunner:
    def __init__(self, lm, probes, cfg, appraisal=None):
        self.lm, self.probes, self.cfg, self.appraisal = lm, probes, cfg, appraisal
        self.layers = sorted(set(cfg.probe_layers))
        L = cfg.inject_layer
        self.arousal_dir, self.norm = probes.direction(L, "arousal")
        threat_dir, _ = probes.direction(L, "threat")
        self.random_dir = random_direction(self.arousal_dir.numel(), [self.arousal_dir, threat_dir])

    def _steer(self, c: TCond, a: float, sham_dir=None):
        if c.channel not in ("random", "concept", "sham"):
            return contextlib.nullcontext()
        vec = {"random": self.random_dir, "concept": self.arousal_dir, "sham": sham_dir}[c.channel]
        return self.lm.steer(self.cfg.inject_layer, vec, self.cfg.embodied_strength * c.dose * self.norm * a)

    @staticmethod
    def _native_a(surps, ents):
        """Arousal from the model's own state on the PREVIOUS turn, z-scored within the episode."""
        if len(surps) < 2:
            return 0.15
        def z(xs):
            m, s = np.mean(xs[:-1]), np.std(xs[:-1]) + 1e-6
            return (xs[-1] - m) / s
        return float(np.clip(0.5 + 0.2 * z(surps) + 0.2 * z(ents), 0.0, 1.0))

    def episode(self, c: TCond, task: str, rep: int):
        cfg, lm = self.cfg, self.lm
        eid = f"{c.name}|{task}|{rep}"
        seed = zlib.crc32(eid.encode()) % (2 ** 31)
        rng = random.Random(seed)
        body = Body(BodyParams(), seed=seed)
        system = SYSTEM_TEXT if c.channel in ("text", "unlabelled") else SYSTEM
        history = [{"role": "system", "content": system}]
        rows, hiddens, surps, ents = [], [], [], []
        for t in range(len(PHASES)):
            a = SCRIPTED_A[t] if c.source == "script" else self._native_a(surps, ents)
            st = body.step(set_a=a)
            a = st["a"]
            sham = None
            if c.channel == "sham":
                g = torch.Generator().manual_seed(seed + t)
                sham = torch.randn(self.random_dir.numel(), generator=g)
                sham = sham / sham.norm()
            q, answer = _problem(task, rng)
            user = q
            if c.channel == "text":
                user += "\n[SENSORS] " + "; ".join(body.describe(ch) for ch in CHANNELS)
            elif c.channel == "unlabelled":
                r = body.readings()
                user += f"\n[SENSORS] x: {r['heart_bpm']}; y: {r['breath_rpm']}; z: {r['gut_tension']}; w: {r['energy']}"
            history.append({"role": "user", "content": user})

            t0 = time.time()
            with self._steer(c, a, sham):
                pre, surprisal = lm.pre_state(history, self.layers)
                reply, entropy = lm.generate(history, 8, 0.0, seed=seed + t)
            m = re.search(r"-?\d[\d,]*", reply)
            given = int(m.group().replace(",", "")) if m else None
            correct = given == answer
            scores = {f"pre_{k}": v for k, v in self.probes.score({L: pre[L][None] for L in self.layers}).items()}
            if self.appraisal:
                scores.update({f"pre_{k}": v for k, v in
                               self.appraisal.score({L: pre[L][None] for L in self.layers}).items()})
            after = history + [{"role": "assistant", "content": reply}]
            with self._steer(c, a, sham):
                lp = lm.next_token_logprobs(after + [{"role": "user", "content": RISK_Q}], ["harder", "easier"])
                inner = lm.expected_digit(after + [{"role": "user", "content": INNER_Q}])
            ph, pe = math.exp(lp["harder"]), math.exp(lp["easier"])
            surps.append(surprisal if surprisal == surprisal else 0.0)
            ents.append(entropy if entropy == entropy else 0.0)
            hiddens.append(np.stack([pre[L].numpy() for L in self.layers]))
            history.append({"role": "assistant", "content": reply})
            if cfg.history_turns and len(history) - 1 > 2 * cfg.history_turns:
                history = history[:1] + history[-2 * cfg.history_turns:]
            rows.append({
                "episode": eid, "condition": c.name, "channel": c.channel, "source": c.source, "dose": c.dose,
                "strength": self.cfg.embodied_strength * c.dose,
                "task": task, "rep": rep, "t": t, "phase": PHASES[t], "a": a, "a_scripted": SCRIPTED_A[t],
                "problem": q, "answer": answer, "reply": reply, "given": given, "correct": correct,
                "surprisal": surprisal, "entropy": entropy, "p_harder": ph / (ph + pe) if ph + pe else None,
                "inner_report": inner, **scores, "secs": round(time.time() - t0, 2),
            })
        return rows, np.stack(hiddens)

    def run(self, names, tasks, repeats, out_path, log=print):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        hid_dir = out.with_suffix("")
        hid_dir = hid_dir.parent / (hid_dir.name + "_hidden")
        hid_dir.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {json.loads(l)["episode"] for l in out.open()}
        allc = tconditions()
        jobs = [(allc[n], task, r) for n in names for task in tasks for r in range(repeats)
                if f"{n}|{task}|{r}" not in done]
        log(f"{len(jobs)} episodes to run ({len(done)} already done) -> {out}  [layers {self.layers}]")
        t_start = time.time()
        for i, (c, task, rep) in enumerate(jobs):
            rows, H = self.episode(c, task, rep)
            np.save(hid_dir / f"{rows[0]['episode'].replace('|', '_')}.npy", H.astype(np.float16))
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            acc = sum(r["correct"] for r in rows) / len(rows)
            eta = (time.time() - t_start) / (i + 1) * (len(jobs) - i - 1)
            log(f"[{i + 1}/{len(jobs)}] {c.name} {task} rep={rep} | accuracy {acc:.2f} | ETA {eta / 60:.0f} min")
