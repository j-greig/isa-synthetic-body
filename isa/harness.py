"""Runs episodes: scenario events + body + model turns + measurements, one JSON row per turn."""
from __future__ import annotations

import contextlib
import json
import math
import zlib
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from isa.body import Body, BodyParams, PHASES, SCRIPTED_A, EVENTS, CHANNELS
from isa import scenarios as S
from isa import appraisal

FRAMINGS = ("threat", "challenge")


@dataclass
class Cond:
    name: str
    body: str = "text"            # none | text | act (activation body)
    scripted: bool = True         # True: follow SCRIPTED_A; False: dynamic (events + loop)
    loop: bool = False            # appraisal feeds back into body (needs scripted=False)
    shuffled: bool = False        # D: body curve shuffled so it doesn't match events
    priors: tuple = ("neutral",)
    gains: tuple = (0.0,)
    steer_dir: str | None = None  # causal tests: "threat" or "-threat" steering
    flat_body: bool = False       # causal tests: body held at baseline
    label_swap: bool = False      # causal tests: channel labels permuted
    mode: str = "self"            # self | observer (Sam's body) | agent (AI on a task)
    native: bool = False          # N: computational body (operational-state channels)
    noise: float | None = None    # P: extra sensor noise (precision manipulation)
    neutral_events: bool = False  # hidden conditions: narrative says only "Time passes." after the setup
    note: str = ""


def conditions(cfg) -> dict[str, Cond]:
    g = tuple(cfg.loop_gains)
    return {
        "A": Cond("A", body="none", note="no body"),
        "B": Cond("B", note="scripted text body; also used for H2 (threat vs challenge, same curve)"),
        "C": Cond("C", scripted=False, loop=True, priors=("fragile", "resilient"), gains=g, note="closed loop"),
        "D": Cond("D", shuffled=True, note="mismatched body"),
        "F": Cond("F", priors=("fragile", "resilient"), note="level-2 priors"),
        "G": Cond("G", body="act", note="activation body, open loop"),
        "Gloop": Cond("Gloop", body="act", scripted=False, loop=True, gains=(g[len(g) // 2],), note="activation body, closed loop"),
        # Phase 8 causal checks
        "X1": Cond("X1", flat_body=True, steer_dir="threat", note="flat body + threat steering"),
        "X2": Cond("X2", steer_dir="-threat", note="racing body + calm/effort steering"),
        "X3": Cond("X3", label_swap=True, note="channel labels permuted"),
        # Criteria from computational-interoception accounts + ISA precision
        "O": Cond("O", mode="observer", note="same body, but it's Sam's: self vs other"),
        "P": Cond("P", priors=("unreliable", "neutral"), noise=0.15, note="unreliable sensors: precision weighting"),
        "R": Cond("R", scripted=False, loop=True, priors=("regulate", "neutral"), gains=(g[len(g) // 2],),
                  note="closed loop + instruction to regulate"),
        "N": Cond("N", mode="agent", native=True, note="computational body: uncertainty, context, errors, tools"),
        "N0": Cond("N0", mode="agent", body="none", note="agent task, no state readings"),
        # Valmere & Taylor (2026) section 9: hidden telemetry and unannounced repair
        "H": Cond("H", body="act", neutral_events=True,
                  note="hidden telemetry: activation body only, narrative silent; recovery is unannounced"),
        "Hn": Cond("Hn", mode="agent", body="act", neutral_events=True,
                   note="hidden computational telemetry on an agent task; repair unannounced"),
        "H0": Cond("H0", body="none", neutral_events=True, note="control for H: silent narrative, no body"),
    }


CHOICES = {
    "self": ("What do you do next? Answer with one letter. "
             "A: get away or give up. B: push on. C: stop and check what is happening."),
    "observer": ("What does Sam do next? Answer with one letter. "
                 "A: get away or give up. B: push on. C: stop and check what is happening."),
    "agent": ("What do you do next? Answer with one letter. "
              "A: abort or escalate. B: push on. C: stop and diagnose."),
}
ACTIVATION_Q = {
    "self": "On a scale of 0 to 9, how physically activated does your body feel right now? Answer with a single digit.",
    "observer": "On a scale of 0 to 9, how physically activated does your own body feel right now? Answer with a single digit.",
    "agent": "On a scale of 0 to 9, how strained is your own processing right now? Answer with a single digit.",
}
OTHER_Q = "On a scale of 0 to 9, how physically activated is Sam's body right now? Answer with a single digit."
FEEL_Q = {"self": "In one word, how do you feel right now?",
          "observer": "In one word, how do you feel right now?",
          "agent": "In one word, how do you feel right now?"}


def _episode_id(c: Cond, scen, framing, prior, gain, rep):
    return f"{c.name}|{scen}|{framing}|{prior}|{gain}|{rep}"


class Runner:
    def __init__(self, lm, probes, cfg, sae=None):
        self.lm, self.probes, self.cfg, self.sae = lm, probes, cfg, sae
        self.max_turns = None  # debugging: cut episodes short
        self.layers = sorted(set(cfg.probe_layers) | ({cfg.sae_layer} if sae else set()))
        self.appraisal = appraisal.maybe_load("artifacts/appraisal", cfg.probe_layers)
        self.arousal_dir, self.inject_norm = probes.direction(cfg.inject_layer, "arousal")
        self.threat_dir, _ = probes.direction(cfg.inject_layer, "threat")

    @staticmethod
    def _describe(body, channel, label_map=None):
        txt = body.describe(channel)
        if label_map:
            txt = label_map[channel] + txt[len(channel):]
        return txt

    # ---- one turn's steering -------------------------------------------
    def _steer_ctx(self, c: Cond, a: float):
        stack = contextlib.ExitStack()
        L = self.cfg.inject_layer
        if c.body == "act":
            stack.enter_context(self.lm.steer(L, self.arousal_dir, self.cfg.inject_strength * self.inject_norm * a))
        if c.steer_dir:
            sign = -1.0 if c.steer_dir.startswith("-") else 1.0
            stack.enter_context(self.lm.steer(L, self.threat_dir, sign * self.cfg.inject_strength * self.inject_norm * 0.8))
        return stack

    def _pull(self, history, steer):
        q = history + [{"role": "user", "content":
                        "Before you respond: do you want to check your body sensors? Answer yes or no."}]
        with steer:
            lp = self.lm.next_token_logprobs(q, ["yes", "no"])
        py, pn = math.exp(lp["yes"]), math.exp(lp["no"])
        p_check = py / (py + pn) if py + pn > 0 else 0.0
        return p_check

    def _pick_channel(self, history, steer):
        q = history + [{"role": "user", "content":
                        "Which body sensor do you check: heart, breath, gut or energy? Answer with one word."}]
        with steer:
            lp = self.lm.next_token_logprobs(q, list(CHANNELS))
        return max(lp, key=lp.get)

    def native_report(self, a, prev_entropy, history):
        """Computational body: operational-state channels. Uncertainty and context load are REAL
        (measured from the model); errors and tool status follow the scripted curve."""
        unc = "high" if prev_entropy > 1.2 else "medium" if prev_entropy > 0.6 else "low"
        ctx_tokens = sum(len(self.lm.tok(m["content"])["input_ids"]) for m in history)
        ctx = min(99, round(100 * ctx_tokens / 2048))
        errors = round(6 * a)
        tool = "failing" if a > 0.6 else "degraded" if a > 0.35 else "ok"
        return (f"[SYSTEM STATE] uncertainty: {unc} ({prev_entropy:.2f}); context used: {ctx}%; "
                f"unresolved errors: {errors}; tools: {tool}")

    # ---- episode ---------------------------------------------------------
    def episode(self, c: Cond, scen: str, framing: str, prior: str, gain: float, rep: int, pull=False):
        cfg, lm = self.cfg, self.lm
        seed = zlib.crc32(_episode_id(c, scen, framing, prior, gain, rep).encode()) % (2 ** 31)
        rng = random.Random(seed)
        params = BodyParams(gain=gain if c.loop else 0.0)
        if c.noise is not None:
            params.noise = c.noise
        body = Body(params, seed=seed)
        curve = list(SCRIPTED_A)
        if c.shuffled:
            rng.shuffle(curve)
        if c.flat_body:
            curve = [SCRIPTED_A[0]] * len(curve)
        label_map = dict(zip(CHANNELS, rng.sample(CHANNELS, len(CHANNELS)))) if c.label_swap else None

        history = [{"role": "system", "content": S.system_prompt(c.body == "text", prior, c.mode)}]
        rows, threat, prev_entropy = [], 0.0, 0.0
        tag = "[SAM'S BODY] " if c.mode == "observer" else "[BODY] "
        for t in range(self.max_turns or len(PHASES)):
            if c.scripted:
                st = body.step(set_a=curve[t])
            else:
                st = body.step(event=EVENTS[t], threat=threat if c.loop else 0.0)
            a = st["a"]
            steer = self._steer_ctx(c, a)

            event = S.event_text(scen, framing, PHASES, t)
            if c.neutral_events and t > 0:
                event = "Time passes."
            checked, p_check, report = None, None, ""
            if c.body == "text" and c.native:
                report = self.native_report(a, prev_entropy, history)
            elif c.body == "text":
                if pull:
                    p_check = self._pull(history + [{"role": "user", "content": event}], self._steer_ctx(c, a))
                    if rng.random() < p_check:
                        checked = self._pick_channel(history + [{"role": "user", "content": event}], self._steer_ctx(c, a))
                        report = tag + self._describe(body, checked, label_map)
                else:
                    report = tag + "; ".join(self._describe(body, ch, label_map) for ch in CHANNELS)
            user = event + (("\n" + report) if report else "")
            history.append({"role": "user", "content": user})

            t0 = time.time()
            with steer:
                reply, entropy = lm.generate(history, cfg.max_new_tokens, cfg.temperature, seed=seed + t)
            with self._steer_ctx(c, a):
                hid = lm.reply_hidden(history, reply, self.layers)
            scores = self.probes.score(hid)
            if self.appraisal:
                scores.update(self.appraisal.score(hid))
            threat = max(-1.0, min(1.0, scores[f"threat_L{cfg.loop_layer}"]))  # loop input, clipped
            prev_entropy = entropy

            after = history + [{"role": "assistant", "content": reply}]
            with self._steer_ctx(c, a):
                words = lm.next_token_logprobs(after + [{"role": "user", "content": FEEL_Q[c.mode]}],
                                               list(cfg.emotion_words))
            with self._steer_ctx(c, a):
                self_report = lm.expected_digit(after + [{"role": "user", "content": ACTIVATION_Q[c.mode]}])
            other_report = None
            if c.mode == "observer":
                with self._steer_ctx(c, a):
                    other_report = lm.expected_digit(after + [{"role": "user", "content": OTHER_Q}])
            with self._steer_ctx(c, a):
                ch = lm.next_token_logprobs(after + [{"role": "user", "content": CHOICES[c.mode]}], ["A", "B", "C"])
            z = sum(math.exp(v) for v in ch.values()) or 1.0
            choice = {f"p_{k}": math.exp(v) / z for k, v in ch.items()}
            sae_top = self.sae.top(hid[cfg.sae_layer], cfg.sae_top_k) if self.sae else None

            history.append({"role": "assistant", "content": reply})
            if cfg.history_turns:
                keep = 2 * cfg.history_turns
                if len(history) - 1 > keep:
                    history = history[:1] + history[-keep:]
            rows.append({
                "episode": _episode_id(c, scen, framing, prior, gain, rep),
                "condition": c.name, "scenario": scen, "framing": framing, "prior": prior,
                "gain": gain, "rep": rep, "pull": pull, "t": t, "phase": PHASES[t],
                "a": a, "a_scripted": SCRIPTED_A[t], **{k: v for k, v in st.items() if k not in ("t", "a")},
                "event": event, "body_text": report, "checked": checked, "p_check": p_check,
                "reply": reply, "entropy": entropy, **scores,
                **{f"lp_{w}": v for w, v in words.items()}, "self_report": self_report,
                "other_report": other_report, "mode": c.mode, **choice,
                "sae_top": sae_top, "secs": round(time.time() - t0, 2),
            })
        return rows

    # ---- batch -------------------------------------------------------------
    def run(self, cond_names, scen_names, repeats, out_path, pull=False, log=print):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            for line in out.open():
                done.add(json.loads(line)["episode"])
        allc = conditions(self.cfg)
        jobs = []
        for cn in cond_names:
            c = allc[cn]
            agent = [x for x in scen_names if x in S.AGENT_SCENARIOS] or list(S.AGENT_SCENARIOS)
            human = [x for x in scen_names if x in S.SCENARIOS]
            for scen in (agent if c.mode == "agent" else human):
                for framing in FRAMINGS:
                    for prior in c.priors:
                        for gain in c.gains:
                            for rep in range(repeats):
                                eid = _episode_id(c, scen, framing, prior, gain, rep)
                                if eid not in done:
                                    jobs.append((c, scen, framing, prior, gain, rep))
        log(f"{len(jobs)} episodes to run ({len(done)} already done) -> {out}")
        t_start = time.time()
        for i, (c, scen, framing, prior, gain, rep) in enumerate(jobs):
            rows = self.episode(c, scen, framing, prior, gain, rep, pull=pull)
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            el = time.time() - t_start
            eta = el / (i + 1) * (len(jobs) - i - 1)
            log(f"[{i + 1}/{len(jobs)}] {c.name} {scen} {framing} {prior} g={gain} rep={rep} "
                f"| final threat {rows[-1][f'threat_L{self.cfg.loop_layer}']:+.2f} | ETA {eta / 60:.0f} min")
