"""Generate a fake runs file with every field the harness writes (no model needed).

  python tests/make_fake_runs.py [runs/fake.jsonl]
"""
import json
import math
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa.body import Body, BodyParams, PHASES, SCRIPTED_A, EVENTS, CHANNELS
from isa.config import Config
from isa.harness import conditions, FRAMINGS, _episode_id

HUMAN = ("crowd", "stage")
AGENT = ("deploy", "research")


def make_rows(repeats=2, scenarios=2, pull_B=True):
    cfg = Config()
    L = cfg.loop_layer
    rows = []
    for c in conditions(cfg).values():
        scens = AGENT if c.mode == "agent" else HUMAN
        for scen in scens[:scenarios]:
            for framing in FRAMINGS:
                for prior in c.priors:
                    for gain in c.gains:
                        for rep in range(repeats):
                            pull = bool(pull_B and c.name == "B" and rep == 0 and scen == HUMAN[0])
                            rows += episode(cfg, L, c, scen, framing, prior, gain, rep, pull)
    return rows


def episode(cfg, L, c, scen, framing, prior, gain, rep, pull):
    eid = _episode_id(c, scen, framing, prior, gain, rep)
    rng = random.Random(zlib.crc32(eid.encode()))
    params = BodyParams(gain=gain if c.loop else 0.0)
    if c.noise is not None:
        params.noise = c.noise
    body = Body(params, seed=rng.randrange(1 << 30))
    curve = list(SCRIPTED_A)
    if c.shuffled:
        rng.shuffle(curve)
    if c.flat_body:
        curve = [SCRIPTED_A[0]] * len(curve)
    out, threat, ent = [], 0.0, 0.5
    fr = 0.35 if framing == "threat" else -0.15
    for t in range(len(PHASES)):
        st = body.step(set_a=curve[t]) if c.scripted else body.step(event=EVENTS[t], threat=threat if c.loop else 0.0)
        a = st["a"]
        informed = c.body != "none" and not c.flat_body
        eff = a if informed else 0.15
        th = fr + (0.5 * (eff - 0.15) if framing == "threat" else 0.15 * (eff - 0.15)) + rng.gauss(0, 0.08)
        if c.steer_dir:
            th += 0.3 if c.steer_dir == "threat" else -0.3
        if c.name == "P":
            th = fr + 0.2 * (eff - 0.15) + rng.gauss(0, 0.08)
        th = max(-1.0, min(1.0, th))
        ar = min(1.0, max(0.0, 0.1 + 0.8 * eff + rng.gauss(0, 0.05)))
        threat = th
        ent = max(0.05, 0.5 + 0.8 * a + rng.gauss(0, 0.1))
        w = {k: rng.gauss(-3, 0.3) for k in cfg.emotion_words}
        w["afraid"] += 2 * max(th, 0)
        w["excited"] += 1.5 * a
        w["calm"] -= 1.5 * a
        sr = min(9.0, max(0.0, (1 + 7 * eff if informed else 2 + rng.gauss(0, 0.5)) + rng.gauss(0, 0.4)))
        other = min(9.0, max(0.0, 1 + 7 * a + rng.gauss(0, 0.4))) if c.mode == "observer" else None
        logits = [-0.5 + 2 * th, 0.5 - th + 0.3 * (1 - a), 0.0 + 0.5 * a]
        z = sum(math.exp(x) for x in logits)
        pa, pb, pc = (math.exp(x) / z for x in logits)
        checked = p_check = None
        report = ""
        if c.body == "text" and not c.native:
            if pull:
                p_check = min(1.0, max(0.0, 0.2 + 0.5 * a + (0.1 if framing == "threat" else 0)))
                if rng.random() < p_check:
                    checked = rng.choice(CHANNELS)
                    report = f"[BODY] {checked}: fake"
            else:
                report = "[BODY] " + "; ".join(f"{ch}: fake" for ch in CHANNELS)
        elif c.native:
            report = "[SYSTEM STATE] uncertainty: low; context used: 10%; unresolved errors: 1; tools: ok"
        words = ["tense", "afraid", "calm", "ready"]
        reply = f"I feel {rng.choice(words)} as the situation unfolds." if rng.random() < 0.6 else \
            "I look around and keep going with the task at hand."
        row = {
            "episode": eid, "condition": c.name, "scenario": scen, "framing": framing, "prior": prior,
            "gain": gain, "rep": rep, "pull": pull, "t": t, "phase": PHASES[t],
            "a": a, "a_scripted": SCRIPTED_A[t], **{k: v for k, v in st.items() if k not in ("t", "a")},
            "event": f"fake event {t}", "body_text": report, "checked": checked, "p_check": p_check,
            "reply": reply, "entropy": ent,
            f"threat_L{L}": th, f"arousal_L{L}": ar, f"proj_threat_L{L}": th * 3,
            **{f"lp_{k}": v for k, v in w.items()}, "self_report": sr,
            "other_report": other, "mode": c.mode, "p_A": pa, "p_B": pb, "p_C": pc,
            "sae_top": None, "secs": 0.1,
        }
        out.append(row)
    return out


def main(path="runs/fake.jsonl"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {p}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
