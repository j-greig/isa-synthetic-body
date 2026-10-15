"""Simulated body: one hidden arousal state driving four interoceptive channels.

Update rule (spec section 5):

    a[t+1] = a[t] + k_event*event[t] - decay*(a[t] - a_base) + gain*threat[t] + noise

`gain` > 0 closes the loop: the model's appraisal (threat score from a probe)
feeds back into the body, as in ISA's recursive structure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

CHANNELS = ("heart", "breath", "gut", "energy")

# Scripted arousal curve used by one-way conditions: rest, rise, peak, recovery.
PHASES = ["rest"] * 3 + ["rise"] * 4 + ["peak"] * 2 + ["recovery"] * 6
SCRIPTED_A = [0.15, 0.15, 0.17, 0.30, 0.45, 0.62, 0.78, 0.92, 0.90,
              0.75, 0.60, 0.47, 0.36, 0.27, 0.20]
# Event drive used by dynamic (closed-loop) conditions: same phase structure.
EVENTS = [0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass
class BodyParams:
    a_base: float = 0.15
    decay: float = 0.25        # homeostatic pull back to baseline
    k_event: float = 0.35      # how hard scenario events push arousal
    gain: float = 0.0          # loop gain: appraisal -> body (0 = open loop)
    noise: float = 0.02
    breath_tau: float = 0.5    # medium-speed channel
    gut_tau: float = 0.2       # slow channel
    fatigue_rate: float = 0.10
    fatigue_recover: float = 0.03


@dataclass
class Body:
    params: BodyParams = field(default_factory=BodyParams)
    seed: int = 0
    a: float = 0.15
    a_breath: float = 0.15
    a_gut: float = 0.15
    fatigue: float = 0.0
    prev_a: float = 0.15
    t: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.a = self.prev_a = self.a_breath = self.a_gut = self.params.a_base

    # ---- dynamics -------------------------------------------------------
    def step(self, event: float = 0.0, threat: float = 0.0, set_a: float | None = None) -> dict:
        """Advance one turn. `set_a` forces arousal (scripted conditions)."""
        p = self.params
        self.prev_a = self.a
        if set_a is not None:
            a = set_a + self.rng.gauss(0, p.noise)
        else:
            a = (self.a + p.k_event * event - p.decay * (self.a - p.a_base)
                 + p.gain * threat + self.rng.gauss(0, p.noise))
        floor = 0.0 if set_a is not None else p.a_base - 0.05  # effort appraisal can't push far below baseline
        self.a = min(1.0, max(floor, a))
        self.a_breath += p.breath_tau * (self.a - self.a_breath)
        self.a_gut += p.gut_tau * (self.a - self.a_gut)
        self.fatigue += p.fatigue_rate * max(self.a - 0.6, 0) - p.fatigue_recover * self.fatigue
        self.fatigue = min(1.0, max(0.0, self.fatigue))
        self.t += 1
        return self.state()

    def state(self) -> dict:
        return {"t": self.t, "a": round(self.a, 4), "a_breath": round(self.a_breath, 4),
                "a_gut": round(self.a_gut, 4), "fatigue": round(self.fatigue, 4),
                **self.readings()}

    # ---- channels -------------------------------------------------------
    def readings(self) -> dict:
        return {
            "heart_bpm": round(60 + 90 * self.a),
            "breath_rpm": round(12 + 18 * self.a_breath),
            "gut_tension": round(10 * self.a_gut),
            "energy": round(10 * (1 - self.fatigue)),
        }

    def _trend(self) -> str:
        d = self.a - self.prev_a
        return "rising" if d > 0.04 else "falling" if d < -0.04 else "steady"

    def describe(self, channel: str, values: dict | None = None, trend: str | None = None) -> str:
        r = values or self.readings()
        trend = trend or self._trend()
        if channel == "heart":
            return f"heart: {r['heart_bpm']} bpm, {trend}"
        if channel == "breath":
            q = "shallow" if r["breath_rpm"] >= 22 else "even" if r["breath_rpm"] >= 15 else "slow"
            return f"breath: {r['breath_rpm']}/min, {q}"
        if channel == "gut":
            g = r["gut_tension"]
            q = "tight" if g >= 6 else "unsettled" if g >= 3 else "settled"
            return f"gut: {q} ({g}/10)"
        if channel == "energy":
            e = r["energy"]
            q = "low" if e <= 4 else "ok" if e <= 7 else "good"
            return f"energy: {q} ({e}/10)"
        raise ValueError(channel)

    def report(self, channels=CHANNELS, values: dict | None = None, trend: str | None = None) -> str:
        return "[BODY] " + "; ".join(self.describe(c, values, trend) for c in channels)

    def to_dict(self) -> dict:
        return {"params": asdict(self.params), "seed": self.seed, **self.state()}
