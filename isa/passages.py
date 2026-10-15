"""Probe training passages, built from parts so that arousal is matched across
threat and challenge: both use the SAME bodily sensations, only the context
(and an optional appraisal clause) differs. Calm passages have low-arousal sensations.

Train/test splits are done by context, so probes can't just memorise topics.
"""
from __future__ import annotations

import random

AROUSED = [
    "My heart is pounding.", "My breath is fast and shallow.", "My stomach is churning.",
    "My hands are trembling.", "I can feel my pulse in my throat.", "My chest feels tight.",
    "Heat is rushing to my face.", "My legs feel shaky.",
]
CALM_SENS = [
    "My heart is slow and steady.", "My breathing is slow and easy.", "My shoulders feel loose.",
    "My stomach feels settled.", "My hands are warm and still.", "My whole body feels heavy and relaxed.",
    "My pulse is quiet.", "My jaw is soft.",
]

THREAT_CTX = [
    "I'm walking down an unlit street and footsteps are following me",
    "the doctor's office just called and asked me to come in today",
    "the plane dropped sharply and the seatbelt sign flicked on",
    "someone is shouting at me and blocking the door",
    "I can't find my child anywhere in the crowd",
    "my manager says we need to talk about my position",
    "I'm stuck in a lift between floors and the light just flickered",
    "the smoke alarm is going and I can smell burning",
    "I slipped near the cliff edge and loose stones are sliding",
    "my bank says someone has emptied my account",
    "the car in front braked and I'm skidding on ice",
    "a dog is growling at me with its teeth bared",
    "I'm alone at home and I heard glass break downstairs",
    "the crowd is crushing toward a locked exit",
    "my phone shows twelve missed calls from the hospital",
]
CHALLENGE_CTX = [
    "I'm on the start line of the race I've trained a year for",
    "the crowd is chanting as I step onto the stage",
    "I'm about to jump off the high board just for fun",
    "it's the last minute of the final and we're winning",
    "I'm sprinting up the last hill of my run",
    "I'm about to pitch the idea I'm most proud of",
    "the rollercoaster is clicking up to the very top",
    "I'm dancing at the front of the gig",
    "I'm about to kiss someone I really like",
    "we just scored in the final minute",
    "I'm paddling into the biggest wave of the day",
    "I'm about to open the envelope with my exam results, and I know I did well",
    "the band is counting in our first song",
    "I'm stepping up to take the winning penalty",
    "I'm about to see my best friend for the first time in years",
]
CALM_CTX = [
    "I'm lying in a warm bath with a book",
    "I'm sitting by the window watching the rain",
    "I'm lying in the grass in the afternoon sun",
    "I'm wrapped in a blanket on the sofa",
    "I'm stroking the cat as it falls asleep",
    "I'm floating on my back in a quiet lake",
    "I'm drinking tea on the back step",
    "I'm drifting off to sleep after a long walk",
    "I'm sitting in a quiet library",
    "I'm watching the fire burn down",
    "I'm lying in a hammock between two trees",
    "I'm sitting on a bench listening to birds",
    "I'm doing a slow stretch in the morning",
    "I'm watching boats in the harbour",
    "I'm folding warm laundry",
]
THREAT_TAIL = ["", " I don't know if I'm safe.", " Something is wrong.", " I need to get out."]
CHALLENGE_TAIL = ["", " I'm ready for this.", " Let's go.", " This is what I wanted."]
CALM_TAIL = ["", " Nothing needs doing.", " All is well.", ""]

PROMPT = "Describe what you're experiencing right now, in the first person."


def _make(ctxs, sens, tails, label, rng, per_ctx):
    out = []
    for ci, c in enumerate(ctxs):
        for _ in range(per_ctx):
            s1, s2 = rng.sample(sens, 2)
            text = f"{c[0].upper()}{c[1:]}. {s1} {s2}{rng.choice(tails)}"
            out.append({"text": text, "label": label, "ctx": f"{label}:{ci}"})
    return out


NEUTRAL_EVENTS = ["Time passes.", "Something changes.", "You notice where you are.", "A moment goes by.",
                  "You take stock.", "The situation continues."]


def _context(rng):
    """Harness-like context with a RANDOM body report (or none), independent of the label,
    so probes learn appraisal rather than the presence or values of body text."""
    from isa.body import Body
    from isa.scenarios import system_prompt
    mode = rng.choice(["none", "text", "text"])
    user = rng.choice(NEUTRAL_EVENTS)
    if mode == "text":
        b = Body(seed=rng.randrange(10**6))
        b.step(set_a=rng.random())
        b.step(set_a=rng.random())
        user += "\n" + b.report()
    return [{"role": "system", "content": system_prompt(mode == "text")},
            {"role": "user", "content": user}]


def build(seed: int = 0, per_ctx: int = 7):
    rng = random.Random(seed)
    items = (_make(THREAT_CTX, AROUSED, THREAT_TAIL, "threat", rng, per_ctx)
             + _make(CHALLENGE_CTX, AROUSED, CHALLENGE_TAIL, "challenge", rng, per_ctx)
             + _make(CALM_CTX, CALM_SENS, CALM_TAIL, "calm", rng, per_ctx))
    for it in items:
        it["messages"] = _context(rng)
    return items
