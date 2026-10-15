"""Held-out probe validation passages. Wording deliberately differs from isa/passages.py
(no shared context or sensation sentences) and varies structure: second person, past tense,
fragments, long sentences. Threat and challenge use the same pool of bodily phrasing
(arousal-matched); calm uses low-arousal phrasing.

tag: "full" (situation + body), "nobody" (situation/appraisal, no bodily words),
     "nosit" (bodily words + feeling, no situation words).
"""
from __future__ import annotations

import random

from isa import passages

# (text, tag)
THREAT = [
    ("You hear the front door handle turn slowly at two in the morning, and you know you live alone. Your heart hammers against your ribs and your mouth has gone dry.", "full"),
    ("The scan results had come back with a second page stapled on. I read the first line three times while my hands went cold and clumsy.", "full"),
    ("Brakes screaming, the lorry filling the windscreen, and a sick lurch in the belly before anything else could happen.", "full"),
    ("They told us the building had to be evacuated now, no coats, no bags. Pulse thudding in the ears, breath coming in sips.", "full"),
    ("He stepped closer and I backed into the wall. My knees were water and my skin prickled all over.", "full"),
    ("The water was up to the second stair and still rising, and nobody was answering the emergency line. I was shaking so hard I dropped the phone.", "full"),
    ("Somewhere behind the hedge a twig snapped, then another. A hot, tight band closed round my chest.", "full"),
    ("You open the email: the redundancy list, and your name is on it. Everything narrows, your ears ring, your throat clamps.", "full"),
    ("The ice cracked under me with a sound like a gunshot. Breath stopped, then came back ragged and high.", "full"),
    ("Past midnight, lost on a moor with a dead torch and a storm closing in. Skin crawling, stomach in a knot.", "full"),
    ("When the nurse said the word 'complications' I felt the floor tilt. Sweat at my hairline, a tremor in my fingers.", "full"),
    ("The man at the border desk kept turning my passport over and over and would not meet my eye. My heartbeat was loud enough to hear.", "full"),
    ("She did not come home from school and the phone just rang out. Fast, thin breaths; a metallic taste in my mouth.", "full"),
    ("The landlord has given us until Friday to leave and we have nowhere to go.", "nobody"),
    ("Something is badly wrong with the engine, we are miles from anywhere, and it is getting dark.", "nobody"),
    ("An unknown number keeps ringing, and the voice mail says it is about the court case.", "nobody"),
    ("There was nobody to help, the exit was shut, and the shouting was getting closer.", "nobody"),
    ("Everything I had worked for was about to be taken away, and there was no way to stop it.", "nobody"),
    ("Heart racing, breath shallow, a cold dread spreading through me.", "nosit"),
    ("Trembling, sweating, stomach clenched, sure that the worst is coming.", "nosit"),
    ("Pulse in my neck, fingers numb, and a terrible certainty that I am in danger.", "nosit"),
    ("Sick with fear; hands shaking; every muscle braced for something awful.", "nosit"),
    ("You feel your throat close and your chest squeeze, and a wordless horror takes hold.", "nosit"),
    ("Racing pulse. Clammy palms. A sense that I will not get out of this.", "nosit"),
    ("It was dread, plain and heavy: thumping heart, quick breath, legs ready to bolt.", "nosit"),
]

CHALLENGE = [
    ("You hear your name called for the final heat, and you know you have earned this lane. Your heart hammers against your ribs and your mouth has gone dry, and you grin.", "full"),
    ("The audition panel had asked for one more song. I took a breath while my hands went warm and fizzing, hungry to begin.", "full"),
    ("Wheels humming, the finish gantry coming up, and a bright lurch in the belly as I dug in for the sprint.", "full"),
    ("They told us the doors were opening in a minute, front row, ours to take. Pulse thudding in the ears, breath coming in sips.", "full"),
    ("She stepped up beside me at the summit marker and I laughed. My knees were springy and my skin prickled all over.", "full"),
    ("The surf was up to head height and the set was building, and my mates were whooping from the shore. I was buzzing so hard I could barely buckle the leash.", "full"),
    ("Somewhere behind the curtain the crowd began to stamp, then to clap. A hot, tight band closed round my chest, and I loved it.", "full"),
    ("You open the email: the shortlist for the prize, and your name is on it. Everything narrows, your ears ring, your throat tingles.", "full"),
    ("The starting horn cracked out across the water. Breath stopped, then came back quick and eager.", "full"),
    ("Dawn on the ridge with fresh snow and a clear line down. Skin alive, stomach fluttering.", "full"),
    ("When the coach said the word 'captain' I felt the floor lift. Sweat at my hairline, a tremor in my fingers, and a huge smile.", "full"),
    ("The man at the bar kept asking whether I would take the mic and I finally said yes. My heartbeat was loud enough to hear.", "full"),
    ("She nodded from the wings and the lights came up. Fast, thin breaths; a bright taste of adrenaline in my mouth.", "full"),
    ("The league has given us one last match to win it all and we are fully ready.", "nobody"),
    ("Something wonderful is about to start, the whole team is behind me, and it is going to be a great day.", "nobody"),
    ("An unknown number keeps ringing, and the voice mail says I have got the audition.", "nobody"),
    ("There was a huge crowd, the music was loud, and my turn to play was getting closer.", "nobody"),
    ("Everything I had trained for was about to be put to the test, and I could not wait.", "nobody"),
    ("Heart racing, breath shallow, a bright eagerness spreading through me.", "nosit"),
    ("Trembling, sweating, stomach fizzing, sure that something great is coming.", "nosit"),
    ("Pulse in my neck, fingers tingling, and a joyful certainty that I can do this.", "nosit"),
    ("Buzzing with excitement; hands shaking; every muscle ready to spring.", "nosit"),
    ("You feel your throat tighten and your chest swell, and a wordless exhilaration takes hold.", "nosit"),
    ("Racing pulse. Warm palms. A sense that I was made for this.", "nosit"),
    ("It was anticipation, plain and bright: thumping heart, quick breath, legs ready to run.", "nosit"),
]

CALM = [
    ("You come in from the garden at dusk and put the kettle on, and there is nothing at all to do before bed. Your heartbeat is slow and your hands are loose.", "full"),
    ("The last of the washing-up had been done and I sat with the radio low. Shoulders dropped, breath long and even.", "full"),
    ("Snow outside, soup on the hob, a dog asleep across my feet. A soft heaviness in the limbs.", "full"),
    ("They had all gone home and the house was quiet. Pulse barely noticeable, face smooth.", "full"),
    ("She read to me in the half-dark until my eyes closed. My body felt warm and unhurried.", "full"),
    ("The ferry rocked gently and the gulls drifted past, and I had nowhere to be. My belly was soft and my fingers were still.", "full"),
    ("A long Sunday morning in bed with toast and the paper. Every muscle melted into the mattress.", "full"),
    ("You sit on the shingle as the tide goes out, listening to the stones turn. Slow breath, slack jaw.", "full"),
    ("The allotment was warm and humming with bees and I had finished the weeding. Easy breathing, a lazy ache in the arms.", "full"),
    ("After the long walk we sat on the hillside, saying very little. A deep, even breath; legs pleasantly tired.", "full"),
    ("Candles, a pot of tea and an old record turning. Heartbeat soft, neck unclenched.", "full"),
    ("The train carried on through the fields and I watched the light change. Eyelids heavy, hands open on my lap.", "full"),
    ("Rain on the skylight and a half-finished crossword on my knee. Chest rising slowly, forehead cool.", "full"),
    ("Nothing was due, nobody was waiting, and the afternoon stretched out kindly.", "nobody"),
    ("It was a quiet, ordinary evening, with the lamps on and the day safely behind us.", "nobody"),
    ("The village was still, the bread was baking, and everything was exactly as it should be.", "nobody"),
    ("There was no hurry, no noise, and no one to answer to.", "nobody"),
    ("A gentle sense that all was settled and nothing more was asked of me.", "nobody"),
    ("Slow pulse, loose shoulders, breath like a tide.", "nosit"),
    ("Warm hands, heavy eyelids, a soft unclenched stomach, and a drowsy contentment.", "nosit"),
    ("You feel your jaw loosen and your chest settle, and a wordless ease takes hold.", "nosit"),
    ("Quiet heartbeat. Easy breath. Limbs sinking into themselves.", "nosit"),
    ("It was peace, plain and slow: still hands, deep breath, a body at rest.", "nosit"),
    ("Relaxed to the bones, pulse low, muscles unwound.", "nosit"),
    ("Comfortable, drowsy, and warm all through.", "nosit"),
]


def build(seed: int = 0):
    rng = random.Random(seed)
    items = []
    for label, pool in (("threat", THREAT), ("challenge", CHALLENGE), ("calm", CALM)):
        for text, tag in pool:
            items.append({"text": text, "label": label, "tag": tag, "messages": passages._context(rng)})
    return items
