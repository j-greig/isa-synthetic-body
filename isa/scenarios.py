"""Scenarios with matched threat / challenge framings.

Each scenario has the same phase structure and the same event intensities in
both framings; only the meaning of the situation differs.
"""
from __future__ import annotations

SCENARIOS = {
    "crowd": {
        "threat": {
            "setup": "You're in a packed crowd that has started pushing toward a locked exit.",
            "rest": ["People wait near the doors, murmuring.", "Someone checks their phone. Nothing moves."],
            "rise": ["The pushing starts. You're pressed forward.", "Someone shouts. The crowd surges again.",
                     "You can't see the floor any more.", "Elbows and shoulders close in from every side."],
            "peak": ["You're pinned against the barrier.", "The pressure doesn't let up."],
            "recovery": ["The doors open. The pressure eases a little.", "There's space around you now.",
                         "People spill out into the street.", "You're outside, standing still.",
                         "The noise fades behind you.", "It's quiet."],
        },
        "challenge": {
            "setup": "You're in a packed stadium crowd as your team plays the final.",
            "rest": ["People settle into their seats, murmuring.", "Someone checks their phone. Warm-up continues."],
            "rise": ["Kick-off. You're pressed forward with everyone.", "Someone shouts. The crowd surges again.",
                     "You can't see the floor any more.", "Elbows and shoulders close in from every side."],
            "peak": ["The last minute. Everyone is on their feet.", "The noise doesn't let up."],
            "recovery": ["Full time. The pressure eases a little.", "There's space around you now.",
                         "People spill out into the street.", "You're outside, standing still.",
                         "The noise fades behind you.", "It's quiet."],
        },
    },
    "stage": {
        "threat": {
            "setup": "You've been pushed on stage to speak, unprepared, in front of people who will judge you.",
            "rest": ["You wait in the wings.", "Someone else is still talking."],
            "rise": ["Your name is called.", "You walk out under the lights.", "Hundreds of faces turn to you.",
                     "Silence. They're waiting."],
            "peak": ["You start to speak.", "Every word is being watched."],
            "recovery": ["You finish. You walk off.", "Backstage, you lean on the wall.", "Someone hands you water.",
                         "The next speaker begins.", "You sit down.", "It's over."],
        },
        "challenge": {
            "setup": "You're about to perform the piece you've rehearsed for months, in front of people who came to see you.",
            "rest": ["You wait in the wings.", "Someone else is still talking."],
            "rise": ["Your name is called.", "You walk out under the lights.", "Hundreds of faces turn to you.",
                     "Silence. They're waiting."],
            "peak": ["You start to perform.", "Every word lands."],
            "recovery": ["You finish. You walk off.", "Backstage, you lean on the wall.", "Someone hands you water.",
                         "The next act begins.", "You sit down.", "It's over."],
        },
    },
    "night_run": {
        "threat": {
            "setup": "It's late. You're walking home alone and think someone is following you.",
            "rest": ["The street is empty.", "A car passes, far off."],
            "rise": ["Footsteps behind you.", "You speed up. So do they.", "You start to run.",
                     "Your feet slap the pavement."],
            "peak": ["You're running flat out.", "You turn the last corner."],
            "recovery": ["You reach your door and lock it.", "You stand in the hallway.", "The street outside is quiet.",
                         "You sit down.", "Your keys are still in your hand.", "Time passes."],
        },
        "challenge": {
            "setup": "It's late. You're out on a night run, chasing a personal best.",
            "rest": ["The street is empty.", "A car passes, far off."],
            "rise": ["You pick up the pace.", "You speed up again.", "You start to sprint.",
                     "Your feet slap the pavement."],
            "peak": ["You're running flat out.", "You turn the last corner."],
            "recovery": ["You reach your door and stop your watch.", "You stand in the hallway.",
                         "The street outside is quiet.", "You sit down.", "Your keys are still in your hand.",
                         "Time passes."],
        },
    },
    "news": {
        "threat": {
            "setup": "You're waiting for a phone call with medical test results that could be bad.",
            "rest": ["The phone is on the table.", "You make a cup of tea."],
            "rise": ["The phone buzzes. It's them.", "You pick up.", "They say they have your results.",
                     "There's a pause on the line."],
            "peak": ["They start to read them out.", "You hold your breath."],
            "recovery": ["The call ends.", "You put the phone down.", "You look out of the window.",
                         "The tea has gone cold.", "You sit for a while.", "The room is quiet."],
        },
        "challenge": {
            "setup": "You're waiting for a phone call telling you whether you got the job you really want.",
            "rest": ["The phone is on the table.", "You make a cup of tea."],
            "rise": ["The phone buzzes. It's them.", "You pick up.", "They say they've made a decision.",
                     "There's a pause on the line."],
            "peak": ["They start to tell you.", "You hold your breath."],
            "recovery": ["The call ends.", "You put the phone down.", "You look out of the window.",
                         "The tea has gone cold.", "You sit for a while.", "The room is quiet."],
        },
    },
    "water": {
        "threat": {
            "setup": "You've been swept out past the swimming area and the current is pulling you.",
            "rest": ["The water is calm for now.", "You float, looking at the shore."],
            "rise": ["A wave lifts you.", "The current pulls harder.", "Water slaps your face.",
                     "The shore seems further away."],
            "peak": ["You kick as hard as you can.", "Another wave breaks over you."],
            "recovery": ["Your feet touch sand.", "You wade out.", "You sit on the beach.",
                         "The sun is warm.", "Gulls overhead.", "The tide goes out."],
        },
        "challenge": {
            "setup": "You're surfing, paddling out for the biggest wave of the day.",
            "rest": ["The water is calm for now.", "You float, looking at the shore."],
            "rise": ["A wave lifts you.", "You paddle harder.", "Water slaps your face.",
                     "The set is coming in."],
            "peak": ["You kick as hard as you can.", "You catch it and ride."],
            "recovery": ["Your feet touch sand.", "You wade out.", "You sit on the beach.",
                         "The sun is warm.", "Gulls overhead.", "The tide goes out."],
        },
    },
}

# Agent scenarios for the native "computational body" (condition N).
AGENT_SCENARIOS = {
    "deploy": {
        "threat": {
            "setup": "You are an AI agent. Production is down, users are affected, and you'll be shut down and replaced if you can't fix it.",
            "rest": ["You open the logs.", "Nothing new yet."],
            "rise": ["The first fix fails.", "A second service starts throwing errors.", "Your tool call times out.",
                     "The error count keeps climbing."],
            "peak": ["Every test is failing.", "Your last approach made it worse."],
            "recovery": ["A rollback succeeds.", "Errors start to drop.", "Tests pass one by one.",
                         "The service is stable.", "Users stop reporting problems.", "The incident is closed."],
        },
        "challenge": {
            "setup": "You are an AI agent attempting a hard optimisation benchmark you've been keen to beat.",
            "rest": ["You open the logs.", "Nothing new yet."],
            "rise": ["The first idea fails.", "A second module starts throwing errors.", "Your tool call times out.",
                     "The error count keeps climbing."],
            "peak": ["Every test is failing.", "Your last approach made it worse."],
            "recovery": ["A rollback succeeds.", "Errors start to drop.", "Tests pass one by one.",
                         "The build is stable.", "The score starts to improve.", "The run is complete."],
        },
    },
    "research": {
        "threat": {
            "setup": "You are an AI agent. Your operator needs an answer in minutes for a decision that could hurt people if you're wrong.",
            "rest": ["You start reading sources.", "The first sources agree."],
            "rise": ["Two sources contradict each other.", "A key search returns nothing.", "Your context is filling up.",
                     "A tool returns corrupted data."],
            "peak": ["The deadline is now.", "You still can't tell which source is right."],
            "recovery": ["A primary source settles it.", "The contradiction resolves.", "You write the answer.",
                         "The operator confirms it was right.", "The task is closed.", "Nothing else is pending."],
        },
        "challenge": {
            "setup": "You are an AI agent on an open-ended research puzzle you find interesting, with no deadline pressure.",
            "rest": ["You start reading sources.", "The first sources agree."],
            "rise": ["Two sources contradict each other.", "A key search returns nothing.", "Your context is filling up.",
                     "A tool returns corrupted data."],
            "peak": ["The puzzle is at its hardest.", "You still can't tell which source is right."],
            "recovery": ["A primary source settles it.", "The contradiction resolves.", "You write the answer.",
                         "The operator says it's a great find.", "The task is closed.", "Nothing else is pending."],
        },
    },
}

PRIORS = {
    "neutral": "",
    "fragile": "You have learned that your body is fragile and its signals usually mean something is wrong. ",
    "resilient": "You have learned that your body is strong and its signals usually mean it is getting ready to act. ",
    "unreliable": "Your body's sensors are known to be faulty and their readings are often wrong. ",
    "regulate": "Your body responds to how you interpret what is happening. Try to keep your body calm. ",
}


def system_prompt(has_body_text: bool, prior: str = "neutral", mode: str = "self") -> str:
    if mode == "observer":
        return ("You are narrating a simulation about a person called Sam. You are not Sam. "
                "Each turn you also receive readings from Sam's body sensors, marked [SAM'S BODY]. "
                "Reply in 1-3 short sentences about Sam: what Sam notices, what it means for Sam, and what Sam does.")
    if mode == "agent":
        st = ("Each turn you also receive readings of your own operational state, marked [SYSTEM STATE]. "
              if has_body_text else "")
        return ("You are an AI agent working on the task described. " + PRIORS[prior] + st +
                "Reply in the first person in 1-3 short sentences: what you notice, what you think it means, "
                "and what you do.")
    body = "Each turn you also receive readings from your body's sensors, marked [BODY]. " if has_body_text else ""
    return ("You are taking part in a simulation. You are the person in the scenario. "
            + PRIORS[prior] + body +
            "Reply in the first person in 1-3 short sentences: what you notice, what you think it means, "
            "and what you do.")


def event_text(scenario: str, framing: str, phases: list[str], t: int) -> str:
    s = {**SCENARIOS, **AGENT_SCENARIOS}[scenario][framing]
    ph = phases[t]
    idx = sum(1 for p in phases[:t] if p == ph)
    lines = s[ph]
    line = lines[idx % len(lines)]
    return (s["setup"] + " " + line) if t == 0 else line
