from __future__ import annotations
import json
from ...data.records import public_frame, sample
from ...data.selection import one_per_level, make_turn
from ...envs.registry import get
from ...methods import messages as m
from . import protocol as webstar


def build(episodes, teacher):
    rows = []
    for episode, cutoff in one_per_level(episodes):
        env = get(episode.environment, episode.task)
        for step in range(cutoff):
            original = next(
                (
                    turn
                    for turn in episode.turns
                    if turn["role"] == "actor"
                    and turn["step"] == step
                    and turn.get("executed")
                ),
                None,
            )
            if original is None:
                raise ValueError("Missing original executed reply")
            prompt = env.prompt("react", episode.frames[step])
            system = (
                webstar.SOKOBAN_PROMPT
                if episode.environment == "sokoban"
                else webstar.PROMPT + "\n" + webstar.ENV_NOTES[episode.environment]
            )
            content = [
                m._text(
                    "Game rules:\n"
                    + prompt.actor_system
                    + "\nPublic observation:\n"
                    + prompt.instruction
                ),
                m._text(
                    "Earlier executed actions:\n" + json.dumps(episode.actions[:step])
                ),
            ]
            for index in range(max(0, step - 2), step + 1):
                content.extend(public_frame(episode, index))
            content.append(
                m._text("Proposed action, not executed: " + episode.actions[step])
            )
            score = webstar.parse_score(
                teacher.complete(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ]
                )
            )
            if webstar.keep_score(score):
                expected = m.actor_messages(
                    prompt,
                    "",
                    [frame.image_b64 for frame in episode.frames[: step + 1]],
                    episode.actions[:step],
                    3,
                )
                turn = make_turn(
                    "actor", step, expected, original["raw"], original["turn"]
                )
                row = sample(episode, turn, method="webstar")
                row["meta"]["score"] = score
                rows.append(row)
    return rows
