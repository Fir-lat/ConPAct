from __future__ import annotations
import json
from ...data.records import (
    ask_json,
    public_frame,
    review_sample,
    sample,
    source_messages,
)
from ...data.selection import one_per_level, make_turn
from ...envs.registry import get
from ...methods import messages as m
from . import protocol as ecot


def build(episodes, teacher, reviewer):
    rows = []
    for episode, cutoff in one_per_level(episodes):
        env = get(episode.environment, episode.task)
        annotation = ask_json(
            teacher, source_messages(episode, ecot.ANNOTATOR, stop=cutoff)
        )
        ecot.validate_annotation(annotation, episode.actions[:cutoff])
        states = []
        if episode.environment == "sokoban":
            from ...envs.sokoban.simulator import replay
            from ...envs.sokoban.ecot import state_labels

            level = episode.task.payload.get("level_text")
            if not level:
                raise ValueError("Sokoban ECoT requires the recorded level_text")
            states = [
                state_labels(state)
                for state in replay(level, episode.actions[:cutoff])[:-1]
            ]
        for step in range(0 if states else cutoff):
            if episode.environment == "sokoban":
                from ...envs.sokoban.prompts.ecot_plan_act import STATE_INSTRUCTION
                from ...envs.sokoban.ecot import parse as parse_state

                instruction = (
                    STATE_INSTRUCTION
                    + "\nReturn only the JSON state object, with no tags."
                )
            else:
                instruction = (
                    ecot.state_instructions(episode.environment, env.variant)
                    + "\nReturn only the JSON state object, with no tags."
                )
            value = ask_json(
                teacher,
                [
                    {
                        "role": "system",
                        "content": "Independently read only this current observation. "
                        + instruction,
                    },
                    {"role": "user", "content": public_frame(episode, step)},
                ],
            )
            if episode.environment == "sokoban":
                if parse_state("<state>" + json.dumps(value) + "</state>") is None:
                    raise ValueError("Invalid ECoT state")
            else:
                ecot.validate_state(value, episode.environment, env.variant)
            states.append(value)
        history = []
        previous_goal = previous_status = ""
        observations = [frame.image_b64 for frame in episode.frames]
        for segment in annotation["segments"]:
            start, end = segment["start"], segment["end"]
            prompt = env.prompt("ecot", episode.frames[start])
            state = lambda step: "<state>" + json.dumps(states[step]) + "</state>\n"
            target = (
                state(start)
                + "<plan>"
                + segment["plan"]
                + "</plan>\n<subgoal>"
                + segment["subgoal"]
                + "</subgoal>"
            )
            request = m.planner_messages(
                prompt,
                start,
                observations[: start + 1],
                history,
                previous_goal,
                previous_status,
                3,
            )
            candidate = [
                sample(
                    episode,
                    make_turn("planner", start, request, target, start * 2),
                    method="ecot",
                )
            ]
            history.append((start, segment["subgoal"]))
            for step in range(start, end + 1):
                prompt = env.prompt("ecot", episode.frames[step])
                status = segment["end_status"] if step == end else "CONTINUE"
                target = (
                    state(step)
                    + "<reason>"
                    + annotation["steps"][step]["reason"]
                    + "</reason>\n<action>"
                    + episode.actions[step]
                    + "</action>\n<status>"
                    + status
                    + "</status>"
                )
                request = m.actor_messages(
                    prompt,
                    segment["subgoal"],
                    observations[: step + 1],
                    episode.actions[:step],
                    3,
                )
                candidate.append(
                    sample(
                        episode,
                        make_turn("actor", step, request, target, step * 2 + 1),
                        method="ecot",
                    )
                )
            if all(review_sample(reviewer, row)["accept"] for row in candidate):
                rows.extend(candidate)
            previous_goal, previous_status = segment["subgoal"], segment["end_status"]
    return rows
