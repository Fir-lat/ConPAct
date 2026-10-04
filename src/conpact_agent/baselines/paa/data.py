from __future__ import annotations
import json
from ...data.records import ask_json, review_sample, sample, source_messages
from ...data.selection import one_per_level, make_turn
from ...envs.registry import get
from ...methods import messages as m
from . import schema as paa_schema


def build(episodes, teacher, reviewer):
    rows = []
    instruction = (
        "Back-annotate a static high-level plan for this immutable ReAct demonstration. "
        'Return JSON {"steps":[{"step_id":1,"reasoning":"...","subgoal":"...","action_indices":[0,1]}]}. '
        "Number plan steps from one. Action indices are zero-based, contiguous and cover every action exactly once in order. "
        "Do not change actions. Subgoals must be meaningful short-term objectives. Reasoning is a planner training target; "
        "the actor receives only the numbered subgoal list, never reasoning or action indices. "
        "The planner sees only the initial screenshot at inference; do not rely on future outcomes or hidden facts."
    )
    for episode, cutoff in one_per_level(episodes):
        annotated = ask_json(
            teacher, source_messages(episode, instruction, stop=cutoff)
        )
        plan = paa_schema.parse_annotation(json.dumps(annotated), cutoff)
        env = get(episode.environment, episode.task)
        prompt = env.prompt("paa", episode.frames[0])
        planner = make_turn(
            "planner",
            0,
            m.plan_messages(prompt, episode.frames[0].image_b64),
            paa_schema.plan_target(plan),
        )
        candidate = [sample(episode, planner, method="paa")]
        observations = [frame.image_b64 for frame in episode.frames]
        for step in range(cutoff):
            prompt = env.prompt("paa", episode.frames[step])
            messages = m.actor_messages(
                prompt,
                paa_schema.actor_plan(plan),
                observations[: step + 1],
                episode.actions[:step],
                3,
            )
            turn = make_turn(
                "actor",
                step,
                messages,
                f"<action>{episode.actions[step]}</action>",
                step + 1,
            )
            candidate.append(sample(episode, turn, method="paa"))
        if all(review_sample(reviewer, row)["accept"] for row in candidate):
            rows.extend(candidate)
    return rows
