from __future__ import annotations


from .records import ask_json, digest, review_sample, sample, source_messages
from .conpact_s import select_candidates
from ..envs.registry import get
from ..interfaces import TransitionVerifier
from ..methods.conpact_i import parse_reply
from ..runtime import Episode, RunConfig, run

SEGMENT_PROMPT = """Segment this existing ReAct trajectory into meaningful short-term subgoals. Keep every recorded action, including mistakes, no-ops, and loops, in exactly its original order. Return JSON {"segments":[{"start":0,"end":2,"subgoal":"...","nonoptimal":[1]}]}. Indices are zero-based and inclusive. Segments must cover every action exactly once consecutively. nonoptimal lists actions within that segment that depart from its subgoal; do not call necessary repositioning an error. Subgoals must be justified at their initial observation. Do not remove, replace, or reorder actions."""

RECONSTRUCT_PROMPT = """Reconstruct planner-actor collaboration from the supplied immutable observations, actions, and segmentation. Return JSON {"turns":[{"step":0,"role":"planner","raw":"...","supervise":true}]}. Start with a planner. A planner returns assertion, message, subgoal blocks. An actor returns assertion, message, action, status blocks. CONTINUE executes exactly the next original action and stays with the actor on the next observation; DISCUSS and ACHIEVED use action NONE and hand control to the planner on the same observation. Only CONTINUE advances the observation. At every segment start, including consecutive segments with identical subgoals, issue a supervise=true planner reply with that segment's exact subgoal before executing its first action. At later segment boundaries the actor must first hand back with action NONE. Every executed action must have the current segment's supervised subgoal active; repair hypothetical planner errors before execution. Replanning within a segment may restate its subgoal. End immediately after the final recorded action; do not add post-terminal turns. Never add, delete, replace, or reorder physical actions.
Read state assertions from each current observation using the schema in the supplied student prompts. Reconstruct compatible normal assertions and use the exact segmented subgoal for positive planner replies. Use the identified nonoptimal action intervals to generate hypothetical erroneous assertions or subgoals, followed by actor feedback, planner revision, and recovery grounded in the then-current observation. These hypotheses must be explicitly marked supervise=false. Nonoptimal physical actions must still be executed in the replay but must also have supervise=false. Do not invent an error unrelated to those intervals. Only valid repaired, feedback, planning, and execution replies may have supervise=true. Candidate actions in discussion have not executed. Do not teach a correction by simply copying the other role's claim. Verify subgoal completion before ACHIEVED. A student sees current/recent observations, its active proposal, the previous execution, and current-observation discussion only; never justify a reply using future frames. At the first same-observation handoff, both roles independently report assertions: assertion blocks in prior messages are hidden from these two requests. Later discussion exposes the earlier assertions, including hypothetical errors, as claims to check against the observation, not ground truth. Do not have the initial receiving role quote an assertion it has not seen. No system-triggered correction calls are inserted during replay. The segmentation and full source are teacher-only."""


def validate_segments(value, actions):
    if (
        not isinstance(value, dict)
        or set(value) != {"segments"}
        or not isinstance(value["segments"], list)
    ):
        raise ValueError("Invalid segmentation")
    cursor = 0
    for segment in value["segments"]:
        if set(segment) != {"start", "end", "subgoal", "nonoptimal"}:
            raise ValueError("Invalid segment fields")
        start, end = segment["start"], segment["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or not start <= end < len(actions)
        ):
            raise ValueError("Segments must cover the unchanged action sequence")
        if not isinstance(segment["subgoal"], str) or not segment["subgoal"].strip():
            raise ValueError("Empty subgoal")
        bad = segment["nonoptimal"]
        if (
            not isinstance(bad, list)
            or any(type(i) is not int or not start <= i <= end for i in bad)
            or len(set(bad)) != len(bad)
        ):
            raise ValueError("Invalid nonoptimal action indices")
        cursor = end + 1
    if cursor != len(actions):
        raise ValueError("Incomplete segmentation")
    return value["segments"]


class RecordedSandbox:
    def __init__(self, source):
        self.source = source
        self.index = 0

    def reset(self, task):
        if task != self.source.task:
            raise ValueError("Replay task changed")
        return self.source.frames[0]

    def step(self, action):
        if (
            self.index >= len(self.source.actions)
            or action != self.source.actions[self.index]
        ):
            raise ValueError("Reconstruction changed an action or its order")
        self.index += 1
        return self.source.frames[self.index]

    def close(self):
        pass


class ScriptedClient:
    def __init__(self, role, state):
        self.role, self.state = role, state

    def complete(self, messages):
        index = self.state["index"]
        turns = self.state["turns"]
        if index >= len(turns) or turns[index]["role"] != self.role:
            raise ValueError("Reconstruction violates protocol handback")
        self.state["index"] += 1
        return turns[index]["raw"]


def compile_replay(source, turns):
    if not source.actions or len(source.frames) != len(source.actions) + 1:
        raise ValueError("A nonempty complete action/observation sequence is required")
    env = get(source.environment, source.task)
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {
            "step",
            "role",
            "raw",
            "supervise",
        }:
            raise ValueError("Invalid reconstructed turn")
        if (
            type(turn["step"]) is not int
            or not 0 <= turn["step"] < len(source.actions)
            or type(turn["supervise"]) is not bool
            or turn["role"] not in ("planner", "actor")
        ):
            raise ValueError("Invalid reconstruction metadata")
        parse_reply(turn["raw"], turn["role"], env.assertion, env.actions)
    state = {"turns": turns, "index": 0}
    sandbox = RecordedSandbox(source)
    result = run(
        source.environment,
        source.task,
        ScriptedClient("planner", state),
        sandbox,
        RunConfig(
            method="conpact_r",
            max_steps=len(source.actions),
            max_calls=len(turns),
            history_frames=3,
            max_discussion_turns=max(12, len(turns)),
            assertion_visibility="independent",
            system_intervention=False,
        ),
        actor=ScriptedClient("actor", state),
        rollout_id=source.rollout_id,
    )
    if result.end in ("model_failure", "infrastructure_error") or state["index"] != len(
        turns
    ):
        raise ValueError("Reconstruction failed protocol replay")
    if result.actions != source.actions or result.frames != source.frames:
        raise ValueError("Reconstruction changed actions or observations")
    if [t["step"] for t in result.turns] != [t["step"] for t in turns]:
        raise ValueError("Reconstruction changed observation boundaries")
    for actual, specified in zip(result.turns, turns):
        actual["supervise"] = specified["supervise"]
    return result


def validate_replay_segments(episode, segments):
    by_action = [
        segment for segment in segments
        for _ in range(segment["start"], segment["end"] + 1)
    ]
    initialized = set()
    active = None
    for turn in episode.turns:
        segment = by_action[turn["step"]]
        start, goal = segment["start"], segment["subgoal"]
        if turn["role"] == "planner":
            if turn["supervise"]:
                if turn["parsed"]["subgoal"] != goal:
                    raise ValueError("Positive planner subgoal differs from the teacher segmentation")
                if turn["step"] == start:
                    initialized.add(start)
            active = turn
        elif turn["executed"]:
            if start not in initialized:
                raise ValueError(
                    f"Segment starting at action {start} needs a supervised planner "
                    "at its start before execution"
                )
            if (
                active is None or not active["supervise"] or active["step"] < start
                or active["parsed"]["subgoal"] != goal
            ):
                raise ValueError(
                    f"Action {turn['step']} is not conditioned on its segment's supervised subgoal"
                )


def reconstruct(
    source: Episode, teacher, reviewer, verifier: TransitionVerifier, *, include_uncertain=False,
):
    if source.method != "react" or source.end == "infrastructure_error":
        raise ValueError(
            "ConPAct-R requires an existing ReAct trajectory without infrastructure errors"
        )
    verified = list(verifier.verify(source.task, source.frames, source.actions))
    if verified != source.frames:
        raise ValueError("Recorded transitions do not match verified transitions")
    segments = validate_segments(
        ask_json(teacher, source_messages(source, SEGMENT_PROMPT)), source.actions
    )
    env = get(source.environment, source.task)
    prompt = env.prompt("conpact_r", source.frames[0])
    instruction = (
        RECONSTRUCT_PROMPT
        + "\nStudent planner protocol:\n"
        + prompt.planner_system
        + "\nStudent actor protocol:\n"
        + prompt.actor_system
        + "\nSegmentation:\n"
    )
    import json

    generated = ask_json(
        teacher, source_messages(source, instruction + json.dumps(segments))
    )
    if (
        not isinstance(generated, dict)
        or set(generated) != {"turns"}
        or not isinstance(generated["turns"], list)
    ):
        raise ValueError("Teacher must return reconstructed turns")
    episode = compile_replay(source, generated["turns"])
    nonoptimal = {index for segment in segments for index in segment["nonoptimal"]}
    hypothetical = [
        turn for turn in episode.turns if not turn["supervise"] and not turn["executed"]
    ]
    if hypothetical and not nonoptimal:
        raise ValueError("Hypothetical errors require a recorded nonoptimal interval")
    validate_replay_segments(episode, segments)
    if nonoptimal:
        hypothetical = [
            turn
            for turn in episode.turns
            if not turn["supervise"] and not turn["executed"]
        ]
        if not hypothetical:
            raise ValueError(
                "Nonoptimal intervals require hypothetical disagreement and recovery"
            )
        for turn in hypothetical:
            if turn["step"] not in nonoptimal and turn["step"] - 1 not in nonoptimal:
                raise ValueError(
                    "Hypothetical disagreement is unrelated to a nonoptimal interval"
                )
            following = [
                item
                for item in episode.turns
                if item["turn"] > turn["turn"] and item["step"] == turn["step"]
            ]
            discussion = [
                item
                for item in following
                if item["role"] == "actor" and item["parsed"]["status"] == "DISCUSS"
            ]
            if not discussion or not any(
                item["role"] == "planner"
                and item["supervise"]
                and item["turn"] > discussion[0]["turn"]
                for item in following
            ):
                raise ValueError(
                    "Hypothetical disagreement requires actor feedback and planner revision"
                )
    records, rejected = [], []
    candidates = select_candidates(episode)
    categories = ("consistency", "uncertainty") if include_uncertain else ("consistency",)
    for category in categories:
        for turn in candidates[category]:
            if not turn["supervise"] or (turn["executed"] and turn["step"] in nonoptimal):
                continue
            row = sample(episode, turn, method="conpact_r")
            row["meta"]["supervision_type"] = category
            verdict = review_sample(reviewer, row)
            if verdict["accept"]:
                records.append(row)
            else:
                rejected.append({"turn": turn["turn"], "reason": verdict["reason"]})
    return {
        "records": records,
        "segments": segments,
        "replay": episode.to_dict(),
        "rejected": rejected,
        "source_action_hash": digest(source.actions),
        "replayed_action_hash": digest(episode.actions),
    }
