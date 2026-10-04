from __future__ import annotations

from collections import defaultdict

from .records import review_sample, sample
from ..comparison import AGREE, CONFLICT, compare_assertions
from ..envs.registry import get
from ..methods.conpact_i import parse_reply


def execution_segments(turns):
    """Keep pre-action repairs together; replanning after execution starts a new segment."""
    segments, current = [], []
    has_execution = False
    for turn in turns:
        if turn["role"] == "planner" and not turn.get("correction") and has_execution:
            segments.append(current)
            current = []
            has_execution = False
        current.append(turn)
        if turn.get("executed"):
            has_execution = True
    if current:
        segments.append(current)
    return segments


def select_candidates(episode):
    eligible = {"consistency": [], "uncertainty": []}
    if episode.end == "infrastructure_error":
        return eligible
    if episode.method not in ("conpact_i", "conpact_s", "conpact_r"):
        raise ValueError("ConPAct-S requires planner-actor discussion")
    if not episode.events:
        return eligible
    env = get(episode.environment, episode.task)
    parsed = {}
    for turn in episode.turns:
        if turn.get("error") or turn["role"] not in ("planner", "actor"):
            continue
        try:
            parsed[turn["turn"]] = parse_reply(
                turn["raw"], turn["role"], env.assertion, env.actions
            )
        except ValueError:
            continue
    conflicting_turns = set()
    turns_by_id = {turn["turn"]: turn for turn in episode.turns}
    for event in episode.events:
        checks = event.get("checks")
        if not checks:
            raise ValueError("ConPAct supervision requires comparison checks; collect or replay new trajectories")
        for check in checks:
            p, a = check.get("planner"), check.get("actor")
            if (
                p not in parsed or a not in parsed
                or turns_by_id[p]["role"] != "planner"
                or turns_by_id[a]["role"] != "actor"
                or turns_by_id[p]["step"] != event["step"]
                or turns_by_id[a]["step"] != event["step"]
            ):
                raise ValueError("Invalid same-observation comparison record")
            if compare_assertions(
                env.assertion, parsed[p]["assertion"], parsed[a]["assertion"],
            ).status == CONFLICT:
                conflicting_turns.update((p, a))
    for segment in execution_segments(episode.turns):
        by_step = defaultdict(list)
        for turn in segment:
            if turn["turn"] in parsed:
                by_step[turn["step"]].append(turn)
        comparisons = {}
        for step, turns in by_step.items():
            planners = [t for t in turns if t["role"] == "planner"]
            actors = [t for t in turns if t["role"] == "actor"]
            if planners and actors:
                p, a = planners[-1], actors[-1]
                comparisons[step] = compare_assertions(
                    env.assertion,
                    parsed[p["turn"]]["assertion"], parsed[a["turn"]]["assertion"],
                )
        checks = list(comparisons.values())
        if not checks or any(
            check.status == CONFLICT for check in checks
        ):
            continue
        segment_agrees = all(check.status == AGREE for check in checks)
        turn_ids = {turn["turn"] for turn in segment}
        if any(
            not event.get("complete", False)
            for event in episode.events
            if event.get("planner") in turn_ids
        ):
            continue
        for turn in segment:
            value = parsed.get(turn["turn"])
            if (
                value is None or turn.get("supervise") is False
                or turn["turn"] in conflicting_turns
            ):
                continue
            known = compare_assertions(
                env.assertion, value["assertion"], value["assertion"],
            ).status == AGREE
            category = "consistency" if segment_agrees and known else "uncertainty"
            eligible[category].append(turn)
    return eligible


def select_targets(episode):
    return select_candidates(episode)["consistency"]


def build(episodes, *, include_uncertain=False, reviewer=None):
    if include_uncertain and reviewer is None:
        raise ValueError("Uncertainty supervision requires an independent reviewer")
    rows = []
    for episode in episodes:
        candidates = select_candidates(episode)
        categories = ("consistency", "uncertainty") if include_uncertain else ("consistency",)
        for category in categories:
            for turn in candidates[category]:
                row = sample(episode, turn, method="conpact_s")
                row["meta"]["supervision_type"] = category
                if category == "uncertainty" and not review_sample(reviewer, row)["accept"]:
                    continue
                rows.append(row)
    return rows
