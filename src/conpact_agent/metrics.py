from __future__ import annotations

from collections import defaultdict
import math
import json
from statistics import mean, median

from .comparison import AGREE, CONFLICT, INDETERMINATE, VERSION, compare_assertions
from .envs.crafter.constants import ACHIEVEMENTS
from .envs.minigrid.profile import resolve_env_name as resolve_minigrid_env_name
from .envs.procgen.games import GAMES, resolve_game
from .envs.registry import get

EASY_BOUNDS = {
    "bigfish": (1, 40),
    "chaser": (0.5, 13),
    "coinrun": (5, 10),
    "jumper": (3, 10),
    "maze": (5, 10),
    "miner": (1.5, 13),
    "ninja": (3.5, 10),
}


def average(values):
    return mean(values) if values else None


def _event_comparison(schema, turns, event, phase):
    # Recompute from the recorded replies, including for legacy event dictionaries.
    if event.get(phase) is None or (phase == "final" and not event.get("complete")):
        return None
    prefix = "" if phase == "initial" else "final_"
    planner = turns.get(event.get(prefix + "planner"))
    actor = turns.get(event.get(prefix + "actor"))
    if planner is None or actor is None:
        return None
    if (
        planner.get("role") != "planner" or actor.get("role") != "actor"
        or planner.get("step") != event.get("step")
        or actor.get("step") != event.get("step")
    ):
        return None
    p, a = schema.parse(planner.get("raw", "")), schema.parse(actor.get("raw", ""))
    if p is None or a is None:
        return None
    return compare_assertions(schema, p, a)


def _comparison_metrics(comparisons, prefix):
    full = [value for value in comparisons if value.fully_comparable]
    field_count = sum(len(value.comparable) + len(value.unknown) for value in comparisons)
    values = {
        "valid_pairs": len(comparisons),
        "coverage": len(full) / len(comparisons) if comparisons else None,
        "field_coverage": (
            sum(len(value.comparable) for value in comparisons) / field_count
            if field_count else None
        ),
        "planner_unknown_rate": (
            sum(len(value.planner_unknown) for value in comparisons) / field_count
            if field_count else None
        ),
        "actor_unknown_rate": (
            sum(len(value.actor_unknown) for value in comparisons) / field_count
            if field_count else None
        ),
        "partial_conflict_pairs": sum(
            value.status == CONFLICT and not value.fully_comparable for value in comparisons
        ),
        "indeterminate_pairs": sum(value.status == INDETERMINATE for value in comparisons),
        "discordance": sum(value.status == CONFLICT for value in full) / len(full) if full else None,
    }
    return {prefix + "_" + key: value for key, value in values.items()}


def assertion_metrics(episodes):
    """Report full-schema discordance, abstention coverage, and explicit repair outcomes."""
    comparisons = []
    unavailable = 0
    for episode in episodes:
        if episode.end == "infrastructure_error":
            continue
        events = [event for event in episode.events if event.get("independent")]
        if not events:
            continue
        schema = get(episode.environment, episode.task).assertion
        turns = {turn["turn"]: turn for turn in episode.turns}
        for event in events:
            initial = _event_comparison(schema, turns, event, "initial")
            if initial is None:
                unavailable += 1
                continue
            final = _event_comparison(schema, turns, event, "final")
            comparisons.append((initial, final))
    initials = [initial for initial, _ in comparisons]
    finals = [final for _, final in comparisons if final is not None]
    repair_finals = [final for initial, final in comparisons if initial.status == CONFLICT]
    observed_repairs = [final for final in repair_finals if final is not None]
    repaired = sum(final.status == AGREE for final in observed_repairs)
    return {
        "assertion_metrics_version": VERSION,
        "pairs": sum(value.fully_comparable for value in initials),
        "final_pairs": sum(value.fully_comparable for value in finals),
        "missing_final_pairs": len(initials) - len(finals),
        "unavailable_initial_pairs": unavailable,
        **_comparison_metrics(initials, "initial"),
        **_comparison_metrics(finals, "final"),
        "repair_pairs": len(observed_repairs),
        "repaired_pairs": repaired,
        "repair_indeterminate_pairs": sum(final.status == INDETERMINATE for final in observed_repairs),
        "repair_conflict_pairs": sum(final.status == CONFLICT for final in observed_repairs),
        "repair_missing_final_pairs": len(repair_finals) - len(observed_repairs),
        "repair_rate": repaired / len(observed_repairs) if observed_repairs else None,
    }


def paired_correctness(pairs, schema):
    valid = []
    parsed_pairs = 0
    for planner_raw, actor_raw, truth in pairs:
        planner, actor = schema.parse(planner_raw), schema.parse(actor_raw)
        if planner is None or actor is None:
            continue
        parsed_pairs += 1
        truth = schema.parse("<assertion>" + json.dumps(truth) + "</assertion>")
        if truth is None or not compare_assertions(schema, planner, actor).fully_comparable:
            continue
        if not compare_assertions(schema, truth, truth).fully_comparable:
            continue
        valid.append((planner == truth, actor == truth))
    return {
        "pairs": len(valid),
        "valid_pairs": parsed_pairs,
        "coverage": len(valid) / parsed_pairs if parsed_pairs else None,
        "planner_accuracy": average([p for p, _ in valid]),
        "actor_accuracy": average([a for _, a in valid]),
    }


def adherence(verdicts):
    values = [value for value in verdicts if value in ("FOLLOW", "VIOLATE")]
    return sum(value == "FOLLOW" for value in values) / len(values) if values else None


def reward(episode):
    values = [frame.reward for frame in episode.frames[1:]]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        return None
    return sum(values)


def iqm(values):
    if not values:
        return None
    values = sorted(values)
    trim = int(len(values) * 0.25)
    return mean(values[trim : len(values) - trim])


def summarize(episodes):
    if (
        len({episode.environment for episode in episodes}) > 1
        or len({episode.method for episode in episodes}) > 1
    ):
        raise ValueError("Summarize one environment and method at a time")
    identities = [(episode.task.id, episode.rollout_id) for episode in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate rollout identity")
    valid = [episode for episode in episodes if episode.end != "infrastructure_error"]
    result = {
        "rollouts": len(episodes),
        "included": len(valid),
        "excluded_infrastructure": len(episodes) - len(valid),
        "excluded_fraction": (len(episodes) - len(valid)) / len(episodes)
        if episodes
        else None,
        "success_rate": average([episode.solved for episode in valid]),
        **assertion_metrics(episodes),
    }
    if not valid:
        return result
    name = valid[0].environment
    if name == "sokoban":
        grouped = defaultdict(list)
        for episode in valid:
            grouped[episode.task.id].append(episode.solved)
        counts = {len(values) for values in grouped.values()}
        result.update(
            avg_at_k=average([average(values) for values in grouped.values()]),
            pass_at_k=average([any(values) for values in grouped.values()])
            if len(counts) == 1
            else None,
            pass_power_k=average([all(values) for values in grouped.values()])
            if len(counts) == 1
            else None,
            k=next(iter(counts)) if len(counts) == 1 else None,
        )
    elif name == "minigrid":
        grouped = defaultdict(list)
        for episode in valid:
            grouped[resolve_minigrid_env_name(episode.task.payload)].append(episode)
        returns = [
            average([reward(episode) for episode in rows])
            if all(reward(e) is not None for e in rows)
            else None
            for rows in grouped.values()
        ]
        result.update(
            success_rate=average(
                [
                    average([episode.solved for episode in rows])
                    for rows in grouped.values()
                ]
            ),
            mean_return=average(returns)
            if all(value is not None for value in returns)
            else None,
            successful_steps=average(
                [len(episode.actions) for episode in valid if episode.solved]
            ),
            task_types=len(grouped),
        )
    elif name == "crafter":
        counts = [episode.frames[-1].info.get("achievements", {}) for episode in valid]
        if all(set(ACHIEVEMENTS) <= set(row) for row in counts):
            rates = [
                100 * mean([row[key] > 0 for row in counts]) for key in ACHIEVEMENTS
            ]
            result.update(
                score=math.expm1(mean([math.log1p(value) for value in rates])),
                progress=mean(rates),
            )
        else:
            result.update(score=None, progress=None)
        values = [reward(episode) for episode in valid]
        result["mean_return"] = (
            average(values) if all(value is not None for value in values) else None
        )
    elif name == "procgen":
        grouped = defaultdict(list)
        for episode in valid:
            grouped[resolve_game(episode.task.payload)].append(reward(episode))
        scores = []
        for game in GAMES:
            values = grouped[game]
            if not values or any(value is None for value in values):
                break
            low, high = EASY_BOUNDS[game]
            scores.append((mean(values) - low) / (high - low))
        result.update(
            iqm=iqm(scores) if len(scores) == len(GAMES) else None,
            mean_normalized_return=mean(scores) if len(scores) == len(GAMES) else None,
            median_normalized_return=median(scores)
            if len(scores) == len(GAMES)
            else None,
        )
    elif name == "osworld":
        scores = [episode.metrics.get("score") for episode in valid]
        result["score"] = (
            average(scores) if all(type(v) in (int, float) for v in scores) else None
        )
    return result
