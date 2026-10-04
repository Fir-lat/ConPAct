from __future__ import annotations
from collections import defaultdict
from .records import sample
from ..envs.procgen.games import resolve_game


def episode_return(episode):
    values = [frame.reward for frame in episode.frames[1:]]
    if any(type(v) not in (int, float) for v in values):
        raise ValueError("Reward is unavailable")
    return sum(values)


def expert_cutoff(episode, *, hindsight=False):
    if episode.end == "infrastructure_error" or not episode.actions:
        return None
    if episode.environment in ("sokoban", "minigrid", "osworld"):
        return len(episode.actions) if episode.solved else None
    if episode.environment == "procgen":
        game = resolve_game(episode.task.payload)
        value = episode_return(episode)
        if game in ("coinrun", "jumper", "maze", "ninja"):
            return len(episode.actions) if episode.solved and value >= 10 else None
        positive = [
            i for i, frame in enumerate(episode.frames[1:], 1) if frame.reward > 0
        ]
        return positive[-1] if value > 0 and positive else None
    if episode.environment == "crafter":
        if (
            hindsight
            and episode.frames[-1]
            .info.get("achievements", {})
            .get("make_stone_pickaxe", 0)
            <= 0
        ):
            return None
        if episode_return(episode) <= 0:
            return None
        previous = episode.frames[0].info.get("achievements", {})
        last = None
        for i, frame in enumerate(episode.frames[1:], 1):
            current = frame.info.get("achievements", {})
            if any(
                value > 0 and previous.get(key, 0) == 0
                for key, value in current.items()
            ):
                last = i
            previous = current
        return last
    raise ValueError("Unsupported demonstration environment")


def one_per_level(episodes, *, hindsight=False):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode.environment, episode.task.id)].append(episode)
    selected = []
    for rows in groups.values():
        candidates = [
            (episode, expert_cutoff(episode, hindsight=hindsight)) for episode in rows
        ]
        candidates = [
            (episode, cutoff) for episode, cutoff in candidates if cutoff is not None
        ]
        if not candidates:
            continue
        if rows[0].environment == "crafter":
            candidates.sort(
                key=lambda pair: (-episode_return(pair[0]), pair[0].rollout_id)
            )
        else:
            candidates.sort(key=lambda pair: pair[0].rollout_id)
        selected.append(candidates[0])
    return selected


def ordinary_sft(episodes, method):
    if method not in ("react", "plan_act") or any(
        episode.method != method for episode in episodes
    ):
        raise ValueError("Source method must match the SFT baseline")
    rows = []
    for episode, cutoff in one_per_level(episodes):
        for turn in episode.turns:
            if (
                turn["step"] < cutoff
                and not turn.get("error")
                and turn.get("parsed")
                and (turn["role"] == "planner" or turn.get("executed"))
            ):
                rows.append(sample(episode, turn, method=method + "_sft"))
    return rows


def make_turn(role, step, messages, raw, index=0):
    return dict(
        role=role, step=step, messages=messages, raw=raw, turn=index, error=None
    )
