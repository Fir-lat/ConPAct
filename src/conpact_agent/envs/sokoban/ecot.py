from __future__ import annotations

import json
import re
from typing import Any

from .assertion import CELL_TYPES, NUM_NEIGHBORS
from .simulator import State, cell_type, frame_truth

TAG = "state"


SLOTS = (
    "player",
    "boxes",
    "targets",
    "box_count",
    "target_count",
    "on_goal_count",
    "player_neighbors",
)
COUNT_SLOTS = ("box_count", "target_count", "on_goal_count")
POSITION_SLOTS = ("boxes", "targets")


MOVE_WALK = "walk"
MOVE_PUSH = "push"
MOVE_PUSH_ONTO_GOAL = "push_onto_goal"
MOVE_PUSH_OFF_GOAL = "push_off_goal"
MOVE_BLOCKED = "blocked"

_BLOCK = re.compile(rf"<{TAG}>(.*?)</{TAG}>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def relative(state: State, pos: tuple[int, int]) -> list[int]:

    return [pos[1] - state.player[1], pos[0] - state.player[0]]


def _sorted_relative(state: State, positions) -> list[list[int]]:

    return sorted((relative(state, p) for p in positions), key=lambda c: (c[1], c[0]))


def state_labels(state: State) -> dict[str, Any]:

    truth = frame_truth(state)
    return {
        "player": [0, 0],
        "boxes": _sorted_relative(state, state.boxes),
        "targets": _sorted_relative(state, state.targets),
        "box_count": truth["box_count"],
        "target_count": truth["target_count"],
        "on_goal_count": truth["on_goal_count"],
        "player_neighbors": truth["player_neighbors"],
    }


def render(labels: dict[str, Any]) -> str:

    ordered = {slot: labels[slot] for slot in SLOTS}
    return (
        f"<{TAG}>"
        + json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
        + f"</{TAG}>"
    )


def transition_labels(before: State, after: State, action: str) -> dict[str, Any]:

    if before.player == after.player:
        kind = MOVE_BLOCKED
    elif before.boxes != after.boxes:
        on_before = len(before.boxes & before.targets)
        on_after = len(after.boxes & after.targets)
        if on_after > on_before:
            kind = MOVE_PUSH_ONTO_GOAL
        elif on_after < on_before:
            kind = MOVE_PUSH_OFF_GOAL
        else:
            kind = MOVE_PUSH
    else:
        kind = MOVE_WALK
    return {
        "action": action,
        "move_kind": kind,
        "on_goal_before": len(before.boxes & before.targets),
        "on_goal_after": len(after.boxes & after.targets),
        "player_from": list(before.player),
        "player_to": list(after.player),
        "target_cell": cell_type(before, _ahead(before, action)),
        "solved_after": bool(after.boxes) and after.boxes <= after.targets,
    }


def _ahead(state: State, action: str) -> tuple[int, int]:
    from .simulator import DELTA

    dr, dc = DELTA.get(action, (0, 0))
    return (state.player[0] + dr, state.player[1] + dc)


def _position_list(value: Any) -> list[list[int]] | None:

    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            return None
        for coordinate in item:
            if isinstance(coordinate, bool) or not isinstance(coordinate, int):
                return None
        out.append([item[0], item[1]])
    return sorted(out, key=lambda c: (c[1], c[0]))


def parse(raw: str) -> dict[str, Any] | None:

    if not raw:
        return None
    blocks = _BLOCK.findall(raw)
    if not blocks:
        return None
    body = _FENCE.sub("", blocks[-1].strip())
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    out: dict[str, Any] = {}
    player = _position_list([obj.get("player")])
    if player is None:
        return None
    out["player"] = player[0]

    for slot in POSITION_SLOTS:
        positions = _position_list(obj.get(slot))
        if positions is None:
            return None
        out[slot] = positions

    for slot in COUNT_SLOTS:
        value = obj.get(slot)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        out[slot] = value

    neighbors = obj.get("player_neighbors")
    if not isinstance(neighbors, list) or len(neighbors) != NUM_NEIGHBORS:
        return None
    normalised = []
    for item in neighbors:
        if not isinstance(item, str):
            return None
        item = item.strip().lower()
        if item not in CELL_TYPES:
            return None
        normalised.append(item)
    out["player_neighbors"] = normalised
    return out


def slot_matches(read: dict[str, Any], truth: dict[str, Any]) -> dict[str, bool]:

    return {slot: read.get(slot) == truth.get(slot) for slot in SLOTS}


def equal(read: dict[str, Any], truth: dict[str, Any]) -> bool:

    return all(slot_matches(read, truth).values())
