from __future__ import annotations

import json
import re
from typing import Any

from .simulator import BOX, BOX_ON_TARGET, EMPTY, TARGET, WALL

TAG = "assertion"
SLOTS = ("box_count", "target_count", "on_goal_count", "player_neighbors")
unknown_values = {}
COUNT_SLOTS = ("box_count", "target_count", "on_goal_count")
CELL_TYPES = frozenset({EMPTY, WALL, BOX, TARGET, BOX_ON_TARGET})
NUM_NEIGHBORS = 4

_BLOCK = re.compile(rf"<{TAG}>(.*?)</{TAG}>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


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


def equal(a: dict[str, Any], b: dict[str, Any]) -> bool:

    return all(a.get(slot) == b.get(slot) for slot in SLOTS)
