from __future__ import annotations

import json
import re

TAG = "assertion"
SLOTS = (
    "key_count",
    "front_cell",
    "locked_door_count",
    "closed_door_count",
    "open_door_count",
)
FRONT_CELLS = (
    "empty",
    "wall",
    "door_locked",
    "door_closed",
    "door_open",
    "key",
    "ball",
    "box",
    "goal",
    "lava",
)


def valid(value) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(SLOTS)
        and isinstance(value["front_cell"], str)
        and value["front_cell"] in FRONT_CELLS
        and all(
            type(value[key]) is int and value[key] >= 0
            for key in SLOTS
            if key != "front_cell"
        )
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate assertion key: {key}")
        result[key] = value
    return result


def parse(raw: str) -> dict | None:
    matches = re.findall(r"<assertion>(.*?)</assertion>", raw or "", re.S | re.I)
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0], object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        return None
    return value if valid(value) else None


def equal(a, b) -> bool:
    return valid(a) and valid(b) and a == b


class AssertionSchema:
    tag = TAG
    slots = SLOTS
    unknown_values = {}
    parse = staticmethod(parse)
    equal = staticmethod(equal)


SCHEMA = AssertionSchema()
