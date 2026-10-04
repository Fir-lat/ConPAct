from __future__ import annotations

import json
import re

from .constants import FACINGS, FRONTS, RESOURCES, TOOLS, VITALS

VERSION = "crafter_visual_assertion_v1"
TAG = "assertion"
SLOTS = ("facing", "front", "vitals", "inventory", "tools")
_BLOCK = re.compile(r"<assertion>(.*?)</assertion>", re.DOTALL)


def _count(value):
    return type(value) is int and 0 <= value <= 9


def validate(value):
    if not isinstance(value, dict) or set(value) != set(SLOTS):
        return False
    if value["facing"] not in FACINGS or value["front"] not in FRONTS:
        return False
    vitals = value["vitals"]
    if not isinstance(vitals, list) or len(vitals) != 4 or not all(map(_count, vitals)):
        return False
    for slot, keys in (("inventory", RESOURCES), ("tools", TOOLS)):
        counts = value[slot]
        if not isinstance(counts, dict) or set(counts) != set(keys):
            return False
        if not all(_count(counts[key]) for key in keys):
            return False
    return True


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate assertion key: " + key)
        result[key] = value
    return result


def parse(raw):
    blocks = _BLOCK.findall(raw or "")
    if len(blocks) != 1:
        return None
    try:
        value = json.loads(blocks[0], object_pairs_hook=_unique_pairs)
    except (ValueError, TypeError):
        return None
    return value if validate(value) else None


def equal(a, b):
    return validate(a) and validate(b) and a == b


def from_snapshot(*, semantic, player_pos, inventory, facing, semantic_names):

    if facing not in FACINGS or not isinstance(inventory, dict):
        return None
    if not isinstance(semantic_names, dict):
        return None
    if not all(key in inventory for key in VITALS + RESOURCES + TOOLS):
        return None
    if not isinstance(semantic, (list, tuple)) or not semantic:
        return None
    if not isinstance(player_pos, (list, tuple)) or len(player_pos) != 2:
        return None
    if not all(type(v) is int for v in player_pos):
        return None
    x, y = player_pos
    if not (0 <= x < len(semantic)) or not isinstance(semantic[x], (list, tuple)):
        return None
    if not (0 <= y < len(semantic[x])):
        return None
    dx, dy = dict(up=(0, -1), right=(1, 0), down=(0, 1), left=(-1, 0))[facing]
    x, y = x + dx, y + dy
    if not 0 <= x < len(semantic) or not isinstance(semantic[x], (list, tuple)):
        return None
    if not 0 <= y < len(semantic[x]):
        return None
    cell = semantic[x][y]
    if type(cell) is not int:
        return None
    front = semantic_names.get(cell)
    if front in ("grass", "sand", "path"):
        front = "walkable"
    value = {
        "facing": facing,
        "front": front,
        "vitals": [inventory[k] for k in VITALS],
        "inventory": {k: inventory[k] for k in RESOURCES},
        "tools": {k: inventory[k] for k in TOOLS},
    }
    return value if validate(value) else None


class CrafterAssertionSchema:
    tag = TAG
    slots = SLOTS
    unknown_values = {}
    version = VERSION
    parse = staticmethod(parse)
    equal = staticmethod(equal)


SCHEMA = CrafterAssertionSchema()
