from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .games import require_game
from ...comparison import compare_assertions

TARGETS = {
    "bigfish": "at least one fish visibly smaller than the player",
    "chaser": "at least one uncollected green orb",
    "coinrun": "the goal coin",
    "jumper": "the carrot",
    "maze": "the cheese",
    "miner": "at least one diamond",
    "ninja": "the goal mushroom",
}
HAZARDS = {
    "bigfish": "a larger fish",
    "chaser": "a non-vulnerable enemy",
    "coinrun": "a saw, enemy, or deadly gap",
    "jumper": "spikes",
    "miner": "a boulder or diamond that could fall onto the player",
    "ninja": "a bomb or dangerous gap",
}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate assertion key: {key}")
        obj[key] = value
    return obj


@dataclass(frozen=True)
class ProcgenAssertionSchema:
    game: str
    tag = "assertion"
    correctness_available = False
    correctness_reason = "No reliable pixel-state oracle for this Procgen protocol."

    def __post_init__(self):
        require_game(self.game)

    @property
    def slots(self) -> tuple[str, ...]:
        slots = ("target_visible",)
        if self.game in HAZARDS:
            slots += ("hazard_nearby",)
        if self.game == "miner":
            slots += ("diamonds_visible",)
        if self.game == "bigfish":
            slots += ("my_size",)
        return slots

    def example(self) -> dict:
        result = {key: "unknown" for key in self.slots}
        if self.game == "miner":
            result["diamonds_visible"] = None
        return result

    @property
    def unknown_values(self):
        return {
            key: (None,) if key == "diamonds_visible" else ("unknown",)
            for key in self.slots
        }

    def documentation(self) -> str:
        lines = [
            "Read only the CURRENT screenshot. Use exactly the following JSON keys:",
            f'- target_visible: is {TARGETS[self.game]} visible? "yes", "no", or "unknown".',
        ]
        if self.game in HAZARDS:
            lines.append(
                f'- hazard_nearby: is {HAZARDS[self.game]} visibly close enough to threaten the player or the next immediate movement? "yes", "no", or "unknown".'
            )
        if self.game == "miner":
            lines.append(
                "- diamonds_visible: number of visible diamonds, a nonnegative integer; use null if you cannot count reliably. Do not count hidden diamonds."
            )
        if self.game == "bigfish":
            lines.append(
                '- my_size: relative to the other visible fish: "smallest" (all are larger), "middle" (both larger and smaller fish), "biggest" (none is larger), or "unknown". Use "unknown" if comparison is unclear or no other fish is visible.'
            )
        lines += [
            'Use "no" only when the relevant visible area is clear enough to establish absence; otherwise use "unknown".',
            "A target outside the screenshot may still exist. No visible hazard does not establish a safe route.",
            "These are readings of this image, not predicted action results, movement since an unseen frame, or proof of completion.",
            "Report decision-relevant locations and obstacles in your message; these slots do not describe the entire scene.",
            "Schema example (uncertain observation, not an answer to copy): "
            + json.dumps(self.example()),
        ]
        return "\n".join(lines)

    def parse(self, raw: str) -> dict | None:
        matches = re.findall(r"<assertion>(.*?)</assertion>", raw or "", re.DOTALL)
        if len(matches) != 1:
            return None
        try:
            value = json.loads(matches[0], object_pairs_hook=_unique_object)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, dict) or set(value) != set(self.slots):
            return None
        for key, item in value.items():
            if key == "diamonds_visible":
                if item is not None and (type(item) is not int or item < 0):
                    return None
            else:
                domain = (
                    ("smallest", "middle", "biggest", "unknown")
                    if key == "my_size"
                    else ("yes", "no", "unknown")
                )
                if not isinstance(item, str) or item not in domain:
                    return None
        return value

    def compare(self, a: dict, b: dict):
        return compare_assertions(self, a, b)

    def equal(self, a: dict, b: dict) -> bool | None:
        return self.compare(a, b).equality
