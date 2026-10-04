from __future__ import annotations

from dataclasses import asdict, dataclass

VERSION = "assertion-comparison"
CONFLICT = "CONFLICT"
AGREE = "AGREE"
INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class Comparison:
    status: str
    conflicts: dict
    comparable: tuple[str, ...]
    matching: tuple[str, ...]
    unknown: tuple[str, ...]
    planner_unknown: tuple[str, ...]
    actor_unknown: tuple[str, ...]

    @property
    def fully_comparable(self):
        return not self.unknown

    @property
    def equality(self):
        if self.status == INDETERMINATE:
            return None
        return self.status == AGREE

    def to_dict(self):
        return dict(version=VERSION, **asdict(self), fully_comparable=self.fully_comparable)

    def __bool__(self):
        raise TypeError("Check Comparison.status explicitly")


def compare_assertions(schema, planner, actor) -> Comparison:
    """Compare parsed assertions; unknown sentinels are declared per schema field."""
    slots = tuple(schema.slots)
    if not slots or any(
        not isinstance(value, dict) or not set(slots) <= set(value)
        for value in (planner, actor)
    ):
        raise ValueError("Comparison requires two complete, parsed assertions")
    unknown_values = getattr(schema, "unknown_values", {})

    def unknown_fields(value):
        return tuple(
            key for key in slots
            if any(
                type(value[key]) is type(sentinel) and value[key] == sentinel
                for sentinel in unknown_values.get(key, ())
            )
        )

    planner_unknown, actor_unknown = unknown_fields(planner), unknown_fields(actor)
    unknown = tuple(key for key in slots if key in planner_unknown or key in actor_unknown)
    comparable = tuple(key for key in slots if key not in unknown)
    matching = tuple(key for key in comparable if planner[key] == actor[key])
    conflicts = {
        key: {"planner": planner[key], "actor": actor[key]}
        for key in comparable if planner[key] != actor[key]
    }
    status = CONFLICT if conflicts else INDETERMINATE if unknown else AGREE
    return Comparison(
        status, conflicts, comparable, matching, unknown, planner_unknown, actor_unknown,
    )
