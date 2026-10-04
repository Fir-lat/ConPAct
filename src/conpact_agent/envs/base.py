from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


Assertion = dict[str, Any]


@dataclass(frozen=True)
class Task:
    id: str

    payload: dict[str, Any]

    difficulty: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    image_b64: str
    solved: bool

    progress: int
    progress_total: int

    text: str = ""
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AssertionSchema(Protocol):
    tag: str
    slots: tuple[str, ...]
    unknown_values: dict[str, tuple[Any, ...]]

    def parse(self, raw: str) -> Assertion | None: ...

    def equal(self, a: Assertion, b: Assertion) -> bool | None: ...


@dataclass(frozen=True)
class PromptSet:
    kind: str

    actor_system: str
    planner_system: str = ""
    obs: str = "image"
    discussion_assertion: bool = True
    discussion_direct: bool = False
    discussion_context: str = "json"

    discussion_scope: str = "recent"

    history_mode: str = "frames"
    instruction: str = ""
    instruction_with_history: str = ""

    planner_prev_block: str = ""
    history_subgoal_block: str = ""
    history_subgoal_item: str = ""
    history_frame_block: str = ""
    history_frame_item: str = ""
    current_frame_block: str = ""
    subgoal_fallback: str = ""


def fill(template: str, **values: object) -> str:

    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out
