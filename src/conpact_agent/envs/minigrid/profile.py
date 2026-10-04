from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from ..base import Task

DATA_DIR = Path(__file__).with_name("data")
PROTOCOL_VERSION = "minigrid-v1"
ACTION_INDEX = dict(left=0, right=1, forward=2, pickup=3, drop=4, toggle=5, done=6)


@dataclass(frozen=True)
class TaskProfile:
    env_name: str
    rules: str
    actions: tuple[str, ...]
    max_steps: int
    max_total_calls: int
    eval_levels: int
    native_action_space_n: int

    def action_index(self, action: str) -> int:
        if action not in self.actions:
            raise ValueError(
                f"{action!r} is outside the prompt action set for {self.env_name}"
            )
        index = ACTION_INDEX[action]
        if index >= self.native_action_space_n:
            raise ValueError(f"action {action!r} exceeds the native action space")
        return index

    def validate_native_actions(self, n: int) -> None:
        if type(n) is not int or n != self.native_action_space_n:
            raise ValueError(
                f"{self.env_name}: expected native action_space.n="
                f"{self.native_action_space_n}, got {n!r}"
            )


@lru_cache(maxsize=1)
def _profiles() -> dict[str, TaskProfile]:
    data = json.loads((DATA_DIR / "task_profiles.json").read_text())
    if data["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("MiniGrid profile protocol mismatch")
    profiles = {}
    for name, row in data["profiles"].items():
        profile = TaskProfile(
            env_name=name, **{**row, "actions": tuple(row["actions"])}
        )
        if type(profile.max_steps) is not int or profile.max_steps <= 0:
            raise ValueError(f"invalid action budget for {name}")
        if profile.max_total_calls != 2 * profile.max_steps:
            raise ValueError(f"invalid model call budget for {name}")
        if Path(profile.rules).name != profile.rules:
            raise ValueError(
                "rule filename must be relative to the bundled rules directory"
            )
        for action in profile.actions:
            profile.action_index(action)
        profiles[name] = profile
    return profiles


def get_profile(env_name: str) -> TaskProfile:
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("MiniGrid env_name must be a nonempty supported task ID")
    try:
        return _profiles()[env_name]
    except KeyError:
        raise ValueError(f"unknown MiniGrid task ID {env_name!r}") from None


def resolve_env_name(payload: dict) -> str:
    if "env_name" not in payload:
        raise ValueError("MiniGrid task requires an explicit payload.env_name")
    return get_profile(payload["env_name"]).env_name


def task_ids() -> tuple[str, ...]:
    return tuple(_profiles())


def budget_for_task(task: Task) -> dict[str, int]:
    profile = get_profile(resolve_env_name(task.payload))
    budget = {
        "max_steps": profile.max_steps,
        "max_total_calls": profile.max_total_calls,
    }
    for key, value in budget.items():
        if key in task.payload and task.payload[key] != value:
            raise ValueError(f"{task.id}: task payload disagrees with {key}={value}")
    return budget
