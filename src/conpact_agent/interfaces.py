from __future__ import annotations

from typing import Any, Protocol, Sequence

from .envs.base import Frame, Task


class ModelClient(Protocol):
    def complete(self, messages: list[dict[str, Any]]) -> str: ...


class Sandbox(Protocol):
    def reset(self, task: Task) -> Frame: ...
    def step(self, action: str) -> Frame: ...
    def close(self) -> None: ...


class TransitionVerifier(Protocol):
    def verify(
        self, task: Task, frames: Sequence[Frame], actions: Sequence[str]
    ) -> Sequence[Frame]: ...


class InfrastructureError(RuntimeError):
    pass


class ModelError(RuntimeError):
    pass
