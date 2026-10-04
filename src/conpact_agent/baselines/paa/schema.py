from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass


class PlanError(ValueError):
    pass


class PlanFormatError(PlanError):
    pass


class PlanCoverageError(PlanError):
    pass


@dataclass(frozen=True)
class PlanStep:
    step_id: int
    reasoning: str
    subgoal: str

    action_indices: tuple[int, ...]


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]

    num_actions: int = 0

    def to_dict(self) -> dict:
        return {
            "steps": [asdict(s) for s in self.steps],
            "num_actions": self.num_actions,
        }


_PLAN_BLOCK = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_THOUGHT_BLOCK = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)


_NUMBERED = re.compile(r"^\s*(?:[-*+]\s*)?\*{0,2}(\d+)\*{0,2}[.)]\*{0,2}\s+(.*)$")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _numbered_items(text: str) -> list[tuple[int, str]]:

    items: list[tuple[int, str]] = []
    for line in (text or "").splitlines():
        match = _NUMBERED.match(line)
        if match:
            items.append((int(match.group(1)), match.group(2).strip()))
        elif items and line.strip():
            number, body = items[-1]
            items[-1] = (number, f"{body} {line.strip()}")
    return [(n, t) for n, t in items if t]


def parse_annotation(raw: str, num_actions: int) -> Plan:

    text = (raw or "").strip()
    if not text:
        raise PlanFormatError("empty response")
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise PlanFormatError("no JSON object in the response")
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except ValueError as exc:
        raise PlanFormatError(f"not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise PlanFormatError(f"top level is {type(obj).__name__}, not an object")
    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanFormatError("'steps' is missing or not a non-empty list")

    steps: list[PlanStep] = []
    for position, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            raise PlanFormatError(f"step {position} is not an object")
        subgoal = str(item.get("subgoal", "")).strip()
        reasoning = str(item.get("reasoning", "")).strip()
        if not subgoal:
            raise PlanFormatError(f"step {position} has an empty subgoal")
        if not reasoning:
            raise PlanFormatError(f"step {position} has an empty reasoning")

        step_id = item.get("step_id", position)
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            raise PlanFormatError(f"step {position}: step_id is not an integer")
        if step_id != position:
            raise PlanFormatError(
                f"step at position {position} is numbered {step_id}; steps must "
                f"be numbered 1..N in order"
            )
        indices = item.get("action_indices")
        if not isinstance(indices, list) or not indices:
            raise PlanFormatError(
                f"step {position}: action_indices is missing or empty"
            )
        for value in indices:
            if not isinstance(value, int) or isinstance(value, bool):
                raise PlanFormatError(
                    f"step {position}: action_indices holds a non-integer"
                )

        steps.append(
            PlanStep(
                step_id=step_id,
                reasoning=" ".join(reasoning.split()),
                subgoal=" ".join(subgoal.split()),
                action_indices=tuple(indices),
            )
        )

    plan = Plan(steps=tuple(steps), num_actions=num_actions)
    check_coverage(plan, num_actions)
    return plan


def check_coverage(plan: Plan, num_actions: int) -> None:

    if num_actions <= 0:
        raise PlanCoverageError(f"the trajectory has {num_actions} actions")
    flat: list[int] = []
    for step in plan.steps:
        indices = list(step.action_indices)
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise PlanCoverageError(
                f"step {step.step_id}: action_indices {indices} is not a "
                f"contiguous ascending run"
            )
        if indices[0] < 0 or indices[-1] >= num_actions:
            raise PlanCoverageError(
                f"step {step.step_id}: action_indices {indices} falls outside "
                f"0..{num_actions - 1}"
            )
        flat.extend(indices)
    if flat != list(range(num_actions)):
        raise PlanCoverageError(
            f"the steps cover {flat} but the trajectory has actions "
            f"0..{num_actions - 1}; the segments must join, in order, to the "
            f"whole sequence with no gap, overlap or repetition"
        )


SERIALIZER_VERSION = "paa-plan-v1"

_THOUGHT_OPEN = "<thought>\n"
_THOUGHT_CLOSE = "\n</thought>\n"
_PLAN_OPEN = "<plan>\n"
_PLAN_CLOSE = "\n</plan>"


def plan_target(plan: Plan) -> str:

    thoughts = "\n".join(f"{s.step_id}. {s.reasoning}" for s in plan.steps)
    subgoals = "\n".join(f"{s.step_id}. {s.subgoal}" for s in plan.steps)
    return (
        _THOUGHT_OPEN + thoughts + _THOUGHT_CLOSE + _PLAN_OPEN + subgoals + _PLAN_CLOSE
    )


def actor_plan(plan: Plan) -> str:

    return "\n".join(f"{s.step_id}. {s.subgoal}" for s in plan.steps)


def actor_plan_from_raw(raw: str) -> str:

    block = _PLAN_BLOCK.findall(raw or "")
    if not block:
        raise PlanFormatError("no <plan> block in the planner response")
    items = _numbered_items(block[-1])
    if not items:
        raise PlanFormatError("the <plan> block holds no numbered subgoal")
    return "\n".join(f"{number}. {text}" for number, text in items)


def plan_steps_from_raw(raw: str) -> list[tuple[int, str]]:

    block = _PLAN_BLOCK.findall(raw or "")
    return _numbered_items(block[-1]) if block else []


def reasoning_steps_from_raw(raw: str) -> list[tuple[int, str]]:

    block = _THOUGHT_BLOCK.findall(raw or "")
    return _numbered_items(block[-1]) if block else []


def sha256_of(text: str) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_sha = sha256_of


def plan_hash(plan: Plan) -> str:

    return _sha(json.dumps(plan.to_dict(), sort_keys=True, ensure_ascii=False))


def trajectory_hash(task_id: str, level_string: str, actions: list[str]) -> str:

    return _sha(
        json.dumps(
            {"task_id": task_id, "level": level_string, "actions": list(actions)},
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def board_hash(level_string: str) -> str:

    lines = [line.rstrip() for line in (level_string or "").splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return _sha("\n".join(lines))
