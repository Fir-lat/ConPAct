from __future__ import annotations

import re


STATUSES = ("CONTINUE", "ACHIEVED", "IMPOSSIBLE", "STUCK", "CHANGED")

_SUBGOAL = re.compile(r"<subgoal>(.*?)</subgoal>", re.DOTALL)
_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_STATUS = re.compile(r"<status>(.*?)</status>", re.DOTALL)
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_HINT = re.compile(r"<hint>(.*?)</hint>", re.DOTALL)
_REPORT = re.compile(r"<report>(.*?)</report>", re.DOTALL)


_MILESTONE = re.compile(r"^\s*(?:[-*]\s*)?\*{0,2}(\d+)\*{0,2}[.)]\s+(.*)$")


def _last(pattern: re.Pattern, text: str) -> str | None:
    matches = pattern.findall(text or "")
    return matches[-1].strip() if matches else None


def parse_subgoal(raw: str) -> str | None:
    return _last(_SUBGOAL, raw) or None


def parse_status(raw: str) -> str | None:

    value = _last(_STATUS, raw)
    if value is None:
        return None
    value = value.upper()
    return value if value in STATUSES else None


def parse_plan(raw: str) -> str | None:

    return _last(_PLAN, raw) or None


def parse_hint(raw: str) -> str | None:

    return _last(_HINT, raw) or None


def parse_report(raw: str) -> str | None:

    return _last(_REPORT, raw) or None


def parse_milestones(plan: str) -> list[tuple[int, str]]:

    milestones: list[tuple[int, str]] = []
    for line in (plan or "").splitlines():
        match = _MILESTONE.match(line)
        if match:
            milestones.append((int(match.group(1)), match.group(2).strip()))
        elif milestones and line.strip():
            number, text = milestones[-1]
            milestones[-1] = (number, f"{text} {line.strip()}")
    return [(n, t) for n, t in milestones if t]


def parse_action(raw: str, valid: tuple[str, ...] = ()) -> str | None:

    value = _last(_ACTION, raw)
    if not value:
        return None
    value = value.lower()
    if valid and value not in valid:
        return None
    return value


def _hiplan_unwrap(raw: str) -> str:

    text = (raw or "").strip()
    match = re.fullmatch(r"```(?:xml|text)?\s*\n(.*?)\n```", text, re.S | re.I)
    return match.group(1).strip() if match else text


def valid_hiplan_plan(raw: str) -> bool:
    text = _hiplan_unwrap(raw)
    if not re.fullmatch(
        r"(?:<thought>[^<>]*</thought>\s*)?<plan>[^<>]+</plan>", text, re.S
    ):
        return False
    milestones = parse_milestones(parse_plan(text) or "")
    numbers = [n for n, _ in milestones]
    return bool(numbers) and numbers[0] > 0 and numbers == sorted(set(numbers))


def valid_hiplan_hint(raw: str) -> bool:
    text = _hiplan_unwrap(raw)
    match = re.fullmatch(
        r"<status>(CONTINUE|ACHIEVED)</status>\s*<hint>([^<>]+)</hint>",
        text,
        re.S | re.I,
    )
    if not match:
        return False
    body = match.group(2).strip()
    fields = re.split(
        r"(?:^|\n)\s*(?:[-*]\s*)?(Current State|Current Milestone|Milestone Gap|Action Correction):\s*",
        body,
    )
    return (
        not fields[0].strip()
        and fields[1::2]
        in (
            ["Current State", "Current Milestone", "Milestone Gap"],
            [
                "Current State",
                "Current Milestone",
                "Milestone Gap",
                "Action Correction",
            ],
        )
        and all(value.strip() for value in fields[2::2])
    )


def valid_hiplan_actor(raw: str) -> bool:
    return (
        re.fullmatch(
            r"(?:<thought>[^<>]*</thought>\s*)?<action>[^<>]+</action>",
            _hiplan_unwrap(raw),
            re.S,
        )
        is not None
    )
