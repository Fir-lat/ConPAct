from __future__ import annotations
import re
from ..methods import parse


def action_reply(session, raw, *, status=False, assertion=False):
    action = session.environment.parse_action(raw)
    value = {"action": action}
    if action is None:
        raise ValueError("Missing or invalid action")
    if status:
        value["status"] = parse.parse_status(raw)
        if value["status"] is None:
            raise ValueError("Missing or invalid status")
    if assertion:
        value["assertion"] = session.environment.assertion.parse(raw)
        if value["assertion"] is None:
            raise ValueError("Invalid assertion")
    return value


def planner_reply(session, raw, *, assertion=False):
    goal = parse.parse_subgoal(raw)
    if not goal or len(re.findall(r"<subgoal>", raw)) != 1:
        raise ValueError("Missing or invalid subgoal")
    value = {"subgoal": goal}
    if assertion:
        value["assertion"] = session.environment.assertion.parse(raw)
        if value["assertion"] is None:
            raise ValueError("Invalid assertion")
    return value
