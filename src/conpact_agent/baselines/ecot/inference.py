import re

from .protocol import valid_reply
from ..common import action_reply, planner_reply
from ..plan_act.inference import rollout as plan_act


def validate(session, raw, role):
    env = session.environment
    if env.name == "sokoban":
        from ...envs.sokoban.ecot import parse

        tags = (
            ("state", "plan", "subgoal")
            if role == "planner"
            else ("state", "reason", "action", "status")
        )
        pattern = (
            r"\s*"
            + r"\s*".join("<" + tag + r">([^<>]+)</" + tag + ">" for tag in tags)
            + r"\s*"
        )
        if not re.fullmatch(pattern, raw, re.S) or parse(raw) is None:
            raise ValueError("Invalid ECoT reply")
    elif not valid_reply(raw, role, env.name, env.variant):
        raise ValueError("Invalid ECoT reply")
    return (
        planner_reply(session, raw)
        if role == "planner"
        else action_reply(session, raw, status=True)
    )


def rollout(session):
    return plan_act(
        session,
        planner_parser=lambda raw: validate(session, raw, "planner"),
        actor_parser=lambda raw: validate(session, raw, "actor"),
    )
