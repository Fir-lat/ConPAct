from __future__ import annotations
from ...methods import messages as m
from ..common import action_reply
from . import schema as paa_schema


def rollout(session):
    if session.done():
        return
    plan = session.ask(
        "planner",
        m.plan_messages(session.prompts, session.observations[0]),
        paa_schema.actor_plan_from_raw,
    )
    while not session.done():
        turn = session.ask(
            "actor",
            m.actor_messages(
                session.prompts,
                plan["parsed"],
                session.observations,
                session.episode.actions,
                session.actor_window,
            ),
            lambda raw: action_reply(session, raw),
        )
        session.act(turn["parsed"]["action"], turn)
