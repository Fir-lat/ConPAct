from __future__ import annotations
from ...methods import messages as m
from ..common import action_reply


def rollout(session):
    while not session.done():
        messages = m.actor_messages(
            session.prompts,
            "",
            session.observations,
            session.episode.actions,
            session.actor_window,
        )
        turn = session.ask("actor", messages, lambda raw: action_reply(session, raw))
        session.act(turn["parsed"]["action"], turn)
