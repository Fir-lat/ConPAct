from __future__ import annotations
from ...methods import messages as m
from ..common import action_reply
import copy


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
        context = copy.deepcopy(messages)
        context.extend(
            [
                {"role": "assistant", "content": turn["raw"]},
                {
                    "role": "user",
                    "content": "Critique this candidate using only the current observation and task. Identify errors or improvements. The action has not executed.",
                },
            ]
        )
        critique = session.ask("actor", context)
        context.extend(
            [
                {"role": "assistant", "content": critique["raw"]},
                {
                    "role": "user",
                    "content": "Revise the candidate using this feedback. Return the complete reply in the original action format. The action has not executed.",
                },
            ]
        )
        turn = session.ask("actor", context, lambda raw: action_reply(session, raw))
        session.act(turn["parsed"]["action"], turn)
