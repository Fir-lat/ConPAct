from __future__ import annotations
from ...methods import messages as m
from ..common import action_reply
from ...methods import parse


def rollout(session):
    if session.done():
        return

    def parse_guide(raw):
        if not parse.valid_hiplan_plan(raw):
            raise ValueError("Invalid milestone plan")
        return parse.parse_plan(raw)

    guide = session.ask(
        "planner",
        m.plan_messages(session.prompts, session.observations[0]),
        parse_guide,
    )["parsed"]
    milestones = parse.parse_milestones(guide)
    pointer = 0
    while not session.done():
        while True:
            index, milestone = milestones[pointer]

            def parse_hint(raw):
                if not parse.valid_hiplan_hint(raw):
                    raise ValueError("Invalid milestone hint")
                return {
                    "status": parse.parse_status(raw),
                    "hint": parse.parse_hint(raw),
                }

            hint = session.ask(
                "hint",
                m.hint_messages(
                    session.prompts,
                    guide,
                    index,
                    milestone,
                    session.observations,
                    session.episode.actions,
                    session.config.history_frames,
                ),
                parse_hint,
            )["parsed"]
            if hint["status"] != "ACHIEVED" or pointer == len(milestones) - 1:
                break
            pointer += 1
        request = m.actor_messages(
            session.prompts,
            milestone,
            session.observations,
            session.episode.actions,
            session.actor_window,
            extra=dict(global_plan=guide, milestone_index=index, step_hint=hint["hint"]),
        )
        turn = session.ask("actor", request, lambda raw: action_reply(session, raw))
        session.act(turn["parsed"]["action"], turn)
