from __future__ import annotations
from ...methods import messages as m
from ..common import action_reply
from ..common import planner_reply
from ...comparison import CONFLICT, compare_assertions


def rollout(
    session, *, asserted=False, resampling=False, planner_parser=None, actor_parser=None
):
    planner_parser = planner_parser or (
        lambda raw: planner_reply(session, raw, assertion=asserted)
    )
    actor_parser = actor_parser or (
        lambda raw: action_reply(session, raw, status=True, assertion=asserted)
    )
    subgoal = status = ""
    history = []
    need_plan = True
    while not session.done():
        planner = None
        if need_plan:
            request = m.planner_messages(
                session.prompts,
                session.step,
                session.observations,
                history,
                subgoal,
                status,
                session.planner_window,
            )
            planner = session.ask("planner", request, planner_parser)
            subgoal = planner["parsed"]["subgoal"]
        actor_request = lambda: m.actor_messages(
            session.prompts,
            subgoal,
            session.observations,
            session.episode.actions,
            session.actor_window,
        )
        actor = session.ask("actor", actor_request(), actor_parser)
        if asserted and planner:
            comparison = compare_assertions(
                session.environment.assertion,
                planner["parsed"]["assertion"],
                actor["parsed"]["assertion"],
            )
            event = dict(
                step=session.step,
                planner=planner["turn"],
                actor=actor["turn"],
                initial=comparison.to_dict(),
                final=comparison.to_dict(),
                rounds=0,
                final_planner=planner["turn"],
                final_actor=actor["turn"],
                independent=True,
                complete=False,
            )
            session.episode.events.append(event)
            while (
                resampling
                and comparison.status == CONFLICT
                and event["rounds"] < session.config.discussion_rounds
            ):
                if session.budget.remaining < 2:
                    break
                planner = session.ask(
                    "planner",
                    request,
                    lambda raw: planner_reply(session, raw, assertion=True),
                )
                subgoal = planner["parsed"]["subgoal"]
                actor = session.ask(
                    "actor",
                    actor_request(),
                    lambda raw: action_reply(session, raw, status=True, assertion=True),
                )
                comparison = compare_assertions(
                    session.environment.assertion,
                    planner["parsed"]["assertion"],
                    actor["parsed"]["assertion"],
                )
                event.update(
                    final=comparison.to_dict(),
                    rounds=event["rounds"] + 1,
                    final_planner=planner["turn"],
                    final_actor=actor["turn"],
                )
            event["complete"] = True
        if need_plan:
            history.append((session.step, subgoal))
        status = actor["parsed"]["status"]
        session.act(actor["parsed"]["action"], actor)
        need_plan = status != "CONTINUE"
