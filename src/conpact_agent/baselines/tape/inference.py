from __future__ import annotations
from . import protocol as tape
from .graph import (
    parse_candidate_graph,
    parse_goal_scores,
    parse_return_scores,
    select_return_path,
)


def rollout(session):
    projection = None
    path = []
    graph = None
    while not session.done():
        prompts = session.prompts
        if projection is None:
            projection = session.ask(
                "projector",
                tape.projector_messages(
                    prompts,
                    session.observations,
                    session.episode.actions,
                    session.config.history_frames,
                ),
                tape.parse_state,
            )["parsed"]
        if not path:
            horizon = min(
                session.max_steps - session.step,
                getattr(prompts, "planning_horizon", session.max_steps),
            )
            plans = [
                session.ask(
                    "planner",
                    tape.planner_messages(prompts, projection, horizon),
                    lambda raw: tape.parse_plan(
                        raw, session.environment.actions, projection
                    ),
                )["parsed"]
                for _ in range(session.config.plan_samples)
            ]
            graph = session.ask(
                "graph",
                tape.graph_messages(prompts, projection, plans),
                lambda raw: parse_candidate_graph(
                    raw, session.environment.actions, projection, plans
                ),
            )["parsed"]
            returns = (
                getattr(prompts, "objective_mode", "goal_path") == "predicted_return"
            )
            scores = session.ask(
                "scorer",
                tape.scorer_messages(prompts, graph),
                lambda raw: (
                    parse_return_scores(raw, graph)
                    if returns
                    else parse_goal_scores(raw, graph)
                ),
            )["parsed"]
            path = (
                select_return_path(graph, scores, horizon)
                if returns
                else tape.select_path(graph, scores, horizon)
            )
            if not path:
                session.episode.end = "no_plan"
                return
        edge = path.pop(0)
        turn = session.ask(
            "actor",
            tape.actor_messages(
                prompts,
                edge["action"],
                session.observations,
                session.episode.actions,
                session.actor_window,
            ),
            lambda raw: tape.parse_actor(raw, edge["action"]),
        )
        session.act(turn["parsed"]["action"], turn)
        if session.done():
            return
        projection = session.ask(
            "projector",
            tape.projector_messages(
                session.prompts,
                session.observations,
                session.episode.actions,
                session.config.history_frames,
            ),
            tape.parse_state,
        )["parsed"]
        expected = next(n["state"] for n in graph["nodes"] if n["id"] == edge["target"])
        matched = session.ask(
            "matcher",
            tape.matcher_messages(session.prompts, expected, projection),
            tape.parse_match,
        )["parsed"]
        if matched != "MATCH":
            path = []
