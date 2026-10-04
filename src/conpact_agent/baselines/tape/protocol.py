from __future__ import annotations
import json
import re
import logging

logger = logging.getLogger(__name__)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
VERDICTS = ("MATCH", "MISMATCH", "UNCERTAIN")
from collections import deque
from conpact_agent.envs.base import PromptSet, fill
from ...methods.messages import _text, _board

DEFAULT_PLAN_SAMPLES = 4


def _block(tag: str, raw: str) -> str | None:

    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", raw or "", re.DOTALL)
    return matches[-1].strip() if matches else None


def _json_block(tag: str, raw: str):
    body = _block(tag, raw)
    if body is None:
        body = raw or ""
    try:
        return json.loads(_FENCE.sub("", body).strip())
    except ValueError:
        return None


def parse_state(raw: str) -> str | None:

    return _block("state", raw) or None


def parse_actor(raw: str, selected_action: str) -> dict[str, str]:
    match = re.fullmatch(r"\s*<action>([^<>]+)</action>\s*", raw or "")
    if match is None:
        raise ValueError("TAPE actor must return exactly one action block")
    action = match.group(1).strip()
    if action != selected_action:
        raise ValueError("TAPE actor action differs from the selected action")
    return {"action": action}


def parse_plan(
    raw: str, valid_actions: tuple[str, ...], expected_start: str | None = None
) -> dict | None:

    plan = _json_block("plan", raw)
    if not isinstance(plan, dict):
        return None
    states, actions = plan.get("states"), plan.get("actions")
    if not isinstance(states, list) or not isinstance(actions, list):
        return None
    if len(states) != len(actions) + 1 or not states:
        return None
    if not all(isinstance(s, str) and s.strip() for s in states):
        return None
    if expected_start is not None and states[0].strip() != expected_start.strip():
        return None
    if not isinstance(plan.get("complete"), bool):
        return None
    actions = [a.lower().strip() for a in actions if isinstance(a, str)]
    if len(actions) != len(plan["actions"]):
        return None
    if valid_actions and not all(a in valid_actions for a in actions):
        return None
    return {
        "states": [s.strip() for s in states],
        "actions": actions,
        "complete": bool(plan.get("complete")),
    }


def parse_graph(
    raw: str, valid_actions: tuple[str, ...], expected_start: str | None = None
) -> dict | None:

    graph = _json_block("graph", raw)
    if not isinstance(graph, dict):
        return None
    nodes, edges = graph.get("nodes"), graph.get("edges")
    start = graph.get("start")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes:
        return None
    clean_nodes = []
    seen = set()
    for node in nodes:
        if not isinstance(node, dict):
            return None
        node_id, state = node.get("id"), node.get("state")
        if not isinstance(node_id, str) or not isinstance(state, str):
            return None
        if not node_id.strip() or not state.strip():
            return None
        if node_id in seen:
            return None
        seen.add(node_id)
        clean_nodes.append({"id": node_id, "state": state.strip()})
    if not isinstance(start, str) or start not in seen:
        return None
    if start != "n0":
        return None
    if (
        expected_start is not None
        and next(n["state"] for n in clean_nodes if n["id"] == start)
        != expected_start.strip()
    ):
        return None
    clean_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            return None
        source, target = edge.get("source"), edge.get("target")
        action = edge.get("action")
        if not isinstance(source, str) or not isinstance(target, str):
            return None
        if source not in seen or target not in seen:
            return None
        if not isinstance(action, str):
            return None
        action = action.lower().strip()
        if valid_actions and action not in valid_actions:
            return None
        clean_edges.append(
            {"id": edge.get("id"), "source": source, "target": target, "action": action}
        )
    return {"start": start, "nodes": clean_nodes, "edges": clean_edges}


def parse_scores(raw: str, graph: dict) -> dict[str, int] | None:

    scores = _json_block("scores", raw)
    if not isinstance(scores, list):
        return None
    known = {node["id"] for node in graph["nodes"]}
    out = {node_id: 0 for node_id in known}
    for entry in scores:
        if not isinstance(entry, dict):
            return None
        node_id, score = entry.get("id"), entry.get("score")
        if score not in (-1, 0, 1):
            return None
        if node_id not in known:
            logger.warning("scorer named node %r, which is not in the graph", node_id)
            continue
        out[node_id] = score
    return out


def parse_match(raw: str) -> str:

    match = _json_block("match", raw)
    if not isinstance(match, dict):
        return "UNCERTAIN"
    verdict = match.get("verdict")
    if not isinstance(verdict, str):
        return "UNCERTAIN"
    verdict = verdict.strip().upper()
    return verdict if verdict in VERDICTS else "UNCERTAIN"


def select_path(graph: dict, scores: dict[str, int], budget: int) -> list[dict]:

    start = graph["start"]
    if budget <= 0:
        return []
    outgoing: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        outgoing.setdefault(edge["source"], []).append(edge)
    forbidden = {n for n, s in scores.items() if s == -1 and n != start}
    goals = {n for n, s in scores.items() if s == 1 and n != start}

    queue = deque([(start, [])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if len(path) >= budget:
            continue
        for edge in outgoing.get(node, ()):
            target = edge["target"]
            if target in forbidden or target in seen:
                continue
            extended = path + [edge]
            if target in goals:
                return extended
            seen.add(target)
            queue.append((target, extended))

    return []


def _history(
    prompts: PromptSet,
    instruction: str,
    history_instruction: str,
    frames: list[str],
    actions: list[str],
    window: int,
) -> list[dict]:

    done = len(actions)
    kept = max(0, min(done, window - 1))
    if kept and not history_instruction:
        raise RuntimeError(
            "a window above 1 was asked for but this prompt set has no "
            "with-history instruction, so the history would arrive unannounced"
        )
    content = [_text(history_instruction if kept else instruction)]
    for j in range(done - kept, done):
        content.append(_board(prompts, frames[j]))
        content.append(_text(f"\n<action>{actions[j]}</action>\n"))
    content.append(_board(prompts, frames[done]))
    return content


def projector_messages(
    prompts: PromptSet, frames: list[str], actions: list[str], window: int
) -> list:

    return [
        {"role": "system", "content": [_text(prompts.projector_system)]},
        {
            "role": "user",
            "content": _history(
                prompts,
                prompts.projector_instruction,
                prompts.projector_instruction_with_history,
                frames,
                actions,
                window,
            ),
        },
    ]


def planner_messages(
    prompts: PromptSet, projected_state: str, remaining_steps: int
) -> list:

    system = fill(
        prompts.planner_system,
        projected_state=projected_state,
        remaining_steps=remaining_steps,
    )
    return [
        {"role": "system", "content": [_text(system)]},
        {"role": "user", "content": [_text(prompts.planner_instruction)]},
    ]


def graph_messages(prompts: PromptSet, projected_state: str, plans: list[dict]) -> list:

    rendered = "\n".join(
        fill(
            prompts.candidate_plan_item,
            index=i + 1,
            plan=json.dumps(plan, ensure_ascii=False, indent=1),
        )
        for i, plan in enumerate(plans)
    )
    system = fill(
        prompts.graph_system, projected_state=projected_state, candidate_plans=rendered
    )
    return [
        {"role": "system", "content": [_text(system)]},
        {"role": "user", "content": [_text(prompts.graph_instruction)]},
    ]


def scorer_messages(prompts: PromptSet, graph: dict) -> list:
    system = fill(
        prompts.scorer_system,
        plan_graph=json.dumps(graph, ensure_ascii=False, indent=1),
    )
    return [
        {"role": "system", "content": [_text(system)]},
        {"role": "user", "content": [_text(prompts.scorer_instruction)]},
    ]


def matcher_messages(prompts: PromptSet, expected: str, observed: str) -> list:
    system = fill(
        prompts.matcher_system, expected_state=expected, observed_state=observed
    )
    return [
        {"role": "system", "content": [_text(system)]},
        {"role": "user", "content": [_text(prompts.matcher_instruction)]},
    ]


def actor_messages(
    prompts: PromptSet,
    selected_action: str,
    frames: list[str],
    actions: list[str],
    window: int,
) -> list:

    system = fill(prompts.actor_system, selected_action=selected_action)
    return [
        {"role": "system", "content": [_text(system)]},
        {
            "role": "user",
            "content": _history(
                prompts,
                prompts.instruction,
                prompts.instruction_with_history,
                frames,
                actions,
                window,
            ),
        },
    ]
