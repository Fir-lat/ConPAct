import math
from .protocol import _json_block, parse_graph, parse_scores


def parse_candidate_graph(raw, actions, start, plans):
    graph = parse_graph(raw, actions, start)
    obj = _json_block("graph", raw)
    if graph is None:
        return None
    mapping = obj.get("plan_nodes")
    if not isinstance(mapping, list) or len(mapping) != len(plans):
        return None
    nodes = {n["id"]: n["state"] for n in graph["nodes"]}
    first, transitions = {}, set()
    for plan, ids in zip(plans, mapping):
        if not isinstance(ids, list) or len(ids) != len(plan["states"]):
            return None
        if (
            any(not isinstance(i, str) or i not in nodes for i in ids)
            or ids[0] != graph["start"]
        ):
            return None
        for i, state in zip(ids, plan["states"]):
            first.setdefault(i, state)
        for source, action, target in zip(ids, plan["actions"], ids[1:]):
            transitions.add((source, action, target))
    if first != nodes:
        return None
    actual = [(e["source"], e["action"], e["target"]) for e in graph["edges"]]
    ids = [e["id"] for e in graph["edges"]]
    if (
        len(actual) != len(set(actual))
        or set(actual) != transitions
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
    ):
        return None
    graph["plan_nodes"] = mapping
    return graph


def parse_goal_scores(raw, graph):
    scores = _json_block("scores", raw)
    known = {n["id"] for n in graph["nodes"]}
    if not isinstance(scores, list) or len(scores) != len(known):
        return None
    if any(
        not isinstance(s, dict)
        or not isinstance(s.get("id"), str)
        or type(s.get("score")) is not int
        or s["score"] not in (-1, 0, 1)
        for s in scores
    ):
        return None
    if {s["id"] for s in scores} != known:
        return None
    return parse_scores(raw, graph)


def parse_return_scores(raw, graph):
    obj = _json_block("scores", raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("nodes"), list):
        return None
    known = {n["id"] for n in graph["nodes"]}
    values, terminal = {}, set()
    for node in obj["nodes"]:
        if not isinstance(node, dict):
            return None
        key, value = node.get("id"), node.get("cumulative_reward")
        if (
            not isinstance(key, str)
            or key not in known
            or key in values
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or type(node.get("terminal")) is not bool
        ):
            return None
        values[key] = float(value)
        if node["terminal"]:
            terminal.add(key)
    if set(values) != known or values[graph["start"]] != 0:
        return None
    rewards = {
        e["id"]: values[e["target"]] - values[e["source"]] for e in graph["edges"]
    }
    if not all(math.isfinite(r) for r in rewards.values()):
        return None
    return {
        "cumulative_reward": values,
        "terminal_nodes": sorted(terminal),
        "edge_rewards": rewards,
        "edge_costs": {e["id"]: 1 for e in graph["edges"]},
    }


def select_return_path(graph, scores, horizon):

    if horizon <= 0:
        return []
    terminal = set(scores["terminal_nodes"])
    outgoing = {}
    for i, edge in enumerate(graph["edges"]):
        outgoing.setdefault(edge["source"], []).append((i, edge))
    frontier = {graph["start"]: (0.0, ())}
    best = None
    for length in range(1, horizon + 1):
        following = {}
        for node, (value, indices) in frontier.items():
            if node in terminal:
                continue
            for i, edge in outgoing.get(node, ()):
                candidate = (value + scores["edge_rewards"][edge["id"]], indices + (i,))
                old = following.get(edge["target"])
                if old is None or (-candidate[0], candidate[1]) < (-old[0], old[1]):
                    following[edge["target"]] = candidate
                key = (-candidate[0], length, candidate[1])
                if best is None or key < best:
                    best = key
        frontier = following
        if not frontier:
            break
    return [graph["edges"][i] for i in best[2]] if best is not None else []
