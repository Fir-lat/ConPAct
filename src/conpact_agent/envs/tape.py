from dataclasses import dataclass
from .sokoban.prompts.tape import TapePromptSet

VERSION = "tape_visual_api_v3"
ROLES = {
    "projector": "P",
    "planner": "P",
    "graph": "P",
    "scorer": "P",
    "matcher": "P",
    "actor": "A",
}
RETURN_SCHEMA = "state_cumulative_reward_difference_v1"
GOAL_SCHEMA = "node_goal_feasibility_v1"


@dataclass(frozen=True)
class MultiEnvTapePromptSet(TapePromptSet):
    adaptation_version: str = VERSION
    objective_mode: str = "goal_path"
    planning_horizon: int = 100
    score_schema: str = GOAL_SCHEMA
    projector_window: int = 1
    plan_samples: int = 2


def protocol(
    prompts,
    *,
    plan_samples=2,
    objective_mode=None,
    planning_horizon=None,
    projector_window=None,
):
    mode = objective_mode or getattr(prompts, "objective_mode", "goal_path")
    horizon = (
        planning_horizon
        if planning_horizon is not None
        else getattr(prompts, "planning_horizon", None)
    )
    window = (
        projector_window
        if projector_window is not None
        else getattr(prompts, "projector_window", 1)
    )
    if mode not in ("goal_path", "predicted_return"):
        raise ValueError("unknown TAPE objective mode")
    if type(plan_samples) is not int or plan_samples < 1:
        raise ValueError("plan_samples must be positive")
    if horizon is not None and (type(horizon) is not int or horizon < 1):
        raise ValueError("planning_horizon must be positive")
    if type(window) is not int or window < 1:
        raise ValueError("projector_window must be positive")
    if hasattr(prompts, "adaptation_version") and plan_samples != 2:
        raise ValueError("new-environment TAPE requires exactly two candidates")
    if hasattr(prompts, "adaptation_version") and mode != prompts.objective_mode:
        raise ValueError(
            "objective_mode must match the environment's explicit TAPE profile"
        )
    settings = dict(
        version=getattr(prompts, "adaptation_version", "sokoban_candidate_membership_v1"),
        plan_samples=plan_samples,
        objective_mode=mode,
        planning_horizon=horizon,
        score_schema=RETURN_SCHEMA if mode == "predicted_return" else GOAL_SCHEMA,
        projector_window=window,
        planner_input="independent_projection_text",
        role_models=dict(ROLES),
        execution="validated_actor_action_dispatch",
        solver="finite_horizon_dynamic_programming"
        if mode == "predicted_return"
        else "shortest_goal_BFS",
        cost_schema="one_env_step_per_edge",
        graph_schema="candidate_membership_v1",
    )
    import hashlib, json

    settings["sha256"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    return settings


_STATE = """Describe task-relevant objects, positions and relationships, accessible routes, hazards,
resources, inventory, health and task progress visible in the observation. Preserve all differences
that change action choice, including rewards or achievements already acquired and consumed items.
Use stable descriptions; distinguish uncertainty from absence. Do not invent hidden map or state.
Record achieved facts separately from future intentions. Copy public mission/direction when supplied."""


def build_prompts(*, rules, action_space, objective_mode, observation=""):
    continuous = objective_mode == "predicted_return"
    goal = (
        "Maximize predicted raw cumulative environment reward over the finite planning horizon. "
        "Partial routes remain useful without a terminal goal. Do not substitute achievement "
        "counts, survival bonuses or cross-episode benchmark scores for raw reward."
        if continuous
        else "Fulfill the public mission within the remaining action budget."
    )
    context = rules + "\n\n# Legal primitive actions\n" + action_space
    public = (
        "\n# Current public task/observation text\n" + observation
        if observation
        else ""
    )
    common = context + "\n" + _STATE + public
    score = (
        """Estimate each node's cumulative RAW environment reward relative to the supplied start,
whose cumulative_reward must be exactly 0. The value includes only rewards after this planning
cycle's start, and includes the final transition reward at a predicted terminal. Count a collected
one-time reward only once. Preserve state distinctions needed for reward history; do not add
rewards that were earned before the start. Edge reward is computed by the harness as target value
minus source value, NEVER as the target value alone. Negative and zero values are valid.
Mark terminal=true only for a predicted environment termination/truncation, not a partial endpoint.
Output <scores>{"nodes":[{"id":"n0","cumulative_reward":0,"terminal":false}, ...]}</scores>.
Include every graph node exactly once. Values must be finite numbers. These are predictions,
not observed returns or proof of success."""
        if continuous
        else """Score each node 1 if its description satisfies the mission, -1 if mission completion is
impossible, 0 otherwise (including uncertain). A partial endpoint is not automatically impossible.
Output <scores>[{"id":"n0","score":0,"reason":"..."}, ...]</scores>, exactly once per node.
The environment alone decides success; a predicted 1 must not terminate the real episode."""
    )
    return MultiEnvTapePromptSet(
        kind="tape",
        obs="image",
        objective_mode=objective_mode,
        planning_horizon=8 if continuous else 100,
        score_schema=RETURN_SCHEMA if continuous else GOAL_SCHEMA,
        projector_system="You are the PROJECTOR. Independently read the actual observation.\n"
        + common
        + "\n"
        "Do not choose actions or predict outcomes. An issued action is not evidence of success. "
        "Return <state>concise observed state</state>; if unreadable, <state_error>reason</state_error>.",
        planner_system="You are the PLANNER. Sample one candidate primitive-action route.\n"
        + common
        + "\n"
        + goal
        + "\n"
        "Observed start (copy only the text BETWEEN the delimiters, not their labels):\n<observed_state>\n{projected_state}\n</observed_state>\nMaximum actions in this candidate: {remaining_steps}\n"
        "Copy the observed start VERBATIM as states[0]. Predict a state after EVERY atomic action. "
        "Stop at a predicted real terminal; never add macro actions or repetitions. Include "
        "relevant resource/reward history in predicted states. Incomplete routes are allowed. "
        'Output <plan>{"states":[...],"actions":[...],"complete":false}</plan>. '
        "Use only listed actions. Exactly one more state than actions; complete means the final "
        "state fulfills the mission, not merely positive reward. No route: empty actions.",
        graph_system="You are the GRAPH builder. Merge only supplied candidate states and edges.\n"
        + common
        + "\n"
        "Start: {projected_state}\nCandidates: {candidate_plans}\n"
        "Merge only semantically equivalent situations, preserving inventory, earned rewards, "
        "hazards and mission progress. Keep uncertain equivalences separate. Each node's state "
        "must be the verbatim first occurrence description. Assign n0 to the observed start. "
        "Supply plan_nodes: one list of node IDs per candidate, mapping EVERY state in order. "
        "Retain exactly the distinct candidate transitions under this mapping, no invented "
        "edges, omitted transitions or shortcuts. Unique edge IDs e0, e1, ... . "
        'Output <graph>{"start":"n0","nodes":[{"id":"n0","state":"..."}],'
        '"edges":[{"id":"e0","source":"n0","target":"n1","action":"..."}],'
        '"plan_nodes":[["n0","n1"],...]}</graph>.',
        scorer_system="You are the SCORER. Evaluate only the predicted graph; do not execute or query an environment.\n"
        + common
        + "\nGraph: {plan_graph}\n"
        + score,
        matcher_system="You are the MATCHER. Compare predicted and independently observed descriptions.\n"
        + common
        + "\n"
        "Expected: {expected_state}\nObserved: {observed_state}\n"
        "MATCH requires agreement on all action-relevant facts. Concrete conflict: MISMATCH. "
        "Missing or uncertain essential evidence: UNCERTAIN. Never fill observed gaps from "
        'the prediction. Output <match>{"verdict":"MATCH|MISMATCH|UNCERTAIN","reason":"..."}</match>.',
        actor_system="You are the ACTOR. Emit the solver's selected primitive action exactly.\n"
        + context
        + public
        + "\n"
        "Selected action: {selected_action}\nOutput exactly <action>{selected_action}</action>. "
        "Do not substitute another action or add a thought. This is an API dispatch adaptation.",
        instruction="Emit the selected action exactly. " + observation,
        instruction_with_history="Previous observations/actions, then current observation. Emit the selected action. "
        + observation,
        projector_instruction="Read the current observation independently. "
        + observation,
        projector_instruction_with_history="Previous observations/actions, then current observation. Read only observed facts. "
        + observation,
        planner_instruction="Generate one candidate route from the projection. "
        + observation,
        graph_instruction="Merge the candidates and supply complete plan_nodes provenance. "
        + observation,
        scorer_instruction="Score the predicted graph according to the specified schema. "
        + observation,
        matcher_instruction="Compare the two descriptions. " + observation,
        candidate_plan_item="Candidate {index}:\n{plan}\n",
        current_frame_block="Current frame:\n",
    )
