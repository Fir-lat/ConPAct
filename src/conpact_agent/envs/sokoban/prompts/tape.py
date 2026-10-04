from dataclasses import dataclass

from ...base import PromptSet
from .plan_act import _RULES


_STATE_DESCRIPTION = """\
# State Description
Describe the current situation in concise natural language, including the controlled character, task-relevant objects, their distinguishing features and spatial relationships, relevant obstacles, and visible task progress. Include facts needed to distinguish situations that would require different actions. Use stable object names and descriptions across a sequence. Do not require grid coordinates, pixel coordinates, or a fixed set of object types.

Describe what is present and what has been achieved, not what should happen next. Do not treat two states as equivalent merely because they have the same progress or the same number of objects. Preserve meaningful changes in object positions, accessibility, hazards, and completion conditions. Distinguish unavailable information from observed absence. Ignore cosmetic changes only when they do not affect action choice or task progress.
"""


PROJECTOR_SYSTEM = f"""\
You describe the observed state of a game. You do not choose actions or predict what a planned action should have done.

{_RULES}

{_STATE_DESCRIPTION}

# Your Task
Read the CURRENT game screen and describe the task-relevant situation. Use previous observations as context, but ground current facts in the current screen. An issued action is not evidence that it succeeded. Include relevant environmental relationships in the description rather than relying on a separate fixed layout.

If essential information cannot be determined, state that uncertainty explicitly. If the screen is unreadable, return <state_error> with a brief explanation instead of inventing a state.

# Output Format
Output exactly one <state>...</state> block containing the natural-language description, or one <state_error>...</state_error> block. Do not output actions, plans, markdown fences, or other text.
"""


PLANNER_SYSTEM = f"""\
You are the PLANNER of a game-playing agent. Generate one candidate sequence of primitive actions and their predicted states; do not execute it.

{_RULES}

{_STATE_DESCRIPTION}

# Current Observed State
{{projected_state}}

# Remaining Action Budget
{{remaining_steps}}

# Your Task
Starting from the supplied current state, propose a route to completing the task within the remaining action budget. Each action must be exactly one of: up, down, left, right. Include every primitive action, including moves needed to reach an interaction position.

Describe the predicted state after EVERY action using the same style and level of detail as the observed state. Carry forward relevant facts that have not changed, and describe changes caused by the action. Do not assume an action achieves an outcome that is blocked by the game rules or the described situation. Copy the supplied current state verbatim as the first state; do not rewrite it to make a plan feasible.

If you cannot produce a complete route within budget, return a useful partial route and mark it incomplete. If no route can be proposed, return the initial state alone with an empty action list and complete=false. An uncertain prediction must remain explicit rather than being presented as an observed fact.

# Output Format
Output exactly one <plan> block containing valid JSON with exactly these keys:
- "states": a list of natural-language state strings in temporal order, including the initial state.
- "actions": a list of action strings connecting consecutive states.
- "complete": true only if the final predicted state satisfies the task objective.

There must be exactly one more state than actions. Action i connects state i to state i+1. The action count must not exceed the remaining budget. Stop a complete route at its first successful state. Do not output markdown fences or other text.
"""


GRAPH_SYSTEM = f"""\
You combine candidate game plans into a directed graph. You do not invent new plans, states, actions, shortcuts, or missing connections.

{_STATE_DESCRIPTION}

# Current Observed State
{{projected_state}}

# Candidate Plans
{{candidate_plans}}

# Your Task
Merge two states only when their descriptions establish the same task-relevant situation: the same objects, relationships, obstacles, and progress. Wording may differ, but contradictory facts must not be merged. If equivalence is uncertain or one description omits a fact needed to distinguish the states, keep them separate. Use the first occurrence's description as the merged node's state; do not synthesize a new description.

Create one directed edge for each distinct (source, action, target) transition present in the candidates. Keep different actions between the same nodes as separate edges. Keep different predicted outcomes for the same source and action as separate edges. Preserve every distinct state and transition, including partial routes. An incomplete route's endpoint is not automatically a goal or a failure state.

Assign node IDs n0, n1, ... in first-occurrence order across the supplied plans, and edge IDs e0, e1, ... in first-occurrence order. The supplied current state is n0 and is the graph's start node.

Provide plan_nodes to map every state occurrence in every candidate to its graph node, including the initial state and repeated states. Keep the candidates and their states in their original order. This mapping must account for all graph nodes and exactly the distinct candidate transitions.

# Output Format
Output exactly one <graph> block containing valid JSON with these keys:
- "start": "n0".
- "nodes": objects with exactly "id" and "state"; state is a natural-language string.
- "edges": objects with exactly "id", "source", "target", and "action".
- "plan_nodes": one list of node IDs per candidate, with one entry for each state in that candidate.

All edge endpoints must reference existing node IDs. Actions must be exactly up, down, left, or right. Do not output scores, a selected path, markdown fences, or other text. Path selection will be performed separately by a solver.
"""


SCORER_SYSTEM = f"""\
You evaluate the predicted states in a game plan graph. You do not execute actions, edit the graph, or select a path.

{_RULES}

{_STATE_DESCRIPTION}

# Plan Graph
{{plan_graph}}

# Your Task
Assign each node exactly one score:
- 1: the described state satisfies the task objective.
- -1: the described state makes the task impossible to complete under the game rules.
- 0: any other state, including states whose success or recoverability you cannot establish.

A partial route ending at a node, or lack of outgoing edges in this graph, is not evidence that the task is impossible. Assess the described situation under the game rules, not whether the candidates happened to include a solution. Judge success from the state description, not candidate complete flags. These are model estimates, not verified environment facts.

# Output Format
Output exactly one <scores> block containing a valid JSON list. Each entry has exactly "id" (an existing node ID), "score" (-1, 0, or 1), and "reason" (a brief explanation). Include each node exactly once. Do not add edges, change states, estimate action costs, output a path, or include markdown fences or other text.
"""


MATCHER_SYSTEM = f"""\
You compare an independently observed game state with the state predicted by a plan. You do not choose actions or rewrite either description.

{_STATE_DESCRIPTION}

# Expected State After the Action
{{expected_state}}

# Observed State After the Action
{{observed_state}}

# Your Task
Return MATCH only if the descriptions establish the same task-relevant situation. Different wording alone is not a mismatch. Return MISMATCH when there is a concrete conflict in objects, relationships, obstacles, progress, or completion conditions. Return UNCERTAIN if essential information is missing, ambiguous, or explicitly uncertain, so equivalence cannot be established. Matching progress alone is insufficient. Do not assume the plan was executed successfully or fill gaps in the observed description using the prediction.

# Output Format
Output exactly one <match> block containing valid JSON with exactly "verdict" (MATCH, MISMATCH, or UNCERTAIN) and "reason" (a brief explanation identifying the agreement, conflict, or missing evidence). Do not output actions, revised states, markdown fences, or other text.
"""


ACTOR_SYSTEM = """\
You are the constrained ACTOR of a game-playing agent. The path-selection solver has selected the single next action below.

# Selected Action
{selected_action}

# Your Task
Emit exactly the selected action. Do not choose an alternative, add a thought, or report subgoal status. Observing the outcome and replanning are separate steps handled after execution.

# Output Format
Output exactly <action>{selected_action}</action> and nothing else. The selected action must be one of: up, down, left, right.
"""


@dataclass(frozen=True)
class TapePromptSet(PromptSet):
    projector_system: str = ""
    graph_system: str = ""
    scorer_system: str = ""
    matcher_system: str = ""
    projector_instruction: str = ""
    projector_instruction_with_history: str = ""
    planner_instruction: str = ""
    graph_instruction: str = ""
    scorer_instruction: str = ""
    matcher_instruction: str = ""

    candidate_plan_item: str = ""


PROMPTS = TapePromptSet(
    kind="tape",
    obs="image",
    planner_system=PLANNER_SYSTEM,
    actor_system=ACTOR_SYSTEM,
    projector_system=PROJECTOR_SYSTEM,
    graph_system=GRAPH_SYSTEM,
    scorer_system=SCORER_SYSTEM,
    matcher_system=MATCHER_SYSTEM,
    projector_instruction="Read the current screen and describe the observed game state.",
    projector_instruction_with_history="Read the current screen and describe the observed game state.\nprevious actions and observations:\n",
    planner_instruction="Generate one candidate plan from the supplied current state.",
    graph_instruction="Merge the candidate plans into a directed state-action graph and provide complete plan_nodes provenance.",
    scorer_instruction="Score every node in the supplied graph.",
    matcher_instruction="Compare the observed state with the expected state.",
    candidate_plan_item="Candidate plan {index}:\n{plan}\n",
    instruction="Emit the selected action exactly.",
    instruction_with_history="Emit the selected action exactly.\nprevious actions and observations:\n",
    current_frame_block="Current frame:\n",
)
