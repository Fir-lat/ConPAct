from dataclasses import replace

from .plan_act import PROMPTS as BASE
from .ecot import _COORDS, shared_prefix


STATE_INSTRUCTION = (
    """Read the CURRENT decision frame independently. Do not describe a historical frame.
<state> is a JSON object with exactly these seven keys:
"""
    + _COORDS
)

PLANNER_SYSTEM = (
    shared_prefix(BASE.planner_system)
    + """
# State and remaining plan
First describe the current board in your own state block. Then write a short
remaining plan covering every unresolved box and its intended target. Select
only the next useful stage as the subgoal. The actor receives only that subgoal,
so it must identify the intended objective without referring to your other blocks.

# Output Format
Output exactly three non-empty blocks in this order:
1) <state>...</state>
2) <plan>...</plan>: one or two sentences covering the remaining whole-level solution.
3) <subgoal>...</subgoal>: the current short-term objective for the actor.
Do not output an action or status.
"""
    + STATE_INSTRUCTION
)

ACTOR_SYSTEM = (
    shared_prefix(BASE.actor_system)
    + """
# Independent state reading
Describe the current board yourself from the current screenshot. The planner's
state and remaining plan are not provided. Explain the next action in relation
to the given subgoal using only visible facts; an unhelpful or blocked action
must not be described as making progress.

# Output Format
Output exactly four non-empty blocks in this order:
1) <state>...</state>
2) <reason>...</reason>: a short explanation of the action's local consequence.
3) <action>...</action>: exactly one of up, down, left, right.
4) <status>...</status>: the subgoal's status after the chosen action:
   CONTINUE: the subgoal remains unfinished and feasible; keep pursuing it.
   ACHIEVED: the chosen action completes the subgoal.
   IMPOSSIBLE: the subgoal cannot be accomplished as stated.
   STUCK: you cannot make progress on the subgoal.
   CHANGED: the current board invalidates the subgoal's assumptions.
Any status other than CONTINUE returns control to the planner. Do not output
a new plan or subgoal yourself.
"""
    + STATE_INSTRUCTION
)

PROMPTS = replace(BASE, planner_system=PLANNER_SYSTEM, actor_system=ACTOR_SYSTEM)
