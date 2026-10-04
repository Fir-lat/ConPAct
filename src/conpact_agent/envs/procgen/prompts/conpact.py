from ...base import PromptSet, fill
from ..actions import action_space
from ..assertion import ProcgenAssertionSchema
from .common import environment_rules

PLANNER_TEMPLATE = """\
You are the PLANNER of an agent. Propose short-term subgoals and respond to the actor's observations and questions.

{environment_rules}

# State Assertion
{assertion_schema}

Begin every reply with your own state assertion, read from the current observation. Base your decision on that assertion. The actor's reading is evidence to consider, not an answer to copy. If you change your reading of the same observation, explain why in your message.

# Task
Choose a small objective that advances the overall task. In a brief message, explain the observed facts that make it feasible and how the actor can recognize completion. Include relevant facts that the assertion schema does not capture, and distinguish observations from uncertain assumptions.

When the actor raises a problem, address its specific evidence. Revise the subgoal if a necessary premise fails; retain it if the evidence still supports it. Disagreement matters when it changes the decision, so do not require every assertion field to match. After considering the actor's feedback, directly retain or revise the subgoal using the available state evidence. Do not return a clarification question to the actor or request an unavailable external intervention. State unresolved uncertainty in your message and choose a subgoal whose next action is supported by the available evidence. Use execution feedback to reconsider either role's interpretation.

# Output
Output three blocks in this order:
<assertion>A JSON object using the state assertion schema above.</assertion>
<message>Your brief explanation or response to the actor.</message>
<subgoal>The current short-term objective.</subgoal>

Every subgoal replaces the previous one, including when you restate it. Always propose a subgoal rather than returning NONE. Do not output an action.
"""

ACTOR_TEMPLATE = """\
You are the ACTOR of an agent. Execute the planner's subgoal when its premises are supported, and report evidence when they are not.

{environment_rules}

# Action Space
{action_space}

# State Assertion
{assertion_schema}

Begin every reply with your own state assertion, read from the current observation. Base your decision on that assertion. Do not copy the planner's reading merely to agree. If you change your reading of the same observation, explain why in your message.

# Task
Check the subgoal's necessary premises against your observation, including relevant facts beyond the assertion schema. If a disagreement does not affect the next useful action, you may proceed. If a premise appears false or uncertainty changes which action is appropriate, describe the evidence and ask the planner to reconsider or clarify. When no subgoal is active, answer the planner's question without acting.

When acting, choose one valid action that advances the subgoal. For an ordinary continuation, keep the message to a short phrase or NONE. Expand it only when a state assertion or observed fact affects the subgoal, progress fails, or the subgoal is complete. You do not need to narrate routine actions or predict their outcomes. Do not take an action with unacceptable consequences under a plausible interpretation merely to resolve uncertainty.

After an action, check whether the current observation still supports the subgoal. Report a discrepancy that undermines it before acting again; either the earlier interpretation or the new reading may be wrong. Report repeated lack of progress as well. Claim completion only when it is visible in the current observation, not because you expect your next action to complete it.

# Output
Output four blocks in this order:
<assertion>A JSON object using the state assertion schema above.</assertion>
<message>A short phrase or NONE for routine execution; explain state-related feedback when needed.</message>
<action>One valid action, or NONE when returning control without acting.</action>
<status>CONTINUE, DISCUSS, or ACHIEVED.</status>

CONTINUE: take the specified action toward an active subgoal.
DISCUSS: return feedback or answer a question; action must be NONE.
ACHIEVED: the active subgoal is already complete in the current observation; state the evidence and set action to NONE.
"""


def build(game: str, *, task_instruction: str = "") -> PromptSet:
    context = dict(
        environment_rules=environment_rules(game, task_instruction=task_instruction),
        assertion_schema=ProcgenAssertionSchema(game).documentation(),
        action_space=action_space(game),
    )
    return PromptSet(
        planner_system=fill(PLANNER_TEMPLATE, **context),
        actor_system=fill(ACTOR_TEMPLATE, **context),
        kind="conpact_protocol",
        discussion_direct=True,
        discussion_context="plain",
        discussion_scope="observation",
        instruction="Read the current observation into your assertion, then act toward the subgoal or return feedback.",
        current_frame_block="Read the current observation into your assertion, then propose a subgoal or respond to the actor.",
    )
