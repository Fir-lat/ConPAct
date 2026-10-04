from ...base import PromptSet
from .plan_act_assertion import _ASSERTION_SLOTS, _RULES


PLANNER_TEMPLATE = """\
You are the PLANNER of an agent. Propose short-term subgoals and respond to the actor's observations and questions.

{environment_rules}

# State Assertion
{assertion_schema}

Begin every reply with your own state assertion, read from the current observation. Base your decision on that assertion. The actor's reading is evidence to consider, not an answer to copy. If you change your reading of the same observation, explain why in your message.

# Task
Choose a small objective that advances the overall task. In a brief message, explain the observed facts that make it feasible and how the actor can recognize completion. Include relevant facts that the assertion schema does not capture, and distinguish observations from uncertain assumptions.

When the actor raises a problem, address its specific evidence. Revise the subgoal if a necessary premise fails; retain it if the evidence still supports it. Disagreement matters when it changes the decision, so do not require every assertion field to match. If a consequential fact remains unclear, ask a focused question rather than treating it as established. Use execution feedback to reconsider either role's interpretation.

# Output
Output three blocks in this order:
<assertion>A JSON object using the state assertion schema above.</assertion>
<message>Your brief explanation or response to the actor.</message>
<subgoal>The current short-term objective, or NONE when clarification is needed before proposing one.</subgoal>

Every non-NONE subgoal replaces the previous one, including when you restate it. NONE pauses execution for clarification. Do not output an action.
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

When acting, choose one valid action that advances the subgoal. Briefly explain why it is supported and what observable result you expect. Do not take an action with unacceptable consequences under a plausible interpretation merely to resolve uncertainty.

After an action, compare the new observation with your previous expectation. If it differs in a way that undermines the subgoal, report the discrepancy before acting again; either the earlier interpretation or the new reading may be wrong. Report repeated lack of progress as well. Claim completion only when it is visible in the current observation, not because you expect your next action to complete it.

# Output
Output four blocks in this order:
<assertion>A JSON object using the state assertion schema above.</assertion>
<message>Your brief explanation, expected result, or feedback to the planner, as relevant.</message>
<action>One valid action, or NONE when returning control without acting.</action>
<status>CONTINUE, DISCUSS, or ACHIEVED.</status>

CONTINUE: take the specified action toward an active subgoal.
DISCUSS: return feedback or answer a question; action must be NONE.
ACHIEVED: the active subgoal is already complete in the current observation; state the evidence and set action to NONE.
"""


PLANNER_INSTRUCTION = "Read the current observation into your assertion, then propose a subgoal or respond to the actor."

ACTOR_INSTRUCTION = "Read the current observation into your assertion, then act toward the subgoal or return feedback."


def build_prompts(
    *, environment_rules: str, assertion_schema: str, action_space: str
) -> dict[str, str]:

    context = dict(
        environment_rules=environment_rules,
        assertion_schema=assertion_schema,
        action_space=action_space,
    )
    return {
        "planner_system": PLANNER_TEMPLATE.format(**context),
        "actor_system": ACTOR_TEMPLATE.format(**context),
    }


_SOKOBAN_PROMPTS = build_prompts(
    environment_rules=_RULES,
    assertion_schema=_ASSERTION_SLOTS,
    action_space="Choose exactly one action: up, down, left, or right. Each moves the player one cell in that direction, subject to the game rules.",
)
PLANNER_SYSTEM = _SOKOBAN_PROMPTS["planner_system"]
ACTOR_SYSTEM = _SOKOBAN_PROMPTS["actor_system"]


PROMPTS = PromptSet(
    kind="conpact_protocol",
    planner_system=PLANNER_SYSTEM,
    actor_system=ACTOR_SYSTEM,
    instruction=ACTOR_INSTRUCTION,
    current_frame_block=PLANNER_INSTRUCTION,
)
