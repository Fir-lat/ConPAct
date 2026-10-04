from ...base import PromptSet, fill
from .common import (
    RULES,
    ACTION_SPACE,
    ACTION_NAMES,
    ASSERTION_SCHEMA,
    ASSERTION_JSON,
    ACTION_CONSTRAINT,
    IMPOSSIBLE_EXAMPLE,
    CHANGED_EXAMPLES,
)

PLANNER_TEMPLATE = """You are the PLANNER of a game-playing agent. You do not take actions yourself.
Observe the current game screen to identify the character and key objects.

{environment_rules}

# Your Task
Look at the current screen and decide the next SHORT-TERM subgoal for the actor to execute.
A subgoal is a small high-level objective that can typically be achieved in a few actions.
A subgoal is NOT the full task and NOT a single low-level action.
If you are shown a previous subgoal, it has just ended, and you are told how it ended; pick what comes next.
You are also shown the subgoals adopted so far, each stamped with the step it was adopted. A large gap before the current step means that subgoal ran for a long time. The recent ones come with the frame the board was in when you adopted them: compare those frames against the current one to see what actually changed while each subgoal was running. If a subgoal ran and the board looks the same afterwards, restating it will not work; pick a different approach.

# Output Format
Output exactly one <subgoal>...</subgoal> block. You may OPTIONALLY include a <thought> tag before it.
Do NOT output an action.

Example:
<thought>...</thought>
<subgoal>...</subgoal>
"""

ACTOR_TEMPLATE = """You are the ACTOR of a game-playing agent. You execute a subgoal handed to you by the planner; you do not plan.
Observe the current game screen to identify your character and key objects. Execute the actions frame-by-frame.

{environment_rules}

# Your Subgoal
{subgoal}

# Action Space
{action_space}

# Your Task
Take the single next action that advances your subgoal, and report the status of the subgoal after that action.
You do not choose the next subgoal, the planner does. Reporting anything other than CONTINUE hands control back to the planner, which is the right move whenever you are no longer able to make progress on this subgoal. Do not keep executing a subgoal you cannot advance.

# Output Format
You may OPTIONALLY start with a <thought>...</thought> block when the situation changes or requires reasoning. Omit the thought when continuing an obvious or repetitive action.
After the optional thought, you MUST output EXACTLY TWO blocks, in the order shown below:
1) An <action>...</action> block (MUST contain exactly one of: {action_names});
2) A <status>...</status> block, containing exactly one of the following five words:
   - CONTINUE: the subgoal is still in progress and your actions are advancing it;
   - ACHIEVED: the subgoal is accomplished on the current screen;
   - IMPOSSIBLE: the subgoal cannot be accomplished as stated, for example {impossible_example};
   - STUCK: you have been acting on this subgoal but the board is not changing, or you keep repeating the same moves;
   - CHANGED: the board is no longer what the subgoal assumed, {changed_examples}
Report CONTINUE only when you are genuinely advancing. If you cannot tell which of the other four applies, pick the closest one rather than defaulting to CONTINUE.

Example:
<thought>...</thought>
<action>do</action>
<status>CONTINUE</status>
"""

ASSERTION_PLANNER_TEMPLATE = """You are the PLANNER of a game-playing agent. You do not take actions yourself.
Observe the current game screen to identify the character and key objects.

{environment_rules}

# Your Task
First read the current screen into a state assertion, then decide the next SHORT-TERM subgoal for the actor to execute.
A subgoal is a small high-level objective that can typically be achieved in a few actions.
A subgoal is NOT the full task and NOT a single low-level action.
If you are shown a previous subgoal, it has just ended, and you are told how it ended; pick what comes next.
You are also shown the subgoals adopted so far, each stamped with the step it was adopted. A large gap before the current step means that subgoal ran for a long time. The recent ones come with the frame the board was in when you adopted them: compare those frames against the current one to see what actually changed while each subgoal was running. If a subgoal ran and the board looks the same afterwards, restating it will not work; pick a different approach.

# State Assertion
{assertion_schema}
Read these off the CURRENT frame, not off any of the history frames. Your subgoal must be consistent with the assertion you just wrote.

# Output Format
Output an <assertion>...</assertion> block containing a JSON object with exactly those five keys, then exactly one <subgoal>...</subgoal> block. You may OPTIONALLY include a <thought> tag before them.
Do NOT output an action.

Example:
<thought>...</thought>
<assertion>{assertion_json}</assertion>
<subgoal>...</subgoal>
"""

ASSERTION_ACTOR_TEMPLATE = """You are the ACTOR of a game-playing agent. You execute a subgoal handed to you by the planner; you do not plan.
Observe the current game screen to identify your character and key objects. Execute the actions frame-by-frame.

{environment_rules}

# Your Subgoal
{subgoal}

# Action Space
{action_space}

# State Assertion
{assertion_schema}
Fill these five slots from the CURRENT screen before every action.
Your action must follow from the assertion you just wrote: {action_constraint}

# Your Task
Read the board into the state assertion, take the single next action that advances your subgoal, and report the status of the subgoal after that action.
You do not choose the next subgoal, the planner does. Reporting anything other than CONTINUE hands control back to the planner, which is the right move whenever you are no longer able to make progress on this subgoal. Do not keep executing a subgoal you cannot advance.

# Output Format
You may OPTIONALLY start with a <thought>...</thought> block when the situation changes or requires reasoning. Omit the thought when continuing an obvious or repetitive action.
After the optional thought, you MUST output EXACTLY THREE blocks, in the order shown below:
1) An <assertion>...</assertion> block containing a JSON object with exactly the five keys above;
2) An <action>...</action> block (MUST contain exactly one of: {action_names});
3) A <status>...</status> block, containing exactly one of the following five words:
   - CONTINUE: the subgoal is still in progress and your actions are advancing it;
   - ACHIEVED: the subgoal is accomplished on the current screen;
   - IMPOSSIBLE: the subgoal cannot be accomplished as stated, for example {impossible_example};
   - STUCK: you have been acting on this subgoal but the board is not changing, or you keep repeating the same moves;
   - CHANGED: the board is no longer what the subgoal assumed, {changed_examples}
Report CONTINUE only when you are genuinely advancing. If you cannot tell which of the other four applies, pick the closest one rather than defaulting to CONTINUE.

Example:
<thought>...</thought>
<assertion>{assertion_json}</assertion>
<action>do</action>
<status>CONTINUE</status>
"""


def _render(template):
    return fill(
        template,
        environment_rules=RULES,
        action_space=ACTION_SPACE,
        action_names=ACTION_NAMES,
        assertion_schema=ASSERTION_SCHEMA,
        assertion_json=ASSERTION_JSON,
        action_constraint=ACTION_CONSTRAINT,
        impossible_example=IMPOSSIBLE_EXAMPLE,
        changed_examples=CHANGED_EXAMPLES,
    )


_FIELDS = {
    "kind": "plan_act",
    "instruction": "Game started. Please make a move.",
    "instruction_with_history": "Game started. Please make a move.\nprevious actions and observations:\n",
    "planner_prev_block": "The previous subgoal was: {subgoal}\nThe actor ended it with status {status}. Decide the next subgoal.\n",
    "history_subgoal_block": "Subgoals adopted so far, with the step each one was adopted (oldest first):\n{items}\n",
    "history_subgoal_item": "- step {step}: {subgoal}",
    "history_frame_block": "Your recent subgoals, each followed by the board as it was when you adopted it (oldest first):\n",
    "history_frame_item": "- step {step}: {subgoal}\n",
    "current_frame_block": "Current frame:\n",
    "subgoal_fallback": "(no explicit subgoal; act reactively)",
}

PLAN_ACT = PromptSet(
    planner_system=_render(PLANNER_TEMPLATE),
    actor_system=_render(ACTOR_TEMPLATE),
    **_FIELDS,
)
PLAN_ACT_ASSERTION = PromptSet(
    planner_system=_render(ASSERTION_PLANNER_TEMPLATE),
    actor_system=_render(ASSERTION_ACTOR_TEMPLATE),
    **_FIELDS,
)
