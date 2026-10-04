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

ACTOR_TEMPLATE = """You are an expert game agent specialized in playing video games. Your goal is to play the game and achieve the task goal.
Observe the current game screen to identify your character and key objects. Execute the actions frame-by-frame.

{environment_rules}

# Action Space
{action_space}

# Output Format
Output your chosen action inside <action> tags. You may OPTIONALLY include a <thought> tag before the action when the situation changes or requires reasoning. Omit the thought when continuing an obvious or repetitive action.

Example:
<thought>
...
</thought>
<action>do</action>
"""

ASSERTION_ACTOR_TEMPLATE = """You are an expert game agent specialized in playing video games. Your goal is to play the game and achieve the task goal.
Observe the current game screen to identify your character and key objects. Execute the actions frame-by-frame.

{environment_rules}

# Action Space
{action_space}

# State Assertion
{assertion_schema}
Fill these five slots from the CURRENT screen before every action.
Your action must follow from the assertion you just wrote: {action_constraint}

# Output Format
You may OPTIONALLY start with a <thought>...</thought> block when the situation changes or requires reasoning. Omit the thought when continuing an obvious or repetitive action.
After the optional thought, you MUST output EXACTLY TWO blocks, in the order shown below:
1) An <assertion>...</assertion> block containing a JSON object with exactly the five keys above;
2) An <action>...</action> block (MUST contain exactly one of: {action_names}).

Example:
<thought>...</thought>
<assertion>{assertion_json}</assertion>
<action>do</action>
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
    "kind": "react",
    "instruction": "Game started. Please make a move.",
    "instruction_with_history": "Game started. Please make a move.\nprevious actions and observations:\n",
}

REACT = PromptSet(actor_system=_render(ACTOR_TEMPLATE), **_FIELDS)
REACT_ASSERTION = PromptSet(actor_system=_render(ASSERTION_ACTOR_TEMPLATE), **_FIELDS)
