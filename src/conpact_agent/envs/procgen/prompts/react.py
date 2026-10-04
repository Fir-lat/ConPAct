from ...base import PromptSet, fill
from ..actions import action_space
from .common import environment_rules

ACTOR_TEMPLATE = """\
You are an expert game agent specialized in playing video games. Your goal is to play the game and achieve the task goal.
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
<action>right</action>
"""


def build(game: str, *, task_instruction: str = "") -> PromptSet:
    return PromptSet(
        actor_system=fill(
            ACTOR_TEMPLATE,
            environment_rules=environment_rules(
                game, task_instruction=task_instruction, actor=True
            ),
            action_space=action_space(game),
        ),
        kind="react",
        instruction="Game started. Please make a move.",
        instruction_with_history="Game started. Please make a move.\nprevious actions and observations:\n",
    )
