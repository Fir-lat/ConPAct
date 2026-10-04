from ...tape import build_prompts
from ..actions import action_space
from .common import environment_rules


def build(game: str, *, task_instruction: str = ""):
    return build_prompts(
        rules=environment_rules(game, task_instruction=task_instruction),
        action_space=action_space(game),
        objective_mode="predicted_return",
    )
