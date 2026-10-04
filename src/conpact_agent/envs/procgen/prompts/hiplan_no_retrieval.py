from ...hiplan import build_prompts
from ..actions import action_space
from .common import environment_rules

OBJECTIVE = (
    "Pursue the supplied game's objective and accumulate reward while avoiding failure. "
    "The final milestone must remain an ongoing phase of pursuing the game's objective "
    "and further available reward until the environment ends or the budget is exhausted. "
    "Keep that phase CONTINUE while play remains possible. Milestone completion does not "
    "certify level success; the environment decides termination."
)


def build(game: str, *, task_instruction: str = ""):
    return build_prompts(
        rules=environment_rules(game, task_instruction=task_instruction),
        action_space=action_space(game),
        objective=OBJECTIVE,
    )
