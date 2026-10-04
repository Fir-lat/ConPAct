from ...tape import build_prompts
from .common import environment_parts


def build(env_name: str):
    rules, actions = environment_parts(env_name)
    return build_prompts(
        rules=rules,
        action_space=actions,
        objective_mode="goal_path",
        observation="{observation}",
    )
