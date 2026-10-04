from ...hiplan import build_prompts
from .common import environment_parts

OBJECTIVE = (
    "Fulfill the current Mission using the task's rules and actual action space. "
    "Completion of the milestone list does not certify success; only the environment "
    "decides whether the mission has ended. If the final milestone appears achieved "
    "but the environment is still running, keep following the mission's completion "
    "requirements. Use done only when allowed and required by those task rules."
)


def build(env_name: str):
    rules, actions = environment_parts(env_name)
    return build_prompts(
        rules=rules,
        action_space=actions.removeprefix("# Action Space\n"),
        objective=OBJECTIVE,
        observation="{observation}",
    )
