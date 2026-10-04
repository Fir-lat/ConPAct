from ...sokoban.prompts.plan_act import PROMPTS as PLAN_ACT
from ...sokoban.prompts.plan_act_assertion import PROMPTS as PLAN_ACT_ASSERTION
from .common import adapt_prompt_set


def build(env_name: str, *, assertion: bool = False):
    source = PLAN_ACT_ASSERTION if assertion else PLAN_ACT
    return adapt_prompt_set(source, env_name)
