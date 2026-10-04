from ...sokoban.prompts.conpact import CONPACT
from .common import adapt_prompt_set


def build(env_name: str):
    return adapt_prompt_set(CONPACT, env_name)
