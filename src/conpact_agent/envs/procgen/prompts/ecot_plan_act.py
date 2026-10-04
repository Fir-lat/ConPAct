from conpact_agent.baselines.ecot.protocol import build_prompts


def build(game, *, task_instruction=""):
    return build_prompts("procgen", game, task_instruction)
