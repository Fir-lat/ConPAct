from conpact_agent.baselines.ecot.protocol import build_prompts


def build(env_name):
    return build_prompts("minigrid", env_name)
