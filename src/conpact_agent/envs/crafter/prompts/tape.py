from ...tape import build_prompts
from .common import RULES, ACTION_SPACE

PROMPTS = build_prompts(
    rules=RULES, action_space=ACTION_SPACE, objective_mode="predicted_return"
)
