from ...hiplan import build_prompts
from .common import RULES, ACTION_SPACE

OBJECTIVE = (
    "Pursue further achievements while staying alive within the action and call budget. "
    "The final milestone must be an ongoing phase: keep meeting survival needs and pursue "
    "further achievable resource or crafting progress. Keep that phase CONTINUE while the "
    "environment runs. Completing earlier milestones does not end the episode or certify success."
)
PROMPTS = build_prompts(rules=RULES, action_space=ACTION_SPACE, objective=OBJECTIVE)
