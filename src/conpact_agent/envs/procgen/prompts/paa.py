from ...paa import build_prompts
from ..actions import action_space, valid_actions
from .common import environment_rules
from .react import build as react_prompts


def build(game, *, task_instruction=""):
    react = react_prompts(game, task_instruction=task_instruction)
    return build_prompts(
        rules=environment_rules(game, task_instruction=task_instruction),
        actions=action_space(game),
        action_names=valid_actions(game),
        objective="Pursue this game's objective and attainable reward while avoiding failure. A productive prefix is not proof of level completion. Continue toward the objective while the environment is running.",
        segmentation="Steps may navigate obstacles, reposition, collect available rewards or reach the game's endpoint. Use this game's rules and controls. Do not invent offscreen objects or replace the observed route with an imagined successful one.",
        instruction=react.instruction,
        instruction_with_history=react.instruction_with_history,
        current_frame_block=react.current_frame_block,
    )
