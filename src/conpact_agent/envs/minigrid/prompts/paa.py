from ...paa import build_prompts
from ..profile import get_profile
from .common import environment_parts
from .react import build as react_prompts


def build(env_name):
    rules, actions = environment_parts(env_name)
    react = react_prompts(env_name)
    return build_prompts(
        rules=rules,
        actions=actions,
        action_names=get_profile(env_name).actions,
        objective="Fulfill the supplied Mission under this task's rules. Only the environment determines success. Use done only when available and required by the task rules.",
        segmentation="Steps may navigate, obtain an object, open a door, transport an object, or satisfy the mission's final interaction. Turns change facing without moving; forward depends on facing. Do not invent an inventory or infer a held object only from its absence on the map.",
        instruction=react.instruction,
        instruction_with_history=react.instruction_with_history,
        current_frame_block=react.current_frame_block,
        observation="{observation}",
    )
