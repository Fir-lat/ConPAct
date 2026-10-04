from ...paa import build_prompts
from ..constants import ACTIONS
from .common import RULES, ACTION_SPACE
from .react import REACT

PROMPTS = build_prompts(
    rules=RULES,
    actions=ACTION_SPACE,
    action_names=ACTIONS,
    objective="Pursue resource and crafting achievements while staying alive. Crafter has no single solved state. Continue survival and attainable progress until the environment ends or the budget is exhausted.",
    segmentation="Steps may collect resources, obtain or use crafting stations, craft tools, reposition, or address survival needs. Facing a blocked resource can be a necessary setup action. Do not assume fixed screen coordinates as the camera moves.",
    instruction=REACT.instruction,
    instruction_with_history=REACT.instruction_with_history,
    current_frame_block=REACT.current_frame_block,
)
