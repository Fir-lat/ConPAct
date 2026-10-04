from dataclasses import replace
from .sokoban.prompts import paa as source

VERSION = "paa-static-multi-env-v1"


def build_prompts(
    *,
    rules,
    actions,
    action_names,
    objective,
    segmentation,
    instruction="Game started. Please make a move.",
    instruction_with_history="Game started. Please make a move.\nprevious actions and observations:\n",
    current_frame_block="Current frame:\n",
    observation="",
):
    planner = source.PLANNER_SYSTEM.replace(source._RULES, rules)
    planner = planner.replace(
        "Write the COMPLETE plan for solving this level, in one go, from the initial screen alone.",
        "Write the COMPLETE plan for pursuing the task objective, in one go, from the initial observation alone.",
    )
    planner = planner.replace(
        "You will not see the board again", "You will not see the environment again"
    )
    planner = planner.replace(
        "so it has to cover the level from the current position to every box on a target.",
        "so it has to cover the task from the current state. " + objective,
    )
    planner = planner.replace("Break the solution", "Break the task")
    planner = planner.replace(
        "what each step accomplishes on the board",
        "what each step accomplishes in the environment",
    )
    planner = planner.replace(
        "such as the box nearest the top-left or the target in the lower right;",
        "using visible appearance and relative location;",
    )
    planner = planner.replace(
        "Steps may revisit a box, move a box out of the way and come back to it, or reposition the player between pushes. One step per box is not required and is often wrong.",
        segmentation,
    )
    actor = source.ACTOR_SYSTEM.replace(source._RULES, rules)
    actor = actor.replace(
        source._ACTION_SPACE,
        "# Action Space\n" + actions.removeprefix("# Action Space\n"),
    )
    actor = actor.replace("before the level started", "at the start of the episode")
    actor = actor.replace(
        "If the board is no longer", "If the environment is no longer"
    )
    actor = actor.replace(
        "make progress on the level anyway", "make progress on the task anyway"
    )
    actor = actor.replace("up, down, left, right", ", ".join(action_names))
    actor = actor.replace(
        "<action>down</action>", "<action>" + action_names[0] + "</action>"
    )
    annotator = source.ANNOTATOR_SYSTEM.replace(source._RULES, rules)
    annotator = annotator.replace(
        "a solved Sokoban game", "an observed game trajectory or productive prefix"
    )
    annotator = annotator.replace(
        "The complete sequence of actions that solved the level",
        "The complete retained sequence of actually executed actions",
    )
    annotator = annotator.replace(
        "The last screen is the solved board.",
        "The last screen is the observed endpoint; it need not be a solved level.",
    )
    annotator = annotator.replace("this solution", "this observed sequence").replace(
        "the solution actually shows", "the sequence actually shows"
    )
    annotator = annotator.replace("on the board", "in the environment")
    annotator = annotator.replace(
        "such as the box nearest the top-left or the target in the lower right.",
        "using visible appearance and relative location.",
    )
    annotator = annotator.replace(
        "A step may move a box partway, move a box out of the way, reposition the player between pushes, or come back to a box that an earlier step already moved. Do not force one step per box.",
        segmentation,
    )
    annotator += (
        "\n"
        + objective
        + "\nDo not claim unseen achievements, rewards or full completion. Describe only objectives supported by the supplied observations and actions.\n"
    )
    prefix = observation + "\n" if observation else ""
    return replace(
        source.PROMPTS,
        planner_system=planner,
        actor_system=actor,
        annotator_system=annotator,
        repair_coverage=source.REPAIR_COVERAGE.replace(
            "The level was solved in", "The retained sequence contains"
        ).replace("solution you were given", "sequence you were given"),
        planner_instruction=prefix + source.PROMPTS.planner_instruction,
        instruction=instruction,
        instruction_with_history=instruction_with_history,
        current_frame_block=current_frame_block,
    )
