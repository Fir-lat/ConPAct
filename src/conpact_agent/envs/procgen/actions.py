from .games import (
    ACTION_DESCRIPTIONS,
    ACTION_NAMES,
    ACTION_REMAP,
    EFFECTIVE_ACTIONS,
    require_game,
)


def valid_actions(game: str) -> tuple[str, ...]:
    return tuple(ACTION_NAMES[i] for i in EFFECTIVE_ACTIONS[require_game(game)])


def action_index(game: str, action: str) -> int:
    if not isinstance(action, str) or action.lower() not in valid_actions(game):
        raise ValueError(f"invalid action {action!r} for {game}")
    return ACTION_NAMES.index(action.lower())


def canonicalize_legacy_index(game: str, index: int) -> int:
    require_game(game)
    if type(index) is not int or not 0 <= index < len(ACTION_NAMES):
        raise ValueError(f"invalid native action index {index!r}")
    return ACTION_REMAP[game].get(index, index)


def action_space(game: str) -> str:
    lines = ["Choose exactly one action per step:"]
    for action in valid_actions(game):
        description = ACTION_DESCRIPTIONS[game][action]
        if action == "noop":
            description = (
                "Apply no directional or special input; the environment still advances."
            )
        lines.append(f"- {action}: {description}")
    lines.append(
        "A combined name such as right_up is one native action, not a sequence."
        if "right_up" in valid_actions(game)
        else "Only the listed cardinal movements and noop are available."
    )
    return "\n".join(lines)


def example_action(game: str) -> str:
    return "right_up" if "right_up" in valid_actions(game) else "right"
