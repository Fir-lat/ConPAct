from ..games import GAME_RULES, OBJECTIVES, require_game


def environment_rules(
    game: str, *, task_instruction: str = "", actor: bool = False
) -> str:
    require_game(game)
    control = (
        "You control the player character."
        if actor
        else "The player character moves in the game world."
    )
    objective = "- " + OBJECTIVES[game]
    if task_instruction and task_instruction != OBJECTIVES[game]:
        objective += "\n- " + task_instruction
    return (
        f"# Game\nYou are playing {game}, a procedurally generated game from the Procgen environment.\n"
        f"{control}\n\n## Objective\n{objective}\n\n## Rules\n{GAME_RULES[game]}"
    )
