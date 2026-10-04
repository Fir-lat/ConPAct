from __future__ import annotations


DIRECTION_NAMES = {
    0: "right (east)",
    1: "down (south)",
    2: "left (west)",
    3: "up (north)",
}


def observation_text(*, mission: str, direction: int) -> str:
    if not isinstance(mission, str) or not mission.strip():
        raise ValueError(
            "MiniGrid requires the mission from the current reset/step response"
        )
    if type(direction) is not int or direction not in DIRECTION_NAMES:
        raise ValueError("MiniGrid requires an integer direction in 0..3")
    return (
        "Current observation metadata:\n"
        f"Mission: {mission}\n"
        f"You are currently facing {DIRECTION_NAMES[direction]}.\n"
        "These fields describe the CURRENT screenshot, not the history frames."
    )
