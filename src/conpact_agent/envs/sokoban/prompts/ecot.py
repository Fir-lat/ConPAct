_PROTOCOL_HEADING = "\n# Output Format\n"

_COORDS = """\
   - player: always [0, 0]. Every coordinate in this block is written as [dx, dy] RELATIVE TO THE PLAYER: dx counts cells to the RIGHT, dy counts cells DOWNWARD. The cell one to the left and two above the player is [-1, -2].
   - boxes: the position of every box on the board, counting boxes already on a target.
   - targets: the position of every target location, counting targets already covered by a box.
     Sort both lists by dy first, then by dx.
   - box_count: how many boxes are on the board in total, counting boxes already on a target.
   - target_count: how many target locations there are in total, counting targets already covered by a box.
   - on_goal_count: how many boxes are currently sitting on a target.
   - player_neighbors: what is in the four cells directly adjacent to the player, in the order UP, RIGHT, DOWN, LEFT. Each entry is exactly one of: cell, wall, box, target, box_on_target. "cell" means empty floor."""


def shared_prefix(system: str) -> str:

    at = system.find(_PROTOCOL_HEADING)
    if at < 0:
        raise ValueError(f"no {_PROTOCOL_HEADING.strip()!r} section to replace")
    return system[:at]
