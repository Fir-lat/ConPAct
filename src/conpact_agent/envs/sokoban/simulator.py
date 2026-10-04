from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


NEIGHBOR_ORDER = ("up", "right", "down", "left")


EMPTY = "cell"
WALL = "wall"
BOX = "box"
TARGET = "target"
BOX_ON_TARGET = "box_on_target"


SYM_WALL = "#"
SYM_PLAYER = "@"
SYM_BOX = "$"
SYM_TARGET = "?"


RENDER = {
    "wall": "#",
    "floor": ".",
    "target": "?",
    "box": "$",
    "box_on_target": "*",
    "player": "@",
    "player_on_target": "+",
}


class LevelError(ValueError):
    pass


@dataclass(frozen=True)
class State:
    walls: frozenset[tuple[int, int]]
    targets: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]

    cells: frozenset[tuple[int, int]]
    nrows: int
    ncols: int


def parse_level(level_text: str) -> State:
    return _parse_board(level_text, spaces_are_floor=True, require_enclosure=True)


def _parse_board(text: str, *, spaces_are_floor: bool, require_enclosure: bool) -> State:
    if not isinstance(text, str) or not text:
        raise LevelError("expected a nonempty Sokoban board")
    walls: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    cells: set[tuple[int, int]] = set()
    players: list[tuple[int, int]] = []

    lines = text.splitlines()
    symbols = set(RENDER.values()) | {" "}
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch not in symbols:
                raise LevelError(f"unsupported Sokoban symbol {ch!r} at {(r, c)}")
            if ch == " " and not spaces_are_floor:
                continue
            pos = (r, c)
            cells.add(pos)
            if ch == RENDER["wall"]:
                walls.add(pos)
            if ch in (RENDER["player"], RENDER["player_on_target"]):
                players.append(pos)
            if ch in (RENDER["box"], RENDER["box_on_target"]):
                boxes.add(pos)
            if ch in (RENDER["target"], RENDER["box_on_target"], RENDER["player_on_target"]):
                targets.add(pos)

    if len(players) != 1:
        raise LevelError(f"expected exactly one player, found {len(players)}")

    if require_enclosure:
        for pos in cells - walls:
            for dr, dc in DELTA.values():
                nb = (pos[0] + dr, pos[1] + dc)
                if nb not in cells:
                    raise LevelError(
                        f"non-wall cell {pos} has neighbour {nb} outside the grid; "
                        "player_neighbors cannot be grounded on this level"
                    )

    return State(
        walls=frozenset(walls),
        targets=frozenset(targets),
        boxes=frozenset(boxes),
        player=players[0],
        cells=frozenset(cells),
        nrows=len(lines),
        ncols=max((len(l) for l in lines), default=0),
    )


def push(state: State, action: str) -> State:

    if action not in DELTA:
        return state
    dr, dc = DELTA[action]
    pr, pc = state.player
    target_pos = (pr + dr, pc + dc)

    if target_pos in state.walls:
        return state
    if target_pos in state.boxes:
        behind = (target_pos[0] + dr, target_pos[1] + dc)
        if behind in state.walls or behind in state.boxes:
            return state
        boxes = set(state.boxes)
        boxes.discard(target_pos)
        boxes.add(behind)
        return _moved(state, target_pos, frozenset(boxes))
    return _moved(state, target_pos, state.boxes)


def _moved(
    state: State, player: tuple[int, int], boxes: frozenset[tuple[int, int]]
) -> State:
    return State(
        walls=state.walls,
        targets=state.targets,
        boxes=boxes,
        player=player,
        cells=state.cells,
        nrows=state.nrows,
        ncols=state.ncols,
    )


def replay(level_text: str, actions: Sequence[str]) -> list[State]:

    state = parse_level(level_text)
    states = [state]
    for action in actions:
        state = push(state, action)
        states.append(state)
    return states


def cell_type(state: State, pos: tuple[int, int]) -> str:
    if pos in state.walls:
        return WALL
    if pos in state.boxes:
        return BOX_ON_TARGET if pos in state.targets else BOX
    if pos in state.targets:
        return TARGET
    return EMPTY


def frame_truth(state: State) -> dict:

    pr, pc = state.player
    neighbors = []
    for direction in NEIGHBOR_ORDER:
        dr, dc = DELTA[direction]
        neighbors.append(cell_type(state, (pr + dr, pc + dc)))
    return {
        "box_count": len(state.boxes),
        "target_count": len(state.targets),
        "on_goal_count": len(state.boxes & state.targets),
        "player_neighbors": neighbors,
    }


def render_ascii(state: State) -> str:

    rows = []
    for r in range(state.nrows):
        row = []
        for c in range(state.ncols):
            pos = (r, c)
            if pos not in state.cells:
                row.append(" ")
            elif pos in state.walls:
                row.append(RENDER["wall"])
            elif pos == state.player:
                row.append(
                    RENDER["player_on_target"]
                    if pos in state.targets
                    else RENDER["player"]
                )
            elif pos in state.boxes:
                row.append(
                    RENDER["box_on_target"] if pos in state.targets else RENDER["box"]
                )
            elif pos in state.targets:
                row.append(RENDER["target"])
            else:
                row.append(RENDER["floor"])
        rows.append("".join(row).rstrip())
    return "\n".join(rows)


def parse_ascii(text: str) -> State:
    return _parse_board(text, spaces_are_floor=False, require_enclosure=False)


def solved(state: State) -> bool:
    return bool(state.boxes) and state.boxes <= state.targets


def board_equal(a: State, b: State) -> bool:

    return (
        a.walls == b.walls
        and a.targets == b.targets
        and a.boxes == b.boxes
        and a.player == b.player
    )
