GAMES = ("bigfish", "chaser", "coinrun", "jumper", "maze", "miner", "ninja")

OBJECTIVES = {
    "bigfish": "Eat smaller fish to grow, avoid larger fish, and outgrow all fish to finish.",
    "chaser": "Collect all green orbs while avoiding enemies unless they are vulnerable.",
    "coinrun": "Reach and collect the coin at the far right while avoiding hazards.",
    "jumper": "Find and collect the carrot while avoiding spikes.",
    "maze": "Navigate the maze and collect the cheese.",
    "miner": "Dig through dirt, collect diamonds, and reach the exit without being crushed.",
    "ninja": "Reach the mushroom by crossing platforms and avoiding bombs.",
}

GAME_RULES = {
    "bigfish": "You are a fish that grows by eating smaller fish. Touching a larger fish kills "
    "you. Get small reward per edible fish; large reward when you outgrow all fish and "
    "finish.",
    "chaser": "Pacman-like maze game: collect all green orbs to clear level. Large stars make "
    "enemies temporarily vulnerable. Avoid non-vulnerable enemies or die.",
    "coinrun": "Platformer: start left, reach the coin at far right. Avoid saws, enemies, and "
    "deadly gaps/chasms.",
    "jumper": "Open-world platformer: find and collect the carrot (main reward). Use movement and "
    "double-jump to navigate tricky layouts; avoid spikes.",
    "maze": "Mouse-in-maze task: navigate and find the single cheese. Movement is 4-directional "
    "(up/down/left/right).",
    "miner": "BoulderDash-like game with gravity: dig dirt, collect diamonds, then reach exit. "
    "Falling boulders/diamonds are lethal.",
    "ninja": "Platformer: jump across narrow ledges, avoid bombs, optionally clear bombs with "
    "throwing stars at different angles. Reach mushroom at end to finish.",
}

ACTION_COMBOS = (
    ("LEFT", "DOWN"),
    ("LEFT",),
    ("LEFT", "UP"),
    ("DOWN",),
    (),
    ("UP",),
    ("RIGHT", "DOWN"),
    ("RIGHT",),
    ("RIGHT", "UP"),
    ("D",),
    ("A",),
    ("W",),
    ("S",),
    ("Q",),
    ("E",),
)

ACTION_NAMES = (
    "left_down",
    "left",
    "left_up",
    "down",
    "noop",
    "up",
    "right_down",
    "right",
    "right_up",
    "d",
    "a",
    "w",
    "s",
    "q",
    "e",
)

EFFECTIVE_ACTIONS = {
    "bigfish": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "chaser": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "coinrun": (1, 2, 4, 5, 7, 8),
    "jumper": (1, 2, 4, 5, 7, 8),
    "maze": (1, 3, 4, 5, 7),
    "miner": (1, 3, 4, 5, 7),
    "ninja": (1, 2, 4, 5, 7, 8, 9, 10, 11, 12),
}

ACTION_REMAP = {
    "bigfish": {9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4},
    "chaser": {9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4},
    "coinrun": {0: 1, 3: 4, 6: 7, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4},
    "jumper": {0: 1, 3: 4, 6: 7, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4},
    "maze": {0: 1, 2: 1, 6: 7, 8: 7, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4},
    "miner": {0: 1, 2: 1, 6: 7, 8: 7, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4},
    "ninja": {0: 1, 3: 4, 6: 7, 13: 9, 14: 9},
}

ACTION_DESCRIPTIONS = {
    "bigfish": {
        "left_down": "Swim down-left",
        "left": "Swim left",
        "left_up": "Swim up-left",
        "down": "Swim down",
        "noop": "Stay still",
        "up": "Swim up",
        "right_down": "Swim down-right",
        "right": "Swim right",
        "right_up": "Swim up-right",
    },
    "chaser": {
        "left_down": "Move down-left",
        "left": "Move left",
        "left_up": "Move up-left",
        "down": "Move down",
        "noop": "Stay still",
        "up": "Move up",
        "right_down": "Move down-right",
        "right": "Move right",
        "right_up": "Move up-right",
    },
    "coinrun": {
        "left": "Run left",
        "left_up": "Jump while moving left",
        "noop": "Stand still",
        "up": "Jump straight up",
        "right": "Run right",
        "right_up": "Jump while moving right",
    },
    "jumper": {
        "left": "Move left",
        "left_up": "Jump while moving left (press again in mid-air to double-jump)",
        "noop": "Stand still",
        "up": "Jump straight up (press again in mid-air to double-jump)",
        "right": "Move right",
        "right_up": "Jump while moving right (press again in mid-air to double-jump)",
    },
    "maze": {
        "left": "Move one step left",
        "down": "Move one step down",
        "noop": "Stay still",
        "up": "Move one step up",
        "right": "Move one step right",
    },
    "miner": {
        "left": "Move left, digging through dirt in the way",
        "down": "Move down, digging through dirt in the way",
        "noop": "Stay still",
        "up": "Move up, digging through dirt in the way",
        "right": "Move right, digging through dirt in the way",
    },
    "ninja": {
        "left": "Move left",
        "left_up": "Jump while moving left",
        "noop": "Stand still",
        "up": "Jump straight up",
        "right": "Move right",
        "right_up": "Jump while moving right",
        "d": "Throw star right",
        "a": "Throw star left",
        "w": "Throw star up",
        "s": "Throw star down",
    },
}


def require_game(game: str) -> str:
    if game not in GAMES:
        raise ValueError(f"unsupported Procgen game {game!r}; expected one of {GAMES}")
    return game


def resolve_game(payload: dict) -> str:
    """Resolve the supported task aliases without guessing a game or ignoring conflicts."""
    names = {
        key: require_game(payload[key])
        for key in ("game", "env_name")
        if key in payload
    }
    if not names:
        raise ValueError("Procgen task requires payload.game or payload.env_name")
    if len(set(names.values())) != 1:
        raise ValueError(
            f"Conflicting Procgen identifiers: game={names['game']!r}, "
            f"env_name={names['env_name']!r}"
        )
    return next(iter(names.values()))
