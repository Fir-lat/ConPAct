from dataclasses import replace
from .rules import RULES
import re

from ...sokoban.prompts.plan_act_assertion import (
    _ASSERTION_SLOTS as SOKOBAN_ASSERTION_SLOTS,
)
from ..profile import get_profile

OBSERVATION = """# Observation
The screenshot shows the whole level from above. Your character is a red triangle pointing in the direction you face.
Lighter cells mark the agent's current field of view; darker cells are also rendered and belong to the visible full map.
Read objects in both light and dark cells. The Mission and facing direction are supplied with the current screenshot.
Left and right are turns in place; forward moves relative to your facing direction.
A carried object is not drawn on the board. Do not invent a visible inventory or infer a held key merely from the absence of keys.
"""

ASSERTION_SLOTS = """A state assertion is how you commit to what you see before you decide. It has exactly five slots:
- key_count: how many keys are drawn on the board, including darker rendered cells. A carried key is not drawn and does not count.
- front_cell: what is in the single cell directly ahead of the triangle, using its facing direction. Exactly one of: empty, wall, door_locked, door_closed, door_open, key, ball, box, goal, lava. Use wall for the outer boundary.
- locked_door_count: how many visible locked doors there are, drawn as solid filled cells with a keyhole.
- closed_door_count: how many visible closed but unlocked doors there are, drawn as hollow double-edged squares.
- open_door_count: how many visible open doors there are, drawn as thin strips at the edge of walkable cells.
All four counts are non-negative integers. Count doors in darker rendered cells too; levels without doors have zero in all three door slots."""

ASSERTION_EXAMPLE = '<assertion>{"key_count": 0, "front_cell": "empty", "locked_door_count": 0, "closed_door_count": 0, "open_door_count": 0}</assertion>'


ENVIRONMENT_CLAUSES = (
    (
        "the box it names is against a wall or in a corner in the direction it would have to move",
        "the object it names is behind a locked door you have no key for or is not present on the board",
    ),
    (
        "a box or target is not where the subgoal implies, or another box moved and a different one now matters more",
        "a door or object is not where the subgoal implies, or another object moved and a different one now matters more",
    ),
    (
        "you cannot move into a neighbor you read as Wall, and you can only push a box that is in the direction you are moving",
        "you cannot move forward into a front_cell you read as wall, door_locked or door_closed, and pickup requires a key, ball or box directly in front of you",
    ),
)


def section(system: str, heading: str) -> str:
    pattern = rf"^# {re.escape(heading)}\n.*?(?=^# |\Z)"
    matches = re.findall(pattern, system, re.M | re.S)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {heading!r} section")
    return matches[0]


def game_context(env_name: str) -> str:
    profile = get_profile(env_name)
    rules = RULES[profile.rules]
    return OBSERVATION.strip() + "\n\n" + rules


def environment_parts(env_name: str) -> tuple[str, str]:
    rules, action_space = game_context(env_name).split("\n# Action Space\n", 1)
    return rules.strip(), "# Action Space\n" + action_space.strip()


def adapt_system(system: str, env_name: str) -> str:
    if not system:
        return system
    profile = get_profile(env_name)
    rules, actions = environment_parts(env_name)
    has_actions = "\n# Action Space\n" in system
    game = section(system, "Game")
    action_block = section(system, "Action Space") if has_actions else None

    system = system.replace(game, "{game_rules}\n\n", 1)
    if action_block:
        system = system.replace(action_block, "{action_space}\n\n", 1)
    system = system.replace(SOKOBAN_ASSERTION_SLOTS, ASSERTION_SLOTS)
    for old, new in ENVIRONMENT_CLAUSES:
        system = system.replace(old, new)
    for old, new in (
        ("Fill these four slots", "Fill these five slots"),
        ("exactly those four keys", "exactly those five keys"),
        ("exactly the four keys above", "exactly the five keys above"),
        ("up, down, left, right", ", ".join(profile.actions)),
        ("<action>down</action>", "<action>forward</action>"),
    ):
        system = system.replace(old, new)
    system = re.sub(
        r"<assertion>\{[^\n]*\}</assertion>", lambda _: ASSERTION_EXAMPLE, system
    )
    system = system.replace(
        "{game_rules}", rules if has_actions else rules + "\n\n" + actions
    )
    return system.replace("{action_space}", actions)


def adapt_prompt_set(source, env_name: str):

    instructions = {}
    for field in ("instruction", "instruction_with_history", "current_frame_block"):
        value = getattr(source, field)
        instructions[field] = "{observation}\n" + value if value else value
    updates = dict(
        actor_system=adapt_system(source.actor_system, env_name),
        planner_system=adapt_system(source.planner_system, env_name),
        **instructions,
    )
    return replace(source, **updates)
