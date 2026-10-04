import json
from ..constants import ACTIONS, FACINGS, FRONTS, RESOURCES, TOOLS

RULES = """# Game
You are playing Crafter, a 2D open-world survival game. Collect resources, craft tools, and survive against enemies.
The player stands at the centre of the visible map. The bottom strip is the inventory: health, food, drink and energy on the left, followed by materials and tools with their counts.

## Objective
Progress through the achievement tree, from basic resource collection to advanced tool crafting, while staying alive.

## Observation
The visible map moves with the player; a screen location is not a fixed world location.

## Rules
**Facing:** A move toward an empty grass, sand or path cell moves the player one cell and turns them that way. A move into a blocked cell (such as a tree, stone, water, table or animal) only turns the player toward it. Turning toward a blocked cell is how to line up a 'do'; it is a valid action. Repeating that move while already facing the obstruction does not clear it. 'do' interacts with the cell currently faced. Lava can be entered but kills the player.

**Resources (use 'do' facing the cell):**
- Tree gives wood.
- Stone and coal each require a wood pickaxe.
- Iron requires a stone pickaxe.
- Diamond requires an iron pickaxe.
- Water restores drink.
- Grass has a 10% chance to give a sapling.

**Placing:** The target cell in front must have no object. A table costs 2 wood and a furnace costs 4 stone; both go on grass, sand or path. A stone costs 1 stone and can also be placed on water or lava. A plant costs 1 sapling and only goes on grass.

**Crafting:** A table must be in one of the eight immediately surrounding cells.
- Wood pickaxe / wood sword: 1 wood each.
- Stone pickaxe / stone sword: 1 wood + 1 stone each.
- Iron pickaxe / iron sword: both table and furnace nearby, 1 wood + 1 coal + 1 iron each.
Crafting a tool already owned consumes materials again.

**Survival:** Health, food, drink and energy range from 0 to 9. Eat cows or ripe plants for food, drink water, and sleep to restore energy. Depleted needs can reduce health. Sleep can persist across steps until energy is restored or damage wakes the player.

**Enemies:** Zombies attack nearby; skeletons shoot arrows. Observe threats before collecting or crafting."""

_ACTION_DESCRIPTIONS = (
    "Do nothing for one environment step.",
    "Move one cell left if possible; otherwise face left.",
    "Move one cell right if possible; otherwise face right.",
    "Move one cell up if possible; otherwise face up.",
    "Move one cell down if possible; otherwise face down.",
    "Interact with the cell faced: collect, drink, eat or attack.",
    "Sleep to restore energy.",
    "Place stone in front, consuming 1 stone.",
    "Place a table in front, consuming 2 wood.",
    "Place a furnace in front, consuming 4 stone.",
    "Plant a sapling on grass in front, consuming 1 sapling.",
    "Craft a wood pickaxe.",
    "Craft a stone pickaxe.",
    "Craft an iron pickaxe.",
    "Craft a wood sword.",
    "Craft a stone sword.",
    "Craft an iron sword.",
)
ACTION_SPACE = "Choose exactly one action per step:\n" + "\n".join(
    f"- {name}: {description}"
    for name, description in zip(ACTIONS, _ACTION_DESCRIPTIONS)
)

ASSERTION_EXAMPLE = {
    "facing": "up",
    "front": "tree",
    "vitals": [9, 8, 7, 9],
    "inventory": {"wood": 3, "stone": 1, "coal": 0, "iron": 0},
    "tools": dict(zip(TOOLS, (1, 0, 0, 1, 0, 0))),
}
ASSERTION_JSON = json.dumps(ASSERTION_EXAMPLE)
ASSERTION_SCHEMA = f"""A state assertion is how you commit to what you see before you decide. It has exactly five slots:
- facing: which way the player character is looking. Exactly one of: {", ".join(FACINGS)}.
- front: what is in the single cell the player is facing. Exactly one of: {", ".join(FRONTS)}. "walkable" means empty grass, sand or path. "plant" covers either growth stage; "arrow" means an arrow occupying that cell. Lava is separate from walkable.
- vitals: the four survival numbers on the left of the HUD strip, as a list of exactly four integers in screen order: [health, food, drink, energy]. Each ranges 0-9.
- inventory: how many wood, stone, coal and iron the player is carrying, read off the HUD strip. Exactly these keys: {", ".join(RESOURCES)}. An item absent from the HUD is 0.
- tools: how many of each tool the player owns, read off the HUD strip. Exactly these keys: {", ".join(TOOLS)}. A tool absent from the HUD is 0.
All material and tool counts are integers from 0 to 9."""

ACTION_NAMES = ", ".join(ACTIONS)
ACTION_CONSTRAINT = "turning toward a blocked cell is allowed, 'do' acts on the cell you face and requires the appropriate tool, and crafting requires the materials and nearby stations."
IMPOSSIBLE_EXAMPLE = (
    "the resource it names requires a tool you do not own or materials you do not have"
)
CHANGED_EXAMPLES = "the resource it names is gone, or something more urgent appeared, such as a zombie next to you or a vital dropping to zero."

ACTOR_INSTRUCTION = "Game started. Please make a move."
ACTOR_HISTORY = (
    "Game started. Please make a move.\nprevious actions and observations:\n"
)
