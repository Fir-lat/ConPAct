def require(value, message):
    if not value:
        raise ValueError(message)


GOAL_TYPES = {
    "minigrid": ("approach_object", "open_door", "pick_up_object", "reach_landmark"),
    "procgen": ("reach_landmark", "collect_item", "defeat_enemy"),
    "crafter": ("obtain_resource", "craft_tool", "place_object", "restore_need"),
}


ENV_RULES = {
    "minigrid": (
        "Left/right turn in place; forward moves relative to facing. Doors have colors and "
        "open/closed/locked states. Carried items are not drawn; disappearance alone is not "
        "proof of pickup. Do not propose merely facing an empty cell or rotating. A navigation "
        "goal must identify a useful visible object or stable landmark by unambiguous language."
    ),
    "procgen": (
        "Scrolling moves the screen: screen coordinates are not stable world locations. "
        "Use identifiable persistent landmarks, item collection or enemy defeat. Mere survival, "
        "arbitrary movement, camera scrolling and moving enemies are not achievements. "
        "An item disappearing offscreen is not proof it was collected. Native actions may "
        "advance multiple physics frames; evaluate only the archived observation boundaries."
    ),
    "crafter": (
        "The player stays centered while the map scrolls. HUD counts are visible. Movement "
        "into an obstruction may turn without moving; do interacts with the faced cell. "
        "A missing object is not proof of collection. Verify HUD/tool changes visually. "
        "Do not equate a screen position with a fixed world position."
    ),
}


MENU_SYSTEM = """Define a small, fixed menu of meaningful partial game objectives from the
INITIAL observation and the original game instructions ONLY. You cannot see the rollout or
its outcome. Return at most six goals that are simpler than, distinct from, and useful
toward the original objective. Goals must be false initially and their completion must be
visually decidable. Each goal must be understandable from a later CURRENT screenshot alone
plus the goal sentence. Do not refer to a past frame, an arbitrary location, hidden state,
a numeric pixel coordinate, an unspecified object, or an original task outcome as a new goal.
Do not weaken the goal to doing any action, making any progress, surviving, or rotating.
Keep a relevant goal even if it is hard; never invent missing objects to fill the menu.
Return JSON {"goals":[{"id":1,"type":"one allowed type","text":"one imperative sentence"}]}.
An empty goals list is valid if nothing meets these criteria.
"""


ACHIEVED_SYSTEM = """Review the archived game frames and the REAL executed actions.
For every menu goal reached by this attempt, return its earliest satisfying frame:
{"achieved":[{"id":1,"first_frame":5,"evidence":"specific visible evidence"}]}.
Frame zero is before any action; action i connects frame i to i+1.
Only use the listed goal IDs. Do not silently weaken a goal. Reject goals already satisfied
at frame zero, goals equivalent to the original objective, goals not interpretable from a
current observation, or goals whose completion is ambiguous. Do not infer success from
intent, action names, hidden rewards or a reset screen. Return an empty list when no goal
has visual evidence. You may see future evidence here; it is never a training input.
"""


RELEVANCE_SYSTEM = """Given the exact hindsight goal and its first satisfying frame, classify
EVERY preceding action as relevant or irrelevant. Relevant means needed or helpful in the
actual route to THIS goal; irrelevant means a detour, a wasted move, or progress undone
before completion. Judge turns, positioning and setup in the context of subsequent actions.
Do not treat all actions as relevant just because the final frame satisfies the goal.
Return JSON {"relevance":["relevant","irrelevant",...]} in action order, exactly one label
per action. If the claimed goal is not supported, return {"relevance":[],"reject":true}.
Do not write a new action, a new goal or a synthetic trajectory.
"""


def validate_menu(value, env):
    require(
        isinstance(value, dict) and isinstance(value.get("goals"), list),
        "expected goals list",
    )
    goals = value["goals"]
    require(len(goals) <= 6, "at most six menu goals")
    ids, texts = set(), set()
    for goal in goals:
        require(set(goal) == {"id", "type", "text"}, "goal needs exactly id/type/text")
        require(
            type(goal["id"]) is int and goal["id"] > 0 and goal["id"] not in ids,
            "unique positive goal ID required",
        )
        require(goal["type"] in GOAL_TYPES[env], "goal type outside this environment")
        text = goal["text"]
        require(
            isinstance(text, str)
            and 5 <= len(text) <= 600
            and "\n" not in text
            and "<" not in text,
            "invalid goal text",
        )
        require(text not in texts, "duplicate goal sentence")
        ids.add(goal["id"])
        texts.add(text)


def validate_achieved(value, menu, count):
    require(
        isinstance(value, dict) and isinstance(value.get("achieved"), list),
        "expected achieved list",
    )
    seen = set()
    for item in value["achieved"]:
        require(
            isinstance(item, dict) and set(item) == {"id", "first_frame", "evidence"},
            "invalid achievement schema",
        )
        require(
            type(item["id"]) is int
            and item["id"] in {g["id"] for g in menu}
            and item["id"] not in seen,
            "invalid or duplicate goal ID",
        )
        index = item["first_frame"]
        require(
            type(index) is int and 0 < index <= count,
            "first satisfying frame outside observed action prefix",
        )
        require(
            isinstance(item["evidence"], str) and item["evidence"].strip(),
            "missing visual evidence",
        )
        seen.add(item["id"])


def validate_relevance(value, count):
    require(isinstance(value, dict), "relevance must be an object")
    labels = value.get("relevance")
    if value.get("reject") is True:
        require(labels == [], "rejected goal must not carry positive labels")
        return
    require(
        isinstance(labels, list) and len(labels) == count,
        "one relevance label per prefix action required",
    )
    require(
        all(x in ("relevant", "irrelevant") for x in labels), "unknown relevance label"
    )


GOAL_TYPES["sokoban"] = ("place_box", "cover_target", "reach_position")
ENV_RULES["sokoban"] = (
    "Push boxes without pulling. Targets already covered are not new hindsight goals. Never accept a deadlocked box as progress."
)
