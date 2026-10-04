from dataclasses import replace
import json
import re

ARM = "ecot_plan_act"
VERSION = "ecot-multi-visual-v1"
STATUSES = ("CONTINUE", "ACHIEVED", "IMPOSSIBLE", "STUCK", "CHANGED")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def schema(env, game=None):
    if env == "crafter":
        from conpact_agent.envs.crafter.assertion import SCHEMA
        from conpact_agent.envs.crafter.prompts.common import ASSERTION_SCHEMA

        template = dict(
            facing="up",
            front="walkable",
            vitals=[0] * 4,
            inventory=dict.fromkeys(("wood", "stone", "coal", "iron"), 0),
            tools=dict.fromkeys(
                (
                    "wood_pickaxe",
                    "stone_pickaxe",
                    "iron_pickaxe",
                    "wood_sword",
                    "stone_sword",
                    "iron_sword",
                ),
                0,
            ),
        )
        return SCHEMA, ASSERTION_SCHEMA, template
    if env == "minigrid":
        from conpact_agent.envs.minigrid.assertion import SCHEMA
        from conpact_agent.envs.minigrid.prompts.common import ASSERTION_SLOTS

        return (
            SCHEMA,
            ASSERTION_SLOTS,
            dict(
                key_count=0,
                front_cell="empty",
                locked_door_count=0,
                closed_door_count=0,
                open_door_count=0,
            ),
        )
    if env == "procgen":
        from conpact_agent.envs.procgen.assertion import ProcgenAssertionSchema

        value = ProcgenAssertionSchema(game)
        return value, value.documentation(), value.example()
    raise ValueError("unsupported ECoT environment: " + env)


def state_instructions(env, game=None):
    return (
        schema(env, game)[1].replace("state assertion", "state description")
        + "\nWrite these fields as one JSON object inside <state>...</state>. "
        "Use JSON null for any unreadable or ambiguous field or count; never invent a value. "
        "Preserve all keys, including keys of nested objects. Read only the current "
        "screenshot and public observation text, never hidden inventory or offscreen state."
    )


def validate_state(value, env, game=None):
    base, _, template = schema(env, game)

    def fill_unknown(item, example):
        if isinstance(example, dict):
            if not isinstance(item, dict) or set(item) != set(example):
                raise ValueError("state keys differ from the environment schema")
            return {k: fill_unknown(item[k], v) for k, v in example.items()}
        if isinstance(example, list):
            if not isinstance(item, list) or len(item) != len(example):
                raise ValueError("state list shape differs from schema")
            return [fill_unknown(x, y) for x, y in zip(item, example)]
        return example if item is None else item

    checked = fill_unknown(value, template)
    if base.parse("<assertion>" + json.dumps(checked) + "</assertion>") is None:
        raise ValueError("invalid visual state value")
    return value


def build_prompts(env, game=None, task_instruction=""):
    if env == "crafter":
        from conpact_agent.envs.crafter.prompts.plan_act import PLAN_ACT as base
        from conpact_agent.envs.crafter.constants import ACTIONS as actions
    elif env == "minigrid":
        from conpact_agent.envs.minigrid.prompts.plan_act import build
        from conpact_agent.envs.minigrid.profile import get_profile

        base, actions = build(game), get_profile(game).actions
    elif env == "procgen":
        from conpact_agent.envs.procgen.prompts.plan_act import build
        from conpact_agent.envs.procgen.actions import valid_actions

        base, actions = (
            build(game, task_instruction=task_instruction),
            valid_actions(game),
        )
    else:
        raise ValueError(env)
    state = "\n# Independent current state\n" + state_instructions(env, game)
    planner = (
        base.planner_system.split("# Your Task", 1)[0]
        + """# Your Task
Read the CURRENT decision frame independently. Describe the current state, write
a short remaining plan toward the task objective, and select the next useful
short-term subgoal. The remaining plan covers the broader objective; the subgoal
is one feasible stage with a clear completion condition, usually several actions.
The actor receives only the subgoal, so it must make sense without your state or
plan. Use previous subgoals, outcomes and their adoption frames when supplied to
understand progress and choose what comes next. Never treat a historical frame
as the current observation. For open-ended games, plan useful survival/reward
progress without inventing a terminal success condition.

# Output Format
Output exactly three non-empty blocks in this order:
<state>JSON object</state>
<plan>One or two sentences covering the remaining task or useful progression.</plan>
<subgoal>The next short-term objective.</subgoal>
Do not output an action or status.
"""
        + state
    )
    actor = (
        base.actor_system.split("# Your Task", 1)[0]
        + """# Your Task
Read the CURRENT screenshot independently. The planner's state and remaining
plan are not provided. Explain the next action in relation to the supplied
subgoal using visible facts. Do not describe a blocked or unhelpful action as
progress. Take exactly one legal action and report the subgoal's status AFTER
that chosen action. The action executes before a non-CONTINUE status hands
control back to the planner on the next observation.

# Output Format
Output exactly four non-empty blocks in this order:
<state>JSON object</state>
<reason>A short explanation of the action's expected local consequence.</reason>
<action>Exactly one of: """
        + ", ".join(actions)
        + """</action>
<status>CONTINUE, ACHIEVED, IMPOSSIBLE, STUCK, or CHANGED</status>
CONTINUE: the subgoal remains unfinished and feasible.
ACHIEVED: the chosen action completes the subgoal.
IMPOSSIBLE: the subgoal cannot be accomplished as stated.
STUCK: you cannot make progress on the subgoal.
CHANGED: the observed situation invalidates the subgoal's assumptions.
A data prefix ending or a reward arriving is not by itself proof of subgoal
completion. Do not output a new plan or subgoal yourself.
"""
        + state
    )
    return replace(base, planner_system=planner, actor_system=actor)


def valid_reply(raw, role, env, game=None):
    tags = (
        ("state", "plan", "subgoal")
        if role == "planner"
        else ("state", "reason", "action", "status")
    )
    pattern = (
        r"\s*" + r"\s*".join("<" + t + r">([^<>]+)</" + t + ">" for t in tags) + r"\s*"
    )
    match = re.fullmatch(pattern, raw or "", re.S)
    if not match or any(not s.strip() for s in match.groups()):
        return False
    try:
        validate_state(
            json.loads(match.group(1), object_pairs_hook=unique_object), env, game
        )
    except (ValueError, TypeError):
        return False
    return role == "planner" or match.group(4) in STATUSES


ANNOTATOR = """You annotate an EXISTING game demonstration. Never change, add,
remove or reorder actions. Pictures are chronological: frame t is BEFORE action
t; the final image is AFTER the last action. Use only pictures, public task text
and supplied actions. No hidden environment state or original model reasoning
is available. Keep mistakes, blocked attempts and detours; describe them honestly.

Return JSON with exactly two keys:
segments: [{start:0,end:2,plan:"...",subgoal:"...",end_status:"ACHIEVED"}, ...]
steps: [{step:0,action:"original action",reason:"...",effect:"..."}, ...]
Use quoted JSON keys. Segment indices are inclusive and must cover all actions
consecutively exactly once. Group actions into meaningful feasible short-term
objectives, usually several actions, with identifiable completion conditions.
Do not use an attempted primitive action as a subgoal. Keep mistakes with the
useful objective they fail to advance. Adjacent segments must have different
subgoals. All natural language is English without XML/HTML tags.

Each segment's plan is one or two sentences at its START frame covering the
whole remaining task, or useful survival/reward progression in open-ended games.
The subgoal is only the current stage and is the ONLY text handed to the actor.
It must remain interpretable from each current screenshot within that segment.
Do not list future primitive actions or depend on hidden item IDs/inventory.

end_status describes the subgoal AFTER the segment's last recorded action:
ACHIEVED only if its stated completion condition is met; CHANGED if the premise
changed; STUCK if progress stalled; IMPOSSIBLE if infeasible. Only the final
segment may end CONTINUE, when the prefix ends with the objective unfinished.
The end of the recording or a positive reward does not by itself mean ACHIEVED.
This may be a reward-bearing prefix, not a solved episode.

Each reason is independently usable by an actor seeing only its current image,
public observation text and the current subgoal. Phrase the next action's effect
as expected, not already observed. Do not mention past attempts, action indices,
later screenshots, hidden state, or hindsight. Do not invent exploration intent
to rationalize a mistake. Distinguish necessary repositioning from a needless
detour. Never infer velocity from a single still image. In MiniGrid left/right
turn in place and done may complete a task; in Crafter a move can only turn toward
an obstacle and repeated do may be needed; autonomous enemies and resource decay
are not achievements caused by the agent. Effect is a short before/after visual
audit description, and may say uncertain; it is not included in student targets.
Review each reason against its OWN before-action frame before returning JSON.
"""


def validate_annotation(value, actions):
    if not isinstance(value, dict) or set(value) != {"segments", "steps"}:
        raise ValueError("expected segments and steps")
    if not isinstance(value["segments"], list) or not value["segments"]:
        raise ValueError("empty segments")
    cursor, previous = 0, None

    def prose(text):
        return (
            isinstance(text, str)
            and bool(text.strip())
            and not re.search(r"[<>]", text)
        )

    for j, seg in enumerate(value["segments"]):
        if not isinstance(seg, dict) or set(seg) != {
            "start",
            "end",
            "plan",
            "subgoal",
            "end_status",
        }:
            raise ValueError("invalid segment schema")
        if (
            type(seg["start"]) is not int
            or type(seg["end"]) is not int
            or not cursor == seg["start"] <= seg["end"] < len(actions)
        ):
            raise ValueError(
                "segments must cover consecutive action indices exactly once"
            )
        if (
            not all(prose(seg[k]) for k in ("plan", "subgoal"))
            or seg["subgoal"] == previous
        ):
            raise ValueError("empty/tagged text or duplicate adjacent subgoal")
        if seg["end_status"] not in STATUSES or (
            seg["end_status"] == "CONTINUE" and j != len(value["segments"]) - 1
        ):
            raise ValueError("only final segment may remain CONTINUE")
        cursor, previous = seg["end"] + 1, seg["subgoal"]
    if (
        cursor != len(actions)
        or not isinstance(value["steps"], list)
        or len(value["steps"]) != len(actions)
    ):
        raise ValueError("incomplete action coverage")
    for i, (step, action) in enumerate(zip(value["steps"], actions)):
        if not isinstance(step, dict) or set(step) != {
            "step",
            "action",
            "reason",
            "effect",
        }:
            raise ValueError("invalid step schema")
        if (
            type(step["step"]) is not int
            or step["step"] != i
            or step["action"] != action
        ):
            raise ValueError("teacher changed an action/index")
        if not all(prose(step[k]) for k in ("reason", "effect")):
            raise ValueError("empty/tagged reason or effect")
    return value
