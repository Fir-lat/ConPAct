from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib
import re

from .base import PromptSet, Task

METHODS = (
    "react",
    "plan_act",
    "hiplan",
    "tape",
    "self_refine",
    "assertion",
    "resampling",
    "paa",
    "hsl",
    "webstar",
    "ecot",
    "conpact_i",
    "conpact_s",
    "conpact_r",
)
ENVIRONMENTS = ("sokoban", "crafter", "procgen", "minigrid", "osworld")
ARMS = dict(
    react="react",
    plan_act="plan_act",
    hiplan="hiplan_no_retrieval",
    tape="tape",
    self_refine="react",
    assertion="plan_act_assertion",
    resampling="plan_act_assertion",
    paa="paa_static",
    hsl="react",
    webstar="react",
    ecot="ecot_plan_act",
    conpact_i="conpact",
    conpact_s="conpact",
    conpact_r="conpact",
)


@dataclass(frozen=True)
class Environment:
    name: str
    variant: str
    actions: tuple[str, ...]
    assertion: object
    prompts: dict[str, PromptSet]
    max_steps: int
    max_calls: int

    def prompt(self, method, frame):
        prompt = self.prompts[method]
        if self.name == "minigrid":
            from .minigrid.observation import observation_text

            observation = observation_text(
                mission=frame.info["mission"], direction=frame.info["direction"]
            )
            from dataclasses import fields

            prompt = replace(
                prompt,
                **{
                    f.name: getattr(prompt, f.name).replace(
                        "{observation}", observation
                    )
                    for f in fields(prompt)
                    if isinstance(getattr(prompt, f.name), str)
                },
            )
        elif self.name == "osworld":
            from .osworld.prompts import bind

            prompt = bind(prompt, frame.info["instruction"])
        return prompt

    def validate_action(self, action):
        if self.actions:
            return isinstance(action, str) and action in self.actions
        if self.name != "osworld" or not isinstance(action, str) or not action.strip():
            return False
        if action in ("WAIT", "DONE", "FAIL"):
            return True
        if action.upper() == "NONE":
            return False
        try:
            ast.parse(action)
            return True
        except (SyntaxError, ValueError):
            return False

    def parse_action(self, raw):
        values = re.findall(r"<action>(.*?)</action>", raw or "", re.S)
        if len(values) != 1:
            return None
        value = values[0].strip()
        if self.actions:
            value = value.lower()
        return value if self.validate_action(value) else None


def _sokoban():
    from .sokoban import assertion
    from .sokoban.prompts.conpact import CONPACT
    from .sokoban.prompts.react import PROMPTS as react
    from .sokoban.prompts.plan_act import PROMPTS as plan_act
    from .sokoban.prompts.plan_act_assertion import PROMPTS as asserted

    modules = {
        "hiplan_no_retrieval": "hiplan_no_retrieval",
        "tape": "tape",
        "paa_static": "paa",
        "ecot_plan_act": "ecot_plan_act",
    }
    prompts = {
        "react": react,
        "plan_act": plan_act,
        "plan_act_assertion": asserted,
        "conpact": CONPACT,
    }
    for arm, module in modules.items():
        prompts[arm] = importlib.import_module(
            f"{__package__}.sokoban.prompts.{module}"
        ).PROMPTS
    assertion.slots = assertion.SLOTS
    return prompts, assertion, ("up", "down", "left", "right"), 30, 60


def get(name: str, task: Task | None = None) -> Environment:
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {name}")
    payload = task.payload if task else {}
    variant = ""
    if name == "sokoban":
        prompts, schema, actions, steps, calls = _sokoban()
    elif name == "crafter":
        from .crafter.assertion import SCHEMA as schema
        from .crafter.constants import ACTIONS as actions
        from .crafter.prompts.react import REACT
        from .crafter.prompts.plan_act import PLAN_ACT, PLAN_ACT_ASSERTION
        from .crafter.prompts.conpact import CONPACT

        prompts = {
            "react": REACT,
            "plan_act": PLAN_ACT,
            "plan_act_assertion": PLAN_ACT_ASSERTION,
            "conpact": CONPACT,
        }
        for arm, module in [
            ("hiplan_no_retrieval", "hiplan_no_retrieval"),
            ("tape", "tape"),
            ("paa_static", "paa"),
            ("ecot_plan_act", "ecot_plan_act"),
        ]:
            prompts[arm] = importlib.import_module(
                f"{__package__}.crafter.prompts.{module}"
            ).PROMPTS
        steps, calls = 80, 160
    elif name == "procgen":
        from .procgen.assertion import ProcgenAssertionSchema
        from .procgen.actions import valid_actions
        from .procgen.games import resolve_game

        variant = resolve_game(payload)
        schema, actions = ProcgenAssertionSchema(variant), valid_actions(variant)
        prompts = {}
        for arm, module in [
            ("react", "react"),
            ("plan_act", "plan_act"),
            ("conpact", "conpact"),
            ("hiplan_no_retrieval", "hiplan_no_retrieval"),
            ("tape", "tape"),
            ("paa_static", "paa"),
            ("ecot_plan_act", "ecot_plan_act"),
        ]:
            prompts[arm] = importlib.import_module(
                f"{__package__}.procgen.prompts.{module}"
            ).build(variant)
        steps, calls = 80, 160
    elif name == "minigrid":
        from .minigrid.assertion import SCHEMA as schema
        from .minigrid.profile import get_profile, resolve_env_name

        variant = resolve_env_name(payload)
        profile = get_profile(variant)
        actions, steps, calls = (
            profile.actions,
            profile.max_steps,
            profile.max_total_calls,
        )
        prompts = {}
        for arm, module in [
            ("react", "react"),
            ("plan_act", "plan_act"),
            ("conpact", "conpact"),
            ("hiplan_no_retrieval", "hiplan_no_retrieval"),
            ("tape", "tape"),
            ("paa_static", "paa"),
            ("ecot_plan_act", "ecot_plan_act"),
        ]:
            prompts[arm] = importlib.import_module(
                f"{__package__}.minigrid.prompts.{module}"
            ).build(variant)
        from .minigrid.prompts.plan_act import build

        prompts["plan_act_assertion"] = build(variant, assertion=True)
    else:
        from .osworld.assertion import SCHEMA as schema
        from .osworld.prompts import PROMPTS as prompts

        actions, steps, calls = (), 15, 30
    if "plan_act_assertion" not in prompts:
        block = (
            "\n# State Assertion\n"
            + schema.documentation()
            + "\nBegin with <assertion>JSON</assertion>, then use the usual output format.\n"
        )
        base = prompts["plan_act"]
        prompts = dict(
            prompts,
            plan_act_assertion=replace(
                base,
                planner_system=base.planner_system + block,
                actor_system=base.actor_system + block,
            ),
        )
    return Environment(
        name,
        variant,
        tuple(actions),
        schema,
        {method: prompts[arm] for method, arm in ARMS.items() if arm in prompts},
        steps,
        calls,
    )
