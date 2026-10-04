from dataclasses import replace
import re

from ...sokoban.prompts.react import PROMPTS as REACT
from ...sokoban.prompts.plan_act_assertion import PROMPTS as PLAN_ACT_ASSERTION
from .common import adapt_prompt_set, section


def build(env_name: str, *, assertion: bool = False):
    source = REACT
    if assertion:
        state = section(PLAN_ACT_ASSERTION.actor_system, "State Assertion")
        output = section(PLAN_ACT_ASSERTION.actor_system, "Output Format")
        output = output.replace("EXACTLY THREE blocks", "EXACTLY TWO blocks")
        output = re.sub(
            r"\n3\) A <status>.*?\n\nExample:", "\n\nExample:", output, flags=re.S
        )
        output = output.replace("\n<status>CONTINUE</status>", "")
        source = replace(
            REACT,
            actor_system=REACT.actor_system.replace(
                section(REACT.actor_system, "Output Format"), state + output, 1
            ),
        )
    return adapt_prompt_set(source, env_name)
