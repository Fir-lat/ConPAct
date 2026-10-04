from dataclasses import replace
from .conpact_protocol import PROMPTS as BASE


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("baseline prompt changed; review the prompt replacement")
    return text.replace(old, new)


LISTING = {
    "planner": "Output three blocks in this order:",
    "actor": "Output four blocks in this order:",
}


planner = replace_once(
    BASE.planner_system,
    "If a consequential fact remains unclear, ask a focused question rather than treating it as established.",
    "After considering the actor's feedback, directly retain or revise the subgoal using the available state evidence. Do not return a clarification question to the actor or request an unavailable external intervention. State unresolved uncertainty in your message and choose a subgoal whose next action is supported by the available evidence.",
)
planner = replace_once(
    planner,
    "The current short-term objective, or NONE when clarification is needed before proposing one.",
    "The current short-term objective.",
)
planner = replace_once(
    planner,
    "Every non-NONE subgoal replaces the previous one, including when you restate it. NONE pauses execution for clarification. Do not output an action.",
    "Every subgoal replaces the previous one, including when you restate it. Always propose a subgoal rather than returning NONE. Do not output an action.",
)
DECISIVE = replace(BASE, planner_system=planner, discussion_direct=True)

actor = replace_once(
    BASE.actor_system,
    "Briefly explain why it is supported and what observable result you expect.",
    "For an ordinary continuation, keep the message to a short phrase or NONE. Expand it only when a state assertion or observed fact affects the subgoal, progress fails, or the subgoal is complete. You do not need to narrate routine actions or predict their outcomes.",
)
actor = replace_once(
    actor,
    "After an action, compare the new observation with your previous expectation. If it differs in a way that undermines the subgoal, report the discrepancy before acting again; either the earlier interpretation or the new reading may be wrong.",
    "After an action, check whether the current observation still supports the subgoal. Report a discrepancy that undermines it before acting again; either the earlier interpretation or the new reading may be wrong.",
)
actor = replace_once(
    actor,
    "Your brief explanation, expected result, or feedback to the planner, as relevant.",
    "A short phrase or NONE for routine execution; explain state-related feedback when needed.",
)
BRIEF = replace(BASE, actor_system=actor)


COMBINED = replace(
    BASE,
    planner_system=DECISIVE.planner_system,
    actor_system=BRIEF.actor_system,
    discussion_direct=True,
    discussion_context="plain",
)


CONPACT = replace(COMBINED, discussion_scope="observation")
