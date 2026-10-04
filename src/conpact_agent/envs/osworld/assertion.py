import json
import re
from ...comparison import compare_assertions

VERSION = "osworld_visual_assertion_v1"
TAG = "assertion"
SLOTS = ("active_app", "overlay", "target_visible", "goal_state")
DOMAINS = {
    "active_app": (
        "chrome",
        "writer",
        "calc",
        "impress",
        "gimp",
        "vscode",
        "thunderbird",
        "vlc",
        "files",
        "terminal",
        "settings",
        "desktop",
        "other",
        "unknown",
    ),
    "overlay": ("none", "menu", "dialog", "unknown"),
    "target_visible": ("yes", "partial", "no", "unknown"),
    "goal_state": ("met", "unmet", "unknown"),
}


def definition():
    return {
        "version": VERSION,
        "slots": list(SLOTS),
        "domains": {key: list(values) for key, values in DOMAINS.items()},
        "unknown": "unknown",
        "scope": "current screenshot and original task",
    }


def validate(value):
    return (
        isinstance(value, dict)
        and set(value) == set(SLOTS)
        and all(
            isinstance(value[key], str) and value[key] in DOMAINS[key] for key in SLOTS
        )
    )


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate assertion key: " + key)
        result[key] = value
    return result


def parse(raw):
    blocks = re.findall(r"<assertion>(.*?)</assertion>", raw or "", re.S)
    if len(blocks) != 1:
        return None
    try:
        value = json.loads(blocks[0], object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return None
    return value if validate(value) else None


def compare(a, b):
    if not validate(a) or not validate(b):
        raise ValueError("comparison requires two valid OSWorld assertions")
    return compare_assertions(SCHEMA, a, b)


def equal(a, b):
    return compare(a, b).equality


def documentation():
    return """A state assertion is how you commit to what you see before you decide. It has exactly four slots:
- active_app: the foreground application. Choose chrome, writer, calc, impress, gimp, vscode, thunderbird, vlc, files, terminal, settings, desktop, other, or unknown. For an application-owned dialog, report its owning application if identifiable. "other" means an identifiable application outside this list.
- overlay: none, menu, dialog, or unknown. Choose dialog if a dialog is open, otherwise menu if a menu is open, otherwise none. Count a file picker as a dialog, not as the files application.
- target_visible: yes, partial, no, or unknown. The target is what the ORIGINAL TASK asks to modify, create, or inspect, not the planner's current subgoal or the next button to click. Choose yes if all target objects or regions are visible, partial if only some or part are visible, no if none are visible, and unknown if you cannot tell.
- goal_state: met, unmet, or unknown. Choose unmet when the screenshot clearly shows at least one task requirement is not satisfied. Choose met only when the screenshot alone establishes that all task requirements are satisfied. Otherwise choose unknown.
Read only the CURRENT screenshot, using the original task to identify the target and requirements. Do not fill missing observations from history or predicted action results. A completed input field, closed dialog, or previously issued action does not by itself prove that a file was saved or that off-screen content is correct.
Every value must be one of the listed strings. Use unknown for an unreadable or unobservable value; it is a valid answer. goal_state is your visual reading, not the official task score.
Example format (an uncertain observation, not an answer to copy):
{"active_app":"unknown","overlay":"unknown","target_visible":"unknown","goal_state":"unknown"}"""


class OSWorldAssertionSchema:
    tag = TAG
    slots = SLOTS
    unknown_values = {key: ("unknown",) for key in SLOTS}
    version = VERSION
    correctness_available = False
    correctness_reason = "No screenshot-level truth oracle; task evaluator scores do not validate assertions."
    parse = staticmethod(parse)
    equal = staticmethod(equal)
    compare = staticmethod(compare)
    validate = staticmethod(validate)
    definition = staticmethod(definition)
    documentation = staticmethod(documentation)


SCHEMA = OSWorldAssertionSchema()
